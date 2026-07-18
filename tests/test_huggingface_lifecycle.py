import copy
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from flightrecorder.huggingface_lifecycle import (
    EXACT_DEPENDENCIES,
    HuggingFaceLifecycleError,
    attach_identity,
    attach_runtime_identity,
    canonical_sha256,
    build_job_completion,
    build_publication_receipt,
    build_reviewed_training_plan,
    file_record,
    validate_job_completion,
    validate_publication_receipt,
)
from scripts.import_huggingface_job_completion import verify_hub_snapshot_identity
from flightrecorder.cloud_training import (
    build_cloud_training_artifact_manifest,
    build_cloud_training_launch_plan,
    build_cloud_training_launch_receipt,
    build_cloud_training_preflight,
    build_cloud_training_status_receipt,
)
from flightrecorder.schema_registry import check_schema_contract
from flightrecorder.validation import (
    validate_cloud_training_launch_plan,
    validate_cloud_training_launch_receipt,
    validate_cloud_training_status_receipt,
)


def _commit_huggingface_checkout(root: Path, repo_id: str) -> str:
    commands = (
        ["git", "init", "-q", str(root)],
        ["git", "-C", str(root), "config", "user.name", "HFR Test"],
        ["git", "-C", str(root), "config", "user.email", "hfr@example.test"],
        ["git", "-C", str(root), "add", "."],
        ["git", "-C", str(root), "commit", "-q", "-m", "Record immutable completion"],
        ["git", "-C", str(root), "remote", "add", "origin", f"https://huggingface.co/{repo_id}.git"],
    )
    for command in commands:
        subprocess.run(command, check=True, capture_output=True, text=True)
    return subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


class HuggingFaceLifecycleTests(unittest.TestCase):
    def test_hub_snapshot_path_binds_repo_and_completion_revision(self):
        publication = {"repo_id": "owner/adapter", "repo_type": "model"}
        with tempfile.TemporaryDirectory() as temp_dir:
            valid = Path(temp_dir) / "checkout"
            valid.mkdir(parents=True)
            (valid / "artifact.txt").write_text("immutable\n", encoding="utf-8")
            revision = _commit_huggingface_checkout(valid, "owner/adapter")
            evidence = verify_hub_snapshot_identity(valid.resolve(), revision, publication)
            self.assertTrue(evidence["git_commit_verified"])
            with self.assertRaisesRegex(HuggingFaceLifecycleError, "checked-out Git commit"):
                verify_hub_snapshot_identity(valid.resolve(), "b" * 40, publication)

    def _artifacts(self, root: Path, reviewed_plan: dict) -> list[dict]:
        artifacts = []
        for role, name in (
            ("training_plan", "reviewed_training_plan.json"),
            ("trainer_log", "trainer.stdout.log"),
            ("trainer_error_log", "trainer.stderr.log"),
            ("training_metrics", "training_metrics.json"),
            ("adapter_manifest", "adapter_manifest.json"),
            ("adapter", "adapter_model.safetensors"),
            ("training_result", "training_result.json"),
            ("trainer_argv", "trainer_argv.json"),
        ):
            path = root / name
            if role == "training_plan":
                path.write_text(json.dumps(reviewed_plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            else:
                path.write_text(f"{role}\n", encoding="utf-8")
            artifacts.append(file_record(path, base_dir=root, role=role))
        return artifacts

    def _runtime(self, reviewed_plan: dict | None = None) -> dict:
        runtime = {
            "context": "training_job",
            "observed_at": "2026-07-18T00:00:30+00:00",
            "python": "3.11.9",
            "python_implementation": "CPython",
            "platform": "test-linux",
            "dependencies": {
                row.split("==", 1)[0]: {
                    "actual": row.split("==", 1)[1],
                    "measurement_source": "importlib.metadata",
                }
                for row in EXACT_DEPENDENCIES
            },
            "container": {
                "measurement_source": "runtime_probe",
                "detected": True,
                "isolation": "container",
                "marker_paths": ["/.dockerenv"],
                "cgroup_sha256": "6" * 64,
                "os_release_sha256": "7" * 64,
                "image_digest": "sha256:" + "a" * 64,
                "image_digest_source": "environment:HF_JOB_IMAGE_DIGEST",
            },
            "hardware": {
                "measurement_source": "runtime_probe",
                "machine": "x86_64",
                "processor": "test-cpu",
                "logical_cpu_count": 8,
                "accelerator_kind": "cuda",
                "accelerator_count": 1,
            },
            "cuda": {
                "available": True,
                "runtime_version": "12.4",
                "driver_version": "550.54.15",
                "cudnn_version": "9.1.0",
                "device_count": 1,
                "devices": [
                    {
                        "index": 0,
                        "name": "NVIDIA A10G",
                        "compute_capability": "8.6",
                    }
                ],
            },
            "measured_artifacts": {"trainer_script_sha256": "c" * 64},
        }
        if reviewed_plan is not None:
            planned_runtime = reviewed_plan["runtime"]
            runtime["container"]["image_digest"] = planned_runtime["container_image"].rsplit("@", 1)[-1]
            runtime["measured_artifacts"]["trainer_script_sha256"] = planned_runtime["trainer_script_sha256"]
        runtime["library_identity_sha256"] = canonical_sha256(runtime["dependencies"])
        runtime["cuda_identity_sha256"] = canonical_sha256(runtime["cuda"])
        runtime["container_identity_sha256"] = canonical_sha256(runtime["container"])
        runtime["hardware_identity_sha256"] = canonical_sha256(runtime["hardware"])
        runtime["runtime_identity_sha256"] = canonical_sha256(
            {
                "context": runtime["context"],
                "python": runtime["python"],
                "python_implementation": runtime["python_implementation"],
                "platform": runtime["platform"],
                "library_identity_sha256": runtime["library_identity_sha256"],
                "cuda_identity_sha256": runtime["cuda_identity_sha256"],
                "container_identity_sha256": runtime["container_identity_sha256"],
                "hardware_identity_sha256": runtime["hardware_identity_sha256"],
                "measured_artifacts": runtime["measured_artifacts"],
            }
        )
        return runtime

    def _reviewed_plan(self, root: Path) -> dict:
        root.mkdir(parents=True, exist_ok=True)
        source_plan_path = root / "source_training_plan.json"
        model_manifest_path = root / "model_manifest.json"
        dataset_manifest_path = root / "dataset_manifest.json"
        trainer_path = root / "trainer.py"
        data_path = root / "training.jsonl"
        source_plan = {
            "schema_version": "hfr.agentic_lora_training_plan.v1",
            "passed": True,
            "recommendation": "launch_allowed",
            "mode": "fr_sft_dpo",
            "model": "test/base-model",
            "smoke": {"row_limit": 0},
            "tracking": {
                "report_to": ["trackio"],
                "trackio_project": "hfr-test",
                "trackio_space_id": "",
                "run_name_prefix": "test",
            },
            "hyperparameters": {
                "sft_epochs": 1.0,
                "dpo_epochs": 1.0,
                "sft_learning_rate": 0.0001,
                "dpo_learning_rate": 0.00001,
                "batch_size": 1,
                "gradient_accumulation_steps": 1,
                "gradient_checkpointing": True,
                "max_steps": 1,
                "max_length": 128,
                "lora_r": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.05,
                "assistant_only_loss": True,
                "seed": 42,
                "data_seed": 42,
                "save_steps": 1,
                "save_total_limit": 1,
            },
        }
        model_manifest = {
            "model_id": "test/base-model",
            "source": {"revision": "3" * 40},
            "compatibility": {
                "tokenizer": {"repo_id": "test/base-model", "revision": "4" * 40},
                "chat_template": {"sha256": "5" * 64},
            },
        }
        source_plan_path.write_text(json.dumps(source_plan), encoding="utf-8")
        model_manifest_path.write_text(json.dumps(model_manifest), encoding="utf-8")
        dataset_manifest_path.write_text("{}\n", encoding="utf-8")
        trainer_path.write_text("pass\n", encoding="utf-8")
        data_path.write_text("{}\n", encoding="utf-8")
        return build_reviewed_training_plan(
            source_plan=source_plan,
            source_plan_record=file_record(source_plan_path, base_dir=root, role="source_training_plan"),
            model_manifest=model_manifest,
            model_manifest_record=file_record(model_manifest_path, base_dir=root, role="model_manifest"),
            dataset_manifest_record=file_record(dataset_manifest_path, base_dir=root, role="dataset_manifest"),
            trainer_script_record=file_record(trainer_path, base_dir=root, role="trainer_script"),
            data_artifacts=[file_record(data_path, base_dir=root, role="training_data")],
            reviewer="test-reviewer",
            reviewed_at="2026-07-18T00:00:00+00:00",
            dataset_repo="test-user/private-data",
            model_repo="test-user/private-model",
            flavor="a10g-large",
            timeout="4h",
            container_image="registry.example/hfr@sha256:" + "a" * 64,
            created_at="2026-07-18T00:00:00+00:00",
        )

    def test_completed_receipt_binds_private_publication_and_all_artifact_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reviewed_plan = self._reviewed_plan(root)
            artifacts = self._artifacts(root, reviewed_plan)
            runtime = self._runtime(reviewed_plan)
            publication = build_publication_receipt(
                repo_id="test-user/private-adapter",
                repo_type="model",
                revision="d" * 40,
                artifacts=artifacts,
                private=True,
                source_plan_sha256=reviewed_plan["identity"]["sha256"],
                reviewed_plan_record=artifacts[0],
                runtime_observation=runtime,
                base_dir=root,
                created_at="2026-07-18T00:00:00+00:00",
            )
            completion = build_job_completion(
                status="completed",
                plan_record=artifacts[0],
                dataset_revision="f" * 40,
                job_id="job-123",
                exit_code=0,
                artifacts=artifacts,
                publication_receipt=publication,
                runtime_observation=runtime,
                resume={"enabled": False, "available_after_run": False, "artifact_revision": "d" * 40},
                created_at="2026-07-18T00:01:00+00:00",
                base_dir=root,
            )

            self.assertEqual(validate_publication_receipt(publication, root), [])
            self.assertEqual(validate_job_completion(completion, root), [])
            for schema_name, payload in (
                ("huggingface_publication_receipt", publication),
                ("huggingface_job_completion", completion),
            ):
                checked = check_schema_contract(payload, name_or_id=schema_name)
                self.assertTrue(checked["passed"], checked["errors"])

            (root / "training_metrics.json").write_text("drift\n", encoding="utf-8")
            self.assertTrue(any("fingerprint is stale" in error for error in validate_job_completion(completion, root)))

    def test_completion_requires_observed_cuda_driver_runtime_and_library_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reviewed_plan = self._reviewed_plan(root)
            artifacts = self._artifacts(root, reviewed_plan)
            runtime = self._runtime(reviewed_plan)
            publication = build_publication_receipt(
                repo_id="test-user/private-adapter",
                repo_type="model",
                revision="d" * 40,
                artifacts=artifacts,
                private=True,
                source_plan_sha256=reviewed_plan["identity"]["sha256"],
                reviewed_plan_record=artifacts[0],
                runtime_observation=runtime,
                base_dir=root,
            )
            completion = build_job_completion(
                status="completed",
                plan_record=artifacts[0],
                dataset_revision="f" * 40,
                job_id="job-runtime",
                exit_code=0,
                artifacts=artifacts,
                publication_receipt=publication,
                runtime_observation=runtime,
                resume={"enabled": False},
                base_dir=root,
            )

            for field, expected_error in (
                ("driver_version", "runtime_observation.cuda.driver_version must be non-empty"),
                ("runtime_version", "runtime_observation.cuda.runtime_version must be non-empty"),
                ("cudnn_version", "runtime_observation.cuda.cudnn_version must be non-empty"),
            ):
                drifted = copy.deepcopy(completion)
                drifted["runtime_observation"]["cuda"].pop(field)
                drifted = attach_identity(drifted)
                self.assertIn(expected_error, validate_job_completion(drifted, root))

            publication_without_driver = copy.deepcopy(publication)
            publication_without_driver["runtime_observation"]["cuda"].pop("driver_version")
            publication_without_driver = attach_identity(publication_without_driver)
            self.assertIn(
                "runtime_observation.cuda.driver_version must be non-empty",
                validate_publication_receipt(publication_without_driver, root),
            )

            drifted = copy.deepcopy(completion)
            drifted["runtime_observation"]["dependencies"]["torch"]["actual"] = "unknown"
            drifted = attach_identity(drifted)
            errors = validate_job_completion(drifted, root)
            self.assertIn("runtime_observation.library_identity_sha256 does not match dependencies", errors)
            self.assertIn("runtime_observation dependency drift: torch", errors)

            forged = copy.deepcopy(completion)
            forged["runtime_observation"]["hardware_flavor"] = reviewed_plan["runtime"]["hardware_flavor"]
            forged = attach_identity(forged)
            self.assertIn(
                "runtime_observation fields do not match the measured-runtime contract",
                validate_job_completion(forged, root),
            )

            declaration_drift = copy.deepcopy(completion)
            declaration_drift["runtime_declaration"]["hardware_flavor"] = "t4-small"
            declaration_drift["runtime_declaration"]["declaration_identity_sha256"] = canonical_sha256(
                {
                    key: value
                    for key, value in declaration_drift["runtime_declaration"].items()
                    if key != "declaration_identity_sha256"
                }
            )
            declaration_drift = attach_identity(declaration_drift)
            self.assertIn(
                "runtime_declaration must exactly match the reviewed plan",
                validate_job_completion(declaration_drift, root),
            )

            measured_digest_drift = copy.deepcopy(completion)
            observation = measured_digest_drift["runtime_observation"]
            observation["container"]["image_digest"] = "sha256:" + "e" * 64
            measured_digest_drift["runtime_observation"] = attach_runtime_identity(observation)
            measured_digest_drift = attach_identity(measured_digest_drift)
            self.assertIn(
                "runtime_observation container image digest conflicts with the reviewed declaration",
                validate_job_completion(measured_digest_drift, root),
            )

    def test_plan_path_hash_replay_is_mandatory_without_base_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reviewed_plan = self._reviewed_plan(root)
            artifacts = self._artifacts(root, reviewed_plan)
            runtime = self._runtime(reviewed_plan)
            publication = build_publication_receipt(
                repo_id="test-user/private-adapter",
                repo_type="model",
                revision="d" * 40,
                artifacts=artifacts,
                private=True,
                source_plan_sha256=reviewed_plan["identity"]["sha256"],
                reviewed_plan_record=artifacts[0],
                runtime_observation=runtime,
                base_dir=root,
            )
            completion = build_job_completion(
                status="completed",
                plan_record=artifacts[0],
                dataset_revision="f" * 40,
                job_id="job-plan-binding",
                exit_code=0,
                artifacts=artifacts,
                publication_receipt=publication,
                runtime_observation=runtime,
                resume={"enabled": False},
                base_dir=root,
            )

            self.assertIn(
                "reviewed_plan path/hash binding requires base_dir",
                validate_job_completion(completion),
            )
            self.assertIn(
                "reviewed_plan path/hash binding requires base_dir",
                validate_publication_receipt(publication),
            )
            self.assertEqual(validate_job_completion(completion, root), [])
            (root / "reviewed_training_plan.json").write_text("{}\n", encoding="utf-8")
            errors = validate_job_completion(completion, root)
            self.assertIn("reviewed_plan fingerprint is stale", errors)

    def test_completed_receipt_rejects_manifest_without_adapter_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reviewed_plan = self._reviewed_plan(root)
            artifacts = [row for row in self._artifacts(root, reviewed_plan) if row["role"] != "adapter"]
            runtime = self._runtime(reviewed_plan)
            publication = build_publication_receipt(
                repo_id="test-user/private-adapter",
                repo_type="model",
                revision="d" * 40,
                artifacts=artifacts,
                private=True,
                source_plan_sha256=reviewed_plan["identity"]["sha256"],
                reviewed_plan_record=artifacts[0],
                runtime_observation=runtime,
                base_dir=root,
            )
            with self.assertRaisesRegex(ValueError, "durable adapter"):
                build_job_completion(
                    status="completed",
                    plan_record=artifacts[0],
                    dataset_revision="f" * 40,
                    job_id="job-123",
                    exit_code=0,
                    artifacts=artifacts,
                    publication_receipt=publication,
                    runtime_observation=runtime,
                    resume={"enabled": False},
                    base_dir=root,
                )

    def test_failed_and_interrupted_receipts_preserve_resume_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reviewed_plan = self._reviewed_plan(root)
            artifacts = self._artifacts(root, reviewed_plan)
            runtime = self._runtime(reviewed_plan)
            publication = build_publication_receipt(
                repo_id="test-user/private-adapter",
                repo_type="model",
                revision="d" * 40,
                artifacts=artifacts,
                private=True,
                source_plan_sha256=reviewed_plan["identity"]["sha256"],
                reviewed_plan_record=artifacts[0],
                runtime_observation=runtime,
                base_dir=root,
            )
            resume = {
                "enabled": True,
                "phase": "dpo",
                "revision": "1" * 40,
                "checkpoint_path": "lifecycle/checkpoint-100",
                "checkpoint_manifest_sha256": "2" * 64,
                "available_after_run": True,
                "artifact_revision": "1" * 40,
            }
            for status, exit_code in (("failed", 1), ("interrupted", 130)):
                completion = build_job_completion(
                    status=status,
                    plan_record=artifacts[0],
                    dataset_revision="f" * 40,
                    job_id=f"job-{status}",
                    exit_code=exit_code,
                    artifacts=artifacts,
                    publication_receipt=publication,
                    runtime_observation=runtime,
                    resume=resume,
                    failure={"class": status, "message": f"job {status}", "retryable": True},
                    base_dir=root,
                )
                self.assertEqual(validate_job_completion(completion, root), [])
                self.assertTrue(completion["resume"]["available_after_run"])

            drifted = copy.deepcopy(completion)
            drifted["resume"]["revision"] = "main"
            self.assertIn("resume.revision must be immutable", validate_job_completion(drifted, root))

    def test_public_repository_receipt_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reviewed_plan = self._reviewed_plan(root)
            artifacts = self._artifacts(root, reviewed_plan)
            with self.assertRaisesRegex(ValueError, "private repository"):
                build_publication_receipt(
                    repo_id="test-user/public-adapter",
                    repo_type="model",
                    revision="d" * 40,
                    artifacts=artifacts,
                    private=False,
                    source_plan_sha256=reviewed_plan["identity"]["sha256"],
                    reviewed_plan_record=artifacts[0],
                    runtime_observation=self._runtime(reviewed_plan),
                    base_dir=root,
                )

    def test_offline_import_emits_canonical_training_and_cloud_completion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            canonical = root / "canonical"
            canonical.mkdir()
            example = Path(__file__).resolve().parents[1] / "examples" / "agentic_training"
            plan = canonical / "source_plan.json"
            trainer_preflight = canonical / "trainer_preflight.json"
            launch_check = canonical / "trainer_launch_check.json"
            shutil.copy2(example / "plans" / "sft_then_dpo_plan.json", plan)
            shutil.copy2(example / "trainer_preflight.json", trainer_preflight)
            shutil.copy2(example / "trainer_launch_check.json", launch_check)
            upload = canonical / "reviewed_payload.json"
            upload.write_text("{}\n", encoding="utf-8")

            preflight_path = canonical / "preflight.json"
            preflight_path.write_text(
                json.dumps(
                    build_cloud_training_preflight(
                        provider_id="huggingface_jobs",
                        agentic_training_plan_path=plan,
                        trainer_preflight_path=trainer_preflight,
                        trainer_launch_check_path=launch_check,
                        region="provider_default",
                        gpu_class="a10g",
                        max_cost_usd=10,
                        output_base_dir=canonical,
                        created_at="2026-07-18T00:00:00+00:00",
                    ),
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            artifact_manifest_path = canonical / "artifact_manifest.json"
            artifact_manifest_path.write_text(
                json.dumps(
                    build_cloud_training_artifact_manifest(
                        provider_id="huggingface_jobs",
                        upload_paths=[upload],
                        expected_downloads=["completion_receipt.json"],
                        output_base_dir=canonical,
                        created_at="2026-07-18T00:00:00+00:00",
                    ),
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            launch_plan_path = canonical / "launch_plan.json"
            launch_plan_path.write_text(
                json.dumps(
                    build_cloud_training_launch_plan(
                        preflight_path=preflight_path,
                        artifact_manifest_path=artifact_manifest_path,
                        output_base_dir=canonical,
                        created_at="2026-07-18T00:00:00+00:00",
                    ),
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            launch_receipt_path = canonical / "launch_receipt.json"
            launch_receipt_path.write_text(
                json.dumps(
                    build_cloud_training_launch_receipt(
                        launch_plan_path=launch_plan_path,
                        output_base_dir=canonical,
                        created_at="2026-07-18T00:00:00+00:00",
                    ),
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            status_path = canonical / "status_receipt.json"
            status_path.write_text(
                json.dumps(
                    build_cloud_training_status_receipt(
                        launch_receipt_path=launch_receipt_path,
                        output_base_dir=canonical,
                        created_at="2026-07-18T00:00:00+00:00",
                    ),
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            for path, validator in (
                (launch_plan_path, validate_cloud_training_launch_plan),
                (launch_receipt_path, validate_cloud_training_launch_receipt),
                (status_path, validate_cloud_training_status_receipt),
            ):
                validation = validator(path)
                self.assertEqual(validation.errors + validation.warnings, [], (path, validation.errors, validation.warnings))

            snapshot = root / "snapshot"
            lifecycle = snapshot / "lifecycle" / "run"
            lifecycle.mkdir(parents=True)
            reviewed_plan = self._reviewed_plan(root / "reviewed_sources")
            artifacts = self._artifacts(lifecycle, reviewed_plan)
            for row in artifacts:
                row["path"] = "lifecycle/run/" + row["path"]
            runtime = self._runtime(reviewed_plan)
            publication = build_publication_receipt(
                repo_id="test-user/private-adapter",
                repo_type="model",
                revision="d" * 40,
                artifacts=artifacts,
                private=True,
                source_plan_sha256=reviewed_plan["identity"]["sha256"],
                reviewed_plan_record=artifacts[0],
                runtime_observation=runtime,
                base_dir=snapshot,
                created_at="2026-07-18T00:00:00+00:00",
            )
            completion = build_job_completion(
                status="completed",
                plan_record=artifacts[0],
                dataset_revision="f" * 40,
                job_id="hf-job-123",
                exit_code=0,
                artifacts=artifacts,
                publication_receipt=publication,
                runtime_observation=runtime,
                resume={"enabled": False, "available_after_run": False, "artifact_revision": "d" * 40},
                created_at="2026-07-18T00:00:00+00:00",
                base_dir=snapshot,
            )
            completion_path = lifecycle / "completion_receipt.json"
            completion_path.write_text(json.dumps(completion, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            completion_revision = _commit_huggingface_checkout(
                snapshot,
                "test-user/private-adapter",
            )
            imported = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve().parents[1] / "scripts" / "import_huggingface_job_completion.py"),
                    "--snapshot-dir",
                    str(snapshot),
                    "--completion-receipt",
                    str(completion_path),
                    "--completion-revision",
                    completion_revision,
                    "--agentic-training-plan",
                    str(example / "plans" / "sft_then_dpo_plan.json"),
                    "--runtime-preflight",
                    str(example / "runtime_preflight" / "ready.json"),
                    "--agentic-training-flow",
                    str(example / "agentic_training_flow.json"),
                    "--cloud-launch-plan",
                    str(launch_plan_path),
                    "--cloud-launch-receipt",
                    str(launch_receipt_path),
                    "--cloud-status-receipt",
                    str(status_path),
                    "--gpu-class",
                    "a10g",
                    "--reported-cost-usd",
                    "1.25",
                    "--out-dir",
                    str(canonical),
                ],
                cwd=Path(__file__).resolve().parents[1],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(imported.returncode, 0, imported.stderr + imported.stdout)
            training_result = json.loads((canonical / "agentic_training_result.json").read_text(encoding="utf-8"))
            cloud_completion = json.loads(
                (canonical / "cloud_training_completion_receipt.json").read_text(encoding="utf-8")
            )
            self.assertEqual(training_result["schema_version"], "hfr.agentic_training_result.v1")
            self.assertTrue(training_result["passed"])
            self.assertEqual(cloud_completion["schema_version"], "hfr.cloud_training_completion_receipt.v1")
            self.assertTrue(cloud_completion["passed"])
            raw = json.loads((canonical / "huggingface_raw_provider_result.json").read_text(encoding="utf-8"))
            self.assertEqual(raw["completion_revision"], completion_revision)
            self.assertTrue(raw["snapshot_provenance"]["git_commit_verified"])
            self.assertEqual(raw["snapshot_provenance"]["repo_id"], "test-user/private-adapter")
            self.assertEqual(raw["snapshot_provenance"]["source"], "huggingface_git_checkout")
            import_receipt = json.loads(
                (canonical / "huggingface_canonical_import.json").read_text(encoding="utf-8")
            )
            checked = check_schema_contract(import_receipt, name_or_id="huggingface_canonical_import")
            self.assertTrue(checked["passed"], checked["errors"])
            self.assertEqual(import_receipt["snapshot_provenance"]["revision"], completion_revision)


if __name__ == "__main__":
    unittest.main()
