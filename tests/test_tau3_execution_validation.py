from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

from flightrecorder.tau3_evaluation import analyze_tau3_evaluation
from flightrecorder.tau3_execution_validation import (
    validate_tau3_benchmark_result_bundle,
    validate_tau3_training_result_bundle,
)
from flightrecorder.tau3_mlx_training import Tau3MlxTrainingConfig, run_tau3_mlx_training
from tests.test_tau3_mlx_training import _fake_model, _install_fake_python, _mixture_variant, _protocol_config


class Tau3ExecutionValidationTests(unittest.TestCase):
    def test_training_bundle_passes_with_single_locked_updated_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build_execution_bundle(root)

            result = validate_tau3_training_result_bundle(root, strict=True)

            self.assertTrue(result["passed"], result)
            self.assertEqual(result["schema_version"], "hfr.validation.v1")

    def test_training_bundle_passes_with_relocated_real_mlx_runner_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runner_root = root / "runner"
            bundle_root = root / "bundle"
            runner_root.mkdir()
            bundle_root.mkdir()
            _install_fake_python(runner_root, "success")
            model, identity = _fake_model(runner_root)
            protocol = _protocol_config(runner_root, identity)
            output = runner_root / "out"

            run_tau3_mlx_training(
                mixture_dir=_mixture_variant(runner_root, protocol_path=protocol),
                protocol_path=protocol,
                model_path=model,
                model_identity_path=identity,
                output_dir=output,
                workspace_root=runner_root,
                config=Tau3MlxTrainingConfig(iters=2, timeout_seconds=5),
            )
            mlx_config = read_json(output / "mlx_lora_config.json")
            receipt = read_json(output / "training_receipt.json")
            self.assertEqual(mlx_config["adapter_path"], "adapter")
            self.assertEqual(receipt["adapter"]["path"], "adapter")
            self.assertIn(str((output / "adapter").resolve()), receipt["command"])

            build_execution_bundle(bundle_root)
            candidate_dir = bundle_root / "training" / "candidate-a"
            shutil.rmtree(candidate_dir)
            shutil.copytree(output, candidate_dir)
            protocol_ref = write_hashed_json(bundle_root, "protocol.json", read_json(protocol))
            receipt_ref = {"path": "training/candidate-a/training_receipt.json", "sha256": sha256_file(candidate_dir / "training_receipt.json")}
            manifest = read_json(bundle_root / "manifest.json")
            development_adapter_ref = manifest["benchmark"]["development_arms"][1]
            selection_ref = build_candidate_selection_report(bundle_root, receipt_ref, development_adapter_ref)
            lock_ref = build_candidate_lock(bundle_root, receipt_ref, protocol_ref, development_adapter_ref, selection_ref)
            manifest["protocol"] = protocol_ref
            manifest["training"]["selected_receipt"] = receipt_ref
            manifest["training"]["candidate_receipts"] = [{**receipt_ref, "candidate_id": "candidate-a"}]
            manifest["training"]["candidate_selection_report"] = selection_ref
            manifest["training"]["candidate_locks"] = [lock_ref]
            write_json(bundle_root / "manifest.json", manifest)

            result = validate_tau3_training_result_bundle(bundle_root, strict=True)

            self.assertTrue(result["passed"], result)

    def test_benchmark_bundle_passes_for_development_and_sealed_grid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build_execution_bundle(root)

            result = validate_tau3_benchmark_result_bundle(root, strict=True)

            self.assertTrue(result["passed"], result)

    def test_training_bundle_fails_on_tampered_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build_execution_bundle(root)
            manifest_path = root / "manifest.json"
            manifest = read_json(manifest_path)
            manifest["training"]["selected_receipt"]["sha256"] = "0" * 64
            write_json(manifest_path, manifest)

            result = validate_tau3_training_result_bundle(root, strict=True)

            self.assertFalse(result["passed"])
            self.assertIn("sha256 mismatch", json.dumps(result))

    def test_training_bundle_fails_on_adapter_file_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build_execution_bundle(root)
            (root / "training" / "candidate-a" / "adapter" / "adapter_model.safetensors").write_bytes(b"changed")

            result = validate_tau3_training_result_bundle(root, strict=True)

            self.assertFalse(result["passed"])
            self.assertIn("adapter file sha256 mismatch", json.dumps(result))

    def test_training_bundle_fails_on_telemetry_loss_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build_execution_bundle(root)

            def mutate(receipt: dict[str, Any]) -> None:
                receipt["losses"]["train"] = [1.0]

            update_training_receipt_and_lock(root, mutate)

            result = validate_tau3_training_result_bundle(root, strict=True)

            self.assertFalse(result["passed"])
            self.assertIn("telemetry train losses do not replay", json.dumps(result))

    def test_benchmark_bundle_fails_on_raw_result_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build_execution_bundle(root)
            raw = root / "benchmark" / "sealed" / "adapter" / "results" / "airline" / "seed-101" / "results.json"
            raw.write_text('{"simulations":[{"reward_info":{"reward":0.0}}]}\n', encoding="utf-8")

            result = validate_tau3_benchmark_result_bundle(root, strict=True)

            self.assertFalse(result["passed"])
            self.assertIn("raw result sha256 mismatch", json.dumps(result))

    def test_benchmark_bundle_propagates_training_artifact_failures(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build_execution_bundle(root)
            (root / "training" / "candidate-a" / "adapter" / "adapter_model.safetensors").write_bytes(b"changed")

            result = validate_tau3_benchmark_result_bundle(root, strict=True)

            self.assertFalse(result["passed"])
            self.assertIn("training.candidate_adapter_tree", json.dumps(result))
            self.assertIn("training result validation failed", json.dumps(result))

    def test_benchmark_bundle_fails_when_lock_is_after_sealed_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build_execution_bundle(root)
            update_candidate_lock_time(root, "2026-07-23T02:00:00Z")

            result = validate_tau3_benchmark_result_bundle(root, strict=True)

            self.assertFalse(result["passed"])
            self.assertIn("candidate lock must predate sealed", json.dumps(result))

    def test_benchmark_bundle_rejects_development_comparator_arm(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build_execution_bundle(root)
            manifest_path = root / "manifest.json"
            manifest = read_json(manifest_path)
            manifest["benchmark"]["development_arms"].append(build_arm(root, "development", "comparator_1", manifest["protocol"]["sha256"], None))
            write_json(manifest_path, manifest)

            result = validate_tau3_benchmark_result_bundle(root, strict=True)

            self.assertFalse(result["passed"])
            self.assertIn("development benchmark must not contain comparator arms", json.dumps(result))

    def test_training_bundle_requires_candidate_selection_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build_execution_bundle(root)
            manifest_path = root / "manifest.json"
            manifest = read_json(manifest_path)
            manifest["training"].pop("candidate_selection_report")
            write_json(manifest_path, manifest)

            result = validate_tau3_training_result_bundle(root, strict=True)

            self.assertFalse(result["passed"])
            self.assertIn("candidate_selection_report", json.dumps(result))

    def test_benchmark_bundle_rejects_skeleton_public_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build_execution_bundle(root)
            manifest_path = root / "manifest.json"
            manifest = read_json(manifest_path)
            report_ref = write_hashed_json(root, "public-evaluation-report.json", {"schema_version": "hfr.tau3_public_evaluation_report.v1", "safety": {"passed": True}})
            manifest["benchmark"]["public_report"] = report_ref
            write_json(manifest_path, manifest)

            result = validate_tau3_benchmark_result_bundle(root, strict=True)

            self.assertFalse(result["passed"])
            self.assertIn("must be hfr.tau3_evaluation.v1", json.dumps(result))

    def test_benchmark_bundle_replays_promotion_checks_and_effects(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build_execution_bundle(root)
            manifest_path = root / "manifest.json"
            manifest = read_json(manifest_path)
            report_path = root / manifest["benchmark"]["public_report"]["path"]
            report = read_json(report_path)
            report["effects"]["base"]["estimate"] = 0.5
            write_json(report_path, report)
            manifest["benchmark"]["public_report"]["sha256"] = sha256_file(report_path)
            manifest["benchmark"]["public_report"]["size"] = report_path.stat().st_size
            write_json(manifest_path, manifest)

            result = validate_tau3_benchmark_result_bundle(root, strict=True)

            self.assertFalse(result["passed"])
            self.assertIn("public evaluation effects does not replay", json.dumps(result))

    def test_benchmark_bundle_rejects_substituted_report_source_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build_execution_bundle(root)
            manifest_path = root / "manifest.json"
            manifest = read_json(manifest_path)
            report_path = root / manifest["benchmark"]["public_report"]["path"]
            report = read_json(report_path)
            report["source_artifacts"]["adapter"][0]["sha256"] = "0" * 64
            write_json(report_path, report)
            manifest["benchmark"]["public_report"]["sha256"] = sha256_file(report_path)
            manifest["benchmark"]["public_report"]["size"] = report_path.stat().st_size
            write_json(manifest_path, manifest)

            result = validate_tau3_benchmark_result_bundle(root, strict=True)

            self.assertFalse(result["passed"])
            self.assertIn("must exactly bind sealed raw result hashes", json.dumps(result))

    def test_benchmark_bundle_requires_run_receipt_sha256(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build_execution_bundle(root)
            manifest_path = root / "manifest.json"
            manifest = read_json(manifest_path)
            arm_path = root / manifest["benchmark"]["sealed_arms"][0]["path"]
            arm = read_json(arm_path)
            arm["run_receipts"][0].pop("receipt_sha256")
            write_json(arm_path, arm)
            manifest["benchmark"]["sealed_arms"][0]["sha256"] = sha256_file(arm_path)
            write_json(manifest_path, manifest)

            result = validate_tau3_benchmark_result_bundle(root, strict=True)

            self.assertFalse(result["passed"])
            self.assertIn("missing receipt_sha256", json.dumps(result))

    def test_manifest_rejects_private_absolute_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build_execution_bundle(root)
            manifest_path = root / "manifest.json"
            manifest = read_json(manifest_path)
            manifest["protocol"]["path"] = "/Users/example/local/tau3/protocol.json"
            write_json(manifest_path, manifest)

            result = validate_tau3_training_result_bundle(root, strict=True)

            self.assertFalse(result["passed"])
            self.assertIn("absolute/private paths", json.dumps(result))

    def test_manifest_rejects_root_escape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build_execution_bundle(root)
            manifest_path = root / "manifest.json"
            manifest = read_json(manifest_path)
            manifest["training"]["selected_receipt"]["path"] = "../training_receipt.json"
            write_json(manifest_path, manifest)

            result = validate_tau3_training_result_bundle(root, strict=True)

            self.assertFalse(result["passed"])
            self.assertIn("below bundle root", json.dumps(result))

    def test_manifest_requires_clean_code_attestation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build_execution_bundle(root)
            manifest_path = root / "manifest.json"
            manifest = read_json(manifest_path)
            manifest["code_revision"]["tracked_worktree_clean"] = False
            write_json(manifest_path, manifest)

            result = validate_tau3_training_result_bundle(root, strict=True)

            self.assertFalse(result["passed"])
            self.assertIn("tracked_worktree_clean", json.dumps(result))

    def test_cli_accepts_bundle_and_strict_flags(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build_execution_bundle(root)

            for script in ("scripts/validate_tau3_training_result.py", "scripts/validate_tau3_benchmark_result.py"):
                with self.subTest(script=script):
                    proc = subprocess.run(
                        [sys.executable, script, "--bundle", str(root), "--strict"],
                        cwd=Path(__file__).resolve().parents[1],
                        text=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                    )
                    self.assertEqual(proc.returncode, 0, proc.stderr + proc.stdout)


def build_execution_bundle(root: Path) -> None:
    protocol = {"schema_version": "hfr.tau3_protocol_config.v1", "protocol_manifest": {"signature": "fixture"}}
    protocol_ref = write_hashed_json(root, "protocol.json", protocol)
    receipt_ref = build_training_receipts(root, protocol_ref)
    dev_refs = [
        build_arm(root, "development", "base", protocol_ref["sha256"], None),
        build_arm(root, "development", "adapter", protocol_ref["sha256"], None),
    ]
    selection_ref = build_candidate_selection_report(root, receipt_ref, dev_refs[1])
    lock_ref = build_candidate_lock(root, receipt_ref, protocol_ref, dev_refs[1], selection_ref)
    sealed_refs = [build_arm(root, "sealed", arm, protocol_ref["sha256"], lock_ref) for arm in ARMS]
    report_ref = build_public_report(root, sealed_refs)
    manifest = {
        "schema_version": "hfr.tau3_execution_bundle.v1",
        "code_revision": {"flight_recorder_git_commit": "a" * 40, "tracked_worktree_clean": True},
        "protocol": protocol_ref,
        "training": {
            "selected_candidate_id": "candidate-a",
            "selected_receipt": receipt_ref,
            "candidate_receipts": [{**receipt_ref, "candidate_id": "candidate-a"}],
            "candidate_selection_report": selection_ref,
            "candidate_locks": [lock_ref],
        },
        "benchmark": {
            "development_arms": dev_refs,
            "sealed_arms": sealed_refs,
            "public_report": report_ref,
        },
    }
    write_json(root / "manifest.json", manifest)


def build_training_receipts(root: Path, protocol_ref: dict[str, str]) -> dict[str, str]:
    training_dir = root / "training" / "candidate-a"
    adapter_dir = training_dir / "adapter"
    adapter_dir.mkdir(parents=True)
    adapter_file = adapter_dir / "adapter_model.safetensors"
    adapter_file.write_bytes(b"abc")
    adapter_files = [{"path": "adapter_model.safetensors", "size": adapter_file.stat().st_size, "sha256": sha256_file(adapter_file), "kind": "adapter"}]
    adapter = {"path": "adapter", "file_count": 1, "files": adapter_files, "tree_sha256": tree_sha256(adapter_files)}
    binding = {
        "protocol": {
            "path": "protocol.json",
            "sha256": protocol_ref["sha256"],
            "protocol_signature": "6" * 64,
            "protocol_signature_provenance": {
                "source": "protocol_file_sha256_content_seal",
                "algorithm": "sha256",
            },
        },
        "model": {
            "identity_sha256": "c" * 64,
            "tree_sha256": "d" * 64,
            "model_id": "mlx-community/Qwen3.5-9B-4bit",
            "revision": "8b2b98c00a6b4d291155e4890773ca8f769aee53",
        },
        "dataset": {
            "manifest_sha256": "e" * 64,
            "files_sha256": "f" * 64,
            "source_binding_sha256": "1" * 64,
        },
        "recipe": {"recipe_sha256": "2" * 64, "rank": 16, "iters": 5},
    }
    config = {"iters": 5, "learning_rate": 0.00001, "rank": 16, "seed": 17}
    checks = [{"id": "fixture", "passed": True, "actual": True, "expected": True}]
    mlx_config_ref = write_hashed_json(
        root,
        "training/candidate-a/mlx_lora_config.json",
        {
            "model": "training/base-model",
            "train": True,
            "fine_tune_type": "lora",
            "data": "training/mixture",
            "adapter_path": "adapter",
            "iters": 5,
            "learning_rate": 0.00001,
            "num_layers": 16,
            "batch_size": 1,
            "grad_accumulation_steps": 1,
            "steps_per_report": 1,
            "steps_per_eval": 1,
            "val_batches": -1,
            "save_every": 5,
            "max_seq_length": 8192,
            "seed": 17,
            "mask_prompt": True,
            "grad_checkpoint": True,
            "clear_cache_threshold": 0,
            "report_to": None,
            "test": False,
            "lora_parameters": {"rank": 16, "scale": 20.0, "dropout": 0.0},
        },
    )
    prelaunch = {
        "schema_version": "hfr.tau3_mlx_training_run.v1",
        "phase": "prelaunch",
        "created_at": "2026-07-23T00:00:00Z",
        "bundle": {"kind": "mixture", "path": "training/mixture", "sha256": "3" * 64},
        "output_dir": ".",
        "command": ["mlx_lm.lora"],
        "config": config,
        "mlx_lora_config": {**local_ref("mlx_lora_config.json", mlx_config_ref), "read_only": True},
        "training_binding": binding,
        "checks": checks,
        "weights_updated": False,
        "terminal_status": "prelaunch",
    }
    prelaunch_ref = write_hashed_json(root, "training/candidate-a/prelaunch_receipt.json", prelaunch)
    telemetry_path = root / "training" / "candidate-a" / "telemetry.jsonl"
    telemetry_path.write_text(
        '{"stream":"stdout","text":"train loss 1.0","time":"2026-07-23T00:01:00Z"}\n'
        '{"stream":"stdout","text":"train loss 0.5","time":"2026-07-23T00:02:00Z"}\n',
        encoding="utf-8",
    )
    telemetry_ref = {"path": "telemetry.jsonl", "sha256": sha256_file(telemetry_path), "read_only": True}
    final = {
        "schema_version": "hfr.tau3_mlx_training_run.v1",
        "phase": "final",
        "created_at": "2026-07-23T00:10:00Z",
        "bundle": {"kind": "mixture", "path": "training/mixture", "sha256": "3" * 64},
        "output_dir": ".",
        "prelaunch_receipt": {**local_ref("prelaunch_receipt.json", prelaunch_ref), "read_only": True},
        "telemetry": {**telemetry_ref, "event_count": 2},
        "command": ["mlx_lm.lora"],
        "config": config,
        "mlx_lora_config": {**local_ref("mlx_lora_config.json", mlx_config_ref), "read_only": True},
        "training_binding": binding,
        "checks": checks,
        "terminal_status": "success",
        "exit_code": 0,
        "timed_out": False,
        "interrupted": False,
        "elapsed_seconds": 1.0,
        "peak_child_rss_kb": 1,
        "losses": {"train": [1.0, 0.5], "validation": [], "last_train": 0.5, "last_validation": None},
        "adapter": adapter,
        "adapter_weight_file_count": 1,
        "weights_updated": True,
        "schema_checked": True,
    }
    return write_hashed_json(root, "training/candidate-a/training_receipt.json", final)


def build_candidate_lock(
    root: Path,
    receipt_ref: dict[str, str],
    protocol_ref: dict[str, str],
    development_adapter_ref: dict[str, str],
    selection_ref: dict[str, str],
) -> dict[str, str]:
    receipt = read_json(root / receipt_ref["path"])
    binding = receipt["training_binding"]
    development_adapter = read_json(root / development_adapter_ref["path"])
    identity_ref = development_adapter["candidate_identity"]
    lock = {
        "schema_version": "hfr.tau3_candidate_lock.v1",
        "created_at": "2026-07-23T00:20:00Z",
        "selected_candidate_id_hash": tree_value_sha256("candidate-a"),
        "candidate_identity_sha256": identity_ref["sha256"],
        "development_selection_report_sha256": selection_ref["sha256"],
        "development_benchmark_manifest_sha256": development_adapter_ref["sha256"],
        "training_receipt_sha256": receipt_ref["sha256"],
        "endpoint_model_sha256": "b" * 64,
        "protocol_sha256": protocol_ref["sha256"],
        "adapter_tree_sha256": receipt["adapter"]["tree_sha256"],
        "recipe_sha256": binding["recipe"]["recipe_sha256"],
        "base_identity_sha256": binding["model"]["identity_sha256"],
        "base_tree_sha256": binding["model"]["tree_sha256"],
        "dataset_manifest_sha256": binding["dataset"]["manifest_sha256"],
        "dataset_files_sha256": binding["dataset"]["files_sha256"],
        "source_binding_sha256": binding["dataset"]["source_binding_sha256"],
        "protocol_signature": binding["protocol"]["protocol_signature"],
        "hashes_only": True,
        "sealed_access_authorized": True,
        "local_paths_included": False,
        "raw_payload_included": False,
    }
    return write_hashed_json(root, "candidate-lock.json", lock)


def build_candidate_selection_report(root: Path, receipt_ref: dict[str, str], development_adapter_ref: dict[str, str]) -> dict[str, str]:
    development_adapter = read_json(root / development_adapter_ref["path"])
    identity_ref = development_adapter["candidate_identity"]
    identity = read_json((root / development_adapter_ref["path"]).parent / identity_ref["path"])
    report = {
        "schema_version": "hfr.tau3_candidate_selection.v1",
        "schema_checked": True,
        "created_at": "2026-07-23T00:19:00Z",
        "passed": True,
        "selected_candidate_id": "candidate-a",
        "selection_policy": {
            "primary_metric": "development_macro_pass1",
            "required_domains": list(DOMAINS),
            "bootstrap_samples": 200,
            "bootstrap_seed": 7,
            "confidence_level": 0.95,
            "non_inferiority_margin": 0.03,
            "safety_non_inferiority_margin": 0.01,
        },
        "base": {"arm_id": "base"},
        "candidates": [
            {
                "candidate_id": "candidate-a",
                "eligible": True,
                "metrics": {"macro_pass1": {"candidate": 1.0}},
                "artifacts": {
                    "development_manifest": development_adapter_ref,
                    "training_receipt": receipt_ref,
                },
                "candidate_identity": {
                    "sha256": identity_ref["sha256"],
                    "identity_sha256": tree_value_sha256(identity),
                    "endpoint_model_sha256": identity["endpoint_model_sha256"],
                },
            }
        ],
        "eligible_candidate_count": 1,
        "selection": {
            "candidate_id": "candidate-a",
            "rank": 1,
            "macro_pass1": 1.0,
            "candidate_identity_sha256": identity_ref["sha256"],
            "candidate_identity_canonical_sha256": tree_value_sha256(identity),
        },
    }
    return write_hashed_json(root, "candidate-selection-report.json", report)


def update_training_receipt_and_lock(root: Path, mutate: Any) -> None:
    manifest_path = root / "manifest.json"
    manifest = read_json(manifest_path)
    receipt_path = root / "training" / "candidate-a" / "training_receipt.json"
    receipt = read_json(receipt_path)
    mutate(receipt)
    write_json(receipt_path, receipt)
    receipt_ref = {"path": "training/candidate-a/training_receipt.json", "sha256": sha256_file(receipt_path)}
    manifest["training"]["selected_receipt"] = receipt_ref
    manifest["training"]["candidate_receipts"] = [{**receipt_ref, "candidate_id": "candidate-a"}]
    lock_path = root / "candidate-lock.json"
    lock = read_json(lock_path)
    lock["training_receipt_sha256"] = receipt_ref["sha256"]
    write_json(lock_path, lock)
    manifest["training"]["candidate_locks"] = [{"path": "candidate-lock.json", "sha256": sha256_file(lock_path)}]
    write_json(manifest_path, manifest)


def update_candidate_lock_time(root: Path, created_at: str) -> None:
    manifest_path = root / "manifest.json"
    manifest = read_json(manifest_path)
    lock_path = root / "candidate-lock.json"
    lock = read_json(lock_path)
    lock["created_at"] = created_at
    write_json(lock_path, lock)
    lock_ref = {"path": "candidate-lock.json", "sha256": sha256_file(lock_path)}
    manifest["training"]["candidate_locks"] = [lock_ref]
    for arm_ref in manifest["benchmark"]["sealed_arms"]:
        arm_path = root / arm_ref["path"]
        arm = read_json(arm_path)
        arm_lock_path = arm_path.parent / "candidate-lock.json"
        write_json(arm_lock_path, lock)
        arm_lock_ref = {"path": "candidate-lock.json", "sha256": sha256_file(arm_lock_path)}
        arm["candidate_lock"] = arm_lock_ref
        prelaunch_path = arm_path.parent / arm["prelaunch_receipt"]["path"]
        prelaunch = read_json(prelaunch_path)
        prelaunch["candidate_lock"] = arm_lock_ref
        write_json(prelaunch_path, prelaunch)
        arm["prelaunch_receipt"]["sha256"] = sha256_file(prelaunch_path)
        write_json(arm_path, arm)
        arm_ref["sha256"] = sha256_file(arm_path)
    write_json(manifest_path, manifest)


def build_arm(root: Path, mode: str, arm_id: str, protocol_sha256: str, lock_ref: dict[str, str] | None) -> dict[str, str]:
    arm_dir = root / "benchmark" / mode / arm_id
    arm_dir.mkdir(parents=True)
    protocol_record = write_hashed_json(arm_dir, "protocol.json", read_json(root / "protocol.json"))
    source_ref = write_hashed_json(arm_dir, "source.json", {"split": "development", "rows": []}) if mode == "development" else None
    identity_ref = write_hashed_json(
        arm_dir,
        "candidate-identity.json",
        {
            "candidate_id": "candidate-a",
            "adapter": "adapter",
            "endpoint_model_sha256": "b" * 64,
            "adapter_tree_sha256": selected_adapter_tree(root),
            "training_receipt_sha256": sha256_file(root / "training" / "candidate-a" / "training_receipt.json"),
        },
    ) if mode == "development" and arm_id == "adapter" else None
    arm_lock_ref = None
    lock_payload = None
    if mode == "sealed" and lock_ref is not None:
        lock_payload = read_json(root / lock_ref["path"])
        arm_lock_ref = write_hashed_json(arm_dir, "candidate-lock.json", lock_payload)
    sealed_task_count_ref = None
    if mode == "sealed":
        sealed_task_count_ref = write_hashed_json(arm_dir, "sealed-task-count-manifest.json", sealed_source_manifest())
        sealed_task_count_ref["hashes_only"] = True
        sealed_task_count_ref["task_count"] = 3
    arm_identity = adapter_arm_identity(arm_id, mode, identity_ref, arm_lock_ref, lock_payload) if arm_id == "adapter" else {"arm_id": arm_id, "source": "protocol_model_freeze", "endpoint_model_sha256": "8" * 64}
    run_refs = []
    for domain in DOMAINS:
        for seed in SEEDS:
            raw_path = arm_dir / "results" / domain / f"seed-{seed}" / "results.json"
            write_json(raw_path, tau_raw_result(domain, seed))
            result_sha = sha256_file(raw_path)
            receipt = {
                "schema_version": "hfr.tau3_benchmark_run.v1",
                "phase": "domain_seed",
                "created_at": "2026-07-23T00:30:00Z",
                "protocol_sha256": protocol_sha256,
                "arm_identity": arm_identity,
                "mode": mode,
                "arm_id": arm_id,
                "domain": domain,
                "seed": seed,
                "command": ["tau2", domain],
                "result_path": f"results/{domain}/seed-{seed}/results.json",
                "result_sha256": result_sha,
                "result_summary": result_summary(),
                "exit_code": 0,
                "timed_out": False,
                "duration_seconds": 0.1,
                "terminal_status": "completed",
                "stdout_tail": "",
                "stderr_tail": "",
                "training_started": False,
                "sealed_payload_accessed": False,
                "sealed_task_ids_materialized": False,
            }
            receipt_path = arm_dir / f"run-{domain}-seed{seed}.json"
            write_json(receipt_path, receipt)
            receipt_sha = sha256_file(receipt_path)
            run_refs.append(
                {
                    "path": f"run-{domain}-seed{seed}.json",
                    "receipt_sha256": receipt_sha,
                    "domain": domain,
                    "seed": seed,
                    "terminal_status": "completed",
                    "result_path": f"results/{domain}/seed-{seed}/results.json",
                    "result_sha256": result_sha,
                    "result_summary": result_summary(),
                }
            )
    prelaunch = {
        "schema_version": "hfr.tau3_benchmark_run.v1",
        "phase": "prelaunch",
        "created_at": "2026-07-23T00:25:00Z" if mode == "sealed" else "2026-07-23T00:05:00Z",
        "tau_revision": "a" * 40,
        "protocol": protocol_record,
        "protocol_sha256": protocol_sha256,
        "mode": mode,
        "arm_id": arm_id,
        "arm_identity": arm_identity,
        "agent": endpoint(),
        "user_simulator": endpoint(),
        "reviewer": endpoint(),
        "config": config(),
        "source": source_ref if mode == "development" else None,
        "sealed_task_count_manifest": sealed_task_count_ref if mode == "sealed" else None,
        "candidate_lock": arm_lock_ref if mode == "sealed" else None,
        "candidate_identity": identity_ref,
        "task_selection": task_selection(mode),
        "loopback_only": True,
        "training_started": False,
        "sealed_payload_accessed": False,
        "sealed_task_ids_materialized": False,
    }
    prelaunch_ref = write_hashed_json(root, f"benchmark/{mode}/{arm_id}/prelaunch_receipt.json", prelaunch)
    prelaunch_ref = local_ref("prelaunch_receipt.json", prelaunch_ref)
    manifest = {
        "schema_version": "hfr.tau3_benchmark_run.v1",
        "phase": "final",
        "created_at": "2026-07-23T01:00:00Z",
        "tau_revision": "a" * 40,
        "protocol": protocol_record,
        "protocol_sha256": protocol_sha256,
        "mode": mode,
        "arm_id": arm_id,
        "arm_identity": arm_identity,
        "agent": endpoint(),
        "user_simulator": endpoint(),
        "reviewer": endpoint(),
        "config": config(),
        "source": source_ref if mode == "development" else None,
        "sealed_task_count_manifest": sealed_task_count_ref if mode == "sealed" else None,
        "candidate_lock": arm_lock_ref if mode == "sealed" else None,
        "candidate_identity": identity_ref,
        "task_selection": task_selection(mode),
        "prelaunch_receipt": prelaunch_ref,
        "run_count": 12,
        "success_count": 12,
        "failure_count": 0,
        "run_receipts": run_refs,
        "loopback_only": True,
        "training_started": False,
        "sealed_payload_accessed": False,
        "sealed_task_ids_materialized": False,
    }
    return write_hashed_json(root, f"benchmark/{mode}/{arm_id}/manifest.json", manifest)


def endpoint() -> dict[str, Any]:
    return {
        "context_window": 16384,
        "endpoint_hash": "7" * 64,
        "loopback": True,
        "max_tokens": 1024,
        "model_sha256": "8" * 64,
        "temperature": 0.0,
        "top_p": 1.0,
    }


def config() -> dict[str, Any]:
    return {
        "agent": "llm_agent",
        "auto_review": True,
        "communication_protocol_enforced": True,
        "context_window": 16384,
        "domains": list(DOMAINS),
        "hallucination_retries": 0,
        "max_concurrency": 1,
        "max_errors": 10,
        "max_retries": 0,
        "max_steps": 30,
        "num_trials": 1,
        "resume": False,
        "review_mode": "full",
        "seeds": list(SEEDS),
        "test_time_search": False,
        "timeout_seconds": 600,
        "user": "user_simulator",
    }


def task_selection(mode: str) -> dict[str, Any]:
    if mode == "sealed":
        return {
            "official_split": "test",
            "task_ids_in_command": False,
            "task_payload_accessed": False,
            "domains": list(DOMAINS),
            "sealed_task_count": 3,
            "task_count_by_domain": None,
        }
    return {
        "official_split": "train",
        "logical_split": "development",
        "task_ids_in_command": True,
        "task_payload_accessed": False,
        "domains": list(DOMAINS),
        "sealed_task_count": None,
        "task_count_by_domain": {domain: 1 for domain in DOMAINS},
        "task_id_sha256_by_domain": {domain: ["9" * 64] for domain in DOMAINS},
    }


def sealed_source_manifest() -> dict[str, Any]:
    return {
        "schema_version": "hfr.tau3_sealed_source_manifest.v1",
        "source_revision": "a" * 40,
        "hashes_only": True,
        "task_count": 3,
        "entries": [
            {"task_id_sha256": "1" * 64, "prompt_sha256": "2" * 64, "task_sha256": "3" * 64},
            {"task_id_sha256": "4" * 64, "prompt_sha256": "5" * 64, "task_sha256": "6" * 64},
            {"task_id_sha256": "7" * 64, "prompt_sha256": "8" * 64, "task_sha256": "9" * 64},
        ],
    }


def result_summary() -> dict[str, Any]:
    return {
        "prompt_token_ceiling": 16384,
        "prompt_token_ceiling_checked": True,
        "prompt_token_ceiling_exceeded": False,
        "prompt_token_observation_count": 1,
        "reward_sum": 1.0,
        "simulation_count": 1,
        "success_count": 1,
    }


def adapter_arm_identity(arm_id: str, mode: str, identity_ref: dict[str, str] | None, lock_ref: dict[str, str] | None, lock: dict[str, Any] | None) -> dict[str, Any]:
    adapter = {
        "path": "adapter",
        "file_count": 1,
        "files": [{"path": "adapter_model.safetensors", "size": 3, "sha256": "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad", "kind": "adapter"}],
        "tree_sha256": selected_adapter_tree_from_files(),
    }
    if mode == "sealed":
        return {
            "arm_id": arm_id,
            "source": "candidate_lock",
            "candidate_lock_sha256": lock_ref["sha256"] if lock_ref else None,
            "candidate_identity_sha256": lock["candidate_identity_sha256"] if lock else None,
            "endpoint_model_sha256": "b" * 64,
            "adapter": adapter,
        }
    return {
        "arm_id": arm_id,
        "source": "candidate_identity",
        "candidate_id": "candidate-a",
        "candidate_identity_sha256": identity_ref["sha256"] if identity_ref else None,
        "candidate_record_sha256": "5" * 64,
        "endpoint_model_sha256": "b" * 64,
        "adapter": adapter,
    }


def selected_adapter_tree(root: Path) -> str:
    return read_json(root / "training" / "candidate-a" / "training_receipt.json")["adapter"]["tree_sha256"]


def selected_adapter_tree_from_files() -> str:
    records = [{"path": "adapter_model.safetensors", "size": 3, "sha256": "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad", "kind": "adapter"}]
    return tree_sha256(records)


def tau_raw_result(domain: str, seed: int) -> dict[str, Any]:
    index = SEEDS.index(seed)
    task_id = f"{domain}-{index}"
    return {
        "timestamp": "2026-07-23T00:30:00",
        "info": {
            "git_commit": "a" * 40,
            "num_trials": 1,
            "max_steps": 30,
            "max_errors": 10,
            "max_retries": 0,
            "auto_resume": False,
            "auto_review": True,
            "review_mode": "full",
            "review_model": "fixed-reviewer",
            "hallucination_retries": 0,
            "seed": seed,
            "text_streaming_config": {"chunk_by": "words", "chunk_size": 1},
            "retrieval_config": None,
            "user_info": {
                "implementation": "user_simulator",
                "llm": "fixed-user-sim",
                "llm_args": {
                    "api_base": "http://127.0.0.1:18081/v1",
                    "api_key": "local",
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "max_tokens": 1024,
                    "num_retries": 0,
                },
            },
            "agent_info": {
                "implementation": "llm_agent",
                "llm": "arm-specific-model",
                "llm_args": {
                    "api_base": "http://127.0.0.1:18080/v1",
                    "api_key": "local",
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "max_tokens": 1024,
                    "num_retries": 0,
                },
            },
            "environment_info": {"domain_name": domain, "policy": f"{domain} policy"},
        },
        "tasks": [{"id": task_id, "user_scenario": {"instructions": {"task_instructions": "hidden"}}, "evaluation_criteria": {"nl_assertions": ["hidden"]}}],
        "simulations": [
            {
                "id": f"sim-{domain}-{seed}",
                "task_id": task_id,
                "trial": 0,
                "seed": seed,
                "termination_reason": "user_stop",
                "reward_info": {"reward": 1.0, "db_check": {"db_match": True, "db_reward": 1.0}, "reward_basis": ["DB", "COMMUNICATE"]},
                "messages": [{"role": "user", "content": "raw transcript"}],
                "raw_data": {"provider": "payload"},
                "review": {"errors": []},
            }
        ],
    }


def build_public_report(root: Path, sealed_refs: list[dict[str, str]]) -> dict[str, Any]:
    arm_paths: dict[str, list[Path]] = {arm: [] for arm in ARMS}
    for ref in sealed_refs:
        manifest_path = root / ref["path"]
        manifest = read_json(manifest_path)
        arm = manifest["arm_id"]
        for receipt_ref in manifest["run_receipts"]:
            receipt = read_json(manifest_path.parent / receipt_ref["path"])
            arm_paths[arm].append(manifest_path.parent / receipt["result_path"])
    analyze_tau3_evaluation(
        arm_result_paths=arm_paths,
        out_path=root / "public-evaluation-report.json",
        mode="sealed",
        expected_tau_revision="a" * 40,
        created_at="2026-07-23T01:05:00Z",
        bootstrap_samples=200,
        bootstrap_seed=7,
    )
    report_path = root / "public-evaluation-report.json"
    return {"path": "public-evaluation-report.json", "sha256": sha256_file(report_path), "size": report_path.stat().st_size}


def local_ref(path: str, ref: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {"path": path, "sha256": ref["sha256"]}
    if "size" in ref:
        result["size"] = ref["size"]
    return result


def write_hashed_json(root: Path, rel: str, payload: dict[str, Any]) -> dict[str, Any]:
    path = root / rel
    write_json(path, payload)
    return {"path": rel, "sha256": sha256_file(path), "size": path.stat().st_size}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_sha256(records: list[dict[str, Any]]) -> str:
    import hashlib

    digest = hashlib.sha256()
    for record in records:
        digest.update(json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    return digest.hexdigest()


def tree_value_sha256(value: Any) -> str:
    import hashlib

    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


DOMAINS = ("airline", "retail", "telecom")
SEEDS = (101, 202, 303, 404)
ARMS = ("adapter", "base", "comparator_1", "comparator_2")


if __name__ == "__main__":
    unittest.main()
