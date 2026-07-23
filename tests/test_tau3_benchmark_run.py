from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from flightrecorder import tau3_sealed_authorization as sealed_auth_module
from flightrecorder.schema_registry import check_schema_contract
from flightrecorder.tau3_benchmark_run import (
    Tau3BenchmarkConfig,
    Tau3BenchmarkEndpoint,
    Tau3BenchmarkRunError,
    _command_timeout_seconds,
    _development_tasks_by_domain,
    _reviewer_environment,
    _tau2_argv,
    run_tau3_benchmark_arm,
)
from flightrecorder.tau3_sealed_authorization import create_tau3_sealed_authorization
from flightrecorder.tau3_sealed_authorization import validate_tau3_sealed_authorization


class Tau3BenchmarkRunTests(unittest.TestCase):
    def test_whole_command_timeout_scales_beyond_the_per_task_timeout(self):
        config = Tau3BenchmarkConfig(
            mode="development",
            arm_id="base",
            protocol_path=Path("protocol.json"),
            timeout_seconds=600,
            command_timeout_padding_seconds=30,
        )
        self.assertEqual(
            _command_timeout_seconds(
                protocol={"sealed_manifest": {"task_count": 100}},
                config=config,
                domain="airline",
                tasks_by_domain={"airline": ["1", "2", "3"]},
            ),
            1830,
        )
        self.assertEqual(
            _command_timeout_seconds(
                protocol={"sealed_manifest": {}},
                config=config,
                domain="airline",
                tasks_by_domain=None,
                sealed_task_count=100,
            ),
            60030,
        )

        with self.assertRaisesRegex(Tau3BenchmarkRunError, "positive command task count"):
            _command_timeout_seconds(protocol={"sealed_manifest": {"task_count": 100}}, config=config, domain="airline", tasks_by_domain=None)

    def test_development_runs_domain_seed_commands_and_resumes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = self._repo(root)
            source = self._development_source(root)
            protocol = self._protocol(root, source=source)
            tau2 = self._fake_tau2(root, reward=0.0)
            endpoint = self._endpoint("local/base", 18080)

            manifest = run_tau3_benchmark_arm(
                out_dir=root / "out",
                tau_repo=repo,
                tau_venv_bin=tau2,
                expected_tau_revision=self.revision,
                agent=endpoint,
                user=self._endpoint("local/user", 18081),
                reviewer=self._endpoint("local/reviewer", 18082),
                config=Tau3BenchmarkConfig(
                    mode="development",
                    arm_id="base",
                    protocol_path=protocol,
                    source_split=source,
                    timeout_seconds=2,
                ),
                created_at="2026-07-22T00:00:00Z",
            )

            self.assertEqual(manifest["success_count"], 12)
            self.assertEqual(manifest["failure_count"], 0)
            self.assertEqual(manifest["task_selection"]["official_split"], "train")
            self.assertEqual(manifest["task_selection"]["task_count_by_domain"], {"airline": 1, "retail": 1, "telecom": 1})
            command = json.loads((root / "out" / "run-airline-seed101.json").read_text(encoding="utf-8"))["command"]
            self.assertIn("--task-ids", command)
            self.assertEqual(command[command.index("--task-ids") + 1], "air-1")
            self.assertIn("--auto-review", command)
            self.assertEqual(command[command.index("--review-mode") + 1], "full")
            self.assertNotIn("--auto-resume", command)
            self.assertNotIn('"api_key":"local"', json.dumps(command))
            self.assertIn("endpoint_hash", manifest["agent"])
            self.assertIn("model_sha256", manifest["agent"])
            self.assertEqual(manifest["protocol_sha256"], self._sha256(protocol))
            self.assertEqual(manifest["arm_identity"]["arm_id"], "base")
            self.assertNotIn("model", manifest["agent"])
            self.assertNotIn("api_base", manifest["agent"])

            resumed = run_tau3_benchmark_arm(
                out_dir=root / "out",
                tau_repo=repo,
                tau_venv_bin=tau2,
                expected_tau_revision=self.revision,
                agent=endpoint,
                user=self._endpoint("local/user", 18081),
                reviewer=self._endpoint("local/reviewer", 18082),
                config=Tau3BenchmarkConfig(
                    mode="development",
                    arm_id="base",
                    protocol_path=protocol,
                    source_split=source,
                    timeout_seconds=2,
                ),
            )
            self.assertEqual(resumed["run_count"], 12)

    def test_sealed_never_uses_source_or_task_ids_and_requires_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = self._repo(root)
            tau2 = self._fake_tau2(root, reward=1.0)
            adapter = self._adapter(root)
            sealed_manifest = self._sealed_source_manifest(root, task_count=100)
            protocol = self._protocol(root, sealed_manifest=sealed_manifest)
            lock = self._candidate_lock(root, adapter=adapter, protocol=protocol)
            authorization = self._sealed_authorization(root, lock=lock, protocol=protocol, sealed_manifest=sealed_manifest)
            endpoint = self._endpoint("local/adapter", 18080, adapter_path=adapter)

            manifest = run_tau3_benchmark_arm(
                out_dir=root / "sealed-out",
                tau_repo=repo,
                tau_venv_bin=tau2,
                expected_tau_revision=self.revision,
                agent=endpoint,
                user=self._endpoint("local/user", 18081),
                reviewer=self._endpoint("local/reviewer", 18082),
                config=Tau3BenchmarkConfig(
                    mode="sealed",
                    arm_id="adapter",
                    protocol_path=protocol,
                    sealed_task_count_manifest=sealed_manifest,
                    sealed_authorization=authorization,
                    sealed_authorization_sha256=self._sha256(authorization),
                    candidate_lock=lock,
                    candidate_lock_sha256=self._sha256(lock),
                    timeout_seconds=2,
                ),
            )

            self.assertIsNone(manifest["source"])
            self.assertEqual(manifest["candidate_lock"]["sha256"], self._sha256(lock))
            self.assertEqual(manifest["candidate_lock"]["path"], "inputs/candidate_lock.json")
            self.assertEqual(manifest["sealed_task_count_manifest"]["sha256"], self._sha256(sealed_manifest))
            self.assertEqual(manifest["sealed_task_count_manifest"]["path"], "inputs/sealed_task_count_manifest.json")
            self.assertEqual(manifest["sealed_task_count_manifest"]["task_count"], 100)
            self.assertEqual(manifest["sealed_authorization"]["sha256"], self._sha256(authorization))
            self.assertEqual(manifest["sealed_authorization"]["path"], "inputs/sealed_authorization.json")
            self.assertEqual(manifest["sealed_authorization"]["candidate_lock_sha256"], self._sha256(lock))
            self.assertEqual(manifest["protocol"]["path"], "inputs/protocol.json")
            self.assertEqual(manifest["prelaunch_receipt"]["path"], "prelaunch_receipt.json")
            self.assertEqual(manifest["arm_identity"]["candidate_identity_sha256"], "b" * 64)
            self.assertEqual(manifest["arm_identity"]["adapter"]["tree_sha256"], self._tree_sha256(adapter))
            self.assertFalse(manifest["task_selection"]["task_ids_in_command"])
            self.assertEqual(manifest["task_selection"]["sealed_task_count"], 100)
            command = json.loads((root / "sealed-out" / "run-airline-seed202.json").read_text(encoding="utf-8"))["command"]
            self.assertEqual(command[command.index("--task-split-name") + 1], "test")
            self.assertNotIn("--task-ids", command)

            tampered = root / "sealed-tampered"
            shutil.copytree(root / "sealed-out", tampered)
            (tampered / "inputs" / "candidate_lock.json").write_text('{"tampered":true}\n', encoding="utf-8")
            with self.assertRaisesRegex(Tau3BenchmarkRunError, "staged candidate lock drifted"):
                run_tau3_benchmark_arm(
                    out_dir=tampered,
                    tau_repo=repo,
                    tau_venv_bin=tau2,
                    expected_tau_revision=self.revision,
                    agent=endpoint,
                    user=self._endpoint("local/user", 18081),
                    reviewer=self._endpoint("local/reviewer", 18082),
                    config=Tau3BenchmarkConfig(
                        mode="sealed",
                        arm_id="adapter",
                        protocol_path=protocol,
                        sealed_task_count_manifest=sealed_manifest,
                        sealed_authorization=authorization,
                        sealed_authorization_sha256=self._sha256(authorization),
                        candidate_lock=lock,
                        candidate_lock_sha256=self._sha256(lock),
                        timeout_seconds=2,
                    ),
                )

            with self.assertRaisesRegex(Tau3BenchmarkRunError, "candidate-lock"):
                run_tau3_benchmark_arm(
                    out_dir=root / "sealed-missing-lock",
                    tau_repo=repo,
                    tau_venv_bin=tau2,
                    expected_tau_revision=self.revision,
                    agent=endpoint,
                    user=self._endpoint("local/user", 18081),
                    reviewer=self._endpoint("local/reviewer", 18082),
                    config=Tau3BenchmarkConfig(
                        mode="sealed",
                        arm_id="adapter",
                        protocol_path=protocol,
                        sealed_task_count_manifest=sealed_manifest,
                        sealed_authorization=authorization,
                    ),
                )

            with self.assertRaisesRegex(Tau3BenchmarkRunError, "sealed-authorization"):
                run_tau3_benchmark_arm(
                    out_dir=root / "sealed-missing-authorization",
                    tau_repo=repo,
                    tau_venv_bin=tau2,
                    expected_tau_revision=self.revision,
                    agent=endpoint,
                    user=self._endpoint("local/user", 18081),
                    reviewer=self._endpoint("local/reviewer", 18082),
                    config=Tau3BenchmarkConfig(
                        mode="sealed",
                        arm_id="adapter",
                        protocol_path=protocol,
                        sealed_task_count_manifest=sealed_manifest,
                        candidate_lock=lock,
                    ),
                )

            drifted = root / "sealed-count-drifted"
            shutil.copytree(root / "sealed-out", drifted)
            (drifted / "inputs" / "sealed_task_count_manifest.json").write_text('{"tampered":true}\n', encoding="utf-8")
            with self.assertRaisesRegex(Tau3BenchmarkRunError, "staged sealed task-count manifest drifted"):
                run_tau3_benchmark_arm(
                    out_dir=drifted,
                    tau_repo=repo,
                    tau_venv_bin=tau2,
                    expected_tau_revision=self.revision,
                    agent=endpoint,
                    user=self._endpoint("local/user", 18081),
                    reviewer=self._endpoint("local/reviewer", 18082),
                    config=Tau3BenchmarkConfig(
                        mode="sealed",
                        arm_id="adapter",
                        protocol_path=protocol,
                        sealed_task_count_manifest=sealed_manifest,
                        sealed_authorization=authorization,
                        sealed_authorization_sha256=self._sha256(authorization),
                        candidate_lock=lock,
                        candidate_lock_sha256=self._sha256(lock),
                        timeout_seconds=2,
                    ),
                )

            auth_drifted = root / "sealed-auth-drifted"
            shutil.copytree(root / "sealed-out", auth_drifted)
            (auth_drifted / "inputs" / "sealed_authorization.json").write_text('{"tampered":true}\n', encoding="utf-8")
            with self.assertRaisesRegex(Tau3BenchmarkRunError, "staged sealed authorization drifted"):
                run_tau3_benchmark_arm(
                    out_dir=auth_drifted,
                    tau_repo=repo,
                    tau_venv_bin=tau2,
                    expected_tau_revision=self.revision,
                    agent=endpoint,
                    user=self._endpoint("local/user", 18081),
                    reviewer=self._endpoint("local/reviewer", 18082),
                    config=Tau3BenchmarkConfig(
                        mode="sealed",
                        arm_id="adapter",
                        protocol_path=protocol,
                        sealed_task_count_manifest=sealed_manifest,
                        sealed_authorization=authorization,
                        sealed_authorization_sha256=self._sha256(authorization),
                        candidate_lock=lock,
                        candidate_lock_sha256=self._sha256(lock),
                        timeout_seconds=2,
                    ),
                )

            source_auth_mutated = root / "sealed-source-auth-mutated"

            def mutate_source_and_validate(**kwargs):
                authorization_path = Path(kwargs["authorization_path"])
                self.assertEqual(authorization_path, source_auth_mutated / "inputs" / "sealed_authorization.json")
                authorization.chmod(0o600)
                authorization.write_text('{"tampered_source":true}\n', encoding="utf-8")
                return {
                    "sha256": self._sha256(authorization_path),
                    "size": authorization_path.stat().st_size,
                    "authorized": True,
                    "candidate_lock_sha256": self._sha256(lock),
                    "protocol_sha256": self._sha256(protocol),
                    "sealed_source_sha256": self._sha256(sealed_manifest),
                    "task_count": 100,
                    "arms": ["adapter", "base", "comparator_1", "comparator_2"],
                    "seeds": [101, 202, 303, 404],
                }

            with patch("flightrecorder.tau3_benchmark_run.validate_tau3_sealed_authorization", side_effect=mutate_source_and_validate):
                run_tau3_benchmark_arm(
                    out_dir=source_auth_mutated,
                    tau_repo=repo,
                    tau_venv_bin=tau2,
                    expected_tau_revision=self.revision,
                    agent=endpoint,
                    user=self._endpoint("local/user", 18081),
                    reviewer=self._endpoint("local/reviewer", 18082),
                    config=Tau3BenchmarkConfig(
                        mode="sealed",
                        arm_id="adapter",
                        protocol_path=protocol,
                        sealed_task_count_manifest=sealed_manifest,
                        sealed_authorization=authorization,
                        sealed_authorization_sha256=self._sha256(authorization),
                        candidate_lock=lock,
                        candidate_lock_sha256=self._sha256(lock),
                        timeout_seconds=2,
                    ),
                )
            self.assertNotEqual(self._sha256(authorization), self._sha256(source_auth_mutated / "inputs" / "sealed_authorization.json"))

    def test_development_adapter_uses_candidate_identity_without_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = self._repo(root)
            source = self._development_source(root)
            adapter = self._adapter(root)
            identity = root / "candidate-identity.json"
            endpoint = self._endpoint("local/dev-adapter", 18080, adapter_path=adapter)
            endpoint_hash = hashlib_sha256(endpoint.model.encode("utf-8")).hexdigest()
            identity.write_text(
                json.dumps(
                    {
                        "schema_version": "hfr.tau3_candidate_identity.v1",
                        "created_at": "2026-07-22T00:00:00Z",
                        "candidate_id": "candidate-1",
                        "training_receipt_sha256": "1" * 64,
                        "final_training_receipt_sha256": "1" * 64,
                        "adapter_tree_sha256": self._tree_sha256(adapter),
                        "endpoint_model_sha256": endpoint_hash,
                        "training_binding": {
                            "protocol_sha256": "2" * 64,
                            "protocol_signature": "3" * 64,
                            "model_freeze_sha256": "4" * 64,
                            "recipe_space_sha256": "5" * 64,
                            "mlx_qlora_plan_sha256": "6" * 64,
                            "base_identity_sha256": "7" * 64,
                            "base_tree_sha256": "8" * 64,
                            "dataset_manifest_sha256": "9" * 64,
                            "dataset_files_sha256": "a" * 64,
                            "source_binding_sha256": "b" * 64,
                            "recipe_sha256": "c" * 64,
                        },
                        "adapter_identity": {
                            "adapter_tree_sha256": self._tree_sha256(adapter),
                            "tree_sha256": self._tree_sha256(adapter),
                            "file_count": 3,
                            "adapter_weight_file_count": 1,
                            "declared_file_set_sha256": self._tree_sha256(adapter),
                            "replayed_file_set_sha256": self._tree_sha256(adapter),
                        },
                        "governance": {
                            "training_receipt_schema_checked": True,
                            "training_receipt_final": True,
                            "training_receipt_success": True,
                            "training_weights_updated": True,
                            "adapter_files_replayed": True,
                            "endpoint_model_hash_only": True,
                            "hashes_only": True,
                            "local_paths_included": False,
                            "absolute_paths_included": False,
                            "raw_endpoint_model_included": False,
                            "raw_training_receipt_included": False,
                            "public_safe": True,
                            "private_material_included": False,
                            "sealed_access_authorized": False,
                        },
                        "schema_checked": True,
                        "read_only": True,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            protocol = self._protocol(root, source=source)
            tau2 = self._fake_tau2(root, reward=1.0)

            manifest = run_tau3_benchmark_arm(
                out_dir=root / "dev-adapter",
                tau_repo=repo,
                tau_venv_bin=tau2,
                expected_tau_revision=self.revision,
                agent=endpoint,
                user=self._endpoint("local/user", 18081),
                reviewer=self._endpoint("local/reviewer", 18082),
                config=Tau3BenchmarkConfig(
                    mode="development",
                    arm_id="adapter",
                    protocol_path=protocol,
                    source_split=source,
                    candidate_identity=identity,
                    candidate_identity_sha256=self._sha256(identity),
                    timeout_seconds=2,
                ),
            )

            self.assertIsNone(manifest["candidate_lock"])
            self.assertEqual(manifest["candidate_identity"]["sha256"], self._sha256(identity))
            self.assertEqual(manifest["candidate_identity"]["path"], "inputs/candidate_identity.json")
            self.assertEqual(manifest["candidate_identity"]["candidate_id"], "candidate-1")
            self.assertEqual(manifest["source"]["path"], "inputs/development_source.json")
            self.assertEqual(manifest["arm_identity"]["source"], "candidate_identity")
            self.assertEqual(manifest["arm_identity"]["adapter"]["tree_sha256"], self._tree_sha256(adapter))
            command = json.loads((root / "dev-adapter" / "run-airline-seed101.json").read_text(encoding="utf-8"))["command"]
            agent_args = json.loads(command[command.index("--agent-llm-args") + 1])
            self.assertEqual(agent_args["extra_body"], {"adapters": str(adapter.resolve())})

            with self.assertRaisesRegex(Tau3BenchmarkRunError, "candidate-identity"):
                run_tau3_benchmark_arm(
                    out_dir=root / "missing-dev-identity",
                    tau_repo=repo,
                    tau_venv_bin=tau2,
                    expected_tau_revision=self.revision,
                    agent=endpoint,
                    user=self._endpoint("local/user", 18081),
                    reviewer=self._endpoint("local/reviewer", 18082),
                    config=Tau3BenchmarkConfig(mode="development", arm_id="adapter", protocol_path=protocol, source_split=source),
                )

            invalid_identity = root / "invalid-candidate-identity.json"
            invalid_identity.write_text(
                json.dumps({"schema_version": "hfr.tau3_candidate_selection.v1", "candidate": {"endpoint_model_sha256": endpoint_hash, "adapter_tree_sha256": self._tree_sha256(adapter)}}) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(Tau3BenchmarkRunError, "candidate identity must use hfr.tau3_candidate_identity.v1"):
                run_tau3_benchmark_arm(
                    out_dir=root / "invalid-dev-identity",
                    tau_repo=repo,
                    tau_venv_bin=tau2,
                    expected_tau_revision=self.revision,
                    agent=endpoint,
                    user=self._endpoint("local/user", 18081),
                    reviewer=self._endpoint("local/reviewer", 18082),
                    config=Tau3BenchmarkConfig(mode="development", arm_id="adapter", protocol_path=protocol, source_split=source, candidate_identity=invalid_identity),
                )

            with self.assertRaisesRegex(Tau3BenchmarkRunError, "adapter-path|adapter path"):
                run_tau3_benchmark_arm(
                    out_dir=root / "missing-adapter-path",
                    tau_repo=repo,
                    tau_venv_bin=tau2,
                    expected_tau_revision=self.revision,
                    agent=self._endpoint("local/dev-adapter", 18080),
                    user=self._endpoint("local/user", 18081),
                    reviewer=self._endpoint("local/reviewer", 18082),
                    config=Tau3BenchmarkConfig(mode="development", arm_id="adapter", protocol_path=protocol, source_split=source, candidate_identity=identity),
                )

    def test_sealed_task_count_manifest_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = self._repo(root)
            tau2 = self._fake_tau2(root, reward=1.0)
            adapter = self._adapter(root)
            sealed_manifest = self._sealed_source_manifest(root, task_count=100)
            protocol = self._protocol(root, sealed_manifest=sealed_manifest)
            lock = self._candidate_lock(root, adapter=adapter, protocol=protocol)
            authorization = self._sealed_authorization(root, lock=lock, protocol=protocol, sealed_manifest=sealed_manifest)
            endpoint = self._endpoint("local/adapter", 18080, adapter_path=adapter)

            with self.assertRaisesRegex(Tau3BenchmarkRunError, "sealed-task-count-manifest"):
                run_tau3_benchmark_arm(
                    out_dir=root / "sealed-missing-count",
                    tau_repo=repo,
                    tau_venv_bin=tau2,
                    expected_tau_revision=self.revision,
                    agent=endpoint,
                    user=self._endpoint("local/user", 18081),
                    reviewer=self._endpoint("local/reviewer", 18082),
                    config=Tau3BenchmarkConfig(
                        mode="sealed",
                        arm_id="adapter",
                        protocol_path=protocol,
                        candidate_lock=lock,
                        candidate_lock_sha256=self._sha256(lock),
                        timeout_seconds=2,
                    ),
                )

            bad_hash_manifest = self._sealed_source_manifest(root, task_count=99)
            with self.assertRaisesRegex(Tau3BenchmarkRunError, "sealed task-count manifest sha256"):
                run_tau3_benchmark_arm(
                    out_dir=root / "sealed-bad-hash",
                    tau_repo=repo,
                    tau_venv_bin=tau2,
                    expected_tau_revision=self.revision,
                    agent=endpoint,
                    user=self._endpoint("local/user", 18081),
                    reviewer=self._endpoint("local/reviewer", 18082),
                    config=Tau3BenchmarkConfig(
                        mode="sealed",
                        arm_id="adapter",
                        protocol_path=protocol,
                        sealed_task_count_manifest=bad_hash_manifest,
                        sealed_authorization=authorization,
                        sealed_authorization_sha256=self._sha256(authorization),
                        candidate_lock=lock,
                        candidate_lock_sha256=self._sha256(lock),
                        timeout_seconds=2,
                    ),
                )

            payload_manifest = self._sealed_source_manifest(root, task_count=100, entry_extra={"raw_id": "air-1"})
            with self.assertRaisesRegex(Tau3BenchmarkRunError, "schema failed|hash-only"):
                run_tau3_benchmark_arm(
                    out_dir=root / "sealed-payload",
                    tau_repo=repo,
                    tau_venv_bin=tau2,
                    expected_tau_revision=self.revision,
                    agent=endpoint,
                    user=self._endpoint("local/user", 18081),
                    reviewer=self._endpoint("local/reviewer", 18082),
                    config=Tau3BenchmarkConfig(
                        mode="sealed",
                        arm_id="adapter",
                        protocol_path=protocol,
                        sealed_task_count_manifest=payload_manifest,
                        sealed_authorization=authorization,
                        sealed_authorization_sha256=self._sha256(authorization),
                        candidate_lock=lock,
                        candidate_lock_sha256=self._sha256(lock),
                        timeout_seconds=2,
                    ),
                )

    def test_sealed_authorization_schema_and_governance_gates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._repo(root)
            adapter = self._adapter(root)
            sealed_manifest = self._sealed_source_manifest(root, task_count=100)
            protocol = self._protocol(root, sealed_manifest=sealed_manifest)
            lock = self._candidate_lock(root, adapter=adapter, protocol=protocol)
            authorization = self._sealed_authorization(root, lock=lock, protocol=protocol, sealed_manifest=sealed_manifest)
            payload = json.loads(authorization.read_text(encoding="utf-8"))

            self.assertTrue(check_schema_contract(payload, name_or_id="tau3_sealed_authorization")["passed"])
            self.assertTrue(payload["authorized"])
            self.assertEqual(payload["candidate_lock"]["sha256"], self._sha256(lock))
            self.assertEqual(payload["protocol"]["sha256"], self._sha256(protocol))
            self.assertEqual(payload["sealed_source"]["manifest_sha256"], self._sha256(sealed_manifest))
            self.assertEqual(payload["sealed_source"]["task_count"], 100)
            self.assertEqual(payload["frozen_contract"]["seeds"], [101, 202, 303, 404])
            self.assertEqual(payload["frozen_contract"]["arms"], ["adapter", "base", "comparator_1", "comparator_2"])
            self.assertFalse(payload["local_paths_included"])
            self.assertFalse(payload["raw_payload_included"])
            self.assertNotIn(str(root), json.dumps(payload, sort_keys=True))

            failed_protocol = json.loads(protocol.read_text(encoding="utf-8"))
            failed_protocol["redaction_attestation"]["passed"] = False
            failed_protocol_path = root / "failed-redaction-protocol.json"
            failed_protocol_path.write_text(json.dumps(failed_protocol) + "\n", encoding="utf-8")
            failed_lock = self._candidate_lock(root, adapter=adapter, protocol=failed_protocol_path)
            with self.assertRaisesRegex(ValueError, "redaction gate"):
                create_tau3_sealed_authorization(
                    candidate_lock=failed_lock,
                    protocol=failed_protocol_path,
                    sealed_source_manifest=sealed_manifest,
                    out=root / "failed-auth.json",
                    created_at="2026-07-22T00:00:01+00:00",
                )

            replay_tampered = dict(payload)
            replay_tampered["created_at"] = "2026-07-21T23:59:59+00:00"
            replay_tampered_path = root / "replay-tampered-auth.json"
            replay_tampered_path.write_text(json.dumps(replay_tampered, sort_keys=True) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "created after candidate lock"):
                validate_tau3_sealed_authorization(
                    authorization_path=replay_tampered_path,
                    candidate_lock_path=lock,
                    protocol_path=protocol,
                    sealed_source_manifest_path=sealed_manifest,
                    arm_id="adapter",
                    seeds=(101, 202, 303, 404),
                    expected_tau_revision=self.revision,
                )

            real_fstat = sealed_auth_module.os.fstat
            seen = {"count": 0}

            def drifting_fstat(fd):
                stat_result = real_fstat(fd)
                seen["count"] += 1
                if seen["count"] == 2:
                    lock.write_text(lock.read_text(encoding="utf-8") + " ", encoding="utf-8")
                    return real_fstat(fd)
                return stat_result

            with patch.object(sealed_auth_module.os, "fstat", side_effect=drifting_fstat):
                with self.assertRaisesRegex(ValueError, "changed while being read|read size mismatch"):
                    create_tau3_sealed_authorization(
                        candidate_lock=lock,
                        protocol=protocol,
                        sealed_source_manifest=sealed_manifest,
                        out=root / "toctou-auth.json",
                        created_at="2026-07-22T00:00:01+00:00",
                    )

    def test_sealed_task_count_manifest_rejects_symlink_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = self._repo(root)
            tau2 = self._fake_tau2(root, reward=1.0)
            adapter = self._adapter(root)
            sealed_manifest = self._sealed_source_manifest(root, task_count=100)
            protocol = self._protocol(root, sealed_manifest=sealed_manifest)
            lock = self._candidate_lock(root, adapter=adapter, protocol=protocol)
            authorization = self._sealed_authorization(root, lock=lock, protocol=protocol, sealed_manifest=sealed_manifest)
            endpoint = self._endpoint("local/adapter", 18080, adapter_path=adapter)

            sealed_link = root / "sealed-link.json"
            os.symlink(sealed_manifest, sealed_link)
            with self.assertRaisesRegex(Tau3BenchmarkRunError, "sealed task-count manifest must not contain symlink"):
                run_tau3_benchmark_arm(
                    out_dir=root / "sealed-source-symlink",
                    tau_repo=repo,
                    tau_venv_bin=tau2,
                    expected_tau_revision=self.revision,
                    agent=endpoint,
                    user=self._endpoint("local/user", 18081),
                    reviewer=self._endpoint("local/reviewer", 18082),
                    config=Tau3BenchmarkConfig(
                        mode="sealed",
                        arm_id="adapter",
                        protocol_path=protocol,
                        sealed_task_count_manifest=sealed_link,
                        sealed_authorization=authorization,
                        sealed_authorization_sha256=self._sha256(authorization),
                        candidate_lock=lock,
                        candidate_lock_sha256=self._sha256(lock),
                        timeout_seconds=2,
                    ),
                )

            inputs_symlink_out = root / "sealed-inputs-symlink"
            inputs_symlink_out.mkdir()
            symlink_target = root / "inputs-target"
            symlink_target.mkdir()
            os.symlink(symlink_target, inputs_symlink_out / "inputs")
            with self.assertRaisesRegex(Tau3BenchmarkRunError, "destination must not contain symlink"):
                run_tau3_benchmark_arm(
                    out_dir=inputs_symlink_out,
                    tau_repo=repo,
                    tau_venv_bin=tau2,
                    expected_tau_revision=self.revision,
                    agent=endpoint,
                    user=self._endpoint("local/user", 18081),
                    reviewer=self._endpoint("local/reviewer", 18082),
                    config=Tau3BenchmarkConfig(
                        mode="sealed",
                        arm_id="adapter",
                        protocol_path=protocol,
                        sealed_task_count_manifest=sealed_manifest,
                        sealed_authorization=authorization,
                        sealed_authorization_sha256=self._sha256(authorization),
                        candidate_lock=lock,
                        candidate_lock_sha256=self._sha256(lock),
                        timeout_seconds=2,
                    ),
                )

            staged_symlink_out = root / "sealed-staged-symlink"
            staged_inputs = staged_symlink_out / "inputs"
            staged_inputs.mkdir(parents=True)
            (staged_inputs / "protocol.json").write_text(protocol.read_text(encoding="utf-8"), encoding="utf-8")
            os.symlink(sealed_manifest, staged_inputs / "sealed_task_count_manifest.json")
            with self.assertRaisesRegex(Tau3BenchmarkRunError, "staged sealed task-count manifest destination must not contain symlink"):
                run_tau3_benchmark_arm(
                    out_dir=staged_symlink_out,
                    tau_repo=repo,
                    tau_venv_bin=tau2,
                    expected_tau_revision=self.revision,
                    agent=endpoint,
                    user=self._endpoint("local/user", 18081),
                    reviewer=self._endpoint("local/reviewer", 18082),
                    config=Tau3BenchmarkConfig(
                        mode="sealed",
                        arm_id="adapter",
                        protocol_path=protocol,
                        sealed_task_count_manifest=sealed_manifest,
                        sealed_authorization=authorization,
                        sealed_authorization_sha256=self._sha256(authorization),
                        candidate_lock=lock,
                        candidate_lock_sha256=self._sha256(lock),
                        timeout_seconds=2,
                    ),
                )

    def test_development_rejects_sealed_task_count_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = self._repo(root)
            source = self._development_source(root)
            sealed_manifest = self._sealed_source_manifest(root, task_count=100)
            protocol = self._protocol(root, source=source)
            tau2 = self._fake_tau2(root, reward=1.0)
            endpoint = self._endpoint("local/base", 18080)

            with self.assertRaisesRegex(Tau3BenchmarkRunError, "development mode must not receive a sealed task-count manifest"):
                run_tau3_benchmark_arm(
                    out_dir=root / "dev-sealed-count",
                    tau_repo=repo,
                    tau_venv_bin=tau2,
                    expected_tau_revision=self.revision,
                    agent=endpoint,
                    user=endpoint,
                    reviewer=endpoint,
                    config=Tau3BenchmarkConfig(
                        mode="development",
                        arm_id="base",
                        protocol_path=protocol,
                        source_split=source,
                        sealed_task_count_manifest=sealed_manifest,
                        timeout_seconds=2,
                    ),
                )

    def test_refuses_remote_endpoint_changed_resume_and_unreceipted_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = self._repo(root)
            source = self._development_source(root)
            protocol = self._protocol(root, source=source)
            tau2 = self._fake_tau2(root, reward=1.0)
            endpoint = self._endpoint("local/base", 18080)
            with self.assertRaisesRegex(Tau3BenchmarkRunError, "loopback"):
                run_tau3_benchmark_arm(
                    out_dir=root / "remote",
                    tau_repo=repo,
                    tau_venv_bin=tau2,
                    expected_tau_revision=self.revision,
                    agent=Tau3BenchmarkEndpoint(model="remote", api_base="https://api.example.test/v1"),
                    user=endpoint,
                    reviewer=endpoint,
                    config=Tau3BenchmarkConfig(mode="development", arm_id="base", protocol_path=protocol, source_split=source),
                )

            out = root / "out"
            run_tau3_benchmark_arm(
                out_dir=out,
                tau_repo=repo,
                tau_venv_bin=tau2,
                expected_tau_revision=self.revision,
                agent=endpoint,
                user=endpoint,
                reviewer=endpoint,
                config=Tau3BenchmarkConfig(mode="development", arm_id="base", protocol_path=protocol, source_split=source, timeout_seconds=2),
            )
            with self.assertRaisesRegex(Tau3BenchmarkRunError, "stale or invalid"):
                run_tau3_benchmark_arm(
                    out_dir=out,
                    tau_repo=repo,
                    tau_venv_bin=tau2,
                    expected_tau_revision=self.revision,
                    agent=endpoint,
                    user=endpoint,
                    reviewer=endpoint,
                    config=Tau3BenchmarkConfig(mode="development", arm_id="base", protocol_path=protocol, source_split=source, timeout_seconds=3),
                )

            unreceipted = root / "unreceipted"
            raw = repo / "data" / "simulations" / "hfr-benchmark" / unreceipted.name / "development-base-airline-seed101" / "results.json"
            raw.parent.mkdir(parents=True)
            raw.write_text('{"simulations":[]}\n', encoding="utf-8")
            with self.assertRaisesRegex(Tau3BenchmarkRunError, "without receipt"):
                run_tau3_benchmark_arm(
                    out_dir=unreceipted,
                    tau_repo=repo,
                    tau_venv_bin=tau2,
                    expected_tau_revision=self.revision,
                    agent=endpoint,
                    user=endpoint,
                    reviewer=endpoint,
                    config=Tau3BenchmarkConfig(mode="development", arm_id="base", protocol_path=protocol, source_split=source, timeout_seconds=2),
                )

    def test_command_contract_is_identical_harness(self):
        endpoint = self._endpoint("local/model", 18080)
        argv = _tau2_argv(
            tau2=Path("/tmp/tau2"),
            domain="retail",
            seed=101,
            save_to="fixture",
            agent=endpoint,
            user=endpoint,
            reviewer=endpoint,
            config=Tau3BenchmarkConfig(mode="development", arm_id="base", protocol_path=Path("/tmp/protocol.json"), source_split=Path("/tmp/dev.json")),
            task_ids=["r1", "r2"],
        )
        expected_flags = {
            "--agent": "llm_agent",
            "--num-trials": "1",
            "--max-steps": "30",
            "--max-errors": "10",
            "--max-concurrency": "1",
            "--max-retries": "0",
            "--hallucination-retries": "0",
            "--review-mode": "full",
        }
        for flag, value in expected_flags.items():
            self.assertEqual(argv[argv.index(flag) + 1], value)
        self.assertIn("--auto-review", argv)
        self.assertIn("--enforce-communication-protocol", argv)

    def test_rejects_non_frozen_contract_and_unversioned_development_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = self._repo(root)
            tau2 = self._fake_tau2(root, reward=1.0)
            endpoint = self._endpoint("local/base", 18080)
            source = root / "development.jsonl"
            source.write_text('{"domain":"airline","raw_id":"1"}\n', encoding="utf-8")
            valid_source = self._development_source(root)
            protocol = self._protocol(root, source=valid_source)
            with self.assertRaisesRegex(Tau3BenchmarkRunError, "seeds must be exactly"):
                run_tau3_benchmark_arm(
                    out_dir=root / "bad-seeds",
                    tau_repo=repo,
                    tau_venv_bin=tau2,
                    expected_tau_revision=self.revision,
                    agent=endpoint,
                    user=endpoint,
                    reviewer=endpoint,
                    config=Tau3BenchmarkConfig(mode="development", arm_id="base", protocol_path=protocol, source_split=source, seeds=(101,)),
                )
            with self.assertRaisesRegex(Tau3BenchmarkRunError, "source schema failed"):
                run_tau3_benchmark_arm(
                    out_dir=root / "bad-source",
                    tau_repo=repo,
                    tau_venv_bin=tau2,
                    expected_tau_revision=self.revision,
                    agent=endpoint,
                    user=endpoint,
                    reviewer=endpoint,
                    config=Tau3BenchmarkConfig(mode="development", arm_id="base", protocol_path=protocol, source_split=source),
                )

    def test_development_source_replays_domain_qualified_raw_id_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._repo(root)
            source = self._development_source(root)

            tasks = _development_tasks_by_domain(source, self.revision)
            self.assertEqual(tasks, {"airline": ["air-1"], "retail": ["ret-1"], "telecom": ["tel-1"]})

            payload = json.loads(source.read_text(encoding="utf-8"))
            payload["tasks"][0]["raw_id_sha256"] = hashlib_sha256(b"air-1").hexdigest()
            source.write_text(json.dumps(payload) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(Tau3BenchmarkRunError, "raw task id hash mismatch"):
                _development_tasks_by_domain(source, self.revision)

    def test_completed_receipt_requires_hashed_result_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = self._repo(root)
            source = self._development_source(root)
            protocol = self._protocol(root, source=source)
            tau2 = self._fake_tau2(root, reward=0.0)
            endpoint = self._endpoint("local/base", 18080)
            out = root / "out"
            run_tau3_benchmark_arm(
                out_dir=out,
                tau_repo=repo,
                tau_venv_bin=tau2,
                expected_tau_revision=self.revision,
                agent=endpoint,
                user=endpoint,
                reviewer=endpoint,
                config=Tau3BenchmarkConfig(mode="development", arm_id="base", protocol_path=protocol, source_split=source, timeout_seconds=2),
            )
            (out / "manifest.json").unlink()
            receipt_path = out / "run-airline-seed101.json"
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            receipt["result_sha256"] = None
            receipt_path.write_text(json.dumps(receipt) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(Tau3BenchmarkRunError, "result_sha256"):
                run_tau3_benchmark_arm(
                    out_dir=out,
                    tau_repo=repo,
                    tau_venv_bin=tau2,
                    expected_tau_revision=self.revision,
                    agent=endpoint,
                    user=endpoint,
                    reviewer=endpoint,
                    config=Tau3BenchmarkConfig(mode="development", arm_id="base", protocol_path=protocol, source_split=source, timeout_seconds=2),
                )

    def test_copied_results_replay_after_output_relocation_and_tamper_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = self._repo(root)
            source = self._development_source(root)
            protocol = self._protocol(root, source=source)
            tau2 = self._fake_tau2(root, reward=1.0)
            endpoint = self._endpoint("local/base", 18080)
            out = root / "out"
            manifest = run_tau3_benchmark_arm(
                out_dir=out,
                tau_repo=repo,
                tau_venv_bin=tau2,
                expected_tau_revision=self.revision,
                agent=endpoint,
                user=endpoint,
                reviewer=endpoint,
                config=Tau3BenchmarkConfig(mode="development", arm_id="base", protocol_path=protocol, source_split=source, timeout_seconds=2),
            )
            ref = manifest["run_receipts"][0]
            self.assertRegex(ref["receipt_sha256"], r"^[0-9a-f]{64}$")
            self.assertEqual(ref["result_path"], "results/airline/seed-101/results.json")
            self.assertEqual(manifest["protocol"]["path"], "inputs/protocol.json")
            self.assertEqual(manifest["source"]["path"], "inputs/development_source.json")
            self.assertEqual(manifest["prelaunch_receipt"]["path"], "prelaunch_receipt.json")
            receipt = json.loads((out / ref["path"]).read_text(encoding="utf-8"))
            self.assertEqual(receipt["result_path"], ref["result_path"])
            self.assertTrue((out / ref["result_path"]).is_file())
            self.assertFalse(Path(receipt["result_path"]).is_absolute())

            relocated = root / "relocated-out"
            shutil.copytree(out, relocated)
            replayed = run_tau3_benchmark_arm(
                out_dir=relocated,
                tau_repo=repo,
                tau_venv_bin=tau2,
                expected_tau_revision=self.revision,
                agent=endpoint,
                user=endpoint,
                reviewer=endpoint,
                config=Tau3BenchmarkConfig(mode="development", arm_id="base", protocol_path=protocol, source_split=source, timeout_seconds=2),
            )
            self.assertEqual(replayed["success_count"], 12)

            input_tampered = root / "input-tampered-out"
            shutil.copytree(out, input_tampered)
            (input_tampered / "inputs" / "protocol.json").write_text('{"tampered":true}\n', encoding="utf-8")
            with self.assertRaisesRegex(Tau3BenchmarkRunError, "staged protocol drifted"):
                run_tau3_benchmark_arm(
                    out_dir=input_tampered,
                    tau_repo=repo,
                    tau_venv_bin=tau2,
                    expected_tau_revision=self.revision,
                    agent=endpoint,
                    user=endpoint,
                    reviewer=endpoint,
                    config=Tau3BenchmarkConfig(mode="development", arm_id="base", protocol_path=protocol, source_split=source, timeout_seconds=2),
                )

            (relocated / ref["result_path"]).write_text('{"simulations":[{"reward_info":{"reward":0}}]}\n', encoding="utf-8")
            with self.assertRaisesRegex(Tau3BenchmarkRunError, "generated result drifted"):
                run_tau3_benchmark_arm(
                    out_dir=relocated,
                    tau_repo=repo,
                    tau_venv_bin=tau2,
                    expected_tau_revision=self.revision,
                    agent=endpoint,
                    user=endpoint,
                    reviewer=endpoint,
                    config=Tau3BenchmarkConfig(mode="development", arm_id="base", protocol_path=protocol, source_split=source, timeout_seconds=2),
                )

    def test_rejects_protocol_model_and_source_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = self._repo(root)
            source = self._development_source(root)
            protocol = self._protocol(root, source=source)
            tau2 = self._fake_tau2(root, reward=1.0)
            endpoint = self._endpoint("local/base", 18080)

            bad_protocol = json.loads(protocol.read_text(encoding="utf-8"))
            bad_protocol["tau_revision"]["revision"] = "0" * 40
            bad_protocol_path = root / "bad-protocol.json"
            bad_protocol_path.write_text(json.dumps(bad_protocol) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(Tau3BenchmarkRunError, "protocol Tau revision mismatch"):
                run_tau3_benchmark_arm(
                    out_dir=root / "bad-revision",
                    tau_repo=repo,
                    tau_venv_bin=tau2,
                    expected_tau_revision=self.revision,
                    agent=endpoint,
                    user=endpoint,
                    reviewer=endpoint,
                    config=Tau3BenchmarkConfig(mode="development", arm_id="base", protocol_path=bad_protocol_path, source_split=source, timeout_seconds=2),
                )

            with self.assertRaisesRegex(Tau3BenchmarkRunError, "base model identity does not match protocol"):
                run_tau3_benchmark_arm(
                    out_dir=root / "bad-model",
                    tau_repo=repo,
                    tau_venv_bin=tau2,
                    expected_tau_revision=self.revision,
                    agent=self._endpoint("local/other", 18080),
                    user=endpoint,
                    reviewer=endpoint,
                    config=Tau3BenchmarkConfig(mode="development", arm_id="base", protocol_path=protocol, source_split=source, timeout_seconds=2),
                )

            source.write_text(source.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with self.assertRaisesRegex(Tau3BenchmarkRunError, "development source sha256"):
                run_tau3_benchmark_arm(
                    out_dir=root / "bad-source-hash",
                    tau_repo=repo,
                    tau_venv_bin=tau2,
                    expected_tau_revision=self.revision,
                    agent=endpoint,
                    user=endpoint,
                    reviewer=endpoint,
                    config=Tau3BenchmarkConfig(mode="development", arm_id="base", protocol_path=protocol, source_split=source, timeout_seconds=2),
                )

    def test_model_identity_accepts_exact_openai_absolute_path_and_rejects_suffix(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = self._repo(root)
            source = self._development_source(root)
            protocol = self._protocol(root, source=source)
            tau2 = self._fake_tau2(root, reward=1.0)
            exact_path = (protocol.parent / "local/base").resolve(strict=False)

            manifest = run_tau3_benchmark_arm(
                out_dir=root / "absolute-model",
                tau_repo=repo,
                tau_venv_bin=tau2,
                expected_tau_revision=self.revision,
                agent=self._endpoint(f"openai//{exact_path}", 18080),
                user=self._endpoint("local/user", 18081),
                reviewer=self._endpoint("local/reviewer", 18082),
                config=Tau3BenchmarkConfig(mode="development", arm_id="base", protocol_path=protocol, source_split=source, timeout_seconds=2),
            )
            self.assertEqual(manifest["success_count"], 12)

            ambiguous_suffix = root / "other" / "local" / "base"
            with self.assertRaisesRegex(Tau3BenchmarkRunError, "base model identity does not match protocol"):
                run_tau3_benchmark_arm(
                    out_dir=root / "suffix-model",
                    tau_repo=repo,
                    tau_venv_bin=tau2,
                    expected_tau_revision=self.revision,
                    agent=self._endpoint(f"openai//{ambiguous_suffix}", 18080),
                    user=self._endpoint("local/user", 18081),
                    reviewer=self._endpoint("local/reviewer", 18082),
                    config=Tau3BenchmarkConfig(mode="development", arm_id="base", protocol_path=protocol, source_split=source, timeout_seconds=2),
                )

    def test_model_identity_resolves_existing_workspace_relative_protocol_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = self._repo(root)
            source = self._development_source(root)
            protocol = self._protocol(root, source=source)
            tau2 = self._fake_tau2(root, reward=1.0)
            workspace_model = root / "local" / "base"
            workspace_model.mkdir(parents=True)
            old_cwd = Path.cwd()
            try:
                os.chdir(root)
                manifest = run_tau3_benchmark_arm(
                    out_dir=root / "workspace-relative-model",
                    tau_repo=repo,
                    tau_venv_bin=tau2,
                    expected_tau_revision=self.revision,
                    agent=self._endpoint(f"openai//{workspace_model}", 18080),
                    user=self._endpoint("local/user", 18081),
                    reviewer=self._endpoint("local/reviewer", 18082),
                    config=Tau3BenchmarkConfig(
                        mode="development",
                        arm_id="base",
                        protocol_path=protocol,
                        source_split=source,
                        timeout_seconds=2,
                    ),
                )
            finally:
                os.chdir(old_cwd)
            self.assertEqual(manifest["success_count"], 12)

    def test_over_context_completed_result_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = self._repo(root)
            source = self._development_source(root)
            protocol = self._protocol(root, source=source)
            tau2 = self._fake_tau2(root, reward=1.0, prompt_tokens=16385)
            endpoint = self._endpoint("local/base", 18080)

            manifest = run_tau3_benchmark_arm(
                out_dir=root / "out",
                tau_repo=repo,
                tau_venv_bin=tau2,
                expected_tau_revision=self.revision,
                agent=endpoint,
                user=endpoint,
                reviewer=endpoint,
                config=Tau3BenchmarkConfig(mode="development", arm_id="base", protocol_path=protocol, source_split=source, timeout_seconds=2),
                created_at="2026-07-22T00:00:00Z",
            )

            self.assertEqual(manifest["success_count"], 0)
            summary = manifest["run_receipts"][0]["result_summary"]
            self.assertEqual(summary["prompt_token_ceiling"], 16384)
            self.assertTrue(summary["prompt_token_ceiling_checked"])
            self.assertTrue(summary["prompt_token_ceiling_exceeded"])
            receipt = json.loads((root / "out" / "run-airline-seed101.json").read_text(encoding="utf-8"))
            self.assertEqual(receipt["terminal_status"], "failed")

    def test_missing_prompt_tokens_result_is_not_context_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = self._repo(root)
            source = self._development_source(root)
            protocol = self._protocol(root, source=source)
            tau2 = self._fake_tau2(root, reward=1.0, prompt_tokens=None)
            endpoint = self._endpoint("local/base", 18080)

            manifest = run_tau3_benchmark_arm(
                out_dir=root / "out",
                tau_repo=repo,
                tau_venv_bin=tau2,
                expected_tau_revision=self.revision,
                agent=endpoint,
                user=endpoint,
                reviewer=endpoint,
                config=Tau3BenchmarkConfig(mode="development", arm_id="base", protocol_path=protocol, source_split=source, timeout_seconds=2),
                created_at="2026-07-22T00:00:00Z",
            )

            self.assertEqual(manifest["success_count"], 0)
            summary = manifest["run_receipts"][0]["result_summary"]
            self.assertEqual(summary["prompt_token_observation_count"], 0)
            self.assertFalse(summary["prompt_token_ceiling_checked"])
            receipt = json.loads((root / "out" / "run-airline-seed101.json").read_text(encoding="utf-8"))
            self.assertEqual(receipt["terminal_status"], "failed")

    def test_reviewer_environment_is_loopback_bound_and_strips_credentials(self):
        endpoint = self._endpoint("local/reviewer", 18082)
        with patch.dict("os.environ", {"OPENAI_API_KEY": "secret", "HF_TOKEN": "secret", "PATH": "/bin"}, clear=True):
            env = _reviewer_environment(endpoint)
        self.assertEqual(env["OPENAI_API_BASE"], endpoint.api_base)
        self.assertEqual(env["OPENAI_API_KEY"], "local")
        self.assertNotIn("HF_TOKEN", env)
        self.assertEqual(env["PATH"], "/bin")

    def _repo(self, root: Path) -> Path:
        repo = root / "tau"
        repo.mkdir()
        subprocess.run(["git", "init"], cwd=repo, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
        subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
        (repo / "README.md").write_text("fixture\n", encoding="utf-8")
        subprocess.run(["git", "add", "README.md"], cwd=repo, check=True)
        subprocess.run(["git", "commit", "-m", "fixture"], cwd=repo, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, check=True, text=True, stdout=subprocess.PIPE).stdout.strip()
        return repo

    def _development_source(self, root: Path) -> Path:
        path = root / "development.json"
        payload = {
            "schema_version": "hfr.tau3_source_split.v1",
            "source_revision": self.revision,
            "split": "development",
            "task_schema_version": "tau2.tasks.v1",
            "algorithm": "test-family-split",
            "salt_sha256": "0" * 64,
            "task_count": 3,
            "family_count": 3,
            "family_ids": ["1" * 64, "2" * 64, "3" * 64],
            "tasks": [
                {"domain": "airline", "raw_id": "air-1", "raw_id_sha256": hashlib_sha256(b"airline:air-1").hexdigest(), "task_sha256": "a" * 64, "prompt_sha256": "b" * 64, "family_id": "1" * 64},
                {"domain": "retail", "raw_id": "ret-1", "raw_id_sha256": hashlib_sha256(b"retail:ret-1").hexdigest(), "task_sha256": "c" * 64, "prompt_sha256": "d" * 64, "family_id": "2" * 64},
                {"domain": "telecom", "raw_id": "tel-1", "raw_id_sha256": hashlib_sha256(b"telecom:tel-1").hexdigest(), "task_sha256": "e" * 64, "prompt_sha256": "f" * 64, "family_id": "3" * 64},
            ],
        }
        path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
        return path

    def _sealed_source_manifest(self, root: Path, *, task_count: int, entry_extra: dict[str, object] | None = None) -> Path:
        path = root / f"sealed-{len(list(root.glob('sealed-*.json')))}.json"
        entries = []
        for index in range(task_count):
            entry: dict[str, object] = {
                "task_id_sha256": hashlib_sha256(f"task-id-{index}".encode("utf-8")).hexdigest(),
                "prompt_sha256": hashlib_sha256(f"prompt-{index}".encode("utf-8")).hexdigest(),
                "task_sha256": hashlib_sha256(f"task-{index}".encode("utf-8")).hexdigest(),
            }
            if entry_extra is not None:
                entry.update(entry_extra)
            entries.append(entry)
        payload = {
            "schema_version": "hfr.tau3_sealed_source_manifest.v1",
            "source_revision": self.revision,
            "hashes_only": True,
            "task_count": task_count,
            "entries": entries,
        }
        path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
        return path

    def _candidate_lock(self, root: Path, *, adapter: Path, protocol: Path, created_at: str = "2026-07-22T00:00:00+00:00") -> Path:
        path = root / f"candidate-lock-{len(list(root.glob('candidate-lock-*.json')))}.json"
        endpoint = self._endpoint("local/adapter", 18080, adapter_path=adapter)
        payload = {
            "schema_version": "hfr.tau3_candidate_lock.v1",
            "created_at": created_at,
            "selected_candidate_id_hash": "1" * 64,
            "candidate_identity_sha256": "b" * 64,
            "development_selection_report_sha256": "2" * 64,
            "development_benchmark_manifest_sha256": "3" * 64,
            "training_receipt_sha256": "4" * 64,
            "endpoint_model_sha256": hashlib_sha256(endpoint.model.encode("utf-8")).hexdigest(),
            "adapter_tree_sha256": self._tree_sha256(adapter),
            "recipe_sha256": "5" * 64,
            "base_identity_sha256": hashlib_sha256(b"local/base").hexdigest(),
            "base_tree_sha256": "6" * 64,
            "dataset_manifest_sha256": "7" * 64,
            "dataset_files_sha256": "8" * 64,
            "source_binding_sha256": "9" * 64,
            "protocol_sha256": self._sha256(protocol),
            "protocol_signature": "a" * 64,
            "hashes_only": True,
            "sealed_access_authorized": True,
            "local_paths_included": False,
            "raw_payload_included": False,
        }
        path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
        return path

    def _sealed_authorization(self, root: Path, *, lock: Path, protocol: Path, sealed_manifest: Path) -> Path:
        path = root / f"sealed-authorization-{len(list(root.glob('sealed-authorization-*.json')))}.json"
        create_tau3_sealed_authorization(
            candidate_lock=lock,
            protocol=protocol,
            sealed_source_manifest=sealed_manifest,
            out=path,
            created_at="2026-07-22T00:00:01+00:00",
        )
        return path

    def _protocol(
        self,
        root: Path,
        *,
        source: Path | None = None,
        sealed_manifest: Path | None = None,
        candidate_lock: Path | None = None,
    ) -> Path:
        path = root / f"protocol-{len(list(root.glob('protocol-*.json')))}.json"
        dev_hash = self._sha256(source) if source is not None else "d" * 64
        sealed_hash = self._sha256(sealed_manifest) if sealed_manifest is not None else "e" * 64
        payload = {
            "schema_version": "hfr.tau3_protocol_config.v1",
            "protocol_manifest": {},
            "tau_revision": {
                "revision": self.revision,
                "split_hashes": {"train": "a" * 64, "development": dev_hash, "sealed": sealed_hash},
            },
            "split_manifest": {
                "splits": {
                    "train": {"sha256": "a" * 64, "sealed": False},
                    "development": {"sha256": dev_hash, "sealed": False},
                    "sealed": {"sha256": sealed_hash, "sealed": True},
                }
            },
            "harness_contract": {
                "domains": ["airline", "retail", "telecom"],
                "context_window": 16384,
                "decoding": {"temperature": 0.0, "top_p": 1.0, "max_output_tokens": 1024, "seeds": [101, 202, 303, 404]},
                "turn_limit": 30,
                "retry_policy": "none",
                "no_test_time_search": True,
                "test_time_search": False,
            },
            "model_freeze": {
                "base_model": {
                    "name": "local/base",
                    "revision": "base-rev",
                    "local_identity_sha256": hashlib_sha256(b"local/base").hexdigest(),
                    "local_path": "local/base",
                    "local_identity_path": "identities/base.json",
                },
                "comparators": [
                    {
                        "name": "local/comparator-1",
                        "revision": "cmp1-rev",
                        "local_identity_sha256": hashlib_sha256(b"local/comparator-1").hexdigest(),
                        "local_path": "local/comparator-1",
                        "local_identity_path": "identities/comparator-1.json",
                    },
                    {
                        "name": "local/comparator-2",
                        "revision": "cmp2-rev",
                        "local_identity_sha256": hashlib_sha256(b"local/comparator-2").hexdigest(),
                        "local_path": "local/comparator-2",
                        "local_identity_path": "identities/comparator-2.json",
                    },
                ],
            },
            "budget": {"passed": True, "max_training_hours": 24, "max_benchmark_hours": 48},
            "sealed_manifest": {
                "access_count": 0,
                "manifest_sha256": sealed_hash,
                "prompt_template_hashes": ["f" * 64],
            },
            "mlx_qlora_plan": {},
            "recipe_space": {},
            "candidate_selection_contract": {"passed": True},
            "contamination_attestation": {"passed": True},
            "redaction_attestation": {"passed": True},
            "licenses": [
                {"status": "approved", "training_allowed": True},
                {"status": "approved", "training_allowed": True},
                {"status": "approved", "training_allowed": True},
                {"status": "approved", "training_allowed": True},
            ],
            "environment_manifest": {},
        }
        path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
        return path

    def _fake_tau2(self, root: Path, *, reward: float, prompt_tokens: int | None = 42) -> Path:
        path = root / "tau2"
        usage_row = "{}" if prompt_tokens is None else "{'usage': {'prompt_tokens': prompt_tokens}}"
        path.write_text(
            "#!/usr/bin/env python3\n"
            "import json, pathlib, sys\n"
            "save_to=sys.argv[sys.argv.index('--save-to')+1]\n"
            f"reward={reward!r}\n"
            f"prompt_tokens={prompt_tokens!r}\n"
            f"usage_row={usage_row}\n"
            "out=pathlib.Path.cwd()/'data'/'simulations'/save_to\n"
            "out.mkdir(parents=True, exist_ok=True)\n"
            "row={'reward_info':{'reward':reward}}\n"
            "row.update(usage_row)\n"
            "json.dump({'simulations':[row]}, open(out/'results.json','w'))\n",
            encoding="utf-8",
        )
        path.chmod(0o755)
        return path

    def _endpoint(self, model: str, port: int, *, adapter_path: Path | None = None) -> Tau3BenchmarkEndpoint:
        return Tau3BenchmarkEndpoint(model=model, api_base=f"http://127.0.0.1:{port}/v1", adapter_path=adapter_path)

    def _sha256(self, path: Path) -> str:
        return hashlib_sha256(path.read_bytes()).hexdigest()

    def _adapter(self, root: Path) -> Path:
        adapter = root / f"adapter-{len(list(root.glob('adapter-*')))}"
        adapter.mkdir()
        (adapter / "adapter_config.json").write_text('{"r":16}\n', encoding="utf-8")
        (adapter / "adapters.safetensors").write_bytes(b"adapter")
        checkpoint = adapter / "checkpoint-0001"
        checkpoint.mkdir()
        (checkpoint / "weights.npz").write_bytes(b"checkpoint")
        return adapter

    def _tree_sha256(self, root: Path) -> str:
        records = []
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            rel = path.relative_to(root).as_posix()
            records.append({"path": rel, "size": path.stat().st_size, "sha256": self._sha256(path), "kind": self._fingerprint_kind(rel)})
        digest = hashlib_sha256(b"")
        for record in records:
            digest.update(json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        return digest.hexdigest()

    def _fingerprint_kind(self, rel: str) -> str:
        name = Path(rel).name
        if name in {"adapter_config.json", "config.json"}:
            return "config"
        if "checkpoint" in rel.lower():
            return "checkpoint"
        if Path(rel).suffix in {".safetensors", ".npz", ".bin"}:
            return "adapter"
        return "artifact"


def hashlib_sha256(data: bytes):
    import hashlib

    return hashlib.sha256(data)


if __name__ == "__main__":
    unittest.main()
