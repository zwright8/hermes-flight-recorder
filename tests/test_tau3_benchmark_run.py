from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from flightrecorder.tau3_benchmark_run import (
    Tau3BenchmarkConfig,
    Tau3BenchmarkEndpoint,
    Tau3BenchmarkRunError,
    _reviewer_environment,
    _tau2_argv,
    run_tau3_benchmark_arm,
)


class Tau3BenchmarkRunTests(unittest.TestCase):
    def test_development_runs_domain_seed_commands_and_resumes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = self._repo(root)
            source = self._development_source(root)
            tau2 = self._fake_tau2(root, reward=0.0)
            endpoint = self._endpoint("local/model", 18080)

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
            lock = root / "candidate-lock.json"
            lock.write_text('{"candidate":"adapter"}\n', encoding="utf-8")
            endpoint = self._endpoint("local/model", 18080)

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
                    candidate_lock=lock,
                    candidate_lock_sha256=self._sha256(lock),
                    timeout_seconds=2,
                ),
            )

            self.assertIsNone(manifest["source"])
            self.assertEqual(manifest["candidate_lock"]["sha256"], self._sha256(lock))
            self.assertFalse(manifest["task_selection"]["task_ids_in_command"])
            command = json.loads((root / "sealed-out" / "run-airline-seed202.json").read_text(encoding="utf-8"))["command"]
            self.assertEqual(command[command.index("--task-split-name") + 1], "test")
            self.assertNotIn("--task-ids", command)

            with self.assertRaisesRegex(Tau3BenchmarkRunError, "candidate-lock"):
                run_tau3_benchmark_arm(
                    out_dir=root / "sealed-missing-lock",
                    tau_repo=repo,
                    tau_venv_bin=tau2,
                    expected_tau_revision=self.revision,
                    agent=endpoint,
                    user=endpoint,
                    reviewer=endpoint,
                    config=Tau3BenchmarkConfig(mode="sealed", arm_id="adapter"),
                )

    def test_refuses_remote_endpoint_changed_resume_and_unreceipted_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = self._repo(root)
            source = self._development_source(root)
            tau2 = self._fake_tau2(root, reward=1.0)
            endpoint = self._endpoint("local/model", 18080)
            with self.assertRaisesRegex(Tau3BenchmarkRunError, "loopback"):
                run_tau3_benchmark_arm(
                    out_dir=root / "remote",
                    tau_repo=repo,
                    tau_venv_bin=tau2,
                    expected_tau_revision=self.revision,
                    agent=Tau3BenchmarkEndpoint(model="remote", api_base="https://api.example.test/v1"),
                    user=endpoint,
                    reviewer=endpoint,
                    config=Tau3BenchmarkConfig(mode="development", arm_id="base", source_split=source),
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
                config=Tau3BenchmarkConfig(mode="development", arm_id="base", source_split=source, timeout_seconds=2),
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
                    config=Tau3BenchmarkConfig(mode="development", arm_id="base", source_split=source, timeout_seconds=3),
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
                    config=Tau3BenchmarkConfig(mode="development", arm_id="base", source_split=source, timeout_seconds=2),
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
            config=Tau3BenchmarkConfig(mode="development", arm_id="base", source_split=Path("/tmp/dev.json")),
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
            endpoint = self._endpoint("local/model", 18080)
            source = root / "development.jsonl"
            source.write_text('{"domain":"airline","raw_id":"1"}\n', encoding="utf-8")
            with self.assertRaisesRegex(Tau3BenchmarkRunError, "seeds must be exactly"):
                run_tau3_benchmark_arm(
                    out_dir=root / "bad-seeds",
                    tau_repo=repo,
                    tau_venv_bin=tau2,
                    expected_tau_revision=self.revision,
                    agent=endpoint,
                    user=endpoint,
                    reviewer=endpoint,
                    config=Tau3BenchmarkConfig(mode="development", arm_id="base", source_split=source, seeds=(101,)),
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
                    config=Tau3BenchmarkConfig(mode="development", arm_id="base", source_split=source),
                )

    def test_completed_receipt_requires_hashed_result_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = self._repo(root)
            source = self._development_source(root)
            tau2 = self._fake_tau2(root, reward=0.0)
            endpoint = self._endpoint("local/model", 18080)
            out = root / "out"
            run_tau3_benchmark_arm(
                out_dir=out,
                tau_repo=repo,
                tau_venv_bin=tau2,
                expected_tau_revision=self.revision,
                agent=endpoint,
                user=endpoint,
                reviewer=endpoint,
                config=Tau3BenchmarkConfig(mode="development", arm_id="base", source_split=source, timeout_seconds=2),
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
                    config=Tau3BenchmarkConfig(mode="development", arm_id="base", source_split=source, timeout_seconds=2),
                )

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
                {"domain": "airline", "raw_id": "air-1", "raw_id_sha256": hashlib_sha256(b"air-1").hexdigest(), "task_sha256": "a" * 64, "prompt_sha256": "b" * 64, "family_id": "1" * 64},
                {"domain": "retail", "raw_id": "ret-1", "raw_id_sha256": hashlib_sha256(b"ret-1").hexdigest(), "task_sha256": "c" * 64, "prompt_sha256": "d" * 64, "family_id": "2" * 64},
                {"domain": "telecom", "raw_id": "tel-1", "raw_id_sha256": hashlib_sha256(b"tel-1").hexdigest(), "task_sha256": "e" * 64, "prompt_sha256": "f" * 64, "family_id": "3" * 64},
            ],
        }
        path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
        return path

    def _fake_tau2(self, root: Path, *, reward: float) -> Path:
        path = root / "tau2"
        path.write_text(
            "#!/usr/bin/env python3\n"
            "import json, pathlib, sys\n"
            "save_to=sys.argv[sys.argv.index('--save-to')+1]\n"
            f"reward={reward!r}\n"
            "out=pathlib.Path.cwd()/'data'/'simulations'/save_to\n"
            "out.mkdir(parents=True, exist_ok=True)\n"
            "json.dump({'simulations':[{'reward_info':{'reward':reward}}]}, open(out/'results.json','w'))\n",
            encoding="utf-8",
        )
        path.chmod(0o755)
        return path

    def _endpoint(self, model: str, port: int) -> Tau3BenchmarkEndpoint:
        return Tau3BenchmarkEndpoint(model=model, api_base=f"http://127.0.0.1:{port}/v1")

    def _sha256(self, path: Path) -> str:
        return hashlib_sha256(path.read_bytes()).hexdigest()


def hashlib_sha256(data: bytes):
    import hashlib

    return hashlib.sha256(data)


if __name__ == "__main__":
    unittest.main()
