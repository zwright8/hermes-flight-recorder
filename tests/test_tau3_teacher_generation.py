from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from flightrecorder.tau3_teacher_generation import (
    Tau3Endpoint,
    Tau3TeacherGenerationConfig,
    Tau3TeacherGenerationError,
    _tau2_argv,
    run_tau3_teacher_generation,
)


REVISION = "1d244f5dca42944b67a379b44bfeb9f5748f189d"


class Tau3TeacherGenerationTests(unittest.TestCase):
    def test_binds_teacher_frozen_protocol_and_rejects_revision_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = self._repo(root)
            source = self._source(root, split="train")
            tau2 = self._fake_tau2(root, repo, reward=1.0)
            protocol = root / "protocol.json"
            protocol.write_text(
                json.dumps(
                    {
                        "schema_version": "hfr.tau3_protocol_config.v1",
                        "tau_revision": {"revision": self.revision},
                        "model_freeze": {"teachers": [{"name": "local-teacher"}]},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            endpoint = Tau3Endpoint(model="openai/local", api_base="http://127.0.0.1:1/v1")
            manifest = run_tau3_teacher_generation(
                source_jsonl=source,
                out_dir=root / "out",
                tau_repo=repo,
                tau_venv_bin=tau2,
                expected_tau_revision=self.revision,
                teacher=endpoint,
                user=endpoint,
                protocol_path=protocol,
                config=Tau3TeacherGenerationConfig(max_tasks=1, timeout_seconds=2),
            )
            self.assertEqual(manifest["protocol"]["sha256"], hashlib.sha256(protocol.read_bytes()).hexdigest())

            bad = root / "bad-protocol.json"
            bad.write_text(
                json.dumps(
                    {
                        "schema_version": "hfr.tau3_protocol_config.v1",
                        "tau_revision": {"revision": "0" * 40},
                        "model_freeze": {"teachers": [{"name": "local-teacher"}]},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(Tau3TeacherGenerationError, "revision mismatch"):
                run_tau3_teacher_generation(
                    source_jsonl=source,
                    out_dir=root / "bad-out",
                    tau_repo=repo,
                    tau_venv_bin=tau2,
                    expected_tau_revision=self.revision,
                    teacher=endpoint,
                    user=endpoint,
                    protocol_path=bad,
                    config=Tau3TeacherGenerationConfig(max_tasks=1, timeout_seconds=2),
                )

    def test_success_failure_and_resume_are_receipted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = self._repo(root)
            source = self._source(root, split="train")
            tau2 = self._fake_tau2(root, repo, reward=1.0)
            out = root / "out"
            manifest = run_tau3_teacher_generation(
                source_jsonl=source,
                out_dir=out,
                tau_repo=repo,
                tau_venv_bin=tau2,
                expected_tau_revision=self.revision,
                teacher=Tau3Endpoint(model="openai/local-teacher", api_base="http://127.0.0.1:1/v1"),
                user=Tau3Endpoint(model="openai/local-user", api_base="http://localhost:2/v1"),
                config=Tau3TeacherGenerationConfig(max_tasks=1, timeout_seconds=2),
                created_at="2026-07-22T00:00:00Z",
            )
            self.assertEqual(manifest["success_count"], 1)
            resumed = run_tau3_teacher_generation(
                source_jsonl=source,
                out_dir=out,
                tau_repo=repo,
                tau_venv_bin=tau2,
                expected_tau_revision=self.revision,
                teacher=Tau3Endpoint(model="openai/local-teacher", api_base="http://127.0.0.1:1/v1"),
                user=Tau3Endpoint(model="openai/local-user", api_base="http://localhost:2/v1"),
                config=Tau3TeacherGenerationConfig(max_tasks=1, timeout_seconds=2),
                created_at="2026-07-22T00:00:00Z",
            )
            self.assertEqual(resumed["success_count"], 1)

    def test_rejects_sealed_and_remote_endpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = self._repo(root)
            tau2 = self._fake_tau2(root, repo, reward=1.0)
            with self.assertRaisesRegex(Tau3TeacherGenerationError, "loopback"):
                run_tau3_teacher_generation(
                    source_jsonl=self._source(root, split="train"),
                    out_dir=root / "out",
                    tau_repo=repo,
                    tau_venv_bin=tau2,
                    expected_tau_revision=self.revision,
                    teacher=Tau3Endpoint(model="openai/remote", api_base="https://api.openai.com/v1"),
                    user=Tau3Endpoint(model="openai/local-user", api_base="http://127.0.0.1:2/v1"),
                )
            with self.assertRaisesRegex(Tau3TeacherGenerationError, "sealed/test"):
                run_tau3_teacher_generation(
                    source_jsonl=self._source(root, split="sealed"),
                    out_dir=root / "out2",
                    tau_repo=repo,
                    tau_venv_bin=tau2,
                    expected_tau_revision=self.revision,
                    teacher=Tau3Endpoint(model="openai/local", api_base="http://127.0.0.1:1/v1"),
                    user=Tau3Endpoint(model="openai/local-user", api_base="http://127.0.0.1:2/v1"),
                )

    def test_records_failed_reward(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = self._repo(root)
            source = self._source(root, split="train")
            tau2 = self._fake_tau2(root, repo, reward=0.0)
            manifest = run_tau3_teacher_generation(
                source_jsonl=source,
                out_dir=root / "out",
                tau_repo=repo,
                tau_venv_bin=tau2,
                expected_tau_revision=self.revision,
                teacher=Tau3Endpoint(model="openai/local", api_base="http://127.0.0.1:1/v1"),
                user=Tau3Endpoint(model="openai/local-user", api_base="http://127.0.0.1:2/v1"),
                config=Tau3TeacherGenerationConfig(max_tasks=1, timeout_seconds=2),
            )
            self.assertEqual(manifest["failure_count"], 1)

    def test_auto_agent_maps_logical_development_to_official_train(self):
        root = Path("/tmp/example")
        common = {
            "domain": "airline",
            "split": "development",
            "official_task_split": "train",
            "task_id": "1",
            "task_family": "family-1",
            "task_sha256": "a" * 64,
            "prompt_sha256": "b" * 64,
        }
        endpoint = Tau3Endpoint(model="openai//local/model", api_base="http://127.0.0.1:18082/v1")
        action_argv = _tau2_argv(
            tau2=root / "tau2",
            task={**common, "has_reference_actions": "true"},
            save_to="fixture/action",
            teacher=endpoint,
            user=endpoint,
            cfg=Tau3TeacherGenerationConfig(agent="auto"),
        )
        no_action_argv = _tau2_argv(
            tau2=root / "tau2",
            task={**common, "has_reference_actions": "false"},
            save_to="fixture/no-action",
            teacher=endpoint,
            user=endpoint,
            cfg=Tau3TeacherGenerationConfig(agent="auto"),
        )
        self.assertEqual(action_argv[action_argv.index("--task-split-name") + 1], "train")
        self.assertEqual(action_argv[action_argv.index("--agent") + 1], "llm_agent_gt")
        self.assertEqual(no_action_argv[no_action_argv.index("--agent") + 1], "llm_agent")

    def test_partial_run_resumes_existing_task_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = self._repo(root)
            source = self._source(root, split="train")
            tau2 = self._fake_tau2(root, repo, reward=0.0)
            out = root / "out"
            manifest = run_tau3_teacher_generation(
                source_jsonl=source,
                out_dir=out,
                tau_repo=repo,
                tau_venv_bin=tau2,
                expected_tau_revision=self.revision,
                teacher=Tau3Endpoint(model="openai/local", api_base="http://127.0.0.1:1/v1"),
                user=Tau3Endpoint(model="openai/local-user", api_base="http://127.0.0.1:2/v1"),
                config=Tau3TeacherGenerationConfig(max_tasks=1, timeout_seconds=2),
            )
            (out / "manifest.json").unlink()
            resumed = run_tau3_teacher_generation(
                source_jsonl=source,
                out_dir=out,
                tau_repo=repo,
                tau_venv_bin=tau2,
                expected_tau_revision=self.revision,
                teacher=Tau3Endpoint(model="openai/local", api_base="http://127.0.0.1:1/v1"),
                user=Tau3Endpoint(model="openai/local-user", api_base="http://127.0.0.1:2/v1"),
                config=Tau3TeacherGenerationConfig(max_tasks=1, timeout_seconds=2),
            )
            self.assertEqual(manifest["failure_count"], 1)
            self.assertEqual(resumed["failure_count"], 1)

    def test_final_resume_rejects_changed_config_and_drifted_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = self._repo(root)
            source = self._source(root, split="train")
            tau2 = self._fake_tau2(root, repo, reward=1.0)
            out = root / "out"
            endpoint = Tau3Endpoint(model="openai/local", api_base="http://127.0.0.1:1/v1")
            run_tau3_teacher_generation(
                source_jsonl=source,
                out_dir=out,
                tau_repo=repo,
                tau_venv_bin=tau2,
                expected_tau_revision=self.revision,
                teacher=endpoint,
                user=endpoint,
                config=Tau3TeacherGenerationConfig(max_tasks=1, timeout_seconds=2, seed=101),
            )
            with self.assertRaisesRegex(Tau3TeacherGenerationError, "stale or invalid"):
                run_tau3_teacher_generation(
                    source_jsonl=source,
                    out_dir=out,
                    tau_repo=repo,
                    tau_venv_bin=tau2,
                    expected_tau_revision=self.revision,
                    teacher=endpoint,
                    user=endpoint,
                    config=Tau3TeacherGenerationConfig(max_tasks=1, timeout_seconds=2, seed=202),
                )

            receipt = next(out.glob("task-*.json"))
            result_path = Path(json.loads(receipt.read_text(encoding="utf-8"))["result_path"])
            result_path.write_text('{"simulations": []}\n', encoding="utf-8")
            with self.assertRaisesRegex(Tau3TeacherGenerationError, "generated result drifted"):
                run_tau3_teacher_generation(
                    source_jsonl=source,
                    out_dir=out,
                    tau_repo=repo,
                    tau_venv_bin=tau2,
                    expected_tau_revision=self.revision,
                    teacher=endpoint,
                    user=endpoint,
                    config=Tau3TeacherGenerationConfig(max_tasks=1, timeout_seconds=2, seed=101),
                )

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

    def _source(self, root: Path, *, split: str) -> Path:
        path = root / f"{split}.jsonl"
        row = {
            "domain": "airline",
            "split": split,
            "source_revision": self.revision,
            "task_family": "family-1",
            "task_sha256": "a" * 64,
            "prompt_sha256": "b" * 64,
            "task": {"id": "1", "evaluation_criteria": {"actions": [{"name": "lookup"}]}},
        }
        path.write_text(json.dumps(row) + "\n", encoding="utf-8")
        return path

    def _fake_tau2(self, root: Path, repo: Path, *, reward: float) -> Path:
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


if __name__ == "__main__":
    unittest.main()
