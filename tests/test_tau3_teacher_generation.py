from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from flightrecorder.tau3_teacher_generation import (
    Tau3Endpoint,
    Tau3TeacherGenerationConfig,
    Tau3TeacherGenerationError,
    _nl_assertions_judge_environment,
    _tau2_argv,
    main,
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

            (root / "out" / "manifest.json").unlink()
            protocol_payload = json.loads(protocol.read_text(encoding="utf-8"))
            protocol_payload["model_freeze"]["teachers"][0]["name"] = "changed-teacher-pin"
            protocol.write_text(json.dumps(protocol_payload) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(Tau3TeacherGenerationError, "prelaunch receipt does not match"):
                run_tau3_teacher_generation(
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

    def test_custom_decoding_is_bound_into_manifest_and_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = self._repo(root)
            source = self._source(root, split="train")
            tau2 = self._fake_tau2(root, repo, reward=1.0)
            out = root / "out"
            teacher = Tau3Endpoint(
                model="openai/local-teacher",
                api_base="http://127.0.0.1:1/v1",
                temperature=0.4,
                top_p=0.95,
                max_tokens=768,
            )
            user = Tau3Endpoint(
                model="openai/local-user",
                api_base="http://localhost:2/v1",
                temperature=0.2,
                top_p=0.8,
                max_tokens=384,
            )
            manifest = run_tau3_teacher_generation(
                source_jsonl=source,
                out_dir=out,
                tau_repo=repo,
                tau_venv_bin=tau2,
                expected_tau_revision=self.revision,
                teacher=teacher,
                user=user,
                config=Tau3TeacherGenerationConfig(max_tasks=1, timeout_seconds=2),
                created_at="2026-07-22T00:00:00Z",
            )
            self.assertEqual(manifest["teacher"]["temperature"], 0.4)
            self.assertEqual(manifest["teacher"]["top_p"], 0.95)
            self.assertEqual(manifest["teacher"]["max_tokens"], 768)
            self.assertEqual(manifest["user_simulator"]["temperature"], 0.2)
            self.assertEqual(manifest["user_simulator"]["top_p"], 0.8)
            self.assertEqual(manifest["user_simulator"]["max_tokens"], 384)

            prelaunch = json.loads((out / "prelaunch_receipt.json").read_text(encoding="utf-8"))
            self.assertEqual(prelaunch["teacher"], manifest["teacher"])
            self.assertEqual(prelaunch["user_simulator"], manifest["user_simulator"])
            self.assertEqual(prelaunch["nl_assertions_judge"], manifest["nl_assertions_judge"])
            self.assertEqual(manifest["nl_assertions_judge"]["request_model"], "gpt-4.1-2025-04-14")
            self.assertEqual(manifest["nl_assertions_judge"]["served_model"], teacher.model)
            self.assertEqual(manifest["nl_assertions_judge"]["endpoint"], teacher.api_base)
            self.assertEqual(manifest["nl_assertions_judge"]["temperature"], 0.0)
            self.assertIsNone(manifest["nl_assertions_judge"]["top_p"])
            self.assertIsNone(manifest["nl_assertions_judge"]["max_tokens"])

            receipt = json.loads(next(out.glob("task-*.json")).read_text(encoding="utf-8"))
            command = receipt["command"]
            teacher_args = json.loads(command[command.index("--agent-llm-args") + 1])
            user_args = json.loads(command[command.index("--user-llm-args") + 1])
            self.assertEqual(teacher_args["temperature"], 0.4)
            self.assertEqual(teacher_args["top_p"], 0.95)
            self.assertEqual(teacher_args["max_tokens"], 768)
            self.assertEqual(user_args["temperature"], 0.2)
            self.assertEqual(user_args["top_p"], 0.8)
            self.assertEqual(user_args["max_tokens"], 384)
            self.assertEqual(teacher_args["api_base"], teacher.api_base)
            self.assertEqual(user_args["api_base"], user.api_base)

    def test_nl_assertions_judge_environment_is_local_and_strips_credentials(self):
        endpoint = Tau3Endpoint(
            model="openai/local-teacher",
            api_base="http://127.0.0.1:18082/v1",
        )
        with patch.dict(
            "os.environ",
            {
                "OPENAI_API_KEY": "external-secret",
                "OPENAI_BASE_URL": "https://api.openai.com/v1",
                "ANTHROPIC_API_KEY": "external-secret",
                "HF_TOKEN": "external-secret",
                "AWS_ACCESS_KEY_ID": "external-secret",
                "AWS_SECRET_ACCESS_KEY": "external-secret",
                "GOOGLE_APPLICATION_CREDENTIALS": "/tmp/external-secret.json",
                "GITHUB_PAT": "external-secret",
                "SSH_AUTH_SOCK": "/tmp/external-agent.sock",
                "GENERIC_SECRET": "external-secret",
                "PATH": "/bin",
            },
            clear=True,
        ):
            env = _nl_assertions_judge_environment(endpoint)
        self.assertEqual(env["OPENAI_API_BASE"], endpoint.api_base)
        self.assertEqual(env["OPENAI_API_KEY"], "local")
        self.assertNotIn("OPENAI_BASE_URL", env)
        self.assertNotIn("ANTHROPIC_API_KEY", env)
        self.assertNotIn("HF_TOKEN", env)
        self.assertNotIn("AWS_ACCESS_KEY_ID", env)
        self.assertNotIn("AWS_SECRET_ACCESS_KEY", env)
        self.assertNotIn("GOOGLE_APPLICATION_CREDENTIALS", env)
        self.assertNotIn("GITHUB_PAT", env)
        self.assertNotIn("SSH_AUTH_SOCK", env)
        self.assertNotIn("GENERIC_SECRET", env)
        self.assertEqual(env["PATH"], "/bin")
        self.assertEqual(env["NO_PROXY"], "127.0.0.1,localhost")
        self.assertEqual(env["no_proxy"], "127.0.0.1,localhost")

    def test_generation_subprocess_receives_only_local_judge_credentials(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = self._repo(root)
            source = self._source(root, split="train")
            tau2 = self._fake_tau2(root, repo, reward=1.0)
            endpoint = Tau3Endpoint(
                model="openai/local-teacher",
                api_base="http://127.0.0.1:18082/v1",
            )
            with patch.dict(
                "os.environ",
                {
                    "OPENAI_API_KEY": "external-secret",
                    "OPENAI_BASE_URL": "https://api.openai.com/v1",
                    "ANTHROPIC_API_KEY": "external-secret",
                    "HF_TOKEN": "external-secret",
                    "AWS_ACCESS_KEY_ID": "external-secret",
                    "AWS_SECRET_ACCESS_KEY": "external-secret",
                    "GOOGLE_APPLICATION_CREDENTIALS": "/tmp/external-secret.json",
                    "GITHUB_PAT": "external-secret",
                    "SSH_AUTH_SOCK": "/tmp/external-agent.sock",
                    "GENERIC_SECRET": "external-secret",
                },
            ):
                run_tau3_teacher_generation(
                    source_jsonl=source,
                    out_dir=root / "out",
                    tau_repo=repo,
                    tau_venv_bin=tau2,
                    expected_tau_revision=self.revision,
                    teacher=endpoint,
                    user=Tau3Endpoint(
                        model="openai/local-user",
                        api_base="http://127.0.0.1:18083/v1",
                    ),
                    config=Tau3TeacherGenerationConfig(max_tasks=1, timeout_seconds=2),
                )
            env_path = next((repo / "data" / "simulations" / "hfr-generation" / "out").rglob("judge-env.json"))
            observed = json.loads(env_path.read_text(encoding="utf-8"))
            self.assertEqual(observed["OPENAI_API_BASE"], endpoint.api_base)
            self.assertEqual(observed["OPENAI_API_KEY"], "local")
            self.assertIsNone(observed["OPENAI_BASE_URL"])
            self.assertIsNone(observed["ANTHROPIC_API_KEY"])
            self.assertIsNone(observed["HF_TOKEN"])
            self.assertIsNone(observed["AWS_ACCESS_KEY_ID"])
            self.assertIsNone(observed["AWS_SECRET_ACCESS_KEY"])
            self.assertIsNone(observed["GOOGLE_APPLICATION_CREDENTIALS"])
            self.assertIsNone(observed["GITHUB_PAT"])
            self.assertIsNone(observed["SSH_AUTH_SOCK"])
            self.assertIsNone(observed["GENERIC_SECRET"])

    def test_nl_assertions_judge_rejects_credentialed_or_ambiguous_loopback_urls(self):
        bad_urls = (
            "http://user:pass@127.0.0.1:18082/v1",
            "http://127.0.0.1:18082/v1?key=secret",
            "http://127.0.0.1:18082/v1#secret",
        )
        for api_base in bad_urls:
            with self.subTest(api_base=api_base):
                endpoint = Tau3Endpoint(model="openai/local-teacher", api_base=api_base)
                with self.assertRaises(Tau3TeacherGenerationError):
                    _nl_assertions_judge_environment(endpoint)

    def test_rejects_unsafe_decoding_ranges(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = self._repo(root)
            source = self._source(root, split="train")
            tau2 = self._fake_tau2(root, repo, reward=1.0)
            valid = Tau3Endpoint(model="openai/local", api_base="http://127.0.0.1:1/v1")
            cases = [
                ("teacher temperature", Tau3Endpoint(model="openai/local", api_base="http://127.0.0.1:1/v1", temperature=-0.1), valid),
                ("teacher top_p", Tau3Endpoint(model="openai/local", api_base="http://127.0.0.1:1/v1", top_p=0.0), valid),
                ("teacher max_tokens", Tau3Endpoint(model="openai/local", api_base="http://127.0.0.1:1/v1", max_tokens=0), valid),
                ("user temperature", valid, Tau3Endpoint(model="openai/local-user", api_base="http://127.0.0.1:2/v1", temperature=2.1)),
                ("user top_p", valid, Tau3Endpoint(model="openai/local-user", api_base="http://127.0.0.1:2/v1", top_p=1.1)),
                ("user max_tokens", valid, Tau3Endpoint(model="openai/local-user", api_base="http://127.0.0.1:2/v1", max_tokens=8193)),
            ]
            for message, teacher, user in cases:
                with self.subTest(message=message):
                    with self.assertRaisesRegex(Tau3TeacherGenerationError, message):
                        run_tau3_teacher_generation(
                            source_jsonl=source,
                            out_dir=root / message.replace(" ", "-"),
                            tau_repo=repo,
                            tau_venv_bin=tau2,
                            expected_tau_revision=self.revision,
                            teacher=teacher,
                            user=user,
                            config=Tau3TeacherGenerationConfig(max_tasks=1, timeout_seconds=2),
                        )

    def test_cli_accepts_decoding_controls_with_default_safe_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = self._repo(root)
            source = self._source(root, split="train")
            tau2 = self._fake_tau2(root, repo, reward=1.0)
            out = root / "out"
            exit_code = main(
                [
                    "--source-jsonl",
                    str(source),
                    "--out",
                    str(out),
                    "--tau-repo",
                    str(repo),
                    "--tau-venv-bin",
                    str(tau2),
                    "--expected-tau-revision",
                    self.revision,
                    "--teacher-model",
                    "openai/local-teacher",
                    "--teacher-api-base",
                    "http://127.0.0.1:1/v1",
                    "--teacher-temperature",
                    "0.6",
                    "--teacher-top-p",
                    "0.7",
                    "--teacher-max-tokens",
                    "512",
                    "--user-model",
                    "openai/local-user",
                    "--user-api-base",
                    "http://127.0.0.1:2/v1",
                    "--max-tasks",
                    "1",
                    "--timeout-seconds",
                    "2",
                ]
            )
            self.assertEqual(exit_code, 0)
            manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["teacher"]["temperature"], 0.6)
            self.assertEqual(manifest["teacher"]["top_p"], 0.7)
            self.assertEqual(manifest["teacher"]["max_tokens"], 512)
            self.assertEqual(manifest["user_simulator"]["temperature"], 0.0)
            self.assertEqual(manifest["user_simulator"]["top_p"], 1.0)
            self.assertEqual(manifest["user_simulator"]["max_tokens"], 1024)

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

    def test_partial_resume_rejects_nl_assertions_judge_binding_drift(self):
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
                config=Tau3TeacherGenerationConfig(max_tasks=1, timeout_seconds=2),
            )
            (out / "manifest.json").unlink()
            prelaunch_path = out / "prelaunch_receipt.json"
            prelaunch = json.loads(prelaunch_path.read_text(encoding="utf-8"))
            prelaunch["nl_assertions_judge"]["served_model"] = "drifted-model"
            prelaunch_path.write_text(json.dumps(prelaunch) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(Tau3TeacherGenerationError, "prelaunch receipt does not match"):
                run_tau3_teacher_generation(
                    source_jsonl=source,
                    out_dir=out,
                    tau_repo=repo,
                    tau_venv_bin=tau2,
                    expected_tau_revision=self.revision,
                    teacher=endpoint,
                    user=endpoint,
                    config=Tau3TeacherGenerationConfig(max_tasks=1, timeout_seconds=2),
                )

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
            final_manifest = out / "manifest.json"
            manifest_payload = json.loads(final_manifest.read_text(encoding="utf-8"))
            manifest_payload["nl_assertions_judge"]["served_model"] = "drifted-model"
            final_manifest.write_text(json.dumps(manifest_payload) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(Tau3TeacherGenerationError, "nl_assertions_judge"):
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
            manifest_payload["nl_assertions_judge"]["served_model"] = endpoint.model
            final_manifest.write_text(json.dumps(manifest_payload) + "\n", encoding="utf-8")
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
            "import json, os, pathlib, sys\n"
            "save_to=sys.argv[sys.argv.index('--save-to')+1]\n"
            f"reward={reward!r}\n"
            "out=pathlib.Path.cwd()/'data'/'simulations'/save_to\n"
            "out.mkdir(parents=True, exist_ok=True)\n"
            "json.dump({key: os.environ.get(key) for key in "
            "['OPENAI_API_BASE', 'OPENAI_BASE_URL', 'OPENAI_API_KEY', "
            "'ANTHROPIC_API_KEY', 'HF_TOKEN', 'AWS_ACCESS_KEY_ID', "
            "'AWS_SECRET_ACCESS_KEY', 'GOOGLE_APPLICATION_CREDENTIALS', "
            "'GITHUB_PAT', 'SSH_AUTH_SOCK', 'GENERIC_SECRET']}, "
            "open(out/'judge-env.json','w'))\n"
            "json.dump({'simulations':[{'reward_info':{'reward':reward}}]}, open(out/'results.json','w'))\n",
            encoding="utf-8",
        )
        path.chmod(0o755)
        return path


if __name__ == "__main__":
    unittest.main()
