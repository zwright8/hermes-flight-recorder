import importlib.util
import json
import socket
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

from flightrecorder.cli import main as flightrecorder_main
from flightrecorder.schema_registry import check_schema_file
from flightrecorder.validation import validate_artifacts


ROOT = Path(__file__).resolve().parents[1]
LIFECYCLE_SCRIPT = ROOT / "scripts" / "manage_openai_serving.py"


def _load_script(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


class ServingLifecycleTests(unittest.TestCase):
    def test_committed_agentic_training_serving_lifecycle_is_replayable(self):
        root = ROOT / "examples" / "agentic_training" / "serving_lifecycle" / "managed_mock"
        lifecycle_path = root / "serving_lifecycle.json"
        lifecycle = _read_json(lifecycle_path)

        self.assertEqual(lifecycle["schema_version"], "hfr.serving_lifecycle.v1")
        self.assertTrue(lifecycle["passed"], lifecycle)
        self.assertTrue(lifecycle["ready"])
        self.assertEqual(lifecycle["readiness"], "ready")
        self.assertEqual(lifecycle["profile"], "mock")
        self.assertEqual(lifecycle["engine"], "mock")
        self.assertEqual(lifecycle["environment"]["platform"], "offline-fixture")
        self.assertFalse(lifecycle["adapter_strategy"]["present"])
        self.assertEqual(lifecycle["artifacts"]["serving_profile"], "preflight/serving_profile.json")
        self.assertEqual(lifecycle["artifacts"]["compatibility_report"], "preflight/compatibility_report.json")
        self.assertEqual(lifecycle["artifacts"]["serving_check"], "preflight/serving_check.json")
        self.assertTrue(lifecycle["readiness_probe"]["ready"])
        self.assertTrue(lifecycle["smoke_check"]["passed"])
        self.assertTrue(lifecycle["teardown"]["clean"])

        for artifact in (
            "preflight/serving_profile.json",
            "preflight/compatibility_report.json",
            "preflight/serving_check.json",
            "server.stdout.log",
            "server.stderr.log",
        ):
            self.assertTrue((root / artifact).is_file(), artifact)
        for artifact in (
            "serving_lifecycle.json",
            "preflight/serving_profile.json",
            "preflight/compatibility_report.json",
            "preflight/serving_check.json",
        ):
            schema_result = check_schema_file(root / artifact)
            self.assertTrue(schema_result["passed"], schema_result["errors"])

        validation = validate_artifacts(serving_lifecycle_paths=[lifecycle_path], strict=True)
        self.assertTrue(validation["passed"], validation)

    def test_managed_mock_lifecycle_runs_preflight_and_tears_down(self):
        manage_openai_serving = _load_script(LIFECYCLE_SCRIPT, "manage_openai_serving_success")
        port = _free_port()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "managed"
            adapter = Path(tmp) / "adapter"
            adapter.mkdir()
            (adapter / "adapter_config.json").write_text('{"r": 8}\n', encoding="utf-8")
            with redirect_stdout(StringIO()):
                code = manage_openai_serving.main(
                    [
                        "--profile",
                        "mock",
                        "--host",
                        "127.0.0.1",
                        "--port",
                        str(port),
                        "--base-url",
                        f"http://127.0.0.1:{port}/v1",
                        "--model",
                        "hfr-managed-mock",
                        "--served-model-name",
                        "hfr-managed-mock",
                        "--adapter",
                        str(adapter),
                        "--out",
                        str(out),
                        "--startup-timeout",
                        "10",
                        "--poll-interval",
                        "0.1",
                        "--grace-period",
                        "2",
                        "--require-streaming",
                        "--require-tool-call",
                        "--require-structured-output",
                    ]
                )

            self.assertEqual(code, 0)
            lifecycle = _read_json(out / "serving_lifecycle.json")
            self.assertEqual(lifecycle["schema_version"], "hfr.serving_lifecycle.v1")
            self.assertTrue(lifecycle["passed"], lifecycle)
            self.assertEqual(lifecycle["readiness"], "ready")
            self.assertTrue(lifecycle["readiness_probe"]["ready"])
            self.assertTrue(lifecycle["smoke_check"]["passed"])
            self.assertTrue(lifecycle["teardown"]["clean"])
            self.assertFalse(lifecycle["teardown"]["running_after_teardown"])
            self.assertEqual(lifecycle["artifacts"]["serving_profile"], "preflight/serving_profile.json")
            self.assertEqual(lifecycle["adapter_strategy"]["resolved_strategy"], "mock_suffix")
            self.assertTrue(lifecycle["adapter_strategy"]["launch_command_applies_adapter"])
            self.assertIn("mock_openai_server_ready", lifecycle["logs"]["stdout_tail"])
            profile = _read_json(out / "preflight" / "serving_profile.json")
            self.assertEqual(profile["capabilities"]["streaming"], "supported")
            self.assertEqual(profile["adapter_strategy"]["resolved_strategy"], "mock_suffix")
            self.assertEqual(profile["model_identity"]["adapter_strategy"]["adapter_id"], "adapter")

            schema_result = check_schema_file(out / "serving_lifecycle.json")
            self.assertTrue(schema_result["passed"], schema_result["errors"])
            self.assertEqual(schema_result["schema"]["name"], "serving_lifecycle")

            profile_result = check_schema_file(out / "preflight" / "serving_profile.json")
            self.assertTrue(profile_result["passed"], profile_result["errors"])

            with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                validation_code = flightrecorder_main(["validate", "--serving-lifecycle", str(out / "serving_lifecycle.json"), "--strict"])
            self.assertEqual(validation_code, 0)

    def test_validate_lifecycle_rejects_forged_ready_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            lifecycle_path = root / "serving_lifecycle.json"
            summary_path = root / "validation.json"
            lifecycle = _blocked_lifecycle()
            lifecycle["passed"] = True
            lifecycle["ready"] = True
            lifecycle["readiness"] = "ready"
            _write_json(lifecycle_path, lifecycle)

            with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                code = flightrecorder_main(
                    [
                        "validate",
                        "--serving-lifecycle",
                        str(lifecycle_path),
                        "--strict",
                        "--out",
                        str(summary_path),
                    ]
                )

            self.assertEqual(code, 1)
            summary = _read_json(summary_path)
            errors = "\n".join(error for target in summary["targets"] for error in target["errors"])
            self.assertIn("serving_lifecycle.passed", errors)
            self.assertIn("serving_lifecycle.ready", errors)
            self.assertIn("serving_lifecycle.readiness expected 'blocked'", errors)

    def test_validate_lifecycle_rejects_unknown_control_plane_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            lifecycle_path = root / "serving_lifecycle.json"
            lifecycle = _blocked_lifecycle()
            lifecycle["provider_console_url"] = "https://example.invalid/job"
            lifecycle["readiness_probe"]["provider_call"] = {"url": "https://example.invalid/ready"}
            lifecycle["smoke_check"]["signed_url"] = "https://example.invalid/smoke"
            lifecycle["teardown"]["provider_call"] = {"url": "https://example.invalid/teardown"}
            _write_json(lifecycle_path, lifecycle)

            schema_result = check_schema_file(lifecycle_path)
            self.assertFalse(schema_result["passed"], schema_result)
            validation = validate_artifacts(serving_lifecycle_paths=[lifecycle_path], strict=True)

            self.assertFalse(validation["passed"])
            errors = "\n".join(error for target in validation["targets"] for error in target["errors"])
            self.assertIn("serving_lifecycle contains unknown field(s): ['provider_console_url']", errors)
            self.assertIn(
                "serving_lifecycle.readiness_probe contains unknown field(s): ['provider_call']",
                errors,
            )
            self.assertIn("serving_lifecycle.smoke_check contains unknown field(s): ['signed_url']", errors)
            self.assertIn("serving_lifecycle.teardown contains unknown field(s): ['provider_call']", errors)

    def test_validate_lifecycle_rejects_symlinked_preflight_artifact_parent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            real_preflight = root / "preflight_real"
            linked_preflight = root / "preflight"
            lifecycle_path = root / "serving_lifecycle.json"
            summary_path = root / "validation.json"
            real_preflight.mkdir()
            for artifact in ("serving_profile.json", "compatibility_report.json", "serving_check.json"):
                _write_json(real_preflight / artifact, {"schema_version": "hfr.test.v1"})
            try:
                linked_preflight.symlink_to(real_preflight, target_is_directory=True)
            except (NotImplementedError, OSError) as exc:
                self.skipTest(f"symlink creation unavailable: {exc}")
            lifecycle = _passed_lifecycle()
            _write_json(lifecycle_path, lifecycle)

            code = flightrecorder_main(
                [
                    "validate",
                    "--serving-lifecycle",
                    str(lifecycle_path),
                    "--strict",
                    "--out",
                    str(summary_path),
                ]
            )

            self.assertEqual(code, 1)
            summary = _read_json(summary_path)
            errors = "\n".join(error for target in summary["targets"] for error in target["errors"])
            self.assertIn(
                "serving_lifecycle.artifacts.serving_profile must point at a regular non-symlink file when passed.",
                errors,
            )
            self.assertIn(
                "serving_lifecycle.artifacts.compatibility_report must point at a regular non-symlink file when passed.",
                errors,
            )
            self.assertIn(
                "serving_lifecycle.artifacts.serving_check must point at a regular non-symlink file when passed.",
                errors,
            )

    def test_lifecycle_records_blocked_when_process_exits_before_ready(self):
        manage_openai_serving = _load_script(LIFECYCLE_SCRIPT, "manage_openai_serving_failure")
        port = _free_port()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "failed"
            with redirect_stdout(StringIO()):
                code = manage_openai_serving.main(
                    [
                        "--command",
                        "python3 -c 'import sys; sys.exit(3)'",
                        "--base-url",
                        f"http://127.0.0.1:{port}/v1",
                        "--out",
                        str(out),
                        "--startup-timeout",
                        "2",
                        "--poll-interval",
                        "0.1",
                    ]
                )

            self.assertEqual(code, 1)
            lifecycle = _read_json(out / "serving_lifecycle.json")
            self.assertFalse(lifecycle["passed"])
            self.assertEqual(lifecycle["readiness"], "blocked")
            self.assertEqual(lifecycle["readiness_probe"]["summary"], "process_exited_before_ready")
            self.assertEqual(lifecycle["readiness_probe"]["exit_code"], 3)
            self.assertFalse(lifecycle["smoke_check"]["attempted"])
            self.assertTrue(lifecycle["teardown"]["already_exited"])
            self.assertTrue(lifecycle["teardown"]["clean"])

    def test_lifecycle_records_blocked_when_launch_fails(self):
        manage_openai_serving = _load_script(LIFECYCLE_SCRIPT, "manage_openai_serving_launch_failure")
        port = _free_port()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "launch_failed"
            with redirect_stdout(StringIO()):
                code = manage_openai_serving.main(
                    [
                        "--command",
                        "hfr-definitely-missing-serving-command",
                        "--base-url",
                        f"http://127.0.0.1:{port}/v1",
                        "--out",
                        str(out),
                        "--startup-timeout",
                        "1",
                    ]
                )

            self.assertEqual(code, 1)
            lifecycle = _read_json(out / "serving_lifecycle.json")
            self.assertFalse(lifecycle["passed"])
            self.assertEqual(lifecycle["readiness"], "blocked")
            self.assertFalse(lifecycle["teardown"]["attempted"])
            self.assertFalse(lifecycle["teardown"]["clean"])
            self.assertIn("FileNotFoundError", "\n".join(lifecycle["errors"]))
            schema_result = check_schema_file(out / "serving_lifecycle.json")
            self.assertTrue(schema_result["passed"], schema_result["errors"])

    def test_builtin_engine_profiles_render_expected_launch_commands(self):
        manage_openai_serving = _load_script(LIFECYCLE_SCRIPT, "manage_openai_serving_profiles")
        vllm_args = manage_openai_serving.parse_args(
            [
                "--profile",
                "vllm",
                "--model",
                "Qwen/Qwen3-4B-Instruct-2507",
                "--served-model-name",
                "qwen3-flightrecorder",
                "--port",
                "18080",
                "--adapter",
                "adapter-lora",
                "--out",
                "unused",
            ]
        )
        sglang_args = manage_openai_serving.parse_args(
            [
                "--profile",
                "sglang",
                "--model",
                "Qwen/Qwen3-4B-Instruct-2507",
                "--served-model-name",
                "qwen3-flightrecorder",
                "--tensor-parallel-size",
                "2",
                "--port",
                "30000",
                "--adapter",
                "adapter-lora",
                "--out",
                "unused",
            ]
        )

        self.assertEqual(manage_openai_serving._launch_command(vllm_args)[:2], ["vllm", "serve"])
        self.assertIn("--served-model-name", manage_openai_serving._launch_command(vllm_args))
        self.assertEqual(manage_openai_serving._launch_command(sglang_args)[:3], ["python3", "-m", "sglang.launch_server"])
        self.assertIn("--tp-size", manage_openai_serving._launch_command(sglang_args))
        vllm_strategy = manage_openai_serving._adapter_strategy(vllm_args)
        sglang_strategy = manage_openai_serving._adapter_strategy(sglang_args)
        self.assertEqual(vllm_strategy["resolved_strategy"], "engine_args")
        self.assertTrue(vllm_strategy["requires_engine_args"])
        self.assertIn("engine-specific adapter flags", " ".join(vllm_strategy["notes"]))
        self.assertEqual(sglang_strategy["resolved_strategy"], "engine_args")


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: dict):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _blocked_lifecycle() -> dict:
    return {
        "schema_version": "hfr.serving_lifecycle.v1",
        "generated_at": "2026-01-01T00:00:00Z",
        "finished_at": "2026-01-01T00:00:01Z",
        "duration_ms": 1000,
        "profile": "mock",
        "engine": "mock",
        "arm": "candidate",
        "provider": "custom",
        "model": "hfr-managed-mock",
        "served_model_name": "hfr-managed-mock",
        "adapter": "",
        "adapter_strategy": {"present": False, "resolved_strategy": "none"},
        "endpoint": {"base_url": "http://127.0.0.1:18080/v1", "host": "127.0.0.1", "port": 18080},
        "launch": {
            "command": ["python3", "scripts/mock_openai_server.py"],
            "command_display": "python3 scripts/mock_openai_server.py",
            "cwd": ".",
            "env_keys": [],
            "startup_timeout_s": 1.0,
            "poll_interval_s": 0.1,
            "grace_period_s": 1.0,
        },
        "process": {"started": True, "pid": 12345, "exit_code": 3},
        "environment": {"python_version": "3.11.0", "platform": "test"},
        "artifacts_root": ".",
        "passed": False,
        "ready": False,
        "readiness": "blocked",
        "readiness_probe": {
            "ready": False,
            "summary": "process_exited_before_ready",
            "exit_code": 3,
            "attempts": [],
        },
        "smoke_check": {"attempted": False, "passed": False, "summary": "not_started"},
        "teardown": {
            "attempted": True,
            "already_exited": True,
            "terminated": False,
            "killed": False,
            "clean": True,
            "running_after_teardown": False,
        },
        "errors": ["process_exited_before_ready"],
        "artifacts": {
            "serving_lifecycle": "serving_lifecycle.json",
            "stdout_log": "server.stdout.log",
            "stderr_log": "server.stderr.log",
        },
        "logs": {"stdout_tail": "", "stderr_tail": ""},
    }


def _passed_lifecycle() -> dict:
    lifecycle = _blocked_lifecycle()
    artifacts = {
        "serving_profile": "preflight/serving_profile.json",
        "compatibility_report": "preflight/compatibility_report.json",
        "serving_check": "preflight/serving_check.json",
    }
    lifecycle.update(
        {
            "passed": True,
            "ready": True,
            "readiness": "ready",
            "readiness_probe": {"ready": True, "summary": "ready", "attempts": [{"ready": True}]},
            "smoke_check": {
                "attempted": True,
                "passed": True,
                "summary": "ok",
                "readiness": "ready",
                "failed_checks": [],
                "artifacts": artifacts,
            },
            "teardown": {
                "attempted": True,
                "already_exited": False,
                "terminated": True,
                "killed": False,
                "clean": True,
                "running_after_teardown": False,
            },
            "errors": [],
            "artifacts": {
                **lifecycle["artifacts"],
                **artifacts,
            },
        }
    )
    return lifecycle


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])
