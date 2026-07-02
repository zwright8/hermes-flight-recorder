import importlib.util
import json
import socket
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from flightrecorder.schema_registry import check_schema_file


ROOT = Path(__file__).resolve().parents[1]
LIFECYCLE_SCRIPT = ROOT / "scripts" / "manage_openai_serving.py"


def _load_script(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


class ServingLifecycleTests(unittest.TestCase):
    def test_managed_mock_lifecycle_runs_preflight_and_tears_down(self):
        manage_openai_serving = _load_script(LIFECYCLE_SCRIPT, "manage_openai_serving_success")
        port = _free_port()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "managed"
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
            self.assertIn("mock_openai_server_ready", lifecycle["logs"]["stdout_tail"])
            profile = _read_json(out / "preflight" / "serving_profile.json")
            self.assertEqual(profile["capabilities"]["streaming"], "supported")

            schema_result = check_schema_file(out / "serving_lifecycle.json")
            self.assertTrue(schema_result["passed"], schema_result["errors"])
            self.assertEqual(schema_result["schema"]["name"], "serving_lifecycle")

            profile_result = check_schema_file(out / "preflight" / "serving_profile.json")
            self.assertTrue(profile_result["passed"], profile_result["errors"])

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
                "--out",
                "unused",
            ]
        )

        self.assertEqual(manage_openai_serving._launch_command(vllm_args)[:2], ["vllm", "serve"])
        self.assertIn("--served-model-name", manage_openai_serving._launch_command(vllm_args))
        self.assertEqual(manage_openai_serving._launch_command(sglang_args)[:3], ["python3", "-m", "sglang.launch_server"])
        self.assertIn("--tp-size", manage_openai_serving._launch_command(sglang_args))


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])
