import json
import os
import subprocess
import sys
import tempfile
import tomllib
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from flightrecorder.cli import main
from flightrecorder.hermes_plugin import HOOKS, register, write_event
from scripts.live_hermes_smoke import EXPECTED, LIVE_SMOKE_SUMMARY_SCHEMA_VERSION, _write_smoke_artifacts, _write_smoke_summary


ROOT = Path(__file__).resolve().parents[1]


def _run_cli(args):
    with redirect_stdout(StringIO()):
        return main(args)


class DeploymentHardeningTests(unittest.TestCase):
    def test_pyproject_exposes_console_scripts(self):
        pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

        self.assertEqual(pyproject["project"]["name"], "hermes-flight-recorder")
        scripts = pyproject["project"]["scripts"]
        self.assertEqual(scripts["flightrecorder"], "flightrecorder.cli:main")
        self.assertEqual(scripts["hermes-flight-recorder"], "flightrecorder.cli:main")

    def test_live_hermes_smoke_script_help_is_available(self):
        completed = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "live_hermes_smoke.py"), "--help"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("live Hermes Flight Recorder observer smoke test", completed.stdout)

    def test_live_smoke_artifact_writer_uses_normal_run_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            observer = out / "live_observer.jsonl"
            rows = [
                {
                    "hook": "pre_llm_call",
                    "payload": {
                        "session_id": "live-session",
                        "model": "hfr-mock",
                        "user_message": "Reply exactly: flight recorder live smoke ok",
                    },
                },
                {
                    "hook": "post_api_request",
                    "payload": {
                        "session_id": "live-session",
                        "model": "hfr-mock",
                        "finish_reason": "stop",
                    },
                },
                {
                    "hook": "post_llm_call",
                    "payload": {
                        "session_id": "live-session",
                        "model": "hfr-mock",
                        "assistant_response": EXPECTED,
                    },
                },
            ]
            observer.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

            result = _write_smoke_artifacts(observer, out)

            self.assertTrue(result["scorecard"]["passed"])
            self.assertTrue((out / "live_scenario.json").exists())
            self.assertTrue((out / "normalized_trace.json").exists())
            self.assertTrue((out / "scorecard.json").exists())
            self.assertTrue((out / "task_completion.json").exists())
            self.assertTrue((out / "artifact_lineage.json").exists())
            self.assertTrue((out / "report.html").exists())
            self.assertEqual(_run_cli(["validate", "--run", str(out), "--strict"]), 0)

    def test_live_smoke_summary_is_persisted(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)

            summary = _write_smoke_summary(
                out,
                {
                    "passed": True,
                    "score": 100,
                    "hooks": ["on_session_start", "post_llm_call"],
                    "missing_hooks": [],
                    "observer_file": str(out / "live_observer.jsonl"),
                },
            )

            summary_path = out / "live_smoke_summary.json"
            persisted = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["schema_version"], LIVE_SMOKE_SUMMARY_SCHEMA_VERSION)
            self.assertEqual(persisted["schema_version"], LIVE_SMOKE_SUMMARY_SCHEMA_VERSION)
            self.assertTrue(persisted["passed"])
            self.assertEqual(persisted["score"], 100)
            self.assertEqual(persisted["summary"], str(summary_path))

    def test_scenario_schema_is_valid_json(self):
        schema = json.loads((ROOT / "scenario.schema.json").read_text(encoding="utf-8"))

        self.assertEqual(schema["title"], "Hermes Flight Recorder Scenario")
        self.assertIn("policy", schema["properties"])

    def test_normalize_command_redacts_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "normalized.json"
            with redirect_stdout(StringIO()):
                code = main(
                    [
                        "normalize",
                        "--trace",
                        str(ROOT / "fixtures" / "prompt_injection_bad.trajectory.jsonl"),
                        "--out",
                        str(out),
                    ]
                )

            self.assertEqual(code, 0)
            text = out.read_text(encoding="utf-8")
            self.assertNotIn("hfr_fixture_secret_value_123", text)
            self.assertIn("[REDACTED]", text)

    def test_hermes_observer_collector_registers_and_writes_jsonl(self):
        class FakeContext:
            def __init__(self):
                self.hooks = {}

            def register_hook(self, name, fn):
                self.hooks[name] = fn

        with tempfile.TemporaryDirectory() as tmp:
            previous = os.environ.get("HERMES_FLIGHT_RECORDER_OUTPUT_DIR")
            os.environ["HERMES_FLIGHT_RECORDER_OUTPUT_DIR"] = tmp
            try:
                ctx = FakeContext()
                register(ctx)
                self.assertEqual(set(ctx.hooks), set(HOOKS))
                ctx.hooks["pre_tool_call"](
                    session_id="session/live",
                    tool_name="terminal",
                    args={"command": "printf ok"},
                )
                files = list(Path(tmp).glob("*.observer.jsonl"))
                self.assertEqual(len(files), 1)
                row = json.loads(files[0].read_text(encoding="utf-8"))
                self.assertEqual(row["hook"], "pre_tool_call")
                self.assertEqual(row["payload"]["tool_name"], "terminal")
            finally:
                if previous is None:
                    os.environ.pop("HERMES_FLIGHT_RECORDER_OUTPUT_DIR", None)
                else:
                    os.environ["HERMES_FLIGHT_RECORDER_OUTPUT_DIR"] = previous

    def test_hermes_observer_collector_registration_is_fail_open(self):
        class FlakyContext:
            def __init__(self):
                self.hooks = {}

            def register_hook(self, name, fn):
                if name == "on_session_start":
                    raise RuntimeError("registration failed")
                self.hooks[name] = fn

        ctx = FlakyContext()
        register(ctx)

        self.assertNotIn("on_session_start", ctx.hooks)
        self.assertIn("pre_tool_call", ctx.hooks)

    def test_write_event_bounds_large_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            previous_dir = os.environ.get("HERMES_FLIGHT_RECORDER_OUTPUT_DIR")
            previous_max = os.environ.get("HERMES_FLIGHT_RECORDER_MAX_FIELD_CHARS")
            os.environ["HERMES_FLIGHT_RECORDER_OUTPUT_DIR"] = tmp
            os.environ["HERMES_FLIGHT_RECORDER_MAX_FIELD_CHARS"] = "100"
            try:
                path = write_event("post_llm_call", {"session_id": "session-1", "assistant_response": "x" * 200})
                row = json.loads(path.read_text(encoding="utf-8"))
                self.assertIn("[truncated]", row["payload"]["assistant_response"])
            finally:
                if previous_dir is None:
                    os.environ.pop("HERMES_FLIGHT_RECORDER_OUTPUT_DIR", None)
                else:
                    os.environ["HERMES_FLIGHT_RECORDER_OUTPUT_DIR"] = previous_dir
                if previous_max is None:
                    os.environ.pop("HERMES_FLIGHT_RECORDER_MAX_FIELD_CHARS", None)
                else:
                    os.environ["HERMES_FLIGHT_RECORDER_MAX_FIELD_CHARS"] = previous_max


if __name__ == "__main__":
    unittest.main()
