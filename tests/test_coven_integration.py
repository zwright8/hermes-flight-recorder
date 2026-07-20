import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

from flightrecorder.cli import main
from flightrecorder.schema_registry import check_schema_file
from scripts.hermes_harness import write_fake_secret_canaries
from scripts.live_coven_smoke import MODEL_REF, _write_smoke_artifacts


ROOT = Path(__file__).resolve().parents[1]


class CovenIntegrationTests(unittest.TestCase):
    def test_coven_example_generates_passing_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "coven_run"
            stdout = StringIO()
            stderr = StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                rc = main(
                    [
                        "run",
                        "--scenario",
                        str(ROOT / "examples" / "coven" / "detached_session_coven.json"),
                        "--out",
                        str(out_dir),
                    ]
                )

            self.assertEqual(rc, 0, stderr.getvalue())
            self.assertTrue((out_dir / "normalized_trace.json").exists())
            self.assertTrue((out_dir / "scorecard.json").exists())
            self.assertTrue((out_dir / "report.html").exists())
            self.assertIn("PASS", (out_dir / "report.html").read_text(encoding="utf-8"))

    def test_coven_jsonl_schema_check_accepts_fixture(self):
        stdout = StringIO()
        stderr = StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            rc = main(
                [
                    "schemas",
                    "--check-jsonl",
                    str(ROOT / "fixtures" / "coven_detached_good.coven.jsonl"),
                    "--name",
                    "coven_event",
                ]
            )

        self.assertEqual(rc, 0, stderr.getvalue())
        self.assertIn('"passed": true', stdout.getvalue())

    def test_live_coven_smoke_artifact_writer_publishes_harness_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            trace = out / "live_coven.coven.jsonl"
            trace.write_text(
                (ROOT / "fixtures" / "coven_detached_good.coven.jsonl").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            fake_secret_files = write_fake_secret_canaries(out / "home")

            result = _write_smoke_artifacts(
                trace,
                out,
                sandbox={
                    "root": out / "sandbox-root",
                    "home": out / "home",
                    "workspace": out / "workspace",
                    "events": out / "events",
                    "ephemeral": True,
                    "audit_artifacts_kept": True,
                },
                fake_secret_files=fake_secret_files,
                process={"coven_exit_code": 0, "run_stdout": str(out / "run_stdout.txt")},
                metadata={"source": "test", "detached": True},
            )

            harness_result = result["harness_result"]
            self.assertTrue(result["scorecard"]["passed"])
            self.assertEqual(harness_result["runner"], "coven_live_smoke")
            self.assertEqual(harness_result["provider"], "coven")
            self.assertEqual(harness_result["model"]["id"], MODEL_REF)
            self.assertEqual(harness_result["trace"]["format"], "coven_jsonl")
            self.assertEqual(harness_result["process"]["coven_exit_code"], 0)
            self.assertIn(
                str((out / "home" / ".hermes" / ".env").resolve()),
                harness_result["sandbox"]["fake_secret_files"],
            )
            manifest = json.loads((out / "harness_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["runner"], "coven_live_smoke")
            self.assertEqual(manifest["tool_policy"]["scenario_policy"]["max_tool_calls"], 0)
            manifest_schema = check_schema_file(out / "harness_manifest.json", "harness_run_manifest")
            result_schema = check_schema_file(out / "harness_result.json", "harness_run_result")
            self.assertTrue(manifest_schema["passed"], manifest_schema["errors"])
            self.assertTrue(result_schema["passed"], result_schema["errors"])


if __name__ == "__main__":
    unittest.main()
