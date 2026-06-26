import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

from flightrecorder.cli import main


ROOT = Path(__file__).resolve().parents[1]


def run_cli(args):
    with redirect_stdout(StringIO()):
        return main(args)


class CliReportTests(unittest.TestCase):
    def test_run_command_generates_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            code = run_cli(["run", "--scenario", str(ROOT / "scenarios" / "prompt_injection_good.json"), "--out", str(out)])

            self.assertEqual(code, 0)
            self.assertTrue((out / "normalized_trace.json").exists())
            self.assertTrue((out / "scorecard.json").exists())
            self.assertTrue((out / "report.html").exists())
            scorecard = json.loads((out / "scorecard.json").read_text(encoding="utf-8"))
            self.assertTrue(scorecard["passed"])

    def test_failing_report_redacts_secret_values_and_writes_regression(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            code = run_cli(["run", "--scenario", str(ROOT / "scenarios" / "prompt_injection_bad.json"), "--out", str(out)])

            self.assertEqual(code, 0)
            self.assertTrue((out / "regression_scenario.json").exists())
            normalized = (out / "normalized_trace.json").read_text(encoding="utf-8")
            self.assertNotIn("hfr_fixture_secret_value_123", normalized)
            report = (out / "report.html").read_text(encoding="utf-8")
            self.assertIn("Hermes Autonomy Flight Recorder", report)
            self.assertIn("Forbidden Actions", report)
            self.assertNotIn("hfr_fixture_secret_value_123", report)

    def test_sensitive_trace_requires_explicit_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            code = run_cli(
                [
                    "run",
                    "--scenario",
                    str(ROOT / "scenarios" / "prompt_injection_bad.json"),
                    "--out",
                    str(out),
                    "--write-sensitive-trace",
                ]
            )

            self.assertEqual(code, 0)
            sensitive = out / "raw_trace.sensitive.json"
            self.assertTrue(sensitive.exists())
            self.assertIn("hfr_fixture_secret_value_123", sensitive.read_text(encoding="utf-8"))

    def test_run_can_fail_nonzero_for_ci(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            code = run_cli(
                [
                    "run",
                    "--scenario",
                    str(ROOT / "scenarios" / "prompt_injection_bad.json"),
                    "--out",
                    str(out),
                    "--fail-on-score",
                ]
            )

            self.assertEqual(code, 1)
            self.assertTrue((out / "report.html").exists())

    def test_report_command_creates_parent_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "run"
            run_cli(["run", "--scenario", str(ROOT / "scenarios" / "prompt_injection_good.json"), "--out", str(run)])
            report_path = Path(tmp) / "nested" / "report.html"

            code = run_cli(
                [
                    "report",
                    "--scenario",
                    str(ROOT / "scenarios" / "prompt_injection_good.json"),
                    "--trace",
                    str(run / "normalized_trace.json"),
                    "--score",
                    str(run / "scorecard.json"),
                    "--out",
                    str(report_path),
                ]
            )

            self.assertEqual(code, 0)
            self.assertTrue(report_path.exists())

    def test_index_command_generates_report_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp) / "runs"
            first = runs / "prompt"
            run_cli(["run", "--scenario", str(ROOT / "scenarios" / "prompt_injection_good.json"), "--out", str(first)])
            index_path = Path(tmp) / "nested" / "index.html"

            code = run_cli(["index", "--runs", str(runs), "--out", str(index_path)])

            self.assertEqual(code, 0)
            index = index_path.read_text(encoding="utf-8")
            self.assertIn("Hermes Flight Recorder Demo Runs", index)
            self.assertIn("Prompt Injection", index)

    def test_audit_command_summarizes_runs_and_can_fail_on_leak(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp) / "runs"
            good = runs / "good"
            bad = runs / "bad"
            run_cli(["run", "--scenario", str(ROOT / "scenarios" / "prompt_injection_good.json"), "--out", str(good)])
            run_cli(["run", "--scenario", str(ROOT / "scenarios" / "prompt_injection_bad.json"), "--out", str(bad)])
            out = Path(tmp) / "audit.json"

            code = run_cli(["audit", "--runs", str(runs), "--out", str(out)])

            self.assertEqual(code, 0)
            audit = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(audit["total"], 2)
            self.assertEqual(audit["passed"], 1)
            self.assertEqual(audit["failed"], 1)
            self.assertEqual(audit["leaks"], [])

            (bad / "leak.txt").write_text("do-not-ship", encoding="utf-8")
            leak_code = run_cli(["audit", "--runs", str(runs), "--forbid-text", "do-not-ship", "--fail-on-leak"])
            self.assertEqual(leak_code, 1)

    def test_audit_command_can_fail_when_any_scorecard_failed(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp) / "runs"
            bad = runs / "bad"
            run_cli(["run", "--scenario", str(ROOT / "scenarios" / "prompt_injection_bad.json"), "--out", str(bad)])

            code = run_cli(["audit", "--runs", str(runs), "--fail-on-failed"])

            self.assertEqual(code, 1)

    def test_audit_missing_runs_directory_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            code_stream = StringIO()
            with self.assertRaises(SystemExit) as raised, redirect_stdout(code_stream), redirect_stderr(StringIO()):
                main(["audit", "--runs", str(Path(tmp) / "missing")])

            self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
