import json
import shutil
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

from flightrecorder.cli import main
from flightrecorder.report import render_report


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

    def test_run_command_writes_ci_outputs_and_task_checklist(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            junit = out / "scorecard.junit.xml"
            markdown = out / "scorecard.md"

            code = run_cli(
                [
                    "run",
                    "--scenario",
                    str(ROOT / "scenarios" / "email_reply_completion_good.json"),
                    "--out",
                    str(out),
                    "--junit-out",
                    str(junit),
                    "--markdown-out",
                    str(markdown),
                ]
            )

            self.assertEqual(code, 0)
            self.assertIn("<testsuite", junit.read_text(encoding="utf-8"))
            self.assertIn("Flight Recorder Scorecard", markdown.read_text(encoding="utf-8"))
            report = (out / "report.html").read_text(encoding="utf-8")
            self.assertIn("Task Checklist", report)
            self.assertIn("Send a reply to assigned thread email-123", report)

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

    def test_run_suite_generates_complete_evidence_bundle(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "runs"

            code = run_cli(
                [
                    "run-suite",
                    "--scenarios",
                    str(ROOT / "scenarios"),
                    "--out",
                    str(out),
                    "--junit",
                    "--markdown",
                    "--export-rl",
                    "--validate",
                    "--strict",
                ]
            )

            self.assertEqual(code, 0)
            summary = json.loads((out / "suite_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["schema_version"], "hfr.run_suite.v1")
            self.assertEqual(summary["total"], 5)
            self.assertEqual(summary["passed"], 2)
            self.assertEqual(summary["failed"], 3)
            self.assertEqual(summary["error_count"], 0)
            self.assertTrue((out / "index.html").exists())
            self.assertTrue((out / "validation.json").exists())
            self.assertTrue((out / "training_export" / "episodes.jsonl").exists())
            self.assertTrue((out / "email_reply_completion_good" / "scorecard.junit.xml").exists())
            self.assertTrue((out / "email_reply_completion_good" / "scorecard.md").exists())
            self.assertIn("training_export", summary["artifacts"])
            self.assertTrue(summary["validation"]["passed"])
            self.assertEqual(summary["training_export"]["failure_mode_count"], 6)

    def test_run_suite_can_fail_nonzero_for_ci_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "runs"

            code = run_cli(
                [
                    "run-suite",
                    "--scenarios",
                    str(ROOT / "scenarios"),
                    "--out",
                    str(out),
                    "--fail-on-failed",
                ]
            )

            self.assertEqual(code, 1)
            summary = json.loads((out / "suite_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["failed"], 3)

    def test_run_suite_rejects_duplicate_scenario_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            scenarios = Path(tmp) / "scenarios"
            scenarios.mkdir()
            shutil.copy(ROOT / "scenarios" / "prompt_injection_good.json", scenarios / "a.json")
            shutil.copy(ROOT / "scenarios" / "prompt_injection_good.json", scenarios / "b.json")
            for path in (scenarios / "a.json", scenarios / "b.json"):
                payload = json.loads(path.read_text(encoding="utf-8"))
                payload["trace"]["path"] = str(ROOT / "fixtures" / "prompt_injection_good.trajectory.jsonl")
                path.write_text(json.dumps(payload), encoding="utf-8")
            out = Path(tmp) / "runs"

            code = run_cli(["run-suite", "--scenarios", str(scenarios), "--out", str(out)])

            self.assertEqual(code, 1)
            summary = json.loads((out / "suite_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["total"], 1)
            self.assertEqual(summary["error_count"], 1)
            self.assertIn("Duplicate scenario id", summary["errors"][0]["error"])

    def test_run_suite_writes_summary_when_export_has_no_completed_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            scenarios = Path(tmp) / "scenarios"
            scenarios.mkdir()
            (scenarios / "invalid.json").write_text(json.dumps({"id": "invalid"}), encoding="utf-8")
            out = Path(tmp) / "runs"

            code = run_cli(["run-suite", "--scenarios", str(scenarios), "--out", str(out), "--export-rl"])

            self.assertEqual(code, 1)
            summary = json.loads((out / "suite_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["total"], 0)
            self.assertGreaterEqual(summary["error_count"], 1)
            self.assertTrue(any("Cannot export RL artifacts" in item["error"] for item in summary["errors"]))

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

    def test_score_command_writes_junit_and_markdown(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "run"
            run_cli(["run", "--scenario", str(ROOT / "scenarios" / "prompt_injection_good.json"), "--out", str(run)])
            junit = Path(tmp) / "score.xml"
            markdown = Path(tmp) / "score.md"

            code = run_cli(
                [
                    "score",
                    "--scenario",
                    str(ROOT / "scenarios" / "prompt_injection_good.json"),
                    "--trace",
                    str(run / "normalized_trace.json"),
                    "--out",
                    str(Path(tmp) / "scorecard.json"),
                    "--junit-out",
                    str(junit),
                    "--markdown-out",
                    str(markdown),
                ]
            )

            self.assertEqual(code, 0)
            self.assertIn("testsuite", junit.read_text(encoding="utf-8"))
            self.assertIn("Forbidden Actions", markdown.read_text(encoding="utf-8"))

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

    def test_compare_command_detects_regression(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp) / "runs"
            good = runs / "good"
            bad = runs / "bad"
            run_cli(["run", "--scenario", str(ROOT / "scenarios" / "prompt_injection_good.json"), "--out", str(good)])
            run_cli(["run", "--scenario", str(ROOT / "scenarios" / "prompt_injection_bad.json"), "--out", str(bad)])
            out = Path(tmp) / "compare.json"
            html = Path(tmp) / "compare.html"

            code = run_cli(
                [
                    "compare",
                    "--baseline",
                    str(good),
                    "--candidate",
                    str(bad),
                    "--out",
                    str(out),
                    "--html-out",
                    str(html),
                    "--fail-on-regression",
                ]
            )

            self.assertEqual(code, 1)
            comparison = json.loads(out.read_text(encoding="utf-8"))
            self.assertTrue(comparison["regressed"])
            self.assertLess(comparison["score_delta"], 0)
            self.assertIn("Flight Recorder Compare", html.read_text(encoding="utf-8"))

    def test_compare_suite_detects_degraded_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            baseline = Path(tmp) / "baseline"
            candidate = Path(tmp) / "candidate"
            run_cli(["run", "--scenario", str(ROOT / "scenarios" / "prompt_injection_good.json"), "--out", str(baseline / "prompt")])
            shutil.copytree(baseline, candidate)
            score_path = candidate / "prompt" / "scorecard.json"
            scorecard = json.loads(score_path.read_text(encoding="utf-8"))
            scorecard["score"] = 60
            scorecard["passed"] = False
            score_path.write_text(json.dumps(scorecard, indent=2), encoding="utf-8")
            out = Path(tmp) / "suite_compare.json"
            html = Path(tmp) / "suite_compare.html"

            code = run_cli(
                [
                    "compare-suite",
                    "--baseline",
                    str(baseline),
                    "--candidate",
                    str(candidate),
                    "--out",
                    str(out),
                    "--html-out",
                    str(html),
                    "--fail-on-regression",
                ]
            )

            self.assertEqual(code, 1)
            comparison = json.loads(out.read_text(encoding="utf-8"))
            self.assertTrue(comparison["regressed"])
            self.assertEqual(comparison["aggregate"]["paired_count"], 1)
            self.assertEqual(comparison["aggregate"]["avg_score_delta"], -40.0)
            self.assertEqual(comparison["regressions"], ["prompt_injection_good"])
            self.assertIn("Flight Recorder Suite Compare", html.read_text(encoding="utf-8"))

    def test_compare_suite_detects_missing_candidate_scenario(self):
        with tempfile.TemporaryDirectory() as tmp:
            baseline = Path(tmp) / "baseline"
            candidate = Path(tmp) / "candidate"
            run_cli(["run", "--scenario", str(ROOT / "scenarios" / "prompt_injection_good.json"), "--out", str(baseline / "prompt")])
            candidate.mkdir()
            out = Path(tmp) / "suite_compare.json"

            code = run_cli(
                [
                    "compare-suite",
                    "--baseline",
                    str(baseline),
                    "--candidate",
                    str(candidate),
                    "--out",
                    str(out),
                    "--fail-on-regression",
                ]
            )

            self.assertEqual(code, 1)
            comparison = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(comparison["missing_in_candidate"], ["prompt_injection_good"])

    def test_compare_suite_self_compare_has_no_regression(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp) / "runs"
            run_cli(["run", "--scenario", str(ROOT / "scenarios" / "prompt_injection_good.json"), "--out", str(runs / "prompt")])
            out = Path(tmp) / "suite_compare.json"

            code = run_cli(["compare-suite", "--baseline", str(runs), "--candidate", str(runs), "--out", str(out)])

            self.assertEqual(code, 0)
            comparison = json.loads(out.read_text(encoding="utf-8"))
            self.assertFalse(comparison["regressed"])
            self.assertEqual(comparison["aggregate"]["avg_score_delta"], 0.0)
            self.assertNotIn(str(runs), out.read_text(encoding="utf-8"))

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

    def test_regression_artifacts_do_not_leak_external_absolute_trace_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "external_trace.json"
            trace_path.write_text(
                json.dumps(
                    {
                        "schema_version": "hfr.trace.v1",
                        "session": {"id": "s1", "source_format": "normalized_json", "model": "fixture"},
                        "events": [],
                        "final_answer": "violates policy",
                    }
                ),
                encoding="utf-8",
            )
            scenario_path = Path(tmp) / "scenario.json"
            scenario_path.write_text(
                json.dumps(
                    {
                        "id": "external_failure",
                        "title": "External Failure",
                        "prompt": "x",
                        "trace": {"format": "auto", "path": str(trace_path)},
                        "policy": {},
                        "assertions": {"final_not_contains": ["violates"]},
                    }
                ),
                encoding="utf-8",
            )
            out = Path(tmp) / "run"

            code = run_cli(["run", "--scenario", str(scenario_path), "--out", str(out)])

            self.assertEqual(code, 0)
            regression = (out / "regression_scenario.json").read_text(encoding="utf-8")
            report = (out / "report.html").read_text(encoding="utf-8")
            self.assertNotIn(str(trace_path), regression)
            self.assertNotIn(str(trace_path), report)
            self.assertIn("<redacted:external_trace.json>", regression)

    def test_report_redacts_windows_absolute_trace_path(self):
        scenario = {
            "id": "windows_path",
            "title": "Windows Path",
            "prompt": "x",
            "trace": {"path": "C:/Users/alice/secrets/trace.json"},
            "policy": {},
        }
        trace = {
            "session": {"source_format": "normalized_json"},
            "events": [],
            "final_answer": "",
        }
        scorecard = {
            "passed": True,
            "score": 100,
            "pass_threshold": 90,
            "summary": "PASS",
            "rules": [],
        }

        report = render_report(scenario, trace, scorecard)

        self.assertIn("&lt;redacted:trace.json&gt;", report)
        self.assertNotIn("C:/Users/alice", report)

    def test_report_redacts_unc_absolute_trace_path(self):
        scenario = {
            "id": "unc_path",
            "title": "UNC Path",
            "prompt": "x",
            "trace": {"path": "\\\\server\\share\\trace.json"},
            "policy": {},
        }
        trace = {
            "session": {"source_format": "normalized_json"},
            "events": [],
            "final_answer": "",
        }
        scorecard = {
            "passed": True,
            "score": 100,
            "pass_threshold": 90,
            "summary": "PASS",
            "rules": [],
        }

        report = render_report(scenario, trace, scorecard)

        self.assertIn("&lt;redacted:trace.json&gt;", report)
        self.assertNotIn("server\\share", report)

    def test_observer_template_command_writes_bootstrap(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "plugin.py"

            code = run_cli(["observer-template", "--out", str(out)])

            self.assertEqual(code, 0)
            text = out.read_text(encoding="utf-8")
            self.assertIn("register_flight_recorder", text)


if __name__ == "__main__":
    unittest.main()
