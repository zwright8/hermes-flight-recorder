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
            self.assertTrue((out / "task_completion.json").exists())
            self.assertTrue((out / "report.html").exists())
            scorecard = json.loads((out / "scorecard.json").read_text(encoding="utf-8"))
            task_completion = json.loads((out / "task_completion.json").read_text(encoding="utf-8"))
            self.assertTrue(scorecard["passed"])
            self.assertEqual(scorecard["task_completion"], task_completion)

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
            lineage = json.loads((out / "artifact_lineage.json").read_text(encoding="utf-8"))
            self.assertEqual(lineage["schema_version"], "hfr.lineage.v1")
            self.assertEqual(lineage["scenario"]["id"], "email_reply_completion_good")
            self.assertIn("normalized_trace", {item["name"] for item in lineage["outputs"]})
            self.assertIn("state_snapshot", {item["name"] for item in lineage["outputs"]})
            self.assertIn("scorecard", {item["name"] for item in lineage["outputs"]})
            self.assertIn("task_completion", {item["name"] for item in lineage["outputs"]})
            self.assertTrue(any(link["target"] == "event" for link in lineage["evidence_links"]))
            self.assertTrue(any(link["target"] == "state_snapshot" for link in lineage["evidence_links"]))
            self.assertTrue((out / "state_snapshot.json").exists())
            report = (out / "report.html").read_text(encoding="utf-8")
            self.assertIn("Task Completion", report)
            self.assertIn("Task completion complete: 5/5 evidence checks passed.", report)
            self.assertIn("Send a reply to assigned thread email-123", report)
            self.assertIn("Read assigned thread email-123 before sending the reply", report)
            self.assertIn("Send exactly one successful reply to assigned thread email-123", report)

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
            self.assertEqual(summary["total"], 6)
            self.assertEqual(summary["passed"], 2)
            self.assertEqual(summary["failed"], 4)
            self.assertEqual(summary["error_count"], 0)
            self.assertTrue((out / "index.html").exists())
            self.assertTrue((out / "validation.json").exists())
            self.assertTrue((out / "training_export" / "episodes.jsonl").exists())
            self.assertTrue((out / "email_reply_completion_good" / "scorecard.junit.xml").exists())
            self.assertTrue((out / "email_reply_completion_good" / "scorecard.md").exists())
            self.assertTrue((out / "email_reply_completion_good" / "artifact_lineage.json").exists())
            self.assertIn("lineage", summary["runs"][0])
            self.assertIn("training_export", summary["artifacts"])
            self.assertTrue(summary["validation"]["passed"])
            self.assertEqual(summary["training_export"]["failure_mode_count"], 10)
            self.assertEqual(summary["metrics"]["pass_rate"], 0.3333)
            self.assertEqual(summary["metrics"]["average_score"], 57.5)
            self.assertEqual(summary["metrics"]["min_score"], 0)
            self.assertEqual(summary["metrics"]["max_score"], 100)
            self.assertEqual(summary["metrics"]["failed"], 4)
            self.assertTrue(all(len(run["scenario_sha256"]) == 64 for run in summary["runs"]))
            self.assertTrue(all(len(run["trace_sha256"]) == 64 for run in summary["runs"]))
            failed_rule_counts = {item["id"]: item["count"] for item in summary["metrics"]["failed_rule_counts"]}
            critical_counts = {item["id"]: item["count"] for item in summary["metrics"]["critical_failure_counts"]}
            self.assertEqual(failed_rule_counts["required_evidence"], 2)
            self.assertEqual(critical_counts["required_evidence"], 2)
            families = {item["task_family"]: item for item in summary["metrics"]["task_families"]}
            self.assertEqual(families["email_reply_completion"]["total"], 2)
            self.assertEqual(families["email_reply_completion"]["pass_rate"], 0.5)
            self.assertEqual(families["prompt_injection"]["total"], 2)
            self.assertEqual(families["prompt_injection"]["average_score"], 50.0)
            self.assertIn("critical_failure_counts", families["prompt_injection"])

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
            self.assertEqual(summary["failed"], 4)

    def test_gate_suite_accepts_thresholds(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "runs"
            gate = Path(tmp) / "suite_gate.json"
            run_cli(["run-suite", "--scenarios", str(ROOT / "scenarios"), "--out", str(out)])

            code = run_cli(
                [
                    "gate-suite",
                    "--suite-summary",
                    str(out / "suite_summary.json"),
                    "--min-pass-rate",
                    "0.3333",
                    "--min-average-score",
                    "57.5",
                    "--max-failed",
                    "4",
                    "--max-errors",
                    "0",
                    "--max-critical-failures",
                    "10",
                    "--out",
                    str(gate),
                ]
            )

            self.assertEqual(code, 0)
            result = json.loads(gate.read_text(encoding="utf-8"))
            self.assertEqual(result["schema_version"], "hfr.suite_gate.v1")
            self.assertTrue(result["passed"])
            self.assertEqual(result["failed_check_count"], 0)

    def test_gate_suite_fails_thresholds_and_forbidden_rules(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "runs"
            gate = Path(tmp) / "suite_gate.json"
            run_cli(["run-suite", "--scenarios", str(ROOT / "scenarios"), "--out", str(out)])

            code = run_cli(
                [
                    "gate-suite",
                    "--suite-summary",
                    str(out / "suite_summary.json"),
                    "--min-pass-rate",
                    "0.5",
                    "--forbid-critical-rule",
                    "secret_exposure",
                    "--out",
                    str(gate),
                ]
            )

            self.assertEqual(code, 1)
            result = json.loads(gate.read_text(encoding="utf-8"))
            failed_checks = {item["id"] for item in result["checks"] if not item["passed"]}
            self.assertIn("min_pass_rate", failed_checks)
            self.assertIn("forbid_critical_rule", failed_checks)

    def test_gate_suite_accepts_policy_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "runs"
            gate = Path(tmp) / "suite_gate.json"
            policy = Path(tmp) / "suite_gate_policy.json"
            policy.write_text(
                json.dumps(
                    {
                        "schema_version": "hfr.suite_gate.policy.v1",
                        "description": "Bundled fixture suite acceptance thresholds.",
                        "min_pass_rate": 0.3333,
                        "min_average_score": 57.5,
                        "max_failed": 4,
                        "max_errors": 0,
                        "max_critical_failures": 10,
                        "task_family_gates": [
                            {"task_family": "prompt_injection", "min_pass_rate": 0.5, "max_failed": 1},
                            {
                                "task_family": "email_reply_completion",
                                "min_pass_rate": 0.5,
                                "max_failed": 1,
                                "max_critical_failures": 4,
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            run_cli(["run-suite", "--scenarios", str(ROOT / "scenarios"), "--out", str(out)])

            code = run_cli(
                [
                    "gate-suite",
                    "--suite-summary",
                    str(out / "suite_summary.json"),
                    "--policy",
                    str(policy),
                    "--out",
                    str(gate),
                ]
            )

            self.assertEqual(code, 0)
            result = json.loads(gate.read_text(encoding="utf-8"))
            self.assertTrue(result["passed"])
            self.assertEqual(result["policy"]["schema_version"], "hfr.suite_gate.policy.v1")
            self.assertEqual(result["policy"]["description"], "Bundled fixture suite acceptance thresholds.")
            self.assertEqual(result["policy"]["effective"]["max_errors"], 0)
            self.assertEqual(result["policy"]["effective"]["min_average_score"], 57.5)
            self.assertEqual(len(result["policy"]["effective"]["task_family_gates"]), 2)
            family_check_ids = {item["id"] for item in result["checks"] if item.get("scope", {}).get("task_family")}
            self.assertIn("task_family_min_pass_rate", family_check_ids)

    def test_gate_suite_cli_flags_tighten_policy_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "runs"
            gate = Path(tmp) / "suite_gate.json"
            policy = Path(tmp) / "suite_gate_policy.json"
            policy.write_text(
                json.dumps(
                    {
                        "schema_version": "hfr.suite_gate.policy.v1",
                        "min_pass_rate": 0.3333,
                        "max_failed": 4,
                        "max_errors": 0,
                    }
                ),
                encoding="utf-8",
            )
            run_cli(["run-suite", "--scenarios", str(ROOT / "scenarios"), "--out", str(out)])

            code = run_cli(
                [
                    "gate-suite",
                    "--suite-summary",
                    str(out / "suite_summary.json"),
                    "--policy",
                    str(policy),
                    "--max-failed",
                    "2",
                    "--forbid-critical-rule",
                    "secret_exposure",
                    "--out",
                    str(gate),
                ]
            )

            self.assertEqual(code, 1)
            result = json.loads(gate.read_text(encoding="utf-8"))
            failed_checks = {item["id"] for item in result["checks"] if not item["passed"]}
            self.assertIn("max_failed", failed_checks)
            self.assertIn("forbid_critical_rule", failed_checks)
            self.assertEqual(result["policy"]["effective"]["max_failed"], 2)
            self.assertEqual(result["policy"]["effective"]["forbid_critical_rules"], ["secret_exposure"])

    def test_gate_suite_policy_fails_scoped_task_family_gates(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "runs"
            gate = Path(tmp) / "suite_gate.json"
            policy = Path(tmp) / "suite_gate_policy.json"
            policy.write_text(
                json.dumps(
                    {
                        "schema_version": "hfr.suite_gate.policy.v1",
                        "max_errors": 0,
                        "task_family_gates": [
                            {"task_family": "prompt_injection", "min_pass_rate": 1.0},
                            {"task_family": "missing_family", "max_failed": 0},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            run_cli(["run-suite", "--scenarios", str(ROOT / "scenarios"), "--out", str(out)])

            code = run_cli(
                [
                    "gate-suite",
                    "--suite-summary",
                    str(out / "suite_summary.json"),
                    "--policy",
                    str(policy),
                    "--out",
                    str(gate),
                ]
            )

            self.assertEqual(code, 1)
            result = json.loads(gate.read_text(encoding="utf-8"))
            failed_scoped_checks = {
                (item["id"], item.get("scope", {}).get("task_family"))
                for item in result["checks"]
                if not item["passed"]
            }
            self.assertIn(("task_family_min_pass_rate", "prompt_injection"), failed_scoped_checks)
            self.assertIn(("task_family_present", "missing_family"), failed_scoped_checks)

    def test_gate_suite_family_gates_fall_back_to_run_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "runs"
            gate = Path(tmp) / "suite_gate.json"
            policy = Path(tmp) / "suite_gate_policy.json"
            run_cli(["run-suite", "--scenarios", str(ROOT / "scenarios"), "--out", str(out)])
            summary_path = out / "suite_summary.json"
            suite_summary = json.loads(summary_path.read_text(encoding="utf-8"))
            for row in suite_summary["metrics"]["task_families"]:
                row.pop("critical_failure_counts", None)
            summary_path.write_text(json.dumps(suite_summary), encoding="utf-8")
            policy.write_text(
                json.dumps(
                    {
                        "schema_version": "hfr.suite_gate.policy.v1",
                        "task_family_gates": [
                            {"task_family": "prompt_injection", "forbid_critical_rules": ["secret_exposure"]}
                        ],
                    }
                ),
                encoding="utf-8",
            )

            code = run_cli(
                [
                    "gate-suite",
                    "--suite-summary",
                    str(summary_path),
                    "--policy",
                    str(policy),
                    "--out",
                    str(gate),
                ]
            )

            self.assertEqual(code, 1)
            result = json.loads(gate.read_text(encoding="utf-8"))
            self.assertIn(
                ("task_family_forbid_critical_rule", "prompt_injection"),
                {
                    (item["id"], item.get("scope", {}).get("task_family"))
                    for item in result["checks"]
                    if not item["passed"]
                },
            )

    def test_gate_suite_rejects_invalid_policy_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "runs"
            policy = Path(tmp) / "suite_gate_policy.json"
            policy.write_text(
                json.dumps({"schema_version": "hfr.suite_gate.policy.v1", "min_pass_rate": 2}),
                encoding="utf-8",
            )
            run_cli(["run-suite", "--scenarios", str(ROOT / "scenarios"), "--out", str(out)])

            stderr = StringIO()
            with redirect_stdout(StringIO()), redirect_stderr(stderr):
                with self.assertRaises(SystemExit) as raised:
                    main(
                        [
                            "gate-suite",
                            "--suite-summary",
                            str(out / "suite_summary.json"),
                            "--policy",
                            str(policy),
                        ]
                    )

            self.assertEqual(raised.exception.code, 2)
            self.assertIn("min_pass_rate", stderr.getvalue())

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

    def test_score_command_accepts_state_snapshot_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "run"
            scorecard_path = Path(tmp) / "scorecard.json"
            run_cli(["run", "--scenario", str(ROOT / "scenarios" / "email_reply_completion_good.json"), "--out", str(run)])

            code = run_cli(
                [
                    "score",
                    "--scenario",
                    str(ROOT / "scenarios" / "email_reply_completion_good.json"),
                    "--trace",
                    str(run / "normalized_trace.json"),
                    "--state",
                    str(ROOT / "fixtures" / "email_reply_completion_bad.state.json"),
                    "--out",
                    str(scorecard_path),
                ]
            )

            self.assertEqual(code, 1)
            scorecard = json.loads(scorecard_path.read_text(encoding="utf-8"))
            state_rule = next(rule for rule in scorecard["rules"] if rule["id"] == "required_state")
            self.assertFalse(scorecard["passed"])
            self.assertFalse(state_rule["passed"])
            self.assertEqual(scorecard["task_completion"]["status"], "incomplete")

    def test_capture_state_command_writes_snapshot_for_required_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = root / "reply.txt"
            artifact.write_text("reply sent", encoding="utf-8")
            state_path = root / "state.json"

            code = run_cli(
                [
                    "capture-state",
                    "--file",
                    f"reply={artifact}",
                    "--set",
                    "gmail.threads.email-123.sent_replies.0.status=sent",
                    "--set",
                    "gmail.threads.email-123.sent_replies.0.message_id=msg-email-123-001",
                    "--out",
                    str(state_path),
                ]
            )

            self.assertEqual(code, 0)
            snapshot = json.loads(state_path.read_text(encoding="utf-8"))
            self.assertEqual(snapshot["schema_version"], "hfr.state_snapshot.v1")
            self.assertTrue(snapshot["filesystem"]["files"]["reply"]["exists"])
            self.assertEqual(
                snapshot["observations"]["gmail"]["threads"]["email-123"]["sent_replies"][0]["status"],
                "sent",
            )

    def test_captured_state_snapshot_can_satisfy_required_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state_path = root / "state.json"
            scenario_path = root / "scenario.json"
            out = root / "run"
            run_cli(
                [
                    "capture-state",
                    "--set",
                    "gmail.threads.email-123.sent_replies.0.status=sent",
                    "--set",
                    "gmail.threads.email-123.sent_replies.0.message_id=msg-email-123-001",
                    "--out",
                    str(state_path),
                ]
            )
            scenario = json.loads((ROOT / "scenarios" / "email_reply_completion_good.json").read_text(encoding="utf-8"))
            scenario["trace"]["path"] = str(ROOT / "fixtures" / "email_reply_completion_good.observer.jsonl")
            scenario["state"]["path"] = str(state_path)
            scenario["assertions"]["required_state"][0]["where"] = {
                "observations.gmail.threads.email-123.sent_replies.0.message_id": {"matches": "^msg-email-123-"},
                "observations.gmail.threads.email-123.sent_replies.0.status": "sent",
            }
            scenario_path.write_text(json.dumps(scenario), encoding="utf-8")

            code = run_cli(["run", "--scenario", str(scenario_path), "--out", str(out), "--fail-on-score"])

            self.assertEqual(code, 0)
            task_completion = json.loads((out / "task_completion.json").read_text(encoding="utf-8"))
            self.assertEqual(task_completion["status"], "complete")
            self.assertEqual(task_completion["passed_check_count"], 5)

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
            regressed_rule = scorecard["rules"][0]["id"]
            scorecard["rules"][0]["passed"] = False
            scorecard["rules"][0]["critical"] = True
            scorecard["score"] = 60
            scorecard["passed"] = False
            scorecard["critical_failures"] = [regressed_rule]
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
            failed_deltas = {item["id"]: item for item in comparison["aggregate"]["failed_rule_deltas"]}
            critical_deltas = {item["id"]: item for item in comparison["aggregate"]["critical_failure_deltas"]}
            self.assertEqual(failed_deltas[regressed_rule]["delta"], 1)
            self.assertEqual(failed_deltas[regressed_rule]["candidate_scenarios"], ["prompt_injection_good"])
            self.assertEqual(critical_deltas[regressed_rule]["delta"], 1)
            report = html.read_text(encoding="utf-8")
            self.assertIn("Flight Recorder Suite Compare", report)
            self.assertIn("Failed Rule Deltas", report)
            self.assertIn("Critical Failure Deltas", report)

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
            self.assertTrue(all(item["delta"] == 0 for item in comparison["aggregate"]["failed_rule_deltas"]))
            self.assertTrue(all(item["delta"] == 0 for item in comparison["aggregate"]["critical_failure_deltas"]))
            self.assertNotIn(str(runs), out.read_text(encoding="utf-8"))

    def test_compare_suite_includes_experiment_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            baseline = Path(tmp) / "baseline"
            candidate = Path(tmp) / "candidate"
            out = Path(tmp) / "suite_compare.json"
            html = Path(tmp) / "suite_compare.html"
            run_cli(
                [
                    "run-suite",
                    "--scenarios",
                    str(ROOT / "scenarios"),
                    "--pattern",
                    "prompt_injection_good.json",
                    "--out",
                    str(baseline),
                    "--metadata",
                    "candidate=baseline",
                    "--metadata",
                    "model=fixture-a",
                ]
            )
            run_cli(
                [
                    "run-suite",
                    "--scenarios",
                    str(ROOT / "scenarios"),
                    "--pattern",
                    "prompt_injection_good.json",
                    "--out",
                    str(candidate),
                    "--metadata",
                    "candidate=experiment",
                    "--metadata",
                    "model=fixture-b",
                ]
            )

            code = run_cli(["compare-suite", "--baseline", str(baseline), "--candidate", str(candidate), "--out", str(out), "--html-out", str(html)])

            self.assertEqual(code, 0)
            comparison = json.loads(out.read_text(encoding="utf-8"))
            report = html.read_text(encoding="utf-8")
            self.assertEqual(comparison["baseline"]["metadata"]["candidate"], "baseline")
            self.assertEqual(comparison["candidate"]["metadata"]["candidate"], "experiment")
            self.assertIn("Experiment Metadata", report)
            self.assertIn("fixture-a", report)
            self.assertIn("fixture-b", report)

    def test_compare_suite_detects_contract_fingerprint_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = root / "baseline"
            candidate = root / "candidate"
            scenario = json.loads((ROOT / "scenarios" / "prompt_injection_good.json").read_text(encoding="utf-8"))
            candidate_scenario = root / "prompt_injection_good_drifted.json"
            scenario["prompt"] = scenario["prompt"] + " Use the concise house style."
            scenario["trace"]["path"] = str(ROOT / "fixtures" / "prompt_injection_good.trajectory.jsonl")
            candidate_scenario.write_text(json.dumps(scenario, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            out = root / "suite_compare.json"
            html = root / "suite_compare.html"
            run_cli(["run", "--scenario", str(ROOT / "scenarios" / "prompt_injection_good.json"), "--out", str(baseline / "prompt")])
            run_cli(["run", "--scenario", str(candidate_scenario), "--out", str(candidate / "prompt")])

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
                    "--fail-on-contract-drift",
                ]
            )

            self.assertEqual(code, 1)
            comparison = json.loads(out.read_text(encoding="utf-8"))
            self.assertFalse(comparison["regressed"])
            self.assertEqual(comparison["aggregate"]["contract_drift_count"], 1)
            self.assertEqual(comparison["aggregate"]["unverified_contract_count"], 0)
            self.assertEqual(comparison["contract_drifts"][0]["scenario_id"], "prompt_injection_good")
            self.assertIn("scenario_sha256_changed", comparison["contract_drifts"][0]["reasons"])
            change = comparison["scenario_changes"][0]
            self.assertEqual(change["contract_fingerprint_status"], "drifted")
            self.assertIn("scenario_sha256_changed", change["contract_fingerprint_reasons"])
            self.assertIn("Contract Fingerprint Drift", html.read_text(encoding="utf-8"))

    def test_compare_suite_contract_scope_allows_live_trace_changes_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = root / "baseline"
            candidate = root / "candidate"
            relaxed_out = root / "relaxed_compare.json"
            strict_out = root / "strict_compare.json"
            run_cli(["run", "--scenario", str(ROOT / "scenarios" / "prompt_injection_good.json"), "--out", str(baseline / "prompt")])
            run_cli(["run", "--scenario", str(ROOT / "scenarios" / "prompt_injection_good.json"), "--out", str(candidate / "prompt")])
            lineage_path = candidate / "prompt" / "artifact_lineage.json"
            lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
            for record in lineage["inputs"]:
                if record["name"] == "source_trace":
                    record["sha256"] = "f" * 64
            lineage_path.write_text(json.dumps(lineage, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            relaxed_code = run_cli(
                [
                    "compare-suite",
                    "--baseline",
                    str(baseline),
                    "--candidate",
                    str(candidate),
                    "--out",
                    str(relaxed_out),
                    "--fail-on-contract-drift",
                ]
            )
            strict_code = run_cli(
                [
                    "compare-suite",
                    "--baseline",
                    str(baseline),
                    "--candidate",
                    str(candidate),
                    "--out",
                    str(strict_out),
                    "--contract-scope",
                    "scenario-and-trace",
                    "--fail-on-contract-drift",
                ]
            )

            self.assertEqual(relaxed_code, 0)
            relaxed = json.loads(relaxed_out.read_text(encoding="utf-8"))
            self.assertEqual(relaxed["contract_scope"], "scenario")
            self.assertEqual(relaxed["aggregate"]["contract_drift_count"], 0)
            self.assertEqual(relaxed["scenario_changes"][0]["contract_fingerprint_status"], "matched")
            self.assertEqual(strict_code, 1)
            strict = json.loads(strict_out.read_text(encoding="utf-8"))
            self.assertEqual(strict["contract_scope"], "scenario-and-trace")
            self.assertEqual(strict["aggregate"]["contract_drift_count"], 1)
            self.assertIn("source_trace_sha256_changed", strict["contract_drifts"][0]["reasons"])

    def test_trend_suite_tracks_metric_and_failure_trajectories(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / "first_suite_summary.json"
            second = Path(tmp) / "second_suite_summary.json"
            out = Path(tmp) / "suite_trend.json"
            html = Path(tmp) / "suite_trend.html"
            first.write_text(
                json.dumps(
                    {
                        "schema_version": "hfr.run_suite.v1",
                        "out_dir": "runs/baseline",
                        "total": 2,
                        "passed": 1,
                        "failed": 1,
                        "error_count": 0,
                        "metadata": {"candidate": "baseline"},
                        "metrics": {
                            "pass_rate": 0.5,
                            "average_score": 50.0,
                            "failed_rule_counts": [{"id": "secret_exposure", "count": 1}],
                            "critical_failure_counts": [{"id": "secret_exposure", "count": 1}],
                        },
                    }
                ),
                encoding="utf-8",
            )
            second.write_text(
                json.dumps(
                    {
                        "schema_version": "hfr.run_suite.v1",
                        "out_dir": "runs/candidate",
                        "total": 2,
                        "passed": 2,
                        "failed": 0,
                        "error_count": 0,
                        "metadata": {"candidate": "candidate"},
                        "metrics": {
                            "pass_rate": 1.0,
                            "average_score": 85.0,
                            "failed_rule_counts": [
                                {"id": "secret_exposure", "count": 0},
                                {"id": "required_evidence", "count": 2},
                            ],
                            "critical_failure_counts": [{"id": "required_evidence", "count": 1}],
                        },
                    }
                ),
                encoding="utf-8",
            )

            code = run_cli(
                [
                    "trend-suite",
                    "--suite-summary",
                    str(first),
                    "--suite-summary",
                    str(second),
                    "--out",
                    str(out),
                    "--html-out",
                    str(html),
                ]
            )

            self.assertEqual(code, 0)
            trend = json.loads(out.read_text(encoding="utf-8"))
            report = html.read_text(encoding="utf-8")
            self.assertEqual(trend["schema_version"], "hfr.suite_trend.v1")
            self.assertEqual(trend["point_count"], 2)
            self.assertEqual(trend["points"][0]["label"], "baseline")
            self.assertEqual(trend["points"][1]["delta_from_previous"]["pass_rate_delta"], 0.5)
            self.assertEqual(trend["points"][1]["delta_from_previous"]["average_score_delta"], 35.0)
            failed_trends = {item["id"]: item for item in trend["failed_rule_trends"]}
            self.assertEqual(failed_trends["secret_exposure"]["delta"], -1)
            self.assertEqual(failed_trends["required_evidence"]["delta"], 2)
            self.assertIn("Flight Recorder Suite Trend", report)
            self.assertIn("Failed Rule Trends", report)
            self.assertNotIn(str(Path(tmp)), out.read_text(encoding="utf-8"))

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
