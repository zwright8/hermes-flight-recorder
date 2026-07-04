import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from flightrecorder.cli import main
from flightrecorder.schema_registry import check_schema_file


ROOT = Path(__file__).resolve().parents[1]


def run_cli(args):
    with redirect_stdout(StringIO()):
        return main(args)


class TraceObservabilityTests(unittest.TestCase):
    def test_trace_observability_reports_suite_signal(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp) / "runs"
            out = Path(tmp) / "trace_observability.json"
            run_cli(["run-suite", "--scenarios", str(ROOT / "scenarios"), "--out", str(runs)])

            code = run_cli(
                [
                    "trace-observability",
                    "--runs",
                    str(runs),
                    "--out",
                    str(out),
                    "--min-average-events",
                    "2",
                    "--min-event-type-count",
                    "2",
                    "--min-tool-or-api-run-rate",
                    "0.5",
                    "--max-empty-final-answers",
                    "0",
                    "--require-event-type",
                    "assistant_message",
                ]
            )

            self.assertEqual(code, 0)
            observability = json.loads(out.read_text(encoding="utf-8"))
            metrics = observability["metrics"]
            self.assertEqual(observability["schema_version"], "hfr.trace_observability.v1")
            self.assertTrue(observability["passed"])
            self.assertEqual(metrics["run_count"], 7)
            self.assertEqual(metrics["total_event_count"], 42)
            self.assertEqual(metrics["average_event_count"], 6.0)
            self.assertEqual(metrics["event_type_count"], 6)
            self.assertEqual(metrics["final_answer_rate"], 1.0)
            self.assertEqual(metrics["tool_or_api_run_rate"], 0.8571)
            self.assertEqual(metrics["empty_final_answer_count"], 0)
            self.assertEqual({run["scenario_id"] for run in observability["runs"]}, {path.stem for path in (ROOT / "scenarios").glob("*.json")})
            self.assertIn("subagent_claim_bad trace risks: no_tool_or_api_events", observability["warnings"])

            self.assertEqual(run_cli(["validate", "--trace-observability", str(out), "--strict"]), 0)
            schema = check_schema_file(out)
            self.assertTrue(schema["passed"], schema["errors"])
            self.assertEqual(schema["schema"]["name"], "trace_observability")

    def test_trace_observability_fails_unmet_threshold(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp) / "runs"
            out = Path(tmp) / "trace_observability.json"
            run_cli(["run", "--scenario", str(ROOT / "scenarios" / "prompt_injection_good.json"), "--out", str(runs / "good")])

            code = run_cli(
                [
                    "trace-observability",
                    "--runs",
                    str(runs),
                    "--out",
                    str(out),
                    "--min-average-events",
                    "999",
                    "--require-event-type",
                    "missing_event_type",
                ]
            )

            self.assertEqual(code, 1)
            observability = json.loads(out.read_text(encoding="utf-8"))
            self.assertFalse(observability["passed"])
            failed_ids = [check["id"] for check in observability["checks"] if not check["passed"]]
            self.assertIn("min_average_events", failed_ids)
            self.assertIn("require_event_type:missing_event_type", failed_ids)

    def test_validate_rejects_stale_trace_observability_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp) / "runs"
            out = Path(tmp) / "trace_observability.json"
            summary_path = Path(tmp) / "validation.json"
            run_cli(["run", "--scenario", str(ROOT / "scenarios" / "prompt_injection_good.json"), "--out", str(runs / "good")])
            run_cli(["trace-observability", "--runs", str(runs), "--out", str(out)])
            observability = json.loads(out.read_text(encoding="utf-8"))
            observability["metrics"]["run_count"] = 999
            out.write_text(json.dumps(observability, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            code = run_cli(["validate", "--trace-observability", str(out), "--out", str(summary_path)])

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in summary["targets"] for error in target["errors"])
            self.assertIn("trace_observability.metrics.run_count", errors)

    def test_strict_validate_warns_on_absolute_run_dirs(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp) / "runs"
            out = Path(tmp) / "trace_observability.json"
            strict_summary = Path(tmp) / "strict_validation.json"
            permissive_summary = Path(tmp) / "permissive_validation.json"
            run_cli(["run-suite", "--scenarios", str(ROOT / "scenarios"), "--out", str(runs)])
            run_cli(["trace-observability", "--runs", str(runs), "--out", str(out), "--preserve-paths"])

            permissive_code = run_cli(["validate", "--trace-observability", str(out), "--out", str(permissive_summary)])
            strict_code = run_cli(["validate", "--trace-observability", str(out), "--out", str(strict_summary), "--strict"])

            self.assertEqual(permissive_code, 0)
            self.assertEqual(strict_code, 1)
            permissive = json.loads(permissive_summary.read_text(encoding="utf-8"))
            strict = json.loads(strict_summary.read_text(encoding="utf-8"))
            self.assertTrue(permissive["passed"], permissive)
            self.assertGreater(permissive["warning_count"], 0, permissive)
            self.assertFalse(strict["passed"], strict)
            self.assertEqual(strict["error_count"], 0, strict)
            warnings = "\n".join(warning for target in strict["targets"] for warning in target["warnings"])
            self.assertIn("trace_observability.runs[0].run_dir is absolute", warnings)


if __name__ == "__main__":
    unittest.main()
