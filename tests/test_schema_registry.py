import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

from flightrecorder.cli import main
from flightrecorder.schema_registry import check_schema_contract, check_schema_file, list_schema_records, load_schema, write_schema_bundle


ROOT = Path(__file__).resolve().parents[1]


class SchemaRegistryTests(unittest.TestCase):
    def test_catalog_loads_public_artifact_contracts(self):
        records = list_schema_records()
        names = {record["name"] for record in records}

        self.assertIn("scenario", names)
        self.assertIn("trace", names)
        self.assertIn("scorecard", names)
        self.assertIn("task_completion", names)
        self.assertIn("evidence_bundle", names)
        self.assertIn("training_manifest", names)
        self.assertIn("dataset_splits", names)
        self.assertIn("compare_rl_manifest", names)
        self.assertIn("review_manifest", names)
        self.assertIn("reviewed_manifest", names)
        self.assertIn("trainer_preflight", names)
        self.assertIn("trainer_launch_check", names)
        for record in records:
            schema = load_schema(record["name"])
            self.assertEqual(schema["$id"], record["id"])
            self.assertEqual(schema["$schema"], "https://json-schema.org/draft/2020-12/schema")
            self.assertEqual(schema["type"], "object")

    def test_load_schema_accepts_version_and_filename(self):
        by_name = load_schema("trace")
        by_version = load_schema("hfr.trace.v1")
        by_filename = load_schema("trace.v1.schema.json")

        self.assertEqual(by_name, by_version)
        self.assertEqual(by_name, by_filename)
        self.assertEqual(by_name["properties"]["schema_version"]["const"], "hfr.trace.v1")

    def test_schema_check_passes_real_scenario_and_minimal_trace(self):
        scenario_result = check_schema_file(ROOT / "scenarios" / "prompt_injection_good.json", "scenario")
        self.assertTrue(scenario_result["passed"], scenario_result["errors"])

        trace = {
            "schema_version": "hfr.trace.v1",
            "session": {"id": "session-1", "source_format": "observer_jsonl", "model": "unknown"},
            "events": [{"type": "assistant_message", "session_id": "session-1", "text": "ok"}],
            "final_answer": "ok",
        }
        trace_result = check_schema_contract(trace)
        self.assertTrue(trace_result["passed"], trace_result["errors"])
        self.assertEqual(trace_result["schema"]["name"], "trace")

    def test_schema_check_reports_contract_errors(self):
        result = check_schema_contract(
            {
                "schema_version": "hfr.trace.v1",
                "session": {"id": "session-1", "source_format": "observer_jsonl"},
                "events": [{"session_id": "session-1"}],
            }
        )

        self.assertFalse(result["passed"])
        self.assertGreaterEqual(result["error_count"], 3)
        errors = "\n".join(result["errors"])
        self.assertIn("missing required property 'final_answer'", errors)
        self.assertIn("$.session: missing required property 'model'", errors)
        self.assertIn("$.events[0]: missing required property 'type'", errors)

    def test_task_completion_schema_accepts_not_applicable_status(self):
        result = check_schema_contract(
            {
                "schema_version": "hfr.task_completion.v1",
                "status": "not_applicable",
                "passed": True,
                "task_evidence_configured": False,
                "required_check_count": 0,
                "passed_check_count": 0,
                "failed_check_count": 0,
                "blocking_rule_ids": [],
                "checks": [],
                "summary": "No task-completion evidence assertions were configured.",
            }
        )

        self.assertTrue(result["passed"], result["errors"])
        self.assertEqual(result["schema"]["name"], "task_completion")

    def test_write_schema_bundle_writes_catalog_and_selected_schemas(self):
        with tempfile.TemporaryDirectory() as tmp:
            written = write_schema_bundle(tmp, ["trace", "scorecard"])
            names = {path.name for path in written}

            self.assertEqual(names, {"manifest.json", "trace.v1.schema.json", "scorecard.v1.schema.json"})
            catalog = json.loads((Path(tmp) / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual({record["name"] for record in catalog["schemas"]}, {"trace", "scorecard"})

    def test_cli_lists_and_exports_schemas(self):
        with tempfile.TemporaryDirectory() as tmp:
            stdout = StringIO()
            with redirect_stdout(stdout):
                list_code = main(["schemas"])
            self.assertEqual(list_code, 0)
            self.assertIn("trace\thfr.trace.v1\ttrace.v1.schema.json", stdout.getvalue())

            schema_out = Path(tmp) / "trace.schema.json"
            with redirect_stdout(StringIO()):
                export_code = main(["schemas", "--name", "trace", "--out", str(schema_out)])
            self.assertEqual(export_code, 0)
            exported = json.loads(schema_out.read_text(encoding="utf-8"))
            self.assertEqual(exported["properties"]["schema_version"]["const"], "hfr.trace.v1")

            bundle_dir = Path(tmp) / "bundle"
            with redirect_stdout(StringIO()):
                bundle_code = main(["schemas", "--name", "task_completion", "--write-dir", str(bundle_dir)])
            self.assertEqual(bundle_code, 0)
            self.assertTrue((bundle_dir / "manifest.json").exists())
            self.assertTrue((bundle_dir / "task_completion.v1.schema.json").exists())

            with self.assertRaises(SystemExit) as raised:
                with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                    main(["schemas", "--name", "task_completion", "--write-dir", str(bundle_dir)])
            self.assertEqual(raised.exception.code, 2)

    def test_cli_checks_artifact_schema_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "trace.json"
            trace_path.write_text(
                json.dumps(
                    {
                        "schema_version": "hfr.trace.v1",
                        "session": {"id": "session-1", "source_format": "observer_jsonl", "model": "unknown"},
                        "events": [],
                        "final_answer": "",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            stdout = StringIO()
            with redirect_stdout(stdout):
                code = main(["schemas", "--check", str(trace_path)])
            self.assertEqual(code, 0)
            result = json.loads(stdout.getvalue())
            self.assertTrue(result["passed"])
            self.assertEqual(result["schema"]["artifact_schema_version"], "hfr.trace.v1")

            bad_path = Path(tmp) / "bad.json"
            bad_path.write_text(json.dumps({"schema_version": "hfr.trace.v1"}) + "\n", encoding="utf-8")
            with redirect_stdout(StringIO()):
                bad_code = main(["schemas", "--check", str(bad_path)])
            self.assertEqual(bad_code, 1)

    def test_schema_check_passes_trainer_handoff_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = root / "runs"
            gate = root / "training_gate.json"
            preflight = root / "trainer_preflight.json"
            launch_check = root / "trainer_launch_check.json"

            with redirect_stdout(StringIO()):
                self.assertEqual(
                    main(
                        [
                            "run-suite",
                            "--scenarios",
                            str(ROOT / "scenarios"),
                            "--out",
                            str(runs),
                            "--export-rl",
                        ]
                    ),
                    0,
                )
                self.assertEqual(
                    main(
                        [
                            "gate-export",
                            "--training-export",
                            str(runs / "training_export"),
                            "--policy",
                            str(ROOT / "examples" / "training_gate_policy.demo.json"),
                            "--out",
                            str(gate),
                        ]
                    ),
                    0,
                )
                self.assertEqual(
                    main(
                        [
                            "trainer-preflight",
                            "--gate",
                            str(gate),
                            "--training-export",
                            str(runs / "training_export"),
                            "--require-gate",
                            "training_gate",
                            "--trainer-command",
                            "python train.py --dataset runs/training_export",
                            "--out",
                            str(preflight),
                        ]
                    ),
                    0,
                )
                self.assertEqual(
                    main(
                        [
                            "trainer-launch-check",
                            "--preflight",
                            str(preflight),
                            "--require-gate",
                            "training_gate",
                            "--out",
                            str(launch_check),
                        ]
                    ),
                    0,
                )

            preflight_result = check_schema_file(preflight)
            launch_result = check_schema_file(launch_check)

            self.assertTrue(preflight_result["passed"], preflight_result["errors"])
            self.assertEqual(preflight_result["schema"]["name"], "trainer_preflight")
            self.assertTrue(launch_result["passed"], launch_result["errors"])
            self.assertEqual(launch_result["schema"]["name"], "trainer_launch_check")


if __name__ == "__main__":
    unittest.main()
