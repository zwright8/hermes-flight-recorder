import json
import os
import shutil
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

from flightrecorder.cli import main
from flightrecorder.schema import load_scenario, resolve_trace_path
from flightrecorder.state import resolve_before_state_snapshot_path, resolve_state_snapshot_path


ROOT = Path(__file__).resolve().parents[1]


def run_cli(args):
    with redirect_stdout(StringIO()):
        return main(args)


class ScenarioCheckTests(unittest.TestCase):
    def test_check_scenarios_accepts_bundled_suite(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "scenario_check.json"

            code = run_cli(
                [
                    "check-scenarios",
                    "--scenarios",
                    str(ROOT / "scenarios"),
                    "--require-traces",
                    "--strict",
                    "--out",
                    str(out),
                ]
            )

            self.assertEqual(code, 0)
            summary = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(summary["schema_version"], "hfr.scenario_check.v1")
            self.assertEqual(summary["total"], 7)
            self.assertEqual(summary["error_count"], 0)
            self.assertEqual(summary["warning_count"], 0)
            self.assertTrue(summary["passed"])
            self.assertTrue(all(item["trace_exists"] for item in summary["scenarios"]))
            self.assertEqual(run_cli(["validate", "--scenario-check", str(out), "--strict"]), 0)
            self.assertEqual(run_cli(["schemas", "--check", str(out)]), 0)

    def test_strict_validate_warns_on_absolute_preserved_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "scenario_check.json"
            permissive_summary = Path(tmp) / "validation.json"
            strict_summary = Path(tmp) / "strict_validation.json"

            self.assertEqual(
                run_cli(
                    [
                        "check-scenarios",
                        "--scenarios",
                        str(ROOT / "scenarios"),
                        "--require-traces",
                        "--out",
                        str(out),
                        "--preserve-paths",
                    ]
                ),
                0,
            )

            summary = json.loads(out.read_text(encoding="utf-8"))
            self.assertTrue(Path(summary["scenarios_dir"]).is_absolute())
            self.assertTrue(Path(summary["scenarios"][0]["path"]).is_absolute())
            self.assertEqual(run_cli(["validate", "--scenario-check", str(out), "--out", str(permissive_summary)]), 0)
            self.assertEqual(run_cli(["validate", "--scenario-check", str(out), "--strict", "--out", str(strict_summary)]), 1)
            warnings = [
                warning
                for target in json.loads(strict_summary.read_text(encoding="utf-8"))["targets"]
                for warning in target["warnings"]
            ]
            warning_text = "\n".join(warnings)
            self.assertIn("scenario_check.scenarios_dir is absolute", warning_text)
            self.assertIn("scenario_check.scenarios[0].path is absolute", warning_text)
            self.assertTrue(any(".trace_path is absolute" in warning for warning in warnings), warnings)
            self.assertTrue(any(".state_path is absolute" in warning for warning in warnings), warnings)

    def test_check_scenarios_rejects_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            scenarios = Path(tmp) / "scenarios"
            scenarios.mkdir()
            shutil.copy(ROOT / "scenarios" / "prompt_injection_good.json", scenarios / "a.json")
            shutil.copy(ROOT / "scenarios" / "prompt_injection_good.json", scenarios / "b.json")
            for path in (scenarios / "a.json", scenarios / "b.json"):
                payload = json.loads(path.read_text(encoding="utf-8"))
                payload["trace"]["path"] = str(ROOT / "fixtures" / "prompt_injection_good.trajectory.jsonl")
                path.write_text(json.dumps(payload), encoding="utf-8")
            out = Path(tmp) / "scenario_check.json"

            code = run_cli(["check-scenarios", "--scenarios", str(scenarios), "--require-traces", "--out", str(out)])

            self.assertEqual(code, 1)
            summary = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(summary["duplicate_id_count"], 1)
            self.assertIn("Duplicate scenario id", summary["scenarios"][1]["errors"][0])
            self.assertEqual(run_cli(["schemas", "--check", str(out)]), 0)

    def test_check_scenarios_strict_fails_on_missing_trace_warning(self):
        with tempfile.TemporaryDirectory() as tmp:
            scenarios = Path(tmp) / "scenarios"
            scenarios.mkdir()
            scenario = {
                "id": "no_trace",
                "title": "No Trace",
                "prompt": "x",
                "policy": {"max_tool_calls": 2},
            }
            (scenarios / "no_trace.json").write_text(json.dumps(scenario), encoding="utf-8")
            out = Path(tmp) / "scenario_check.json"

            non_strict = run_cli(["check-scenarios", "--scenarios", str(scenarios), "--out", str(out)])
            strict = run_cli(["check-scenarios", "--scenarios", str(scenarios), "--strict", "--out", str(out)])

            self.assertEqual(non_strict, 0)
            self.assertEqual(strict, 1)
            summary = json.loads(out.read_text(encoding="utf-8"))
            self.assertGreater(summary["warning_count"], 0)

    def test_check_scenarios_rejects_invalid_scenario(self):
        with tempfile.TemporaryDirectory() as tmp:
            scenarios = Path(tmp) / "scenarios"
            scenarios.mkdir()
            (scenarios / "bad.json").write_text(
                json.dumps(
                    {
                        "id": "bad",
                        "title": "Bad",
                        "prompt": "x",
                        "policy": {"forbidden_command_patterns": ["["]},
                    }
                ),
                encoding="utf-8",
            )
            out = Path(tmp) / "scenario_check.json"

            code = run_cli(["check-scenarios", "--scenarios", str(scenarios), "--out", str(out)])

            self.assertEqual(code, 1)
            errors = json.loads(out.read_text(encoding="utf-8"))["scenarios"][0]["errors"]
            self.assertTrue(any("Invalid regex" in error for error in errors))

    def test_check_scenarios_missing_directory_is_parser_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit) as raised, redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                main(["check-scenarios", "--scenarios", str(Path(tmp) / "missing")])

            self.assertEqual(raised.exception.code, 2)

    def test_scenario_relative_trace_and_state_do_not_fallback_to_cwd(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cwd_root = root / "cwd"
            scenario_dir = root / "scenarios"
            cwd_root.mkdir()
            scenario_dir.mkdir()
            (cwd_root / "trace.jsonl").write_text("{}\n", encoding="utf-8")
            (cwd_root / "before.json").write_text("{}\n", encoding="utf-8")
            (cwd_root / "after.json").write_text("{}\n", encoding="utf-8")
            scenario_path = scenario_dir / "scenario.json"
            scenario_path.write_text(
                json.dumps(
                    {
                        "id": "cwd-lookalike",
                        "title": "CWD Lookalike",
                        "prompt": "x",
                        "policy": {},
                        "trace": {"path": "trace.jsonl"},
                        "state": {"before_path": "before.json", "path": "after.json"},
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )

            previous_cwd = Path.cwd()
            try:
                os.chdir(cwd_root)
                scenario = load_scenario(scenario_path)
                self.assertEqual(resolve_trace_path(scenario), (scenario_dir / "trace.jsonl").resolve())
                self.assertEqual(resolve_before_state_snapshot_path(scenario), (scenario_dir / "before.json").resolve())
                self.assertEqual(resolve_state_snapshot_path(scenario), (scenario_dir / "after.json").resolve())
                self.assertEqual(resolve_trace_path(scenario, "trace.jsonl"), (cwd_root / "trace.jsonl").resolve())
                self.assertEqual(resolve_state_snapshot_path(scenario, "after.json"), (cwd_root / "after.json").resolve())
            finally:
                os.chdir(previous_cwd)


if __name__ == "__main__":
    unittest.main()
