import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from flightrecorder.agentic_training_runtime import build_agentic_training_runtime_preflight
from flightrecorder.schema_registry import check_schema_contract, check_schema_file, list_schema_records


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_PLAN = ROOT / "examples" / "agentic_training" / "plans" / "sft_then_dpo_plan.json"


class AgenticTrainingRuntimePreflightTests(unittest.TestCase):
    def test_committed_example_plan_can_pass_dependency_scoped_runtime_preflight(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "runtime_preflight.json"
            preflight = build_agentic_training_runtime_preflight(
                plan_path=EXAMPLE_PLAN,
                out_path=out,
                require_modules=["json"],
                skip_default_modules=True,
                created_at="2026-07-02T00:00:00+00:00",
            )

            self.assertTrue(preflight["passed"], preflight["blocked_reasons"])
            self.assertEqual(preflight["recommendation"], "ready_for_tiny_smoke_launch")
            self.assertEqual(preflight["plan_mode"], "sft_then_dpo")
            self.assertEqual(preflight["backend"], "axolotl")
            self.assertEqual({view["name"] for view in preflight["view_checks"]}, {"sft", "dpo"})
            self.assertTrue(all(view["passed"] for view in preflight["view_checks"]))
            self.assertTrue(all(not Path(view["resolved_path"]).is_absolute() for view in preflight["view_checks"]))
            for view in preflight["view_checks"]:
                self.assertIsInstance(view["size_bytes"], int)
            self.assertFalse(preflight["execution_boundary"]["flight_recorder_launched_training"])
            self.assertFalse(preflight["execution_boundary"]["trainer_modules_imported"])
            schema = check_schema_contract(preflight)
            self.assertTrue(schema["passed"], schema["errors"])

    def test_missing_required_module_blocks_tiny_smoke_launch(self):
        preflight = build_agentic_training_runtime_preflight(
            plan_path=EXAMPLE_PLAN,
            require_modules=["hfr_missing_dependency_for_test"],
            skip_default_modules=True,
            created_at="2026-07-02T00:00:00+00:00",
        )

        self.assertFalse(preflight["passed"])
        self.assertEqual(preflight["recommendation"], "block_tiny_smoke_launch")
        self.assertIn("hfr_missing_dependency_for_test", {check["module"] for check in preflight["dependency_checks"] if not check["passed"]})
        self.assertIn("runtime_dependencies_available", {check["id"] for check in preflight["checks"] if not check["passed"]})
        schema = check_schema_contract(preflight)
        self.assertTrue(schema["passed"], schema["errors"])

    def test_missing_selected_view_blocks_tiny_smoke_launch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = json.loads(EXAMPLE_PLAN.read_text(encoding="utf-8"))
            plan["selected_views"][0]["path"] = "missing_sft.jsonl"
            plan_path = root / "plan-with-missing-view.json"
            plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            preflight = build_agentic_training_runtime_preflight(
                plan_path=plan_path,
                require_modules=["json"],
                skip_default_modules=True,
                created_at="2026-07-02T00:00:00+00:00",
            )

            self.assertFalse(preflight["passed"])
            self.assertEqual(preflight["recommendation"], "block_tiny_smoke_launch")
            self.assertIn("selected_views_schema_passed", {check["id"] for check in preflight["checks"] if not check["passed"]})
            failed_views = [view for view in preflight["view_checks"] if not view["passed"]]
            self.assertEqual(failed_views[0]["name"], "sft")
            self.assertIn("view path is not a regular file", failed_views[0]["errors"])
            schema = check_schema_contract(preflight)
            self.assertTrue(schema["passed"], schema["errors"])

    def test_runtime_preflight_rejects_cwd_relative_view_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan_dir = root / "plans"
            cwd_dir = root / "cwd"
            plan_dir.mkdir()
            cwd_dir.mkdir()
            (cwd_dir / "sft.jsonl").write_text(
                (ROOT / "examples" / "agentic_training" / "data" / "sft.jsonl").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            plan = json.loads(EXAMPLE_PLAN.read_text(encoding="utf-8"))
            plan["selected_views"] = [
                {
                    "name": "sft",
                    "path": "sft.jsonl",
                    "row_count": 2,
                    "schema_version": "hfr.rl.sft.v1",
                }
            ]
            plan_path = plan_dir / "plan-with-cwd-lookalike-view.json"
            plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            previous_cwd = Path.cwd()
            try:
                os.chdir(cwd_dir)
                preflight = build_agentic_training_runtime_preflight(
                    plan_path=plan_path,
                    require_modules=["json"],
                    skip_default_modules=True,
                    created_at="2026-07-02T00:00:00+00:00",
                )
            finally:
                os.chdir(previous_cwd)

            self.assertFalse(preflight["passed"])
            self.assertEqual(preflight["recommendation"], "block_tiny_smoke_launch")
            self.assertEqual(preflight["view_checks"][0]["resolved_path"], "sft.jsonl")
            self.assertIn("view path is not a regular file", preflight["view_checks"][0]["errors"])
            schema = check_schema_contract(preflight)
            self.assertTrue(schema["passed"], schema["errors"])

    def test_cli_writes_schema_checkable_runtime_preflight(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "runtime_preflight.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "preflight_agentic_training_runtime.py"),
                    "--plan",
                    str(EXAMPLE_PLAN),
                    "--skip-default-modules",
                    "--require-module",
                    "json",
                    "--created-at",
                    "2026-07-02T00:00:00+00:00",
                    "--out",
                    str(out),
                ],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr + completed.stdout)
            preflight = json.loads(out.read_text(encoding="utf-8"))
            self.assertTrue(preflight["passed"], preflight["blocked_reasons"])
            schema = check_schema_file(out)
            self.assertTrue(schema["passed"], schema["errors"])

    def test_committed_example_runtime_preflight_is_schema_checkable(self):
        preflight_path = ROOT / "examples" / "agentic_training" / "runtime_preflight" / "ready.json"
        preflight = json.loads(preflight_path.read_text(encoding="utf-8"))

        self.assertTrue(preflight["passed"], preflight["blocked_reasons"])
        self.assertEqual(preflight["recommendation"], "ready_for_tiny_smoke_launch")
        self.assertTrue(all(not Path(view["resolved_path"]).is_absolute() for view in preflight["view_checks"]))
        for view in preflight["view_checks"]:
            view_path = ROOT / view["resolved_path"]
            self.assertEqual(view["size_bytes"], view_path.stat().st_size)
        schema = check_schema_file(preflight_path)
        self.assertTrue(schema["passed"], schema["errors"])

    def test_schema_rejects_passed_view_without_size(self):
        preflight = build_agentic_training_runtime_preflight(
            plan_path=EXAMPLE_PLAN,
            require_modules=["json"],
            skip_default_modules=True,
            created_at="2026-07-02T00:00:00+00:00",
        )
        preflight["view_checks"][0].pop("size_bytes")

        schema = check_schema_contract(preflight)

        self.assertFalse(schema["passed"])
        self.assertTrue(any("size_bytes" in error for error in schema["errors"]))

    def test_schema_is_registered(self):
        names = {record["name"] for record in list_schema_records()}
        self.assertIn("agentic_training_runtime_preflight", names)


if __name__ == "__main__":
    unittest.main()
