import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from flightrecorder.next_iteration_schedule import build_next_iteration_schedule, write_next_iteration_schedule
from flightrecorder.schema_registry import check_schema_file, list_schema_records
from flightrecorder.validation import validate_artifacts


ROOT = Path(__file__).resolve().parents[1]


class NextIterationScheduleTests(unittest.TestCase):
    def test_schedule_proposes_next_iteration_without_side_effects(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            loop = self.write_loop_ledger(root / "agentic_loop_ledger.json")
            action = self.write_action_ledger(root / "action_ledger.json")
            improvement = self.write_improvement_ledger(root / "improvement_ledger.json")
            out = root / "next_iteration_schedule.json"

            schedule = build_next_iteration_schedule(
                loop_ledger_path=loop,
                action_ledger_path=action,
                improvement_ledger_path=improvement,
                next_iteration_id="loop-002",
                objective="Close remaining repair pressure",
                schedule={"cadence": "manual"},
                out_path=out,
                created_at="2026-07-03T00:00:00+00:00",
            )
            write_next_iteration_schedule(out, schedule)

            schema = check_schema_file(out)
            self.assertTrue(schema["passed"], schema["errors"])
            validation = validate_artifacts(next_iteration_schedule_paths=[out], strict=True)
            self.assertTrue(validation["passed"], validation)
            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertTrue(payload["passed"], payload["blocked_reasons"])
            self.assertEqual(payload["recommendation"], "create_next_loop_plan")
            self.assertEqual(payload["next_iteration"]["iteration_id"], "loop-002")
            self.assertFalse(payload["next_iteration"]["scheduled"])
            self.assertFalse(payload["execution_boundary"]["automations_created"])
            self.assertEqual(payload["pressure"]["total_open_signal_count"], 6)

    def test_cli_writes_valid_next_iteration_schedule(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            loop = self.write_loop_ledger(root / "agentic_loop_ledger.json")
            action = self.write_action_ledger(root / "action_ledger.json")
            improvement = self.write_improvement_ledger(root / "improvement_ledger.json")
            out = root / "schedule.json"

            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "flightrecorder",
                    "next-iteration-schedule",
                    "--loop-ledger",
                    str(loop),
                    "--action-ledger",
                    str(action),
                    "--improvement-ledger",
                    str(improvement),
                    "--next-iteration-id",
                    "loop-cli-next",
                    "--objective",
                    "CLI next iteration",
                    "--schedule",
                    "cadence=\"manual\"",
                    "--created-at",
                    "2026-07-03T00:00:00+00:00",
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
            validation = validate_artifacts(next_iteration_schedule_paths=[out], strict=True)
            self.assertTrue(validation["passed"], validation)

    def test_schedule_blocks_missing_action_ledger(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            loop = self.write_loop_ledger(root / "agentic_loop_ledger.json")
            improvement = self.write_improvement_ledger(root / "improvement_ledger.json")

            schedule = build_next_iteration_schedule(
                loop_ledger_path=loop,
                action_ledger_path=root / "missing_action_ledger.json",
                improvement_ledger_path=improvement,
                created_at="2026-07-03T00:00:00+00:00",
            )

            self.assertFalse(schedule["passed"])
            self.assertEqual(schedule["readiness"], "blocked")
            self.assertEqual(schedule["recommendation"], "fix_schedule_inputs")
            self.assertIn("action_ledger_present", {check["id"] for check in schedule["checks"] if not check["passed"]})
            self.assertFalse(schedule["execution_boundary"]["codex_threads_created"])

    def test_validation_rejects_scheduler_side_effect_claims(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            loop = self.write_loop_ledger(root / "agentic_loop_ledger.json")
            action = self.write_action_ledger(root / "action_ledger.json")
            improvement = self.write_improvement_ledger(root / "improvement_ledger.json")
            out = root / "schedule.json"
            schedule = build_next_iteration_schedule(
                loop_ledger_path=loop,
                action_ledger_path=action,
                improvement_ledger_path=improvement,
                out_path=out,
                created_at="2026-07-03T00:00:00+00:00",
            )
            schedule["execution_boundary"]["automations_created"] = True
            write_next_iteration_schedule(out, schedule)

            validation = validate_artifacts(next_iteration_schedule_paths=[out], strict=True)
            self.assertFalse(validation["passed"], validation)
            errors = "\n".join(error for target in validation["targets"] for error in target["errors"])
            self.assertIn("execution_boundary.automations_created must be false", errors)

    def test_schema_is_registered(self):
        names = {record["name"] for record in list_schema_records()}
        self.assertIn("next_iteration_schedule", names)

    def write_loop_ledger(self, path: Path) -> Path:
        return self.write_json(
            path,
            {
                "schema_version": "hfr.agentic_loop_ledger.v1",
                "passed": True,
                "metrics": {
                    "latest_iteration_id": "loop-001",
                    "latest_readiness": "planned_fail_closed",
                    "latest_missing_phase_input_count": 2,
                },
            },
        )

    def write_action_ledger(self, path: Path) -> Path:
        return self.write_json(
            path,
            {
                "schema_version": "hfr.action_ledger.v1",
                "passed": True,
                "metrics": {"open_action_count": 1, "new_action_count": 1, "recurring_action_count": 0},
            },
        )

    def write_improvement_ledger(self, path: Path) -> Path:
        return self.write_json(
            path,
            {
                "schema_version": "hfr.improvement_ledger.v1",
                "passed": True,
                "metrics": {
                    "open_work_item_count": 3,
                    "critical_open_work_item_count": 1,
                    "high_open_work_item_count": 1,
                    "resolved_work_item_count": 2,
                },
            },
        )

    def write_json(self, path: Path, payload: dict) -> Path:
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return path


if __name__ == "__main__":
    unittest.main()
