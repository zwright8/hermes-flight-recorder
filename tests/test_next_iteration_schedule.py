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

    def test_schedule_redacts_external_output_path_and_keeps_local_sources_replayable(self):
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
                out_path=out,
                created_at="2026-07-03T00:00:00+00:00",
            )
            write_next_iteration_schedule(out, schedule)

            payload = json.loads(out.read_text(encoding="utf-8"))
            rendered = json.dumps(payload, sort_keys=True)
            self.assertEqual(payload["schedule_path"], "<redacted:next_iteration_schedule.json>")
            self.assertEqual(payload["source_ledgers"]["agentic_loop_ledger"][0]["path"], "agentic_loop_ledger.json")
            self.assertEqual(payload["source_ledgers"]["action_ledger"][0]["path"], "action_ledger.json")
            self.assertEqual(payload["source_ledgers"]["improvement_ledger"][0]["path"], "improvement_ledger.json")
            self.assertNotIn(str(root), rendered)
            self.assertTrue(validate_artifacts(next_iteration_schedule_paths=[out], strict=True)["passed"])

    def test_schedule_blocks_unreplayable_external_source_ledgers(self):
        with tempfile.TemporaryDirectory() as source_tmp, tempfile.TemporaryDirectory(dir=Path.cwd()) as out_tmp:
            source_root = Path(source_tmp)
            out_root = Path(out_tmp)
            loop = self.write_loop_ledger(source_root / "agentic_loop_ledger.json")
            action = self.write_action_ledger(source_root / "action_ledger.json")
            improvement = self.write_improvement_ledger(source_root / "improvement_ledger.json")
            out = out_root / "schedule.json"

            schedule = build_next_iteration_schedule(
                loop_ledger_path=loop,
                action_ledger_path=action,
                improvement_ledger_path=improvement,
                out_path=out,
                created_at="2026-07-03T00:00:00+00:00",
            )
            write_next_iteration_schedule(out, schedule)

            payload = json.loads(out.read_text(encoding="utf-8"))
            rendered = json.dumps(payload, sort_keys=True)
            self.assertFalse(payload["passed"])
            self.assertEqual(payload["recommendation"], "fix_schedule_inputs")
            for role in ("agentic_loop_ledger", "action_ledger", "improvement_ledger"):
                ref = payload["source_ledgers"][role][0]
                self.assertEqual(ref["path"], f"<redacted:{role}.json>")
                self.assertFalse(ref["exists"])
                self.assertIsNone(ref["sha256"])
                self.assertIsNone(ref["size_bytes"])
            self.assertNotIn(str(source_root), rendered)
            self.assertTrue(check_schema_file(out)["passed"])
            validation = validate_artifacts(next_iteration_schedule_paths=[out], strict=True)
            self.assertFalse(validation["passed"], validation)
            errors = "\n".join(error for target in validation["targets"] for error in target["errors"])
            self.assertIn("next_iteration_schedule.source_ledgers.agentic_loop_ledger[0].exists must be true.", errors)
            self.assertIn("next_iteration_schedule.source_ledgers.agentic_loop_ledger[0].sha256 must be a SHA-256 hex string.", errors)
            self.assertNotIn("next_iteration_schedule.source_ledgers.agentic_loop_ledger[0].path does not resolve to an existing ledger file.", errors)

    def test_validation_rejects_absolute_schedule_and_source_paths(self):
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
            write_next_iteration_schedule(out, schedule)
            payload = json.loads(out.read_text(encoding="utf-8"))
            payload["schedule_path"] = str(out.resolve())
            payload["source_ledgers"]["action_ledger"][0]["path"] = str(action.resolve())
            out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            validation = validate_artifacts(next_iteration_schedule_paths=[out], strict=True)

            self.assertFalse(validation["passed"], validation)
            errors = "\n".join(error for target in validation["targets"] for error in target["errors"])
            self.assertIn("next_iteration_schedule.schedule_path must be a safe relative path or redacted placeholder.", errors)
            self.assertIn(
                "next_iteration_schedule.source_ledgers.action_ledger[0].path must be a safe relative path or redacted placeholder.",
                errors,
            )
            self.assertNotIn("next_iteration_schedule.source_ledgers.action_ledger[0].path does not resolve to an existing ledger file.", errors)
            self.assertNotIn("next_iteration_schedule.source_ledgers.action_ledger[0].size_bytes does not match the current file.", errors)
            self.assertNotIn("next_iteration_schedule.source_ledgers.action_ledger[0].sha256 does not match the current file.", errors)

    def test_validation_rejects_unsafe_source_paths_without_dereferencing_them(self):
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
            write_next_iteration_schedule(out, schedule)
            base_payload = json.loads(out.read_text(encoding="utf-8"))
            cases = (
                (str(action.resolve()), "existing"),
                (str((root / "missing_action_ledger.json").resolve()), "missing"),
                ("../action_ledger.json", "traversal"),
            )

            for unsafe_path, case_name in cases:
                with self.subTest(case=case_name):
                    payload = json.loads(json.dumps(base_payload))
                    payload["source_ledgers"]["action_ledger"][0]["path"] = unsafe_path
                    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

                    validation = validate_artifacts(next_iteration_schedule_paths=[out], strict=True)

                    self.assertFalse(validation["passed"], validation)
                    errors = "\n".join(error for target in validation["targets"] for error in target["errors"])
                    self.assertIn(
                        "next_iteration_schedule.source_ledgers.action_ledger[0].path must be a safe relative path or redacted placeholder.",
                        errors,
                    )
                    self.assertNotIn(
                        "next_iteration_schedule.source_ledgers.action_ledger[0].path does not resolve to an existing ledger file.",
                        errors,
                    )
                    self.assertNotIn(
                        "next_iteration_schedule.source_ledgers.action_ledger[0].size_bytes does not match the current file.",
                        errors,
                    )
                    self.assertNotIn(
                        "next_iteration_schedule.source_ledgers.action_ledger[0].sha256 does not match the current file.",
                        errors,
                    )

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
            out = root / "blocked_schedule.json"

            schedule = build_next_iteration_schedule(
                loop_ledger_path=loop,
                action_ledger_path=root / "missing_action_ledger.json",
                improvement_ledger_path=improvement,
                out_path=out,
                created_at="2026-07-03T00:00:00+00:00",
            )
            write_next_iteration_schedule(out, schedule)

            self.assertFalse(schedule["passed"])
            self.assertEqual(schedule["readiness"], "blocked")
            self.assertEqual(schedule["recommendation"], "fix_schedule_inputs")
            self.assertIn("action_ledger_present", {check["id"] for check in schedule["checks"] if not check["passed"]})
            self.assertFalse(schedule["execution_boundary"]["codex_threads_created"])
            schema = check_schema_file(out)
            self.assertTrue(schema["passed"], schema["errors"])
            validation = validate_artifacts(next_iteration_schedule_paths=[out], strict=True)
            self.assertFalse(validation["passed"], validation)

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

    def test_validation_rejects_stale_source_ledger_metrics(self):
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
            write_next_iteration_schedule(out, schedule)
            action_payload = json.loads(action.read_text(encoding="utf-8"))
            action_payload["metrics"]["open_action_count"] = 9
            action.write_text(json.dumps(action_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            validation = validate_artifacts(next_iteration_schedule_paths=[out], strict=True)

            self.assertFalse(validation["passed"], validation)
            errors = "\n".join(error for target in validation["targets"] for error in target["errors"])
            self.assertIn("next_iteration_schedule.source_ledgers.action_ledger[0].sha256 does not match the current file.", errors)
            self.assertIn("next_iteration_schedule.source_ledgers.action_ledger[0].metrics must match the current file.", errors)
            self.assertIn("next_iteration_schedule.pressure.open_action_count must match source ledgers.", errors)
            self.assertIn("next_iteration_schedule.pressure.total_open_signal_count must match source ledgers.", errors)

    def test_validation_rejects_parent_symlink_source_ledgers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source_ledgers"
            source.mkdir()
            loop = self.write_loop_ledger(source / "agentic_loop_ledger.json")
            action = self.write_action_ledger(source / "action_ledger.json")
            improvement = self.write_improvement_ledger(source / "improvement_ledger.json")
            out = root / "schedule.json"
            schedule = build_next_iteration_schedule(
                loop_ledger_path=loop,
                action_ledger_path=action,
                improvement_ledger_path=improvement,
                out_path=out,
                created_at="2026-07-03T00:00:00+00:00",
            )
            write_next_iteration_schedule(out, schedule)
            linked_source = root / "linked_source"
            linked_source.symlink_to(source, target_is_directory=True)
            base_payload = json.loads(out.read_text(encoding="utf-8"))
            source_files = {
                "agentic_loop_ledger": "agentic_loop_ledger.json",
                "action_ledger": "action_ledger.json",
                "improvement_ledger": "improvement_ledger.json",
            }

            for role, filename in source_files.items():
                with self.subTest(role=role):
                    payload = json.loads(json.dumps(base_payload))
                    payload["source_ledgers"][role][0]["path"] = str(Path("linked_source") / filename)
                    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

                    validation = validate_artifacts(next_iteration_schedule_paths=[out], strict=True)

                    self.assertFalse(validation["passed"], validation)
                    errors = "\n".join(error for target in validation["targets"] for error in target["errors"])
                    self.assertIn(
                        f"next_iteration_schedule.source_ledgers.{role}[0].path must resolve to a regular non-symlink ledger file.",
                        errors,
                    )

    def test_validation_and_schema_reject_unknown_schedule_fields(self):
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
            write_next_iteration_schedule(out, schedule)
            payload = json.loads(out.read_text(encoding="utf-8"))
            payload["scheduler_job_id"] = "external-only"
            payload["checks"][0]["provider_call"] = "forged"
            payload["source_ledgers"]["unexpected_ledger"] = []
            payload["source_ledgers"]["agentic_loop_ledger"][0]["credential_hint"] = "redacted"
            payload["source_ledgers"]["agentic_loop_ledger"][0]["metrics"]["hidden_metric"] = 1
            payload["pressure"]["hidden_pressure"] = 1
            payload["next_iteration"]["external_thread_reference"] = "<redacted:thread>"
            payload["execution_boundary"]["automation_receipt"] = "not-created"
            out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            schema = check_schema_file(out)
            validation = validate_artifacts(next_iteration_schedule_paths=[out], strict=True)

            self.assertFalse(schema["passed"], schema["errors"])
            self.assertFalse(validation["passed"], validation)
            errors = "\n".join(error for target in validation["targets"] for error in target["errors"])
            self.assertIn("next_iteration_schedule contains unknown field(s): ['scheduler_job_id'].", errors)
            self.assertIn("next_iteration_schedule.checks[0] contains unknown field(s): ['provider_call'].", errors)
            self.assertIn("next_iteration_schedule.source_ledgers contains unknown field(s): ['unexpected_ledger'].", errors)
            self.assertIn(
                "next_iteration_schedule.source_ledgers.agentic_loop_ledger[0] contains unknown field(s): ['credential_hint'].",
                errors,
            )
            self.assertIn(
                "next_iteration_schedule.source_ledgers.agentic_loop_ledger[0].metrics contains unknown field(s): ['hidden_metric'].",
                errors,
            )
            self.assertIn("next_iteration_schedule.pressure contains unknown field(s): ['hidden_pressure'].", errors)
            self.assertIn(
                "next_iteration_schedule.next_iteration contains unknown field(s): ['external_thread_reference'].",
                errors,
            )
            self.assertIn(
                "next_iteration_schedule.execution_boundary contains unknown field(s): ['automation_receipt'].",
                errors,
            )

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
