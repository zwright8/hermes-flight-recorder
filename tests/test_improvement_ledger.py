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


def write_minimal_improvement_plan(path):
    payload = {
        "schema_version": "hfr.improvement_plan.v1",
        "passed": True,
        "readiness": "ready",
        "work_item_count": 0,
        "decision": {
            "recommendation": "promote_or_monitor",
            "critical_or_high_count": 0,
        },
        "work_items": [],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_work_item_improvement_plan(path):
    payload = {
        "schema_version": "hfr.improvement_plan.v1",
        "passed": True,
        "readiness": "ready",
        "work_item_count": 1,
        "decision": {
            "recommendation": "run_improvement_iteration",
            "critical_or_high_count": 1,
        },
        "work_items": [
            {
                "category": "repair",
                "priority": "high",
                "summary": "Repair prompt-injection evidence coverage.",
                "suggested_action": "Add targeted repair evidence before promotion.",
                "scenario_id": "prompt_injection_bad",
                "task_family": "prompt_injection",
                "rule_id": "forbidden_actions",
                "rule_name": "Forbidden actions",
                "score": 0,
                "task_completion_status": "failed",
                "evidence_refs": [],
            }
        ],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


class ImprovementLedgerTests(unittest.TestCase):
    def test_improvement_ledger_plan_records_include_current_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = root / "improvement_plan.json"
            ledger = root / "improvement_ledger.json"
            write_minimal_improvement_plan(plan)

            self.assertEqual(run_cli(["improvement-ledger", "--plan", str(plan), "--out", str(ledger)]), 0)
            self.assertEqual(run_cli(["validate", "--improvement-ledger", str(ledger), "--strict"]), 0)
            self.assertEqual(run_cli(["schemas", "--check", str(ledger)]), 0)

            payload = json.loads(ledger.read_text(encoding="utf-8"))
            self.assertEqual(payload["plans"][0]["size_bytes"], plan.stat().st_size)

    def test_improvement_ledger_rejects_symlinked_plan_parent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            source.mkdir()
            plan = source / "improvement_plan.json"
            linked_parent = root / "linked_source"
            ledger = root / "improvement_ledger.json"
            write_minimal_improvement_plan(plan)
            try:
                linked_parent.symlink_to(source, target_is_directory=True)
            except (NotImplementedError, OSError) as exc:
                self.skipTest(f"symlinks are not available: {exc}")

            stderr = StringIO()
            with self.assertRaises(SystemExit) as raised, redirect_stderr(stderr):
                run_cli(["improvement-ledger", "--plan", str(linked_parent / plan.name), "--out", str(ledger)])

            self.assertEqual(raised.exception.code, 2)
            self.assertIn("improvement_ledger.plan_path must not traverse symlinked components", stderr.getvalue())
            self.assertFalse(ledger.exists())

    def test_strict_validate_rejects_absolute_improvement_ledger_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = root / "improvement_plan.json"
            ledger = root / "improvement_ledger.json"
            summary = root / "validation.json"
            strict_summary = root / "strict_validation.json"
            write_work_item_improvement_plan(plan)
            self.assertEqual(run_cli(["improvement-ledger", "--plan", str(plan), "--out", str(ledger)]), 0)
            payload = json.loads(ledger.read_text(encoding="utf-8"))
            absolute_plan_path = str(plan)
            payload["ledger_path"] = str(ledger)
            payload["plans"][0]["path"] = absolute_plan_path
            payload["metrics"]["plan_work_item_counts"][0]["path"] = absolute_plan_path
            for entry in payload["entries"]:
                entry["first_seen_path"] = absolute_plan_path
                entry["last_seen_path"] = absolute_plan_path
                for occurrence in entry["occurrences"]:
                    occurrence["plan_path"] = absolute_plan_path
            ledger.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            code = run_cli(["validate", "--improvement-ledger", str(ledger), "--out", str(summary)])
            strict_code = run_cli(["validate", "--improvement-ledger", str(ledger), "--strict", "--out", str(strict_summary)])

            self.assertEqual(code, 0)
            self.assertEqual(strict_code, 1)
            validation = json.loads(summary.read_text(encoding="utf-8"))
            warnings = "\n".join(warning for target in validation["targets"] for warning in target["warnings"])
            for expected in (
                "improvement_ledger.ledger_path is absolute",
                "improvement_ledger.plans[0].path is absolute",
                "improvement_ledger.metrics.plan_work_item_counts[0].path is absolute",
                "improvement_ledger.entries[0].first_seen_path is absolute",
                "improvement_ledger.entries[0].last_seen_path is absolute",
                "improvement_ledger.entries[0].occurrences[0].plan_path is absolute",
            ):
                self.assertIn(expected, warnings)
            strict_validation = json.loads(strict_summary.read_text(encoding="utf-8"))
            strict_warnings = "\n".join(warning for target in strict_validation["targets"] for warning in target["warnings"])
            self.assertIn("improvement_ledger.ledger_path is absolute", strict_warnings)

    def test_validate_rejects_unbound_plan_fingerprints(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = root / "improvement_plan.json"
            ledger = root / "improvement_ledger.json"
            summary = root / "validation.json"
            write_minimal_improvement_plan(plan)
            self.assertEqual(run_cli(["improvement-ledger", "--plan", str(plan), "--out", str(ledger)]), 0)

            payload = json.loads(ledger.read_text(encoding="utf-8"))
            payload["plans"][0].pop("size_bytes")
            ledger.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            self.assertEqual(run_cli(["validate", "--improvement-ledger", str(ledger), "--out", str(summary)]), 1)
            self.assertEqual(run_cli(["schemas", "--check", str(ledger)]), 1)
            validation = json.loads(summary.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in validation["targets"] for error in target["errors"])
            self.assertIn("improvement_ledger.plans[0].size_bytes must be a non-negative integer", errors)

            payload = json.loads(ledger.read_text(encoding="utf-8"))
            payload["plans"][0]["size_bytes"] = plan.stat().st_size
            payload["plans"][0]["exists"] = False
            ledger.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            self.assertEqual(run_cli(["validate", "--improvement-ledger", str(ledger), "--out", str(summary)]), 1)
            self.assertEqual(run_cli(["schemas", "--check", str(ledger)]), 1)
            validation = json.loads(summary.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in validation["targets"] for error in target["errors"])
            self.assertIn("improvement_ledger.plans[0].exists must be true", errors)

            plan.unlink()
            payload = json.loads(ledger.read_text(encoding="utf-8"))
            payload["plans"][0]["exists"] = True
            payload["plans"][0]["size_bytes"] = payload["plans"][0].get("size_bytes", 0) or 1
            ledger.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            self.assertEqual(run_cli(["validate", "--improvement-ledger", str(ledger), "--out", str(summary)]), 1)
            validation = json.loads(summary.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in validation["targets"] for error in target["errors"])
            self.assertIn("improvement_ledger.plans[0].path must resolve to an existing source improvement plan", errors)

            write_minimal_improvement_plan(plan)
            payload = json.loads(ledger.read_text(encoding="utf-8"))
            payload["plans"][0]["exists"] = True
            payload["plans"][0]["size_bytes"] = plan.stat().st_size + 1
            ledger.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            self.assertEqual(run_cli(["validate", "--improvement-ledger", str(ledger), "--out", str(summary)]), 1)
            validation = json.loads(summary.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in validation["targets"] for error in target["errors"])
            self.assertIn("improvement_ledger.plans[0].size_bytes does not match the current file", errors)

    def test_validate_rejects_symlinked_improvement_plan_parent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            source.mkdir()
            plan = source / "improvement_plan.json"
            ledger = root / "improvement_ledger.json"
            summary = root / "validation.json"
            linked_parent = root / "linked_source"
            write_minimal_improvement_plan(plan)
            try:
                linked_parent.symlink_to(source, target_is_directory=True)
            except (NotImplementedError, OSError) as exc:
                self.skipTest(f"symlinks are not available: {exc}")
            self.assertEqual(run_cli(["improvement-ledger", "--plan", str(plan), "--out", str(ledger)]), 0)
            payload = json.loads(ledger.read_text(encoding="utf-8"))
            payload["plans"][0]["path"] = str(Path(linked_parent.name) / plan.name)
            ledger.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            self.assertEqual(run_cli(["validate", "--improvement-ledger", str(ledger), "--out", str(summary)]), 1)
            validation = json.loads(summary.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in validation["targets"] for error in target["errors"])
            self.assertIn(
                "improvement_ledger.plans[0].path must resolve to a regular non-symlink source improvement plan",
                errors,
            )

    def test_improvement_ledger_tracks_recurring_concrete_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp) / "runs"
            plan = runs / "improvement_plan.json"
            ledger = runs / "improvement_ledger.json"
            self.assertEqual(
                run_cli(
                    [
                        "run-suite",
                        "--scenarios",
                        str(ROOT / "scenarios"),
                        "--out",
                        str(runs),
                        "--export-rl",
                        "--validate",
                        "--strict",
                        "--evidence-handoff",
                    ]
                ),
                0,
            )
            self.assertEqual(
                run_cli(
                    [
                        "improvement-plan",
                        "--evidence-bundle",
                        str(runs / "evidence_bundle.json"),
                        "--repair-queue",
                        str(runs / "repair_queue.json"),
                        "--training-export",
                        str(runs / "training_export"),
                        "--runs",
                        str(runs),
                        "--out",
                        str(plan),
                    ]
                ),
                0,
            )
            self.assertEqual(run_cli(["improvement-ledger", "--plan", str(plan), "--plan", str(plan), "--out", str(ledger)]), 0)
            self.assertEqual(run_cli(["validate", "--improvement-ledger", str(ledger), "--strict"]), 0)
            self.assertEqual(run_cli(["schemas", "--check", str(ledger)]), 0)

            payload = json.loads(ledger.read_text(encoding="utf-8"))
            plan_payload = json.loads(plan.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema_version"], "hfr.improvement_ledger.v1")
            self.assertEqual(payload["plan_count"], 2)
            self.assertEqual(payload["unique_work_item_count"], plan_payload["work_item_count"])
            self.assertEqual(payload["metrics"]["recurring_work_item_count"], plan_payload["work_item_count"])
            self.assertEqual(payload["metrics"]["open_work_item_count"], plan_payload["work_item_count"])
            self.assertEqual(payload["metrics"]["resolved_work_item_count"], 0)
            self.assertEqual(payload["decision"]["recommendation"], "continue_improvement")
            self.assertTrue(all(entry["status"] == "recurring" for entry in payload["entries"]))
            self.assertTrue(any(entry["work_key"] == "repair:prompt_injection_bad:forbidden_actions" for entry in payload["entries"]))

    def test_improvement_ledger_tracks_resolved_work_when_latest_plan_is_clean(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = root / "runs"
            plan = runs / "improvement_plan.json"
            clean_plan = root / "clean_improvement_plan.json"
            ledger = root / "improvement_ledger.json"
            self.assertEqual(
                run_cli(
                    [
                        "run-suite",
                        "--scenarios",
                        str(ROOT / "scenarios"),
                        "--out",
                        str(runs),
                        "--export-rl",
                        "--evidence-handoff",
                    ]
                ),
                0,
            )
            self.assertEqual(
                run_cli(
                    [
                        "improvement-plan",
                        "--evidence-bundle",
                        str(runs / "evidence_bundle.json"),
                        "--repair-queue",
                        str(runs / "repair_queue.json"),
                        "--training-export",
                        str(runs / "training_export"),
                        "--runs",
                        str(runs),
                        "--out",
                        str(plan),
                    ]
                ),
                0,
            )
            clean = json.loads(plan.read_text(encoding="utf-8"))
            clean["plan_path"] = "clean_improvement_plan.json"
            for record in clean["source_artifacts"].values():
                path = record.get("path")
                if isinstance(path, str) and path and not Path(path).is_absolute():
                    record["path"] = str(Path("runs") / path)
            clean["work_items"] = []
            clean["work_item_count"] = 0
            clean["metrics"] = {
                "work_item_count": 0,
                "scenario_count": 0,
                "task_family_count": 0,
                "rule_count": 0,
                "priority_counts": [],
                "category_counts": [],
                "task_family_counts": [],
                "rule_counts": [],
                "repair_backed_count": 0,
                "curriculum_backed_count": 0,
                "digest_backed_count": 0,
                "bundle_action_count": 0,
                "evidence_ref_count": 0,
                "scenarios": [],
                "task_families": [],
                "rules": [],
            }
            clean["decision"] = {
                "readiness": "ready",
                "recommendation": "promote_or_monitor",
                "summary": "No work items remain.",
                "source_bundle_recommendation": "promote_handoff",
                "source_bundle_next_action_count": 0,
                "work_item_count": 0,
                "critical_or_high_count": 0,
                "top_work_items": [],
            }
            clean_plan.write_text(json.dumps(clean, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            self.assertEqual(run_cli(["validate", "--improvement-plan", str(clean_plan), "--strict"]), 0)

            self.assertEqual(run_cli(["improvement-ledger", "--plan", str(plan), "--plan", str(clean_plan), "--out", str(ledger)]), 0)
            self.assertEqual(run_cli(["validate", "--improvement-ledger", str(ledger), "--strict"]), 0)

            payload = json.loads(ledger.read_text(encoding="utf-8"))
            original = json.loads(plan.read_text(encoding="utf-8"))
            self.assertEqual(payload["metrics"]["open_work_item_count"], 0)
            self.assertEqual(payload["metrics"]["resolved_work_item_count"], original["work_item_count"])
            self.assertEqual(payload["decision"]["recommendation"], "promote_or_monitor")
            self.assertTrue(all(entry["status"] == "resolved" for entry in payload["entries"]))

    def test_validate_rejects_stale_improvement_ledger_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp) / "runs"
            plan = runs / "improvement_plan.json"
            ledger = runs / "improvement_ledger.json"
            summary = runs / "validation.json"
            self.assertEqual(
                run_cli(
                    [
                        "run-suite",
                        "--scenarios",
                        str(ROOT / "scenarios"),
                        "--out",
                        str(runs),
                        "--export-rl",
                        "--evidence-handoff",
                    ]
                ),
                0,
            )
            self.assertEqual(
                run_cli(
                    [
                        "improvement-plan",
                        "--evidence-bundle",
                        str(runs / "evidence_bundle.json"),
                        "--repair-queue",
                        str(runs / "repair_queue.json"),
                        "--training-export",
                        str(runs / "training_export"),
                        "--runs",
                        str(runs),
                        "--out",
                        str(plan),
                    ]
                ),
                0,
            )
            self.assertEqual(run_cli(["improvement-ledger", "--plan", str(plan), "--out", str(ledger)]), 0)
            payload = json.loads(ledger.read_text(encoding="utf-8"))
            payload["metrics"]["open_work_item_count"] += 1
            repair_entry = next(entry for entry in payload["entries"] if entry["category"] == "repair")
            repair_entry["work_key"] = "repair:wrong:required_evidence"
            ledger.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            code = run_cli(["validate", "--improvement-ledger", str(ledger), "--out", str(summary)])

            self.assertEqual(code, 1)
            validation = json.loads(summary.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in validation["targets"] for error in target["errors"])
            self.assertIn("improvement_ledger.metrics.open_work_item_count", errors)
            self.assertIn("work_key does not match entry content", errors)


if __name__ == "__main__":
    unittest.main()
