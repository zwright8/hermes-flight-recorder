import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

from flightrecorder.cli import main


ROOT = Path(__file__).resolve().parents[1]


def run_cli(args):
    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        return main(args)


class ImprovementLedgerGateTests(unittest.TestCase):
    def test_gate_improvement_ledger_passes_and_fails_with_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp) / "runs"
            plan = runs / "improvement_plan.json"
            ledger = runs / "improvement_ledger.json"
            gate_path = runs / "improvement_ledger_gate.json"
            strict_gate_path = runs / "strict_improvement_ledger_gate.json"

            _build_improvement_plan(runs, plan)
            self.assertEqual(run_cli(["improvement-ledger", "--plan", str(plan), "--plan", str(plan), "--out", str(ledger)]), 0)
            self.assertEqual(
                run_cli(
                    [
                        "gate-improvement-ledger",
                        "--improvement-ledger",
                        str(ledger),
                        "--policy",
                        str(ROOT / "examples" / "improvement_ledger_gate_policy.demo.json"),
                        "--out",
                        str(gate_path),
                    ]
                ),
                0,
            )
            self.assertEqual(run_cli(["validate", "--improvement-ledger-gate", str(gate_path), "--strict"]), 0)

            gate = json.loads(gate_path.read_text(encoding="utf-8"))
            plan_payload = json.loads(plan.read_text(encoding="utf-8"))
            self.assertEqual(gate["schema_version"], "hfr.improvement_ledger_gate.v1")
            self.assertTrue(gate["passed"])
            self.assertEqual(gate["decision"]["readiness"], "ready")
            self.assertEqual(gate["decision"]["recommendation"], "promote_iteration")
            self.assertEqual(gate["decision"]["blocking_check_count"], 0)
            self.assertEqual(gate["decision"]["failed_checks"], [])
            self.assertEqual(gate["decision"]["next_actions"], [])
            self.assertEqual(gate["metrics"]["recurring_work_item_count"], plan_payload["work_item_count"])
            self.assertEqual(gate["policy"]["schema_version"], "hfr.improvement_ledger_gate.policy.v1")
            self.assertEqual(gate["policy"]["effective"]["max_recurring_work_items"], plan_payload["work_item_count"])

            self.assertEqual(
                run_cli(
                    [
                        "gate-improvement-ledger",
                        "--improvement-ledger",
                        str(ledger),
                        "--max-recurring-work-items",
                        "0",
                        "--forbid-open-category",
                        "repair",
                        "--out",
                        str(strict_gate_path),
                    ]
                ),
                1,
            )
            strict_gate = json.loads(strict_gate_path.read_text(encoding="utf-8"))
            self.assertEqual(strict_gate["decision"]["readiness"], "blocked")
            self.assertEqual(strict_gate["decision"]["recommendation"], "block_iteration")
            self.assertEqual(strict_gate["decision"]["next_action_count"], 1)
            self.assertEqual(strict_gate["decision"]["next_actions"][0]["id"], "resolve_failed_checks")
            failed_checks = {check["id"] for check in strict_gate["checks"] if not check["passed"]}
            self.assertIn("max_recurring_work_items", failed_checks)
            self.assertIn("forbid_open_category", failed_checks)
            self.assertEqual(run_cli(["validate", "--improvement-ledger-gate", str(strict_gate_path), "--strict"]), 0)

            strict_gate["failed_check_count"] = 0
            strict_gate_path.write_text(json.dumps(strict_gate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            self.assertEqual(run_cli(["validate", "--improvement-ledger-gate", str(strict_gate_path)]), 1)

    def test_gate_improvement_ledger_can_require_resolved_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = root / "runs"
            plan = runs / "improvement_plan.json"
            clean_plan = root / "clean_improvement_plan.json"
            ledger = root / "improvement_ledger.json"
            gate_path = root / "resolved_improvement_ledger_gate.json"

            _build_improvement_plan(runs, plan)
            _write_clean_plan(plan, clean_plan)
            self.assertEqual(run_cli(["improvement-ledger", "--plan", str(plan), "--plan", str(clean_plan), "--out", str(ledger)]), 0)
            payload = json.loads(ledger.read_text(encoding="utf-8"))
            resolved_key = next(entry["work_key"] for entry in payload["entries"] if entry["status"] == "resolved")

            self.assertEqual(
                run_cli(
                    [
                        "gate-improvement-ledger",
                        "--improvement-ledger",
                        str(ledger),
                        "--min-resolved-work-items",
                        "1",
                        "--max-open-work-items",
                        "0",
                        "--require-resolved-work-key",
                        resolved_key,
                        "--out",
                        str(gate_path),
                    ]
                ),
                0,
            )
            gate = json.loads(gate_path.read_text(encoding="utf-8"))
            self.assertTrue(gate["passed"])
            self.assertEqual(gate["metrics"]["open_work_item_count"], 0)
            self.assertGreater(gate["metrics"]["resolved_work_item_count"], 0)
            self.assertEqual(run_cli(["validate", "--improvement-ledger-gate", str(gate_path), "--strict"]), 0)

    def test_gate_improvement_ledger_writes_output_relative_source_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = root / "src"
            output_dir = root / "out"
            runs = source_dir / "runs"
            plan = runs / "improvement_plan.json"
            ledger = source_dir / "improvement_ledger.json"
            gate_path = output_dir / "improvement_ledger_gate.json"
            summary_path = output_dir / "validation.json"
            source_dir.mkdir()
            output_dir.mkdir()
            _build_improvement_plan(runs, plan)
            self.assertEqual(run_cli(["improvement-ledger", "--plan", str(plan), "--out", str(ledger)]), 0)

            self.assertEqual(run_cli(["gate-improvement-ledger", "--improvement-ledger", str(ledger), "--out", str(gate_path)]), 0)

            gate = json.loads(gate_path.read_text(encoding="utf-8"))
            self.assertEqual(gate["improvement_ledger"], "../src/improvement_ledger.json")
            self.assertEqual(run_cli(["validate", "--improvement-ledger-gate", str(gate_path), "--strict", "--out", str(summary_path)]), 0)

    def test_strict_validate_rejects_absolute_improvement_ledger_gate_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = root / "runs"
            plan = runs / "improvement_plan.json"
            ledger = runs / "improvement_ledger.json"
            gate_path = runs / "improvement_ledger_gate.json"
            summary_path = runs / "validation.json"
            strict_summary_path = runs / "strict_validation.json"
            _build_improvement_plan(runs, plan)
            self.assertEqual(run_cli(["improvement-ledger", "--plan", str(plan), "--out", str(ledger)]), 0)
            self.assertEqual(run_cli(["gate-improvement-ledger", "--improvement-ledger", str(ledger), "--out", str(gate_path)]), 0)
            gate = json.loads(gate_path.read_text(encoding="utf-8"))
            gate["improvement_ledger"] = str(ledger)
            gate["policy"] = {
                "schema_version": "hfr.improvement_ledger_gate.policy.v1",
                "path": str(ROOT / "examples" / "improvement_ledger_gate_policy.demo.json"),
                "effective": {},
            }
            gate_path.write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            code = run_cli(["validate", "--improvement-ledger-gate", str(gate_path), "--out", str(summary_path)])
            strict_code = run_cli(["validate", "--improvement-ledger-gate", str(gate_path), "--strict", "--out", str(strict_summary_path)])

            self.assertEqual(code, 0)
            self.assertEqual(strict_code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            warnings = "\n".join(warning for target in summary["targets"] for warning in target["warnings"])
            self.assertIn("improvement_ledger_gate.improvement_ledger is absolute", warnings)
            self.assertIn("improvement_ledger_gate.policy.path is absolute", warnings)
            strict_summary = json.loads(strict_summary_path.read_text(encoding="utf-8"))
            strict_warnings = "\n".join(warning for target in strict_summary["targets"] for warning in target["warnings"])
            self.assertIn("improvement_ledger_gate.improvement_ledger is absolute", strict_warnings)

    def test_validate_rejects_improvement_ledger_gate_stale_source_ledger_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = root / "runs"
            plan = runs / "improvement_plan.json"
            ledger = runs / "improvement_ledger.json"
            gate_path = runs / "improvement_ledger_gate.json"
            summary_path = runs / "validation.json"
            _build_improvement_plan(runs, plan)
            self.assertEqual(run_cli(["improvement-ledger", "--plan", str(plan), "--out", str(ledger)]), 0)
            self.assertEqual(run_cli(["gate-improvement-ledger", "--improvement-ledger", str(ledger), "--out", str(gate_path)]), 0)
            gate = json.loads(gate_path.read_text(encoding="utf-8"))
            open_count = gate["metrics"]["open_work_item_count"]
            gate["metrics"]["new_work_item_count"] = 0
            gate["metrics"]["recurring_work_item_count"] = open_count
            gate["decision"]["key_metrics"] = gate["metrics"]
            gate_path.write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            code = run_cli(["validate", "--improvement-ledger-gate", str(gate_path), "--strict", "--out", str(summary_path)])

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in summary["targets"] for error in target["errors"])
            self.assertIn("improvement_ledger_gate.metrics must match replayed source ledger metrics", errors)

    def test_validate_rejects_improvement_ledger_gate_forged_check_actuals(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = root / "runs"
            plan = runs / "improvement_plan.json"
            ledger = runs / "improvement_ledger.json"
            gate_path = runs / "improvement_ledger_gate.json"
            summary_path = runs / "validation.json"
            _build_improvement_plan(runs, plan)
            self.assertEqual(run_cli(["improvement-ledger", "--plan", str(plan), "--out", str(ledger)]), 0)
            self.assertEqual(
                run_cli(
                    [
                        "gate-improvement-ledger",
                        "--improvement-ledger",
                        str(ledger),
                        "--max-open-work-items",
                        "0",
                        "--out",
                        str(gate_path),
                    ]
                ),
                1,
            )
            gate = json.loads(gate_path.read_text(encoding="utf-8"))
            gate["checks"][0]["actual"] = 0
            gate["checks"][0]["passed"] = True
            gate["checks"][0]["summary"] = "max_open_work_items: actual=0, max=0"
            gate["failed_check_count"] = 0
            gate["passed"] = True
            gate["decision"]["readiness"] = "ready"
            gate["decision"]["recommendation"] = "promote_iteration"
            gate["decision"]["summary"] = "Improvement-ledger gate is ready: concrete repair pressure is within policy."
            gate["decision"]["blocking_check_count"] = 0
            gate["decision"]["blocking_checks"] = []
            gate_path.write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            code = run_cli(["validate", "--improvement-ledger-gate", str(gate_path), "--strict", "--out", str(summary_path)])

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in summary["targets"] for error in target["errors"])
            self.assertIn("improvement_ledger_gate.checks must match replayed source ledger checks", errors)

    def test_validate_rejects_improvement_ledger_gate_missing_source_ledger(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = root / "runs"
            plan = runs / "improvement_plan.json"
            ledger = runs / "improvement_ledger.json"
            gate_path = runs / "improvement_ledger_gate.json"
            summary_path = runs / "validation.json"
            _build_improvement_plan(runs, plan)
            self.assertEqual(run_cli(["improvement-ledger", "--plan", str(plan), "--out", str(ledger)]), 0)
            self.assertEqual(run_cli(["gate-improvement-ledger", "--improvement-ledger", str(ledger), "--out", str(gate_path)]), 0)
            ledger.unlink()

            code = run_cli(["validate", "--improvement-ledger-gate", str(gate_path), "--strict", "--out", str(summary_path)])

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in summary["targets"] for error in target["errors"])
            self.assertIn("improvement_ledger_gate.improvement_ledger must resolve to an existing improvement ledger", errors)

    def test_validate_rejects_improvement_ledger_gate_parent_symlink_source_ledger(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = root / "runs"
            plan = runs / "improvement_plan.json"
            ledger = runs / "improvement_ledger.json"
            gate_path = runs / "improvement_ledger_gate.json"
            summary_path = runs / "validation.json"
            _build_improvement_plan(runs, plan)
            self.assertEqual(run_cli(["improvement-ledger", "--plan", str(plan), "--out", str(ledger)]), 0)
            self.assertEqual(run_cli(["gate-improvement-ledger", "--improvement-ledger", str(ledger), "--out", str(gate_path)]), 0)
            linked_parent = runs / "linked_source"
            try:
                linked_parent.symlink_to(runs, target_is_directory=True)
            except (NotImplementedError, OSError) as exc:
                self.skipTest(f"symlink creation unavailable: {exc}")
            payload = json.loads(gate_path.read_text(encoding="utf-8"))
            payload["improvement_ledger"] = str(Path("linked_source") / "improvement_ledger.json")
            gate_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            code = run_cli(["validate", "--improvement-ledger-gate", str(gate_path), "--strict", "--out", str(summary_path)])

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in summary["targets"] for error in target["errors"])
            self.assertIn(
                "improvement_ledger_gate.improvement_ledger must resolve to a regular non-symlink improvement ledger.",
                errors,
            )

    def test_gate_improvement_ledger_rejects_wrong_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            not_ledger = Path(tmp) / "not_ledger.json"
            not_ledger.write_text(json.dumps({"schema_version": "hfr.not_a_ledger.v1", "metrics": {}, "entries": []}) + "\n")

            with self.assertRaises(SystemExit) as raised:
                run_cli(
                    [
                        "gate-improvement-ledger",
                        "--improvement-ledger",
                        str(not_ledger),
                        "--max-open-work-items",
                        "0",
                    ]
                )

            self.assertEqual(raised.exception.code, 2)


def _build_improvement_plan(runs: Path, plan: Path) -> None:
    assert run_cli(
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
    ) == 0
    assert run_cli(
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
    ) == 0


def _write_clean_plan(source_plan: Path, clean_plan: Path) -> None:
    clean = json.loads(source_plan.read_text(encoding="utf-8"))
    clean["plan_path"] = "clean_improvement_plan.json"
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


if __name__ == "__main__":
    unittest.main()
