import json
import shutil
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


def _count_rows(values):
    counts = {}
    for value in values:
        key = str(value or "unknown")
        counts[key] = counts.get(key, 0) + 1
    return [{"id": key, "count": counts[key]} for key in sorted(counts)]


class ActionLedgerTests(unittest.TestCase):
    def test_action_ledger_marks_recurring_actions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = root / "runs"
            runs.mkdir()
            gate = root / "failed_gate.json"
            bundle_1 = root / "bundle_1.json"
            bundle_2 = root / "bundle_2.json"
            ledger_path = root / "action_ledger.json"
            gate.write_text(json.dumps({"schema_version": "hfr.test_gate.v1", "passed": False}, sort_keys=True) + "\n", encoding="utf-8")
            self.assertEqual(run_cli(["evidence-bundle", "--runs", str(runs), "--gate", str(gate), "--out", str(bundle_1)]), 1)
            shutil.copyfile(bundle_1, bundle_2)

            code = run_cli(["action-ledger", "--bundle", str(bundle_1), "--bundle", str(bundle_2), "--out", str(ledger_path)])

            self.assertEqual(code, 0)
            ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
            self.assertEqual(ledger["schema_version"], "hfr.action_ledger.v1")
            self.assertTrue(ledger["passed"])
            self.assertEqual(ledger["bundle_count"], 2)
            self.assertEqual(ledger["metrics"]["recurring_action_count"], ledger["unique_action_count"])
            self.assertEqual(ledger["metrics"]["resolved_action_count"], 0)
            self.assertTrue(all(entry["status"] == "recurring" for entry in ledger["entries"]))
            self.assertTrue(all(entry["occurrence_count"] == 2 for entry in ledger["entries"]))
            self.assertTrue(all(len(entry["action_fingerprint"]) == 64 for entry in ledger["entries"]))
            self.assertTrue(all(entry["routing_key"].endswith(entry["action_fingerprint"][:12]) for entry in ledger["entries"]))
            self.assertEqual(run_cli(["validate", "--action-ledger", str(ledger_path), "--strict"]), 0)
            self.assertEqual(run_cli(["schemas", "--check", str(ledger_path)]), 0)

            gate_path = root / "action_ledger_gate.json"
            self.assertEqual(
                run_cli(
                    [
                        "gate-action-ledger",
                        "--action-ledger",
                        str(ledger_path),
                        "--policy",
                        str(ROOT / "examples" / "action_ledger_gate_policy.demo.json"),
                        "--out",
                        str(gate_path),
                    ]
                ),
                0,
            )
            gate_result = json.loads(gate_path.read_text(encoding="utf-8"))
            self.assertEqual(gate_result["schema_version"], "hfr.action_ledger_gate.v1")
            self.assertTrue(gate_result["passed"])
            self.assertEqual(gate_result["decision"]["readiness"], "ready")
            self.assertEqual(gate_result["decision"]["recommendation"], "promote_iteration")
            self.assertEqual(gate_result["decision"]["blocking_check_count"], 0)
            self.assertEqual(gate_result["decision"]["key_metrics"]["recurring_action_count"], gate_result["metrics"]["recurring_action_count"])
            self.assertEqual(gate_result["policy"]["schema_version"], "hfr.action_ledger_gate.policy.v1")
            self.assertEqual(gate_result["policy"]["effective"]["max_recurring_actions"], 6)
            self.assertEqual(run_cli(["validate", "--action-ledger-gate", str(gate_path), "--strict"]), 0)
            self.assertEqual(run_cli(["schemas", "--check", str(gate_path)]), 0)

            strict_gate_path = root / "strict_action_ledger_gate.json"
            self.assertEqual(
                run_cli(
                    [
                        "gate-action-ledger",
                        "--action-ledger",
                        str(ledger_path),
                        "--max-recurring-actions",
                        "0",
                        "--forbid-open-priority",
                        "critical",
                        "--out",
                        str(strict_gate_path),
                    ]
                ),
                1,
            )
            strict_gate = json.loads(strict_gate_path.read_text(encoding="utf-8"))
            self.assertEqual(strict_gate["decision"]["readiness"], "blocked")
            self.assertEqual(strict_gate["decision"]["recommendation"], "block_iteration")
            self.assertEqual(strict_gate["decision"]["blocking_check_count"], strict_gate["failed_check_count"])
            failed_checks = {check["id"] for check in strict_gate["checks"] if not check["passed"]}
            self.assertIn("max_recurring_actions", failed_checks)
            self.assertIn("forbid_open_priority", failed_checks)
            self.assertEqual(run_cli(["validate", "--action-ledger-gate", str(strict_gate_path), "--strict"]), 0)
            self.assertEqual(run_cli(["schemas", "--check", str(strict_gate_path)]), 0)

            strict_gate["failed_check_count"] = 0
            strict_gate_path.write_text(json.dumps(strict_gate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            self.assertEqual(run_cli(["validate", "--action-ledger-gate", str(strict_gate_path)]), 1)

    def test_gate_action_ledger_rejects_wrong_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            not_ledger = root / "not_ledger.json"
            not_ledger.write_text(json.dumps({"schema_version": "hfr.not_a_ledger.v1", "metrics": {}, "entries": []}) + "\n", encoding="utf-8")

            with self.assertRaises(SystemExit) as raised:
                run_cli(
                    [
                        "gate-action-ledger",
                        "--action-ledger",
                        str(not_ledger),
                        "--max-open-actions",
                        "0",
                    ]
                )

            self.assertEqual(raised.exception.code, 2)

    def test_action_ledger_marks_resolved_and_new_actions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = root / "runs"
            runs.mkdir()
            gate_1 = root / "failed_gate_1.json"
            gate_2 = root / "failed_gate_2.json"
            bundle_1 = root / "bundle_1.json"
            bundle_2 = root / "bundle_2.json"
            ledger_path = root / "action_ledger.json"
            gate_1.write_text(json.dumps({"schema_version": "hfr.first_gate.v1", "passed": False}, sort_keys=True) + "\n", encoding="utf-8")
            gate_2.write_text(json.dumps({"schema_version": "hfr.second_gate.v1", "passed": False}, sort_keys=True) + "\n", encoding="utf-8")
            self.assertEqual(run_cli(["evidence-bundle", "--runs", str(runs), "--gate", str(gate_1), "--out", str(bundle_1)]), 1)
            self.assertEqual(run_cli(["evidence-bundle", "--runs", str(runs), "--gate", str(gate_2), "--out", str(bundle_2)]), 1)

            code = run_cli(["action-ledger", "--bundle", str(bundle_1), "--bundle", str(bundle_2), "--out", str(ledger_path)])

            self.assertEqual(code, 0)
            ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
            statuses = {entry["status"] for entry in ledger["entries"]}
            self.assertIn("resolved", statuses)
            self.assertIn("new", statuses)
            self.assertEqual(ledger["metrics"]["open_action_count"], ledger["metrics"]["new_action_count"])
            self.assertGreater(ledger["metrics"]["resolved_action_count"], 0)
            self.assertGreater(ledger["metrics"]["new_action_count"], 0)
            self.assertEqual(run_cli(["validate", "--action-ledger", str(ledger_path), "--strict"]), 0)
            self.assertEqual(run_cli(["schemas", "--check", str(ledger_path)]), 0)

            resolved_key = next(entry["routing_key"] for entry in ledger["entries"] if entry["status"] == "resolved")
            gate_path = root / "resolved_action_ledger_gate.json"
            self.assertEqual(
                run_cli(
                    [
                        "gate-action-ledger",
                        "--action-ledger",
                        str(ledger_path),
                        "--min-resolved-actions",
                        "1",
                        "--require-resolved-action",
                        resolved_key,
                        "--out",
                        str(gate_path),
                    ]
                ),
                0,
            )
            gate_result = json.loads(gate_path.read_text(encoding="utf-8"))
            self.assertTrue(gate_result["passed"])
            self.assertEqual(gate_result["decision"]["recommendation"], "promote_iteration")
            self.assertEqual(gate_result["metrics"]["resolved_action_count"], ledger["metrics"]["resolved_action_count"])
            self.assertEqual(run_cli(["validate", "--action-ledger-gate", str(gate_path), "--strict"]), 0)

    def test_validate_rejects_stale_action_ledger_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = root / "runs"
            runs.mkdir()
            gate = root / "failed_gate.json"
            bundle = root / "bundle.json"
            ledger_path = root / "action_ledger.json"
            summary_path = root / "validation.json"
            gate.write_text(json.dumps({"schema_version": "hfr.test_gate.v1", "passed": False}, sort_keys=True) + "\n", encoding="utf-8")
            run_cli(["evidence-bundle", "--runs", str(runs), "--gate", str(gate), "--out", str(bundle)])
            run_cli(["action-ledger", "--bundle", str(bundle), "--out", str(ledger_path)])
            ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
            ledger["metrics"]["unique_action_count"] = 0
            ledger["entries"][0]["routing_key"] = "stale"
            ledger_path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            code = run_cli(["validate", "--action-ledger", str(ledger_path), "--out", str(summary_path)])

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in summary["targets"] for error in target["errors"])
            self.assertIn("action_ledger.metrics.unique_action_count", errors)
            self.assertIn("routing_key expected", errors)

    def test_validate_rejects_action_ledger_missing_source_bundle_action(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = root / "runs"
            runs.mkdir()
            gate = root / "failed_gate.json"
            bundle = root / "bundle.json"
            ledger_path = root / "action_ledger.json"
            summary_path = root / "validation.json"
            gate.write_text(json.dumps({"schema_version": "hfr.test_gate.v1", "passed": False}, sort_keys=True) + "\n", encoding="utf-8")
            self.assertEqual(run_cli(["evidence-bundle", "--runs", str(runs), "--gate", str(gate), "--out", str(bundle)]), 1)
            self.assertEqual(run_cli(["action-ledger", "--bundle", str(bundle), "--out", str(ledger_path)]), 0)
            self.assertEqual(run_cli(["validate", "--action-ledger", str(ledger_path), "--strict"]), 0)
            ledger = json.loads(ledger_path.read_text(encoding="utf-8"))

            removed_entry = ledger["entries"].pop()
            removed_occurrences = removed_entry["occurrence_count"]
            ledger["unique_action_count"] = len(ledger["entries"])
            ledger["action_count"] -= removed_occurrences
            ledger["metrics"]["unique_action_count"] = len(ledger["entries"])
            ledger["metrics"]["action_count"] -= removed_occurrences
            if removed_entry["open"]:
                ledger["metrics"]["open_action_count"] -= 1
            if removed_entry["status"] == "new":
                ledger["metrics"]["new_action_count"] -= 1
            if removed_entry["status"] == "recurring":
                ledger["metrics"]["recurring_action_count"] -= 1
            if removed_entry["status"] == "resolved":
                ledger["metrics"]["resolved_action_count"] -= 1
            ledger["metrics"]["status_counts"] = _count_rows(entry["status"] for entry in ledger["entries"])
            ledger["metrics"]["priority_counts"] = _count_rows(entry["priority"] for entry in ledger["entries"])
            ledger["metrics"]["artifact_counts"] = _count_rows(entry["artifact"] for entry in ledger["entries"])
            ledger["bundles"][0]["action_count"] -= removed_occurrences
            ledger["metrics"]["bundle_action_counts"][0]["action_count"] = ledger["bundles"][0]["action_count"]
            ledger_path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            code = run_cli(["validate", "--action-ledger", str(ledger_path), "--strict", "--out", str(summary_path)])

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in summary["targets"] for error in target["errors"])
            self.assertIn("action_ledger.bundles[0].action_count expected", errors)
            self.assertIn("action_ledger.entries missing", errors)

    def test_validate_rejects_action_ledger_stale_source_bundle_digest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = root / "runs"
            runs.mkdir()
            gate = root / "failed_gate.json"
            bundle = root / "bundle.json"
            ledger_path = root / "action_ledger.json"
            summary_path = root / "validation.json"
            gate.write_text(json.dumps({"schema_version": "hfr.test_gate.v1", "passed": False}, sort_keys=True) + "\n", encoding="utf-8")
            self.assertEqual(run_cli(["evidence-bundle", "--runs", str(runs), "--gate", str(gate), "--out", str(bundle)]), 1)
            self.assertEqual(run_cli(["action-ledger", "--bundle", str(bundle), "--out", str(ledger_path)]), 0)
            self.assertEqual(run_cli(["validate", "--action-ledger", str(ledger_path), "--strict"]), 0)
            payload = json.loads(bundle.read_text(encoding="utf-8"))
            payload["notes"].append("tampered after ledger generation")
            bundle.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            code = run_cli(["validate", "--action-ledger", str(ledger_path), "--strict", "--out", str(summary_path)])

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in summary["targets"] for error in target["errors"])
            self.assertIn("action_ledger.bundles[0].sha256 does not match current file contents", errors)

    def test_validate_rejects_action_ledger_missing_source_bundle_even_if_exists_flag_is_false(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = root / "runs"
            runs.mkdir()
            gate = root / "failed_gate.json"
            bundle = root / "bundle.json"
            ledger_path = root / "action_ledger.json"
            summary_path = root / "validation.json"
            gate.write_text(json.dumps({"schema_version": "hfr.test_gate.v1", "passed": False}, sort_keys=True) + "\n", encoding="utf-8")
            self.assertEqual(run_cli(["evidence-bundle", "--runs", str(runs), "--gate", str(gate), "--out", str(bundle)]), 1)
            self.assertEqual(run_cli(["action-ledger", "--bundle", str(bundle), "--out", str(ledger_path)]), 0)
            ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
            ledger["bundles"][0]["exists"] = False
            bundle.unlink()
            ledger_path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            code = run_cli(["validate", "--action-ledger", str(ledger_path), "--strict", "--out", str(summary_path)])

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in summary["targets"] for error in target["errors"])
            self.assertIn("action_ledger.bundles[0].path must resolve to an existing evidence bundle", errors)

    def test_validate_rejects_action_ledger_rewritten_source_action_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = root / "runs"
            runs.mkdir()
            gate = root / "failed_gate.json"
            bundle = root / "bundle.json"
            ledger_path = root / "action_ledger.json"
            summary_path = root / "validation.json"
            gate.write_text(json.dumps({"schema_version": "hfr.test_gate.v1", "passed": False}, sort_keys=True) + "\n", encoding="utf-8")
            self.assertEqual(run_cli(["evidence-bundle", "--runs", str(runs), "--gate", str(gate), "--out", str(bundle)]), 1)
            self.assertEqual(run_cli(["action-ledger", "--bundle", str(bundle), "--out", str(ledger_path)]), 0)
            ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
            ledger["entries"][0]["summary"] = "Forged action summary"
            ledger_path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            code = run_cli(["validate", "--action-ledger", str(ledger_path), "--strict", "--out", str(summary_path)])

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in summary["targets"] for error in target["errors"])
            self.assertIn("action_ledger.entries missing", errors)

            self.assertEqual(run_cli(["action-ledger", "--bundle", str(bundle), "--out", str(ledger_path)]), 0)
            ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
            ledger["entries"][0]["evidence"]["forged"] = True
            ledger_path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            code = run_cli(["validate", "--action-ledger", str(ledger_path), "--strict", "--out", str(summary_path)])

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in summary["targets"] for error in target["errors"])
            self.assertIn("action_ledger.entries missing", errors)


if __name__ == "__main__":
    unittest.main()
