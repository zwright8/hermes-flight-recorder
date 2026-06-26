import json
import shutil
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from flightrecorder.cli import main


def run_cli(args):
    with redirect_stdout(StringIO()):
        return main(args)


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


if __name__ == "__main__":
    unittest.main()
