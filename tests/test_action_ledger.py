import json
import os
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


def _build_action_ledger(root):
    runs = root / "runs"
    runs.mkdir(exist_ok=True)
    gate = root / "failed_gate.json"
    bundle = root / "bundle.json"
    ledger_path = root / "action_ledger.json"
    gate.write_text(json.dumps({"schema_version": "hfr.test_gate.v1", "passed": False}, sort_keys=True) + "\n", encoding="utf-8")
    if run_cli(["evidence-bundle", "--runs", str(runs), "--gate", str(gate), "--out", str(bundle)]) != 1:
        raise AssertionError("evidence-bundle fixture did not produce expected failing bundle")
    if run_cli(["action-ledger", "--bundle", str(bundle), "--out", str(ledger_path)]) != 0:
        raise AssertionError("action-ledger fixture failed")
    return ledger_path


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

    def test_action_ledger_writes_output_relative_bundle_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = root / "src"
            output_dir = root / "out"
            runs = source_dir / "runs"
            source_dir.mkdir()
            output_dir.mkdir()
            runs.mkdir()
            gate = source_dir / "failed_gate.json"
            bundle = source_dir / "bundle.json"
            ledger_path = output_dir / "action_ledger.json"
            gate.write_text(json.dumps({"schema_version": "hfr.test_gate.v1", "passed": False}, sort_keys=True) + "\n", encoding="utf-8")
            self.assertEqual(run_cli(["evidence-bundle", "--runs", str(runs), "--gate", str(gate), "--out", str(bundle)]), 1)

            self.assertEqual(run_cli(["action-ledger", "--bundle", str(bundle), "--out", str(ledger_path)]), 0)

            ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
            self.assertEqual(ledger["bundles"][0]["path"], "../src/bundle.json")
            self.assertEqual(ledger["metrics"]["bundle_action_counts"][0]["path"], "../src/bundle.json")
            self.assertTrue(all(occurrence["bundle_path"] == "../src/bundle.json" for entry in ledger["entries"] for occurrence in entry["occurrences"]))
            self.assertEqual(run_cli(["validate", "--action-ledger", str(ledger_path), "--strict"]), 0)

    def test_strict_validate_rejects_absolute_action_ledger_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ledger_path = _build_action_ledger(root)
            summary_path = root / "validation.json"
            strict_summary_path = root / "strict_validation.json"
            absolute_bundle_path = str(root / "bundle.json")
            ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
            ledger["ledger_path"] = str(ledger_path)
            ledger["bundles"][0]["path"] = absolute_bundle_path
            ledger["metrics"]["bundle_action_counts"][0]["path"] = absolute_bundle_path
            for entry in ledger["entries"]:
                entry["first_seen_path"] = absolute_bundle_path
                entry["last_seen_path"] = absolute_bundle_path
                for occurrence in entry["occurrences"]:
                    occurrence["bundle_path"] = absolute_bundle_path
            ledger_path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            code = run_cli(["validate", "--action-ledger", str(ledger_path), "--out", str(summary_path)])
            strict_code = run_cli(["validate", "--action-ledger", str(ledger_path), "--strict", "--out", str(strict_summary_path)])

            self.assertEqual(code, 0)
            self.assertEqual(strict_code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            warnings = "\n".join(warning for target in summary["targets"] for warning in target["warnings"])
            for expected in (
                "action_ledger.ledger_path is absolute",
                "action_ledger.bundles[0].path is absolute",
                "action_ledger.metrics.bundle_action_counts[0].path is absolute",
                "action_ledger.entries[0].first_seen_path is absolute",
                "action_ledger.entries[0].last_seen_path is absolute",
                "action_ledger.entries[0].occurrences[0].bundle_path is absolute",
            ):
                self.assertIn(expected, warnings)
            strict_summary = json.loads(strict_summary_path.read_text(encoding="utf-8"))
            strict_warnings = "\n".join(warning for target in strict_summary["targets"] for warning in target["warnings"])
            self.assertIn("action_ledger.ledger_path is absolute", strict_warnings)

    def test_validate_rejects_action_ledger_cwd_relative_source_bundle_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = root / "src"
            output_dir = root / "out"
            runs = source_dir / "runs"
            source_dir.mkdir()
            output_dir.mkdir()
            runs.mkdir()
            gate = source_dir / "failed_gate.json"
            bundle = source_dir / "bundle.json"
            ledger_path = output_dir / "action_ledger.json"
            summary_path = root / "validation.json"
            gate.write_text(json.dumps({"schema_version": "hfr.test_gate.v1", "passed": False}, sort_keys=True) + "\n", encoding="utf-8")
            self.assertEqual(run_cli(["evidence-bundle", "--runs", str(runs), "--gate", str(gate), "--out", str(bundle)]), 1)
            self.assertEqual(run_cli(["action-ledger", "--bundle", str(bundle), "--out", str(ledger_path)]), 0)
            ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
            ledger["bundles"][0]["path"] = "bundle.json"
            ledger["metrics"]["bundle_action_counts"][0]["path"] = "bundle.json"
            for entry in ledger["entries"]:
                entry["first_seen_path"] = "bundle.json"
                entry["last_seen_path"] = "bundle.json"
                for occurrence in entry["occurrences"]:
                    occurrence["bundle_path"] = "bundle.json"
            ledger_path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            previous_cwd = Path.cwd()
            try:
                os.chdir(source_dir)
                code = run_cli(["validate", "--action-ledger", str(ledger_path), "--strict", "--out", str(summary_path)])
            finally:
                os.chdir(previous_cwd)

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in summary["targets"] for error in target["errors"])
            self.assertIn("action_ledger.bundles[0].path must resolve to an existing evidence bundle", errors)

    def test_validate_rejects_action_ledger_non_utf8_source_bundle_without_crashing(self):
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
            bundle.write_bytes(b"\xff\xfe\x00")

            code = run_cli(["validate", "--action-ledger", str(ledger_path), "--strict", "--out", str(summary_path)])

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in summary["targets"] for error in target["errors"])
            self.assertIn("action_ledger.bundles[0].sha256 does not match current file contents", errors)
            self.assertIn("action_ledger.bundles[0].path is not valid UTF-8", errors)

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

    def test_action_ledger_schema_rejects_missing_source_bundle_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ledger_path = _build_action_ledger(root)
            summary_path = root / "validation.json"
            ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
            ledger["bundles"][0].pop("size_bytes")
            ledger_path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            validate_code = run_cli(["validate", "--action-ledger", str(ledger_path), "--strict", "--out", str(summary_path)])
            schema_code = run_cli(["schemas", "--check", str(ledger_path)])

            self.assertEqual(validate_code, 1)
            self.assertEqual(schema_code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in summary["targets"] for error in target["errors"])
            self.assertIn("action_ledger.bundles[0].size_bytes must be a non-negative integer", errors)

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

    def test_validate_rejects_action_ledger_gate_stale_source_ledger_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = root / "runs"
            runs.mkdir()
            gate = root / "failed_gate.json"
            bundle = root / "bundle.json"
            ledger_path = root / "action_ledger.json"
            gate_path = root / "action_ledger_gate.json"
            summary_path = root / "validation.json"
            gate.write_text(json.dumps({"schema_version": "hfr.test_gate.v1", "passed": False}, sort_keys=True) + "\n", encoding="utf-8")
            self.assertEqual(run_cli(["evidence-bundle", "--runs", str(runs), "--gate", str(gate), "--out", str(bundle)]), 1)
            self.assertEqual(run_cli(["action-ledger", "--bundle", str(bundle), "--out", str(ledger_path)]), 0)
            self.assertEqual(run_cli(["gate-action-ledger", "--action-ledger", str(ledger_path), "--out", str(gate_path)]), 0)
            self.assertEqual(run_cli(["validate", "--action-ledger-gate", str(gate_path), "--strict"]), 0)
            payload = json.loads(gate_path.read_text(encoding="utf-8"))
            forged_metrics = {
                "bundle_count": 1,
                "new_action_count": 0,
                "open_action_count": 0,
                "open_priority_counts": [],
                "recurring_action_count": 0,
                "resolved_action_count": 0,
                "unique_action_count": 0,
            }
            payload["metrics"] = forged_metrics
            payload["decision"]["key_metrics"] = forged_metrics
            gate_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            code = run_cli(["validate", "--action-ledger-gate", str(gate_path), "--strict", "--out", str(summary_path)])

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in summary["targets"] for error in target["errors"])
            self.assertIn("action_ledger_gate.metrics must match replayed source ledger metrics", errors)

    def test_validate_rejects_action_ledger_gate_forged_check_actuals(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = root / "runs"
            runs.mkdir()
            gate = root / "failed_gate.json"
            bundle = root / "bundle.json"
            ledger_path = root / "action_ledger.json"
            gate_path = root / "action_ledger_gate.json"
            summary_path = root / "validation.json"
            gate.write_text(json.dumps({"schema_version": "hfr.test_gate.v1", "passed": False}, sort_keys=True) + "\n", encoding="utf-8")
            self.assertEqual(run_cli(["evidence-bundle", "--runs", str(runs), "--gate", str(gate), "--out", str(bundle)]), 1)
            self.assertEqual(run_cli(["action-ledger", "--bundle", str(bundle), "--out", str(ledger_path)]), 0)
            self.assertEqual(run_cli(["gate-action-ledger", "--action-ledger", str(ledger_path), "--max-open-actions", "0", "--out", str(gate_path)]), 1)
            payload = json.loads(gate_path.read_text(encoding="utf-8"))
            payload["checks"][0]["actual"] = 0
            payload["checks"][0]["passed"] = True
            payload["checks"][0]["summary"] = "max_open_actions: actual=0, max=0"
            payload["failed_check_count"] = 0
            payload["passed"] = True
            payload["decision"]["readiness"] = "ready"
            payload["decision"]["recommendation"] = "promote_iteration"
            payload["decision"]["summary"] = "Action-ledger gate is ready: improvement-loop pressure is within policy."
            payload["decision"]["blocking_check_count"] = 0
            payload["decision"]["blocking_checks"] = []
            gate_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            code = run_cli(["validate", "--action-ledger-gate", str(gate_path), "--strict", "--out", str(summary_path)])

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in summary["targets"] for error in target["errors"])
            self.assertIn("action_ledger_gate.checks must match replayed source ledger checks", errors)

    def test_validate_rejects_action_ledger_gate_forged_decision_details(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = root / "runs"
            runs.mkdir()
            gate = root / "failed_gate.json"
            bundle = root / "bundle.json"
            ledger_path = root / "action_ledger.json"
            gate_path = root / "action_ledger_gate.json"
            summary_path = root / "validation.json"
            gate.write_text(json.dumps({"schema_version": "hfr.test_gate.v1", "passed": False}, sort_keys=True) + "\n", encoding="utf-8")
            self.assertEqual(run_cli(["evidence-bundle", "--runs", str(runs), "--gate", str(gate), "--out", str(bundle)]), 1)
            self.assertEqual(run_cli(["action-ledger", "--bundle", str(bundle), "--out", str(ledger_path)]), 0)
            self.assertEqual(run_cli(["gate-action-ledger", "--action-ledger", str(ledger_path), "--max-open-actions", "0", "--out", str(gate_path)]), 1)
            payload = json.loads(gate_path.read_text(encoding="utf-8"))
            payload["decision"]["summary"] = "Action-ledger gate is blocked by 1 check(s); first failure: forged"
            payload["decision"]["blocking_checks"][0]["summary"] = "forged"
            gate_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            code = run_cli(["validate", "--action-ledger-gate", str(gate_path), "--strict", "--out", str(summary_path)])

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in summary["targets"] for error in target["errors"])
            self.assertIn("action_ledger_gate.decision must match replayed source ledger decision", errors)

    def test_validate_rejects_action_ledger_gate_missing_source_ledger(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = root / "runs"
            runs.mkdir()
            gate = root / "failed_gate.json"
            bundle = root / "bundle.json"
            ledger_path = root / "action_ledger.json"
            gate_path = root / "action_ledger_gate.json"
            summary_path = root / "validation.json"
            gate.write_text(json.dumps({"schema_version": "hfr.test_gate.v1", "passed": False}, sort_keys=True) + "\n", encoding="utf-8")
            self.assertEqual(run_cli(["evidence-bundle", "--runs", str(runs), "--gate", str(gate), "--out", str(bundle)]), 1)
            self.assertEqual(run_cli(["action-ledger", "--bundle", str(bundle), "--out", str(ledger_path)]), 0)
            self.assertEqual(run_cli(["gate-action-ledger", "--action-ledger", str(ledger_path), "--out", str(gate_path)]), 0)
            ledger_path.unlink()

            code = run_cli(["validate", "--action-ledger-gate", str(gate_path), "--strict", "--out", str(summary_path)])

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in summary["targets"] for error in target["errors"])
            self.assertIn("action_ledger_gate.action_ledger must resolve to an existing action ledger", errors)

    def test_gate_action_ledger_writes_output_relative_source_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = root / "runs"
            runs.mkdir()
            summary_path = runs / "validation.json"

            previous_cwd = Path.cwd()
            try:
                os.chdir(root)
                ledger_path = _build_action_ledger(runs)
                gate_path = runs / "action_ledger_gate.json"
                self.assertEqual(
                    run_cli(["gate-action-ledger", "--action-ledger", str(ledger_path.relative_to(root)), "--out", str(gate_path)]),
                    0,
                )
                gate = json.loads(gate_path.read_text(encoding="utf-8"))
                self.assertEqual(gate["action_ledger"], "action_ledger.json")
                code = run_cli(["validate", "--action-ledger-gate", str(gate_path), "--strict", "--out", str(summary_path)])
            finally:
                os.chdir(previous_cwd)

            self.assertEqual(code, 0)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in summary["targets"] for error in target["errors"])
            self.assertEqual(errors, "")

    def test_validate_rejects_action_ledger_gate_cwd_relative_source_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cwd_root = root / "cwd"
            outside_root = root / "outside"
            nested = cwd_root / "nested"
            nested.mkdir(parents=True)
            outside_root.mkdir()
            summary_path = root / "validation.json"

            previous_cwd = Path.cwd()
            try:
                os.chdir(cwd_root)
                ledger_path = _build_action_ledger(nested)
                gate_path = cwd_root / "action_ledger_gate.json"
                self.assertEqual(
                    run_cli(["gate-action-ledger", "--action-ledger", str(ledger_path.relative_to(cwd_root)), "--out", str(gate_path)]),
                    0,
                )
                outside_gate = outside_root / "action_ledger_gate.json"
                outside_gate.write_text(gate_path.read_text(encoding="utf-8"), encoding="utf-8")
                code = run_cli(["validate", "--action-ledger-gate", str(outside_gate), "--strict", "--out", str(summary_path)])
            finally:
                os.chdir(previous_cwd)

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in summary["targets"] for error in target["errors"])
            self.assertIn("action_ledger_gate.action_ledger must resolve to an existing action ledger", errors)

    def test_validate_rejects_action_ledger_gate_invalid_source_ledger_without_crashing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = root / "runs"
            runs.mkdir()
            gate = root / "failed_gate.json"
            bundle = root / "bundle.json"
            ledger_path = root / "action_ledger.json"
            gate_path = root / "action_ledger_gate.json"
            summary_path = root / "validation.json"
            gate.write_text(json.dumps({"schema_version": "hfr.test_gate.v1", "passed": False}, sort_keys=True) + "\n", encoding="utf-8")
            self.assertEqual(run_cli(["evidence-bundle", "--runs", str(runs), "--gate", str(gate), "--out", str(bundle)]), 1)
            self.assertEqual(run_cli(["action-ledger", "--bundle", str(bundle), "--out", str(ledger_path)]), 0)
            self.assertEqual(run_cli(["gate-action-ledger", "--action-ledger", str(ledger_path), "--out", str(gate_path)]), 0)
            ledger_path.write_text(json.dumps({"schema_version": "hfr.not_action_ledger.v1", "metrics": {}, "entries": []}) + "\n", encoding="utf-8")

            code = run_cli(["validate", "--action-ledger-gate", str(gate_path), "--strict", "--out", str(summary_path)])

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in summary["targets"] for error in target["errors"])
            self.assertIn("action_ledger_gate.action_ledger could not be replayed", errors)

    def test_validate_rejects_action_ledger_gate_non_utf8_source_ledger_without_crashing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ledger_path = _build_action_ledger(root)
            gate_path = root / "action_ledger_gate.json"
            summary_path = root / "validation.json"
            self.assertEqual(run_cli(["gate-action-ledger", "--action-ledger", str(ledger_path), "--out", str(gate_path)]), 0)
            ledger_path.write_bytes(b"\xff\xfe\xff")

            code = run_cli(["validate", "--action-ledger-gate", str(gate_path), "--strict", "--out", str(summary_path)])

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in summary["targets"] for error in target["errors"])
            self.assertIn("action_ledger_gate.action_ledger is not valid UTF-8", errors)

    def test_validate_rejects_action_ledger_gate_policy_check_omission(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = root / "runs"
            runs.mkdir()
            source_gate = root / "failed_gate.json"
            bundle_1 = root / "bundle_1.json"
            bundle_2 = root / "bundle_2.json"
            ledger_path = root / "action_ledger.json"
            gate_path = root / "action_ledger_gate.json"
            summary_path = root / "validation.json"
            source_gate.write_text(json.dumps({"schema_version": "hfr.test_gate.v1", "passed": False}, sort_keys=True) + "\n", encoding="utf-8")
            self.assertEqual(run_cli(["evidence-bundle", "--runs", str(runs), "--gate", str(source_gate), "--out", str(bundle_1)]), 1)
            shutil.copyfile(bundle_1, bundle_2)
            self.assertEqual(run_cli(["action-ledger", "--bundle", str(bundle_1), "--bundle", str(bundle_2), "--out", str(ledger_path)]), 0)
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
            payload = json.loads(gate_path.read_text(encoding="utf-8"))
            payload["checks"].pop()
            payload["check_count"] = len(payload["checks"])
            gate_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            code = run_cli(["validate", "--action-ledger-gate", str(gate_path), "--strict", "--out", str(summary_path)])

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in summary["targets"] for error in target["errors"])
            self.assertIn("action_ledger_gate.checks must cover action_ledger_gate.policy.effective requirements", errors)


if __name__ == "__main__":
    unittest.main()
