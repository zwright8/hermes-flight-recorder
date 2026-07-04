import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

from flightrecorder.cli import main


ROOT = Path(__file__).resolve().parents[1]
CONFIDENCE_MANIFEST_FIELDS = (
    "confidence_counts",
    "high_confidence_label_count",
    "medium_or_high_confidence_label_count",
    "low_confidence_label_count",
    "unknown_confidence_label_count",
)


def run_cli(args):
    with redirect_stdout(StringIO()):
        return main(args)


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_completed_labels(review_dir: Path, labels_path: Path) -> None:
    rows = read_jsonl(review_dir / "label_template.jsonl")
    for row in rows:
        row["human_label"] = row["suggested_human_label"]
        row["reviewer"] = "test-reviewer"
        row["reviewer_confidence"] = "high"
        row["reviewed_at"] = "2026-06-26T00:00:00Z"
        row["notes"] = "Accepted suggested label for reviewed-gate fixture coverage."
    labels_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def make_reviewed_export(tmp: str):
    root = Path(tmp)
    runs = root / "runs"
    review = root / "review"
    labels = root / "completed_labels.jsonl"
    reviewed = root / "reviewed"
    run_cli(["run-suite", "--scenarios", str(ROOT / "scenarios"), "--out", str(runs)])
    run_cli(["export-review", "--runs", str(runs), "--out", str(review)])
    write_completed_labels(review, labels)
    run_cli(["apply-review", "--review-export", str(review), "--labels", str(labels), "--out", str(reviewed)])
    return reviewed


class ReviewedGateTests(unittest.TestCase):
    def test_committed_agentic_training_reviewed_gate_is_schema_checkable(self):
        gate_path = ROOT / "examples" / "agentic_training" / "model_grader" / "reviewed_gate.json"
        result = json.loads(gate_path.read_text(encoding="utf-8"))

        self.assertEqual(result["schema_version"], "hfr.reviewed_gate.v1")
        self.assertTrue(result["passed"])
        self.assertEqual(result["failed_check_count"], 0)
        self.assertEqual(result["decision"]["readiness"], "ready")
        self.assertEqual(result["metrics"]["reviewed_label_count"], 2)
        self.assertEqual(result["metrics"]["accepted_count"], 1)
        self.assertEqual(result["metrics"]["rejected_count"], 1)
        self.assertEqual(result["metrics"]["task_families"], ["prompt_injection"])
        self.assertEqual(run_cli(["schemas", "--check", str(gate_path)]), 0)

    def test_gate_reviewed_accepts_demo_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            reviewed = make_reviewed_export(tmp)
            gate = Path(tmp) / "reviewed_gate.json"

            code = run_cli(
                [
                    "gate-reviewed",
                    "--reviewed-export",
                    str(reviewed),
                    "--policy",
                    str(ROOT / "examples" / "reviewed_gate_policy.demo.json"),
                    "--out",
                    str(gate),
                ]
            )

            self.assertEqual(code, 0)
            result = json.loads(gate.read_text(encoding="utf-8"))
            self.assertEqual(result["schema_version"], "hfr.reviewed_gate.v1")
            self.assertTrue(result["passed"])
            self.assertEqual(result["failed_check_count"], 0)
            self.assertEqual(result["decision"]["readiness"], "ready")
            self.assertEqual(result["decision"]["recommendation"], "promote_iteration")
            self.assertGreaterEqual(result["decision"]["key_metrics"]["accepted_count"], 2)
            self.assertEqual(result["policy"]["schema_version"], "hfr.reviewed_gate.policy.v1")
            self.assertGreaterEqual(result["metrics"]["accepted_count"], 2)
            self.assertGreaterEqual(result["metrics"]["rejected_count"], 2)
            self.assertEqual(result["metrics"]["low_confidence_label_count"], 0)
            self.assertEqual(result["metrics"]["unknown_confidence_label_count"], 0)
            self.assertEqual(run_cli(["schemas", "--check", str(gate)]), 0)

    def test_gate_reviewed_fails_thresholds_and_forbidden_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            reviewed = make_reviewed_export(tmp)
            gate = Path(tmp) / "reviewed_gate.json"

            code = run_cli(
                [
                    "gate-reviewed",
                    "--reviewed-export",
                    str(reviewed),
                    "--min-reviewed-labels",
                    "99",
                    "--forbid-label",
                    "reject",
                    "--out",
                    str(gate),
                ]
            )

            self.assertEqual(code, 1)
            result = json.loads(gate.read_text(encoding="utf-8"))
            self.assertEqual(result["decision"]["readiness"], "blocked")
            self.assertEqual(result["decision"]["blocking_check_count"], 2)
            failed_checks = {item["id"] for item in result["checks"] if not item["passed"]}
            self.assertIn("min_reviewed_labels", failed_checks)
            self.assertIn("forbid_label", failed_checks)
            self.assertEqual(run_cli(["schemas", "--check", str(gate)]), 0)

    def test_gate_reviewed_fails_confidence_thresholds(self):
        with tempfile.TemporaryDirectory() as tmp:
            reviewed = make_reviewed_export(tmp)
            gate = Path(tmp) / "reviewed_gate.json"
            manifest_path = reviewed / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["confidence_counts"] = {"high": 1, "medium": 2, "low": 1, "unknown": 2}
            manifest["high_confidence_label_count"] = 1
            manifest["medium_or_high_confidence_label_count"] = 3
            manifest["low_confidence_label_count"] = 1
            manifest["unknown_confidence_label_count"] = 2
            manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")

            code = run_cli(
                [
                    "gate-reviewed",
                    "--reviewed-export",
                    str(reviewed),
                    "--skip-validation",
                    "--min-high-confidence-labels",
                    "2",
                    "--min-medium-or-high-confidence-labels",
                    "6",
                    "--max-low-confidence-labels",
                    "0",
                    "--max-unknown-confidence-labels",
                    "0",
                    "--out",
                    str(gate),
                ]
            )

            self.assertEqual(code, 1)
            result = json.loads(gate.read_text(encoding="utf-8"))
            failed_checks = {item["id"] for item in result["checks"] if not item["passed"]}
            self.assertIn("min_high_confidence_labels", failed_checks)
            self.assertIn("min_medium_or_high_confidence_labels", failed_checks)
            self.assertIn("max_low_confidence_labels", failed_checks)
            self.assertIn("max_unknown_confidence_labels", failed_checks)

    def test_gate_reviewed_treats_missing_confidence_counts_as_unknown(self):
        with tempfile.TemporaryDirectory() as tmp:
            reviewed = make_reviewed_export(tmp)
            gate = Path(tmp) / "reviewed_gate.json"
            manifest_path = reviewed / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for field_name in CONFIDENCE_MANIFEST_FIELDS:
                manifest.pop(field_name, None)
            manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")

            code = run_cli(
                [
                    "gate-reviewed",
                    "--reviewed-export",
                    str(reviewed),
                    "--skip-validation",
                    "--max-unknown-confidence-labels",
                    "0",
                    "--out",
                    str(gate),
                ]
            )

            self.assertEqual(code, 1)
            result = json.loads(gate.read_text(encoding="utf-8"))
            self.assertEqual(result["metrics"]["unknown_confidence_label_count"], result["metrics"]["reviewed_label_count"])
            failed_checks = {item["id"] for item in result["checks"] if not item["passed"]}
            self.assertIn("max_unknown_confidence_labels", failed_checks)

    def test_gate_reviewed_rejects_missing_confidence_counts_with_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            reviewed = make_reviewed_export(tmp)
            gate = Path(tmp) / "reviewed_gate.json"
            manifest_path = reviewed / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for field_name in CONFIDENCE_MANIFEST_FIELDS:
                manifest.pop(field_name, None)
            manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")

            code = run_cli(
                [
                    "gate-reviewed",
                    "--reviewed-export",
                    str(reviewed),
                    "--max-unknown-confidence-labels",
                    "0",
                    "--out",
                    str(gate),
                ]
            )

            self.assertEqual(code, 1)
            result = json.loads(gate.read_text(encoding="utf-8"))
            failed_checks = {item["id"] for item in result["checks"] if not item["passed"]}
            self.assertIn("valid_reviewed_export", failed_checks)
            self.assertIn("max_unknown_confidence_labels", failed_checks)
            self.assertGreater(result["metrics"]["validation"]["error_count"], 0)

    def test_gate_reviewed_fails_invalid_reviewed_export_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            reviewed = make_reviewed_export(tmp)
            gate = Path(tmp) / "reviewed_gate.json"
            labels_path = reviewed / "reviewed_labels.jsonl"
            rows = read_jsonl(labels_path)
            rows[0]["notes"] = "tampered after reviewed export"
            labels_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")

            code = run_cli(["gate-reviewed", "--reviewed-export", str(reviewed), "--out", str(gate)])

            self.assertEqual(code, 1)
            result = json.loads(gate.read_text(encoding="utf-8"))
            failed_checks = {item["id"] for item in result["checks"] if not item["passed"]}
            self.assertIn("valid_reviewed_export", failed_checks)
            self.assertFalse(result["metrics"]["validation"]["passed"])
            self.assertGreater(result["metrics"]["validation"]["error_count"], 0)

    def test_gate_reviewed_can_explicitly_skip_validation_for_legacy_handoffs(self):
        with tempfile.TemporaryDirectory() as tmp:
            reviewed = make_reviewed_export(tmp)
            gate = Path(tmp) / "reviewed_gate.json"
            labels_path = reviewed / "reviewed_labels.jsonl"
            rows = read_jsonl(labels_path)
            rows[0]["notes"] = "tampered legacy reviewed export"
            labels_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")

            code = run_cli(["gate-reviewed", "--reviewed-export", str(reviewed), "--skip-validation", "--out", str(gate)])

            self.assertEqual(code, 0)
            result = json.loads(gate.read_text(encoding="utf-8"))
            self.assertEqual(result["checks"], [])
            self.assertFalse(result["metrics"]["validation"]["available"])

    def test_gate_reviewed_rejects_unknown_policy_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            reviewed = make_reviewed_export(tmp)
            policy = Path(tmp) / "bad_policy.json"
            policy.write_text(
                json.dumps(
                    {
                        "schema_version": "hfr.reviewed_gate.policy.v1",
                        "forbid_labels": ["maybe"],
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaises(SystemExit) as raised, redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                main(["gate-reviewed", "--reviewed-export", str(reviewed), "--policy", str(policy)])

            self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
