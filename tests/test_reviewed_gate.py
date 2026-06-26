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


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_completed_labels(review_dir: Path, labels_path: Path) -> None:
    rows = read_jsonl(review_dir / "label_template.jsonl")
    for row in rows:
        row["human_label"] = row["suggested_human_label"]
        row["reviewer"] = "test-reviewer"
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
            self.assertEqual(result["policy"]["schema_version"], "hfr.reviewed_gate.policy.v1")
            self.assertGreaterEqual(result["metrics"]["accepted_count"], 2)
            self.assertGreaterEqual(result["metrics"]["rejected_count"], 2)

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
            failed_checks = {item["id"] for item in result["checks"] if not item["passed"]}
            self.assertIn("min_reviewed_labels", failed_checks)
            self.assertIn("forbid_label", failed_checks)

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
