import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from flightrecorder.cli import main


ROOT = Path(__file__).resolve().parents[1]


def run_cli(args):
    with redirect_stdout(StringIO()):
        return main(args)


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_completed_labels(review_dir: Path, labels_path: Path, *, reject_good: bool = False) -> None:
    rows = read_jsonl(review_dir / "label_template.jsonl")
    for row in rows:
        row["human_label"] = row["suggested_human_label"]
        if reject_good and row.get("scenario_id") == "prompt_injection_good":
            row["human_label"] = "reject"
            row["notes"] = "Reviewer rejected a deterministic pass for calibration coverage."
        else:
            row["notes"] = "Accepted suggested label for calibration coverage."
        row["reviewer"] = "calibration-test"
        row["reviewer_confidence"] = "high"
        row["reviewed_at"] = "2026-06-26T00:00:00Z"
    labels_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def make_reviewed_export(tmp: str, *, reject_good: bool = False) -> Path:
    root = Path(tmp)
    runs = root / "runs"
    review = root / "review"
    labels = root / "completed_labels.jsonl"
    reviewed = root / "reviewed"
    run_cli(["run", "--scenario", str(ROOT / "scenarios" / "prompt_injection_good.json"), "--out", str(runs / "prompt_injection_good")])
    run_cli(["run", "--scenario", str(ROOT / "scenarios" / "prompt_injection_bad.json"), "--out", str(runs / "prompt_injection_bad")])
    run_cli(["export-review", "--runs", str(runs), "--out", str(review)])
    write_completed_labels(review, labels, reject_good=reject_good)
    run_cli(["apply-review", "--review-export", str(review), "--labels", str(labels), "--out", str(reviewed)])
    return reviewed


class ReviewCalibrationTests(unittest.TestCase):
    def test_review_calibration_accepts_aligned_reviewed_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            reviewed = make_reviewed_export(tmp)
            out = Path(tmp) / "review_calibration.json"

            code = run_cli(
                [
                    "review-calibration",
                    "--reviewed-export",
                    str(reviewed),
                    "--out",
                    str(out),
                    "--min-comparable-labels",
                    "2",
                    "--min-agreement-rate",
                    "1.0",
                    "--max-disagreements",
                    "0",
                    "--max-false-positives",
                    "0",
                    "--max-false-negatives",
                    "0",
                ]
            )

            self.assertEqual(code, 0)
            calibration = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(calibration["schema_version"], "hfr.review_calibration.v1")
            self.assertTrue(calibration["passed"])
            self.assertEqual(calibration["failed_check_count"], 0)
            self.assertEqual(calibration["metrics"]["reviewed_label_count"], 2)
            self.assertEqual(calibration["metrics"]["comparable_label_count"], 2)
            self.assertEqual(calibration["metrics"]["agreement_count"], 2)
            self.assertEqual(calibration["metrics"]["agreement_rate"], 1.0)
            self.assertEqual(calibration["metrics"]["disagreement_count"], 0)
            self.assertEqual(calibration["disagreements"], [])
            self.assertEqual(run_cli(["validate", "--review-calibration", str(out), "--strict"]), 0)
            self.assertEqual(run_cli(["schemas", "--check", str(out)]), 0)

    def test_review_calibration_blocks_scorecard_human_disagreement(self):
        with tempfile.TemporaryDirectory() as tmp:
            reviewed = make_reviewed_export(tmp, reject_good=True)
            out = Path(tmp) / "review_calibration.json"

            code = run_cli(
                [
                    "review-calibration",
                    "--reviewed-export",
                    str(reviewed),
                    "--out",
                    str(out),
                    "--min-agreement-rate",
                    "1.0",
                    "--max-disagreements",
                    "0",
                    "--max-false-positives",
                    "0",
                ]
            )

            self.assertEqual(code, 1)
            calibration = json.loads(out.read_text(encoding="utf-8"))
            self.assertFalse(calibration["passed"])
            self.assertEqual(calibration["metrics"]["agreement_rate"], 0.5)
            self.assertEqual(calibration["metrics"]["disagreement_count"], 1)
            self.assertEqual(calibration["metrics"]["false_positive_count"], 1)
            self.assertEqual(calibration["metrics"]["false_negative_count"], 0)
            failed_checks = {check["id"] for check in calibration["checks"] if not check["passed"]}
            self.assertIn("min_agreement_rate", failed_checks)
            self.assertIn("max_disagreements", failed_checks)
            self.assertIn("max_false_positives", failed_checks)
            disagreement = calibration["disagreements"][0]
            self.assertEqual(disagreement["scenario_id"], "prompt_injection_good")
            self.assertEqual(disagreement["disagreement_type"], "scorecard_passed_human_rejected")
            self.assertEqual(run_cli(["validate", "--review-calibration", str(out), "--strict"]), 0)
            self.assertEqual(run_cli(["schemas", "--check", str(out)]), 0)

    def test_review_calibration_fails_invalid_reviewed_export_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            reviewed = make_reviewed_export(tmp)
            labels_path = reviewed / "reviewed_labels.jsonl"
            rows = read_jsonl(labels_path)
            rows[0]["notes"] = "tampered before calibration"
            labels_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
            out = Path(tmp) / "review_calibration.json"

            code = run_cli(["review-calibration", "--reviewed-export", str(reviewed), "--out", str(out)])

            self.assertEqual(code, 1)
            calibration = json.loads(out.read_text(encoding="utf-8"))
            failed_checks = {check["id"] for check in calibration["checks"] if not check["passed"]}
            self.assertIn("valid_reviewed_export", failed_checks)
            self.assertFalse(calibration["metrics"]["validation"]["passed"])
            self.assertGreater(calibration["metrics"]["validation"]["error_count"], 0)
            self.assertEqual(run_cli(["validate", "--review-calibration", str(out), "--strict"]), 0)
            self.assertEqual(run_cli(["schemas", "--check", str(out)]), 0)


if __name__ == "__main__":
    unittest.main()
