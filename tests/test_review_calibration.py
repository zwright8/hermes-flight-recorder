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

            linked = Path(tmp) / "reviewed_link"
            try:
                linked.symlink_to(reviewed, target_is_directory=True)
            except (NotImplementedError, OSError) as exc:
                self.skipTest(f"symlink unavailable: {exc}")
            stderr = StringIO()
            with self.assertRaises(SystemExit) as raised, redirect_stdout(StringIO()), redirect_stderr(stderr):
                main(["review-calibration", "--reviewed-export", str(linked), "--out", str(Path(tmp) / "linked_calibration.json")])
            self.assertEqual(raised.exception.code, 2)
            self.assertIn("reviewed export must resolve to a regular non-symlink directory", stderr.getvalue())

            redirected = Path(tmp) / "redirected_calibration.json"
            redirected.write_text("{}\n", encoding="utf-8")
            output_link = Path(tmp) / "calibration_output_link.json"
            output_link.symlink_to(redirected)
            stderr = StringIO()
            with self.assertRaises(SystemExit) as raised, redirect_stdout(StringIO()), redirect_stderr(stderr):
                main(["review-calibration", "--reviewed-export", str(reviewed), "--out", str(output_link)])
            self.assertEqual(raised.exception.code, 2)
            self.assertIn("review calibration output must resolve to a regular non-symlink file", stderr.getvalue())

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

    def test_validate_rejects_review_calibration_with_unknown_control_plane_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            reviewed = make_reviewed_export(tmp, reject_good=True)
            out = Path(tmp) / "review_calibration.json"
            validation = Path(tmp) / "validation.json"
            run_cli(
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
            calibration = json.loads(out.read_text(encoding="utf-8"))
            calibration["provider_console_url"] = "https://example.invalid/calibration"
            calibration["source"]["reviewed_labels_signed_url"] = "https://example.invalid/labels"
            calibration["checks"][0]["provider_call"] = {"url": "https://example.invalid/check"}
            calibration["checks"][0]["actual"]["provider_call"] = {"url": "https://example.invalid/check-actual"}
            calibration["checks"][0]["expected"]["provider_call"] = {"url": "https://example.invalid/check-expected"}
            calibration["checks"][1]["scope"] = {"provider_call": "https://example.invalid/check-scope"}
            calibration["metrics"]["calibration_job_url"] = "https://example.invalid/job"
            calibration["metrics"]["validation"]["provider_call"] = {"url": "https://example.invalid/validation"}
            calibration["metrics"]["label_counts"][0]["provider_call"] = {"url": "https://example.invalid/label-count"}
            calibration["metrics"]["mean_score_by_human_label"][0]["provider_call"] = {
                "url": "https://example.invalid/mean-score"
            }
            calibration["disagreements"][0]["review_thread_ref"] = "redacted-thread"
            out.write_text(json.dumps(calibration, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            self.assertNotEqual(run_cli(["schemas", "--check", str(out)]), 0)
            code = run_cli(["validate", "--review-calibration", str(out), "--strict", "--out", str(validation)])

            self.assertEqual(code, 1)
            errors = "\n".join(error for target in json.loads(validation.read_text(encoding="utf-8"))["targets"] for error in target["errors"])
            self.assertIn("review_calibration contains unknown field(s): ['provider_console_url']", errors)
            self.assertIn(
                "review_calibration.source contains unknown field(s): ['reviewed_labels_signed_url']",
                errors,
            )
            self.assertIn("review_calibration.checks[0] contains unknown field(s): ['provider_call']", errors)
            self.assertIn("review_calibration.checks[0].actual contains unknown field(s): ['provider_call']", errors)
            self.assertIn("review_calibration.checks[0].expected contains unknown field(s): ['provider_call']", errors)
            self.assertIn("review_calibration.checks[1] contains unknown field(s): ['scope']", errors)
            self.assertIn("review_calibration.metrics contains unknown field(s): ['calibration_job_url']", errors)
            self.assertIn(
                "review_calibration.metrics.validation contains unknown field(s): ['provider_call']",
                errors,
            )
            self.assertIn(
                "review_calibration.metrics.label_counts[0] contains unknown field(s): ['provider_call']",
                errors,
            )
            self.assertIn(
                "review_calibration.metrics.mean_score_by_human_label[0] contains unknown field(s): ['provider_call']",
                errors,
            )
            self.assertIn(
                "review_calibration.disagreements[0] contains unknown field(s): ['review_thread_ref']",
                errors,
            )

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
