import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from flightrecorder.cli import main
from flightrecorder.schema_registry import check_schema_file, list_schema_records
from flightrecorder.validation import validate_artifacts


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
        row["notes"] = "Accepted suggested label for model-grader fixture coverage."
        row["reviewer"] = "model-grader-test"
        row["reviewer_confidence"] = "high"
        row["reviewed_at"] = "2026-07-03T00:00:00Z"
    labels_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def make_review_flow(tmp: str) -> tuple[Path, Path]:
    root = Path(tmp)
    runs = root / "runs"
    review = root / "review"
    labels = root / "completed_labels.jsonl"
    reviewed = root / "reviewed"
    calibration = root / "review_calibration.json"
    run_cli(["run", "--scenario", str(ROOT / "scenarios" / "prompt_injection_good.json"), "--out", str(runs / "prompt_injection_good")])
    run_cli(["run", "--scenario", str(ROOT / "scenarios" / "prompt_injection_bad.json"), "--out", str(runs / "prompt_injection_bad")])
    run_cli(["export-review", "--runs", str(runs), "--out", str(review)])
    write_completed_labels(review, labels)
    run_cli(["apply-review", "--review-export", str(review), "--labels", str(labels), "--out", str(reviewed)])
    run_cli(
        [
            "review-calibration",
            "--reviewed-export",
            str(reviewed),
            "--out",
            str(calibration),
            "--min-comparable-labels",
            "2",
            "--min-agreement-rate",
            "1.0",
            "--max-disagreements",
            "0",
        ]
    )
    return review, calibration


class ModelGraderTests(unittest.TestCase):
    def test_cli_emits_fail_closed_model_grader_flow(self):
        with tempfile.TemporaryDirectory() as tmp:
            review, calibration = make_review_flow(tmp)
            root = Path(tmp)
            rubric = root / "rubric.json"
            dry_run = root / "dry_run.json"
            blocked_gate = root / "blocked_gate.json"
            passing_gate = root / "passing_gate.json"

            self.assertEqual(
                run_cli(
                    [
                        "model-grader",
                        "rubric",
                        "--review-export",
                        str(review),
                        "--rubric-id",
                        "prompt-injection-rubric",
                        "--criterion",
                        "Ground labels in scorecard rules and observable trace evidence.",
                        "--created-at",
                        "2026-07-03T00:00:00+00:00",
                        "--out",
                        str(rubric),
                    ]
                ),
                0,
            )
            rubric_payload = json.loads(rubric.read_text(encoding="utf-8"))
            self.assertEqual(rubric_payload["schema_version"], "hfr.rubric_spec.v1")
            self.assertEqual(rubric_payload["review_item_count"], 2)
            self.assertFalse(rubric_payload["execution_boundary"]["provider_api_called"])
            self.assert_schema_and_validate(rubric, "rubric_spec")

            self.assertEqual(
                run_cli(
                    [
                        "model-grader",
                        "dry-run",
                        "--review-export",
                        str(review),
                        "--rubric",
                        str(rubric),
                        "--grader-id",
                        "mock-grader-v1",
                        "--provider",
                        "mock",
                        "--created-at",
                        "2026-07-03T00:00:00+00:00",
                        "--out",
                        str(dry_run),
                    ]
                ),
                0,
            )
            dry_payload = json.loads(dry_run.read_text(encoding="utf-8"))
            self.assertEqual(dry_payload["schema_version"], "hfr.model_grader_dry_run.v1")
            self.assertEqual(dry_payload["graded_item_count"], 2)
            self.assertFalse(dry_payload["grader"]["provider_api_called"])
            self.assertFalse(dry_payload["grader"]["paid_model_grader_calls_started"])
            self.assertFalse(dry_payload["training_admission"]["labels_allowed_for_training"])
            self.assertEqual(dry_payload["training_admission"]["labels_admitted_count"], 0)
            self.assertTrue(all(len(row["label_sha256"]) == 64 for row in dry_payload["grader_labels"]))
            self.assert_schema_and_validate(dry_run, "model_grader_dry_run")

            self.assertEqual(
                run_cli(
                    [
                        "model-grader",
                        "gate",
                        "--dry-run",
                        str(dry_run),
                        "--rubric",
                        str(rubric),
                        "--created-at",
                        "2026-07-03T00:00:00+00:00",
                        "--out",
                        str(blocked_gate),
                    ]
                ),
                1,
            )
            blocked_payload = json.loads(blocked_gate.read_text(encoding="utf-8"))
            self.assertFalse(blocked_payload["passed"])
            self.assertEqual(blocked_payload["readiness"], "blocked")
            self.assertFalse(blocked_payload["admission"]["labels_allowed_for_training"])
            self.assertEqual(blocked_payload["admission"]["labels_admitted_count"], 0)
            self.assertIn("review_calibration_present", {check["id"] for check in blocked_payload["checks"] if not check["passed"]})
            self.assert_schema_and_validate(blocked_gate, "model_grader_gate")

            self.assertEqual(
                run_cli(
                    [
                        "model-grader",
                        "gate",
                        "--dry-run",
                        str(dry_run),
                        "--rubric",
                        str(rubric),
                        "--review-calibration",
                        str(calibration),
                        "--min-calibration-agreement-rate",
                        "1.0",
                        "--max-disagreements",
                        "0",
                        "--created-at",
                        "2026-07-03T00:00:00+00:00",
                        "--out",
                        str(passing_gate),
                    ]
                ),
                0,
            )
            passing_payload = json.loads(passing_gate.read_text(encoding="utf-8"))
            self.assertTrue(passing_payload["passed"])
            self.assertTrue(passing_payload["admission"]["labels_allowed_for_training"])
            self.assertEqual(passing_payload["admission"]["labels_admitted_count"], 2)
            self.assertEqual(passing_payload["admission"]["uncalibrated_labels_admitted"], 0)
            self.assertFalse(passing_payload["execution_boundary"]["provider_api_called"])
            self.assert_schema_and_validate(passing_gate, "model_grader_gate")

    def test_schema_names_are_registered(self):
        names = {record["name"] for record in list_schema_records()}
        self.assertIn("rubric_spec", names)
        self.assertIn("model_grader_dry_run", names)
        self.assertIn("model_grader_gate", names)

    def assert_schema_and_validate(self, path: Path, schema_name: str) -> None:
        schema = check_schema_file(path)
        self.assertTrue(schema["passed"], schema["errors"])
        kwargs = {f"{schema_name}_paths": [path]}
        validation = validate_artifacts(**kwargs, strict=True)
        self.assertTrue(validation["passed"], validation)


if __name__ == "__main__":
    unittest.main()
