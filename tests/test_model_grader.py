import hashlib
import json
import shutil
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

from flightrecorder.cli import main
from flightrecorder.review import review_item_sha256
from flightrecorder.schema_registry import check_schema_file, list_schema_records
from flightrecorder.validation import validate_artifacts


ROOT = Path(__file__).resolve().parents[1]


def run_cli(args):
    with redirect_stdout(StringIO()):
        return main(args)


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def model_grader_row_sha256(row: dict) -> str:
    payload = {key: item for key, item in row.items() if key != "label_sha256"}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_completed_labels(review_dir: Path, labels_path: Path) -> None:
    rows = read_jsonl(review_dir / "label_template.jsonl")
    for row in rows:
        row["human_label"] = row["suggested_human_label"]
        row["notes"] = "Accepted suggested label for model-grader fixture coverage."
        row["reviewer"] = "model-grader-test"
        row["reviewer_confidence"] = "high"
        row["reviewed_at"] = "2026-07-03T00:00:00Z"
    labels_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def mark_first_review_item_needs_review(review_dir: Path) -> None:
    items_path = review_dir / "review_items.jsonl"
    rows = read_jsonl(items_path)
    rows[0]["suggested_human_label"] = "needs_review"
    rows[0]["notes"] = "Fixture forces a model-grader disagreement queue item."
    rows[0]["review_item_sha256"] = review_item_sha256(rows[0])
    items_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


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
    def test_disagreement_queue_blocks_schema_valid_semantically_forged_dry_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "model_grader"
            shutil.copytree(ROOT / "examples" / "agentic_training" / "model_grader", root)
            dry_run = root / "dry_run.json"
            forged = json.loads(dry_run.read_text(encoding="utf-8"))
            forged["graded_item_count"] += 1
            dry_run.write_text(json.dumps(forged, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            self.assertTrue(check_schema_file(dry_run)["passed"])
            out = root / "forged_disagreement_queue.json"

            code = run_cli(
                [
                    "model-grader",
                    "disagreement-queue",
                    "--dry-run",
                    str(dry_run),
                    "--out",
                    str(out),
                ]
            )

            self.assertEqual(code, 1)
            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertFalse(payload["passed"])
            self.assertIn("dry_run_receipt_valid", {check["id"] for check in payload["checks"] if not check["passed"]})

    def test_cli_emits_fail_closed_model_grader_flow(self):
        with tempfile.TemporaryDirectory() as tmp:
            review, calibration = make_review_flow(tmp)
            root = Path(tmp)
            artifact_dir = root / "artifacts" / "model_grader"
            rubric = artifact_dir / "rubric.json"
            dry_run = artifact_dir / "dry_run.json"
            blocked_gate = artifact_dir / "blocked_gate.json"
            passing_gate = artifact_dir / "passing_gate.json"

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
            self.assertEqual(rubric_payload["review_export"]["manifest"]["path"], "../../review/manifest.json")
            self.assert_schema_and_validate(rubric, "rubric_spec")

            preserved_rubric = artifact_dir / "preserved_rubric.json"
            self.assertEqual(
                run_cli(
                    [
                        "model-grader",
                        "rubric",
                        "--review-export",
                        str(review),
                        "--rubric-id",
                        "prompt-injection-rubric",
                        "--preserve-paths",
                        "--created-at",
                        "2026-07-03T00:00:00+00:00",
                        "--out",
                        str(preserved_rubric),
                    ]
                ),
                0,
            )
            preserved_payload = json.loads(preserved_rubric.read_text(encoding="utf-8"))
            self.assertNotIn(str(root), json.dumps(preserved_payload, sort_keys=True))
            self.assertFalse(Path(preserved_payload["review_export"]["manifest"]["path"]).is_absolute())
            self.assert_schema_and_validate(preserved_rubric, "rubric_spec")

            forged_rubric_payload = json.loads(json.dumps(rubric_payload))
            forged_rubric_payload["provider_rubric_url"] = "redacted-provider-rubric"
            forged_rubric_payload["criteria"][0]["provider_weight"] = "forged"
            forged_rubric_payload["review_export"]["provider_dataset_id"] = "redacted-provider-dataset"
            forged_rubric_payload["review_export"]["manifest"]["signed_url"] = "redacted-signed-url"
            forged_rubric_payload["review_item_fingerprints"][0]["provider_trace_id"] = "redacted-provider-trace"
            forged_rubric_payload["calibration_requirements"]["paid_grader_allowed"] = True
            forged_rubric_payload["execution_boundary"]["provider_api_receipt"] = "redacted-provider-receipt"
            rubric.write_text(json.dumps(forged_rubric_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            schema = check_schema_file(rubric)
            self.assertFalse(schema["passed"], schema)
            validation = validate_artifacts(rubric_spec_paths=[rubric], strict=True)
            self.assertFalse(validation["passed"], validation)
            errors = "\n".join(error for target in validation["targets"] for error in target["errors"])
            self.assertIn("rubric_spec contains unknown field(s): ['provider_rubric_url'].", errors)
            self.assertIn("rubric_spec.criteria[0] contains unknown field(s): ['provider_weight'].", errors)
            self.assertIn("rubric_spec.review_export contains unknown field(s): ['provider_dataset_id'].", errors)
            self.assertIn("rubric_spec.review_export.manifest contains unknown field(s): ['signed_url'].", errors)
            self.assertIn(
                "rubric_spec.review_item_fingerprints[0] contains unknown field(s): ['provider_trace_id'].",
                errors,
            )
            self.assertIn(
                "rubric_spec.calibration_requirements contains unknown field(s): ['paid_grader_allowed'].",
                errors,
            )
            self.assertIn("rubric_spec.execution_boundary contains unknown field(s): ['provider_api_receipt'].", errors)
            rubric.write_text(json.dumps(rubric_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            redirected_rubric = artifact_dir / "redirected_rubric.json"
            redirected_rubric.write_text("{}\n", encoding="utf-8")
            rubric_output_link = artifact_dir / "rubric_output_link.json"
            try:
                rubric_output_link.symlink_to(redirected_rubric)
            except (NotImplementedError, OSError) as exc:
                self.skipTest(f"symlink unavailable: {exc}")
            self.assert_cli_error(
                [
                    "model-grader",
                    "rubric",
                    "--review-export",
                    str(review),
                    "--rubric-id",
                    "prompt-injection-rubric",
                    "--out",
                    str(rubric_output_link),
                ],
                "model-grader artifact output must resolve to a regular non-symlink file",
            )

            review_link = artifact_dir / "review_link"
            review_link.symlink_to(review, target_is_directory=True)
            self.assert_cli_error(
                [
                    "model-grader",
                    "rubric",
                    "--review-export",
                    str(review_link),
                    "--rubric-id",
                    "prompt-injection-rubric",
                    "--out",
                    str(artifact_dir / "rubric_from_symlinked_review.json"),
                ],
                "review export must resolve to a regular non-symlink directory",
            )

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
            self.assertEqual(dry_payload["source_artifacts"]["rubric_spec"]["path"], "rubric.json")
            self.assertTrue(all(len(row["label_sha256"]) == 64 for row in dry_payload["grader_labels"]))
            self.assert_schema_and_validate(dry_run, "model_grader_dry_run")

            forged_dry_payload = json.loads(json.dumps(dry_payload))
            forged_dry_payload["paid_grader_invoice_id"] = "redacted-invoice"
            forged_dry_payload["checks"][0]["provider_call"] = "forged"
            forged_dry_payload["grader"]["provider_job_id"] = "redacted-provider-job"
            forged_dry_payload["source_artifacts"]["provider_dataset"] = {"path": "redacted-provider-dataset"}
            forged_dry_payload["source_artifacts"]["review_export"]["provider_dataset_id"] = "redacted-provider-dataset"
            forged_dry_payload["source_artifacts"]["review_export"]["manifest"]["signed_url"] = "redacted-signed-url"
            forged_dry_payload["source_artifacts"]["rubric_spec"]["credential_hint"] = "redacted"
            forged_dry_payload["label_counts"][0]["source"] = "forged"
            forged_dry_payload["grader_labels"][0]["provider_trace_id"] = "redacted-provider-trace"
            forged_dry_payload["human_review_overrides"]["provider_queue_id"] = "redacted-provider-queue"
            forged_dry_payload["training_admission"]["trainer_dataset_id"] = "redacted-trainer-dataset"
            forged_dry_payload["execution_boundary"]["paid_model_grader_invoice_id"] = "redacted-invoice"
            dry_run.write_text(json.dumps(forged_dry_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            schema = check_schema_file(dry_run)
            self.assertFalse(schema["passed"], schema)
            validation = validate_artifacts(model_grader_dry_run_paths=[dry_run], strict=True)
            self.assertFalse(validation["passed"], validation)
            errors = "\n".join(error for target in validation["targets"] for error in target["errors"])
            self.assertIn("model_grader_dry_run contains unknown field(s): ['paid_grader_invoice_id'].", errors)
            self.assertIn("model_grader_dry_run.checks[0] contains unknown field(s): ['provider_call'].", errors)
            self.assertIn("model_grader_dry_run.grader contains unknown field(s): ['provider_job_id'].", errors)
            self.assertIn("model_grader_dry_run.source_artifacts contains unknown field(s): ['provider_dataset'].", errors)
            self.assertIn(
                "model_grader_dry_run.source_artifacts.review_export contains unknown field(s): ['provider_dataset_id'].",
                errors,
            )
            self.assertIn(
                "model_grader_dry_run.source_artifacts.review_export.manifest contains unknown field(s): ['signed_url'].",
                errors,
            )
            self.assertIn(
                "model_grader_dry_run.source_artifacts.rubric_spec contains unknown field(s): ['credential_hint'].",
                errors,
            )
            self.assertIn("model_grader_dry_run.label_counts[0] contains unknown field(s): ['source'].", errors)
            self.assertIn("model_grader_dry_run.grader_labels[0] contains unknown field(s): ['provider_trace_id'].", errors)
            self.assertIn(
                "model_grader_dry_run.human_review_overrides contains unknown field(s): ['provider_queue_id'].",
                errors,
            )
            self.assertIn("model_grader_dry_run.training_admission contains unknown field(s): ['trainer_dataset_id'].", errors)
            self.assertIn(
                "model_grader_dry_run.execution_boundary contains unknown field(s): ['paid_model_grader_invoice_id'].",
                errors,
            )
            dry_run.write_text(json.dumps(dry_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            rubric_link = artifact_dir / "rubric_link.json"
            rubric_link.symlink_to(rubric)
            self.assert_validation_error(
                rubric_link,
                "rubric_spec",
                "rubric_spec.path must resolve to a regular non-symlink file",
            )
            self.assert_cli_error(
                [
                    "model-grader",
                    "dry-run",
                    "--review-export",
                    str(review),
                    "--rubric",
                    str(rubric_link),
                    "--grader-id",
                    "mock-grader-v1",
                    "--out",
                    str(artifact_dir / "dry_run_from_symlinked_rubric.json"),
                ],
                "rubric spec must resolve to a regular non-symlink file",
            )

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
            self.assertEqual(passing_payload["metrics"]["dry_run_disagreement_queue_count"], 0)
            self.assertEqual(passing_payload["metrics"]["dry_run_labels_requiring_human_review_count"], 0)
            self.assertFalse(passing_payload["execution_boundary"]["provider_api_called"])
            self.assertEqual(passing_payload["source_artifacts"]["review_calibration"]["path"], "../../review_calibration.json")
            self.assert_schema_and_validate(passing_gate, "model_grader_gate")

            absolute_ref_payload = json.loads(json.dumps(passing_payload))
            absolute_ref_payload["source_artifacts"]["dry_run_receipt"]["path"] = str(dry_run)
            passing_gate.write_text(json.dumps(absolute_ref_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            validation = validate_artifacts(model_grader_gate_paths=[passing_gate])
            self.assertFalse(validation["passed"], validation)
            errors = "\n".join(error for target in validation["targets"] for error in target["errors"])
            self.assertIn("model_grader_gate.source_artifacts.dry_run_receipt.path must be a relative path or redacted placeholder", errors)

            forged_gate_payload = json.loads(json.dumps(passing_payload))
            forged_gate_payload["trainer_handoff_url"] = "redacted-trainer-handoff"
            forged_gate_payload["checks"][0]["admitted_label_source"] = "forged"
            forged_gate_payload["source_artifacts"]["dry_run_receipt"]["provider_job_id"] = "redacted-provider-job"
            forged_gate_payload["source_artifacts"]["unexpected_trainer_receipt"] = json.loads(
                json.dumps(forged_gate_payload["source_artifacts"]["dry_run_receipt"])
            )
            forged_gate_payload["admission"]["uncalibrated_label_source"] = "forged"
            forged_gate_payload["metrics"]["uncalibrated_label_count"] = 1
            forged_gate_payload["execution_boundary"]["labels_written_to_dataset"] = True
            passing_gate.write_text(json.dumps(forged_gate_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            schema = check_schema_file(passing_gate)
            self.assertFalse(schema["passed"], schema)
            validation = validate_artifacts(model_grader_gate_paths=[passing_gate], strict=True)
            self.assertFalse(validation["passed"], validation)
            errors = "\n".join(error for target in validation["targets"] for error in target["errors"])
            self.assertIn("model_grader_gate contains unknown field(s): ['trainer_handoff_url'].", errors)
            self.assertIn("model_grader_gate.checks[0] contains unknown field(s): ['admitted_label_source'].", errors)
            self.assertIn("model_grader_gate.source_artifacts contains unknown field(s): ['unexpected_trainer_receipt'].", errors)
            self.assertIn(
                "model_grader_gate.source_artifacts.dry_run_receipt contains unknown field(s): ['provider_job_id'].",
                errors,
            )
            self.assertIn("model_grader_gate.admission contains unknown field(s): ['uncalibrated_label_source'].", errors)
            self.assertIn("model_grader_gate.metrics contains unknown field(s): ['uncalibrated_label_count'].", errors)
            self.assertIn(
                "model_grader_gate.execution_boundary contains unknown field(s): ['labels_written_to_dataset'].",
                errors,
            )
            passing_gate.write_text(json.dumps(passing_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            calibration_link = artifact_dir / "review_calibration_link.json"
            calibration_link.symlink_to(calibration)
            self.assert_validation_error(
                calibration_link,
                "review_calibration",
                "review_calibration.path must resolve to a regular non-symlink file",
            )
            self.assert_cli_error(
                [
                    "model-grader",
                    "gate",
                    "--dry-run",
                    str(dry_run),
                    "--rubric",
                    str(rubric),
                    "--review-calibration",
                    str(calibration_link),
                    "--out",
                    str(artifact_dir / "gate_from_symlinked_calibration.json"),
                ],
                "review calibration must resolve to a regular non-symlink file",
            )

            stale_rubric = artifact_dir / "stale_rubric.json"
            stale_rubric_payload = dict(rubric_payload)
            stale_rubric_payload["review_export"] = dict(rubric_payload["review_export"])
            stale_rubric_payload["review_export"]["review_items"] = dict(rubric_payload["review_export"]["review_items"])
            stale_rubric_payload["review_export"]["review_items"]["sha256"] = "0" * 64
            stale_rubric.write_text(json.dumps(stale_rubric_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            self.assert_validation_error(stale_rubric, "rubric_spec", "review_items.sha256 does not match the current file")

            stale_rubric_fingerprint = artifact_dir / "stale_rubric_fingerprint.json"
            stale_rubric_fingerprint_payload = json.loads(json.dumps(rubric_payload))
            stale_rubric_fingerprint_payload["review_item_fingerprints"][0]["review_item_sha256"] = "0" * 64
            stale_rubric_fingerprint.write_text(
                json.dumps(stale_rubric_fingerprint_payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            self.assert_validation_error(
                stale_rubric_fingerprint,
                "rubric_spec",
                "review_item_fingerprints must match review_export.review_items",
            )

            stale_dry_run = artifact_dir / "stale_dry_run.json"
            stale_dry_payload = dict(dry_payload)
            stale_dry_payload["source_artifacts"] = dict(dry_payload["source_artifacts"])
            stale_dry_payload["source_artifacts"]["rubric_spec"] = dict(dry_payload["source_artifacts"]["rubric_spec"])
            stale_dry_payload["source_artifacts"]["rubric_spec"]["size_bytes"] += 1
            stale_dry_run.write_text(json.dumps(stale_dry_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            self.assert_validation_error(stale_dry_run, "model_grader_dry_run", "rubric_spec.size_bytes does not match the current file")

            stale_label_dry_run = artifact_dir / "stale_label_dry_run.json"
            stale_label_payload = json.loads(json.dumps(dry_payload))
            stale_label_payload["grader_labels"][0]["label_sha256"] = "0" * 64
            stale_label_dry_run.write_text(json.dumps(stale_label_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            self.assert_validation_error(
                stale_label_dry_run,
                "model_grader_dry_run",
                "label_sha256 does not match label contents",
            )

            stale_source_label_dry_run = artifact_dir / "stale_source_label_dry_run.json"
            stale_source_label_payload = json.loads(json.dumps(dry_payload))
            stale_source_label_payload["grader_labels"][0]["review_item_id"] = "stale-review-item-id"
            stale_source_label_payload["grader_labels"][0]["label_sha256"] = model_grader_row_sha256(
                stale_source_label_payload["grader_labels"][0]
            )
            stale_source_label_dry_run.write_text(
                json.dumps(stale_source_label_payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            self.assert_validation_error(
                stale_source_label_dry_run,
                "model_grader_dry_run",
                "grader_labels must match source_artifacts.review_export.review_items",
            )

            alternate_review = root / "alternate_review"
            shutil.copytree(review, alternate_review)
            alternate_rows = read_jsonl(alternate_review / "review_items.jsonl")
            alternate_rows[0]["suggested_human_label"] = "needs_review"
            alternate_rows[0]["notes"] = "Fixture creates a distinct review export for rubric mismatch validation."
            alternate_rows[0]["review_item_sha256"] = review_item_sha256(alternate_rows[0])
            (alternate_review / "review_items.jsonl").write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in alternate_rows),
                encoding="utf-8",
            )
            alternate_rubric = artifact_dir / "alternate_rubric.json"
            self.assertEqual(
                run_cli(
                    [
                        "model-grader",
                        "rubric",
                        "--review-export",
                        str(alternate_review),
                        "--rubric-id",
                        "alternate-prompt-injection-rubric",
                        "--out",
                        str(alternate_rubric),
                    ]
                ),
                0,
            )
            mismatched_rubric_dry_run = artifact_dir / "mismatched_rubric_dry_run.json"
            mismatched_rubric_payload = json.loads(json.dumps(dry_payload))
            mismatched_rubric_payload["source_artifacts"]["rubric_spec"]["path"] = alternate_rubric.name
            mismatched_rubric_payload["source_artifacts"]["rubric_spec"]["sha256"] = hashlib.sha256(
                alternate_rubric.read_bytes()
            ).hexdigest()
            mismatched_rubric_payload["source_artifacts"]["rubric_spec"]["size_bytes"] = alternate_rubric.stat().st_size
            mismatched_rubric_dry_run.write_text(
                json.dumps(mismatched_rubric_payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            self.assert_validation_error(
                mismatched_rubric_dry_run,
                "model_grader_dry_run",
                "source_artifacts.review_export must match rubric_spec.review_export",
            )

            stale_gate = artifact_dir / "stale_gate.json"
            stale_gate_payload = dict(passing_payload)
            stale_gate_payload["source_artifacts"] = dict(passing_payload["source_artifacts"])
            stale_gate_payload["source_artifacts"]["dry_run_receipt"] = dict(passing_payload["source_artifacts"]["dry_run_receipt"])
            stale_gate_payload["source_artifacts"]["dry_run_receipt"]["sha256"] = "0" * 64
            stale_gate.write_text(json.dumps(stale_gate_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            self.assert_validation_error(
                stale_gate,
                "model_grader_gate",
                "dry_run_receipt.sha256 does not match the current file",
            )

            symlink_dry_run = artifact_dir / "symlink_dry_run.json"
            symlink_dry_payload = dict(dry_payload)
            symlink_dry_payload["source_artifacts"] = dict(dry_payload["source_artifacts"])
            symlink_dry_payload["source_artifacts"]["rubric_spec"] = dict(dry_payload["source_artifacts"]["rubric_spec"])
            symlink_dry_payload["source_artifacts"]["rubric_spec"]["path"] = rubric_link.name
            symlink_dry_run.write_text(json.dumps(symlink_dry_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            self.assert_validation_error(
                symlink_dry_run,
                "model_grader_dry_run",
                "rubric_spec.path must resolve to a regular non-symlink file",
            )

            broken_symlink_dry_run = artifact_dir / "broken_symlink_dry_run.json"
            broken_symlink_dry_payload = dict(dry_payload)
            broken_symlink_dry_payload["source_artifacts"] = dict(dry_payload["source_artifacts"])
            broken_symlink_dry_payload["source_artifacts"]["rubric_spec"] = dict(dry_payload["source_artifacts"]["rubric_spec"])
            broken_rubric_link = artifact_dir / "broken_rubric_link.json"
            broken_rubric_link.symlink_to(artifact_dir / "missing_rubric.json")
            broken_symlink_dry_payload["source_artifacts"]["rubric_spec"]["path"] = broken_rubric_link.name
            broken_symlink_dry_run.write_text(json.dumps(broken_symlink_dry_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            self.assert_validation_error(
                broken_symlink_dry_run,
                "model_grader_dry_run",
                "rubric_spec.path must resolve to a regular non-symlink file",
            )

            symlink_parent_gate = artifact_dir / "symlink_parent_gate.json"
            symlink_parent_gate_payload = dict(passing_payload)
            symlink_parent_gate_payload["source_artifacts"] = dict(passing_payload["source_artifacts"])
            symlink_parent_gate_payload["source_artifacts"]["dry_run_receipt"] = dict(passing_payload["source_artifacts"]["dry_run_receipt"])
            linked_target = artifact_dir / "linked_target"
            linked_target.mkdir()
            (linked_target / dry_run.name).write_text(dry_run.read_text(encoding="utf-8"), encoding="utf-8")
            linked_parent = artifact_dir / "linked_artifacts"
            linked_parent.symlink_to(linked_target, target_is_directory=True)
            symlink_parent_gate_payload["source_artifacts"]["dry_run_receipt"]["path"] = str(Path(linked_parent.name) / dry_run.name)
            symlink_parent_gate.write_text(json.dumps(symlink_parent_gate_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            self.assert_validation_error(
                symlink_parent_gate,
                "model_grader_gate",
                "dry_run_receipt.path must resolve to a regular non-symlink file",
            )

            optional_calibration_gate = artifact_dir / "optional_calibration_symlink_gate.json"
            optional_calibration_payload = dict(passing_payload)
            optional_calibration_payload["source_artifacts"] = dict(passing_payload["source_artifacts"])
            optional_calibration_payload["source_artifacts"]["review_calibration"] = dict(
                passing_payload["source_artifacts"]["review_calibration"]
            )
            optional_calibration_payload["source_artifacts"]["review_calibration"]["exists"] = False
            optional_calibration_payload["source_artifacts"]["review_calibration"]["path"] = calibration_link.name
            optional_calibration_gate.write_text(
                json.dumps(optional_calibration_payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            self.assert_validation_error(
                optional_calibration_gate,
                "model_grader_gate",
                "review_calibration.path must resolve to a regular non-symlink file",
            )

            optional_override_gate = artifact_dir / "optional_override_symlink_gate.json"
            optional_override_payload = dict(passing_payload)
            optional_override_payload["source_artifacts"] = dict(passing_payload["source_artifacts"])
            optional_override_payload["source_artifacts"]["model_grader_override_receipt"] = dict(
                passing_payload["source_artifacts"]["model_grader_override_receipt"]
            )
            override_receipt_link = artifact_dir / "model_grader_override_receipt_link.json"
            override_receipt_link.symlink_to(dry_run)
            optional_override_payload["source_artifacts"]["model_grader_override_receipt"]["exists"] = False
            optional_override_payload["source_artifacts"]["model_grader_override_receipt"]["path"] = override_receipt_link.name
            optional_override_gate.write_text(
                json.dumps(optional_override_payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            self.assert_validation_error(
                optional_override_gate,
                "model_grader_gate",
                "model_grader_override_receipt.path must resolve to a regular non-symlink file",
            )

            dry_run_link = artifact_dir / "dry_run_link.json"
            dry_run_link.symlink_to(dry_run)
            self.assert_validation_error(
                dry_run_link,
                "model_grader_dry_run",
                "model_grader_dry_run.path must resolve to a regular non-symlink file",
            )

            gate_link = artifact_dir / "passing_gate_link.json"
            gate_link.symlink_to(passing_gate)
            self.assert_validation_error(
                gate_link,
                "model_grader_gate",
                "model_grader_gate.path must resolve to a regular non-symlink file",
            )

    def test_gate_blocks_unresolved_model_grader_disagreement_queue(self):
        with tempfile.TemporaryDirectory() as tmp:
            review, calibration = make_review_flow(tmp)
            mark_first_review_item_needs_review(review)
            root = Path(tmp)
            rubric = root / "rubric.json"
            dry_run = root / "dry_run.json"
            disagreement_queue = root / "disagreement_queue.json"
            overrides = root / "overrides.jsonl"
            override_receipt = root / "override_receipt.json"
            gate = root / "gate.json"
            resolved_gate = root / "resolved_gate.json"

            self.assertEqual(
                run_cli(
                    [
                        "model-grader",
                        "rubric",
                        "--review-export",
                        str(review),
                        "--rubric-id",
                        "prompt-injection-rubric",
                        "--created-at",
                        "2026-07-03T00:00:00+00:00",
                        "--out",
                        str(rubric),
                    ]
                ),
                0,
            )
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
                        "--created-at",
                        "2026-07-03T00:00:00+00:00",
                        "--out",
                        str(dry_run),
                    ]
                ),
                0,
            )
            dry_payload = json.loads(dry_run.read_text(encoding="utf-8"))
            self.assertEqual(dry_payload["disagreement_queue"][0]["mock_model_label"], "needs_review")
            self.assertEqual(sum(1 for row in dry_payload["grader_labels"] if row["requires_human_review"]), 1)
            self.assert_schema_and_validate(dry_run, "model_grader_dry_run")

            self.assertEqual(
                run_cli(
                    [
                        "model-grader",
                        "disagreement-queue",
                        "--dry-run",
                        str(dry_run),
                        "--created-at",
                        "2026-07-03T00:00:00+00:00",
                        "--out",
                        str(disagreement_queue),
                    ]
                ),
                0,
            )
            queue_payload = json.loads(disagreement_queue.read_text(encoding="utf-8"))
            self.assertEqual(queue_payload["schema_version"], "hfr.model_grader_disagreement_queue.v1")
            self.assertEqual(queue_payload["readiness"], "ready_for_human_review")
            self.assertEqual(queue_payload["queue_count"], 1)
            self.assertEqual(queue_payload["required_review_item_ids"], [dry_payload["disagreement_queue"][0]["review_item_id"]])
            self.assertEqual(queue_payload["queue"], dry_payload["disagreement_queue"])
            self.assertFalse(queue_payload["training_admission"]["labels_allowed_for_training"])
            self.assertEqual(queue_payload["training_admission"]["labels_admitted_count"], 0)
            self.assert_schema_and_validate(disagreement_queue, "model_grader_disagreement_queue")

            forged_queue_payload = json.loads(json.dumps(queue_payload))
            forged_queue_payload["provider_queue_url"] = "redacted-provider-queue"
            forged_queue_payload["checks"][0]["provider_call"] = "forged"
            forged_queue_payload["source_artifacts"]["dry_run_receipt"]["provider_job_id"] = "redacted-provider-job"
            forged_queue_payload["queue"][0]["provider_review_url"] = "redacted-provider-review"
            forged_queue_payload["override_requirements"]["provider_assignment_id"] = "redacted-provider-assignment"
            forged_queue_payload["training_admission"]["trainer_dataset_id"] = "redacted-trainer-dataset"
            forged_queue_payload["execution_boundary"]["labels_written_to_dataset"] = True
            disagreement_queue.write_text(json.dumps(forged_queue_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            schema = check_schema_file(disagreement_queue)
            self.assertFalse(schema["passed"], schema)
            validation = validate_artifacts(model_grader_disagreement_queue_paths=[disagreement_queue], strict=True)
            self.assertFalse(validation["passed"], validation)
            errors = "\n".join(error for target in validation["targets"] for error in target["errors"])
            self.assertIn("model_grader_disagreement_queue contains unknown field(s): ['provider_queue_url'].", errors)
            self.assertIn("model_grader_disagreement_queue.checks[0] contains unknown field(s): ['provider_call'].", errors)
            self.assertIn(
                "model_grader_disagreement_queue.source_artifacts.dry_run_receipt contains unknown field(s): ['provider_job_id'].",
                errors,
            )
            self.assertIn(
                "model_grader_disagreement_queue.queue[0] contains unknown field(s): ['provider_review_url'].",
                errors,
            )
            self.assertIn(
                "model_grader_disagreement_queue.override_requirements contains unknown field(s): ['provider_assignment_id'].",
                errors,
            )
            self.assertIn(
                "model_grader_disagreement_queue.training_admission contains unknown field(s): ['trainer_dataset_id'].",
                errors,
            )
            self.assertIn(
                "model_grader_disagreement_queue.execution_boundary contains unknown field(s): ['labels_written_to_dataset'].",
                errors,
            )
            disagreement_queue.write_text(json.dumps(queue_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            stale_queue_artifact = root / "stale_disagreement_queue.json"
            stale_queue_payload = json.loads(json.dumps(queue_payload))
            stale_queue_payload["queue"][0]["review_item_sha256"] = "0" * 64
            stale_queue_artifact.write_text(json.dumps(stale_queue_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            self.assert_validation_error(
                stale_queue_artifact,
                "model_grader_disagreement_queue",
                "queue must match source dry-run disagreement_queue",
            )

            unknown_queue_dry_run = root / "unknown_queue_dry_run.json"
            unknown_queue_payload = json.loads(json.dumps(dry_payload))
            unknown_queue_payload["disagreement_queue"][0]["provider_review_url"] = "redacted-provider-review"
            unknown_queue_dry_run.write_text(json.dumps(unknown_queue_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            schema = check_schema_file(unknown_queue_dry_run)
            self.assertFalse(schema["passed"], schema)
            self.assert_validation_error(
                unknown_queue_dry_run,
                "model_grader_dry_run",
                "model_grader_dry_run.disagreement_queue[0] contains unknown field(s): ['provider_review_url'].",
            )

            stale_queue_dry_run = root / "stale_queue_dry_run.json"
            stale_queue_payload = json.loads(json.dumps(dry_payload))
            stale_queue_payload["disagreement_queue"][0]["review_item_sha256"] = "0" * 64
            stale_queue_dry_run.write_text(json.dumps(stale_queue_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            self.assert_validation_error(
                stale_queue_dry_run,
                "model_grader_dry_run",
                "disagreement_queue must match labels requiring human review",
            )

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
                        str(gate),
                    ]
                ),
                1,
            )
            gate_payload = json.loads(gate.read_text(encoding="utf-8"))
            self.assertFalse(gate_payload["passed"])
            self.assertFalse(gate_payload["admission"]["labels_allowed_for_training"])
            self.assertEqual(gate_payload["admission"]["labels_admitted_count"], 0)
            self.assertEqual(gate_payload["metrics"]["dry_run_disagreement_queue_count"], 1)
            self.assertEqual(gate_payload["metrics"]["dry_run_labels_requiring_human_review_count"], 1)
            failed_ids = {check["id"] for check in gate_payload["checks"] if not check["passed"]}
            self.assertIn("dry_run_human_review_queue_resolved", failed_ids)
            self.assert_schema_and_validate(gate, "model_grader_gate")

            queued = dry_payload["disagreement_queue"][0]
            overrides.write_text(
                json.dumps(
                    {
                        "review_item_id": queued["review_item_id"],
                        "review_item_sha256": queued["review_item_sha256"],
                        "human_label": "reject",
                        "reviewer_confidence": "high",
                        "reviewer": "model-grader-test",
                        "reviewed_at": "2026-07-03T00:00:00Z",
                        "notes": "Human override resolves the queued mock grader label.",
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            self.assertEqual(
                run_cli(
                    [
                        "model-grader",
                        "override-receipt",
                        "--dry-run",
                        str(dry_run),
                        "--overrides",
                        str(overrides),
                        "--created-at",
                        "2026-07-03T00:00:00+00:00",
                        "--out",
                        str(override_receipt),
                    ]
                ),
                0,
            )
            override_payload = json.loads(override_receipt.read_text(encoding="utf-8"))
            self.assertTrue(override_payload["passed"], override_payload["blocked_reasons"])
            self.assertEqual(override_payload["metrics"]["resolved_queue_count"], 1)
            self.assertEqual(override_payload["metrics"]["unresolved_queue_count"], 0)
            self.assert_schema_and_validate(override_receipt, "model_grader_override_receipt")

            forged_override_payload = json.loads(json.dumps(override_payload))
            forged_override_payload["trainer_handoff_url"] = "redacted-trainer-handoff"
            forged_override_payload["checks"][0]["provider_call"] = "forged"
            forged_override_payload["source_artifacts"]["unexpected_provider_receipt"] = {"path": "redacted-provider-receipt"}
            forged_override_payload["source_artifacts"]["dry_run_receipt"]["provider_job_id"] = "redacted-provider-job"
            forged_override_payload["source_artifacts"]["override_rows"]["signed_url"] = "redacted-signed-url"
            forged_override_payload["queue"]["provider_queue_id"] = "redacted-provider-queue"
            forged_override_payload["overrides"][0]["provider_review_url"] = "redacted-provider-review"
            forged_override_payload["metrics"]["provider_override_count"] = 1
            forged_override_payload["training_admission"]["trainer_dataset_id"] = "redacted-trainer-dataset"
            forged_override_payload["execution_boundary"]["labels_written_to_dataset"] = True
            override_receipt.write_text(json.dumps(forged_override_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            schema = check_schema_file(override_receipt)
            self.assertFalse(schema["passed"], schema)
            validation = validate_artifacts(model_grader_override_receipt_paths=[override_receipt], strict=True)
            self.assertFalse(validation["passed"], validation)
            errors = "\n".join(error for target in validation["targets"] for error in target["errors"])
            self.assertIn("model_grader_override_receipt contains unknown field(s): ['trainer_handoff_url'].", errors)
            self.assertIn("model_grader_override_receipt.checks[0] contains unknown field(s): ['provider_call'].", errors)
            self.assertIn(
                "model_grader_override_receipt.source_artifacts contains unknown field(s): ['unexpected_provider_receipt'].",
                errors,
            )
            self.assertIn(
                "model_grader_override_receipt.source_artifacts.dry_run_receipt contains unknown field(s): ['provider_job_id'].",
                errors,
            )
            self.assertIn(
                "model_grader_override_receipt.source_artifacts.override_rows contains unknown field(s): ['signed_url'].",
                errors,
            )
            self.assertIn("model_grader_override_receipt.queue contains unknown field(s): ['provider_queue_id'].", errors)
            self.assertIn("model_grader_override_receipt.overrides[0] contains unknown field(s): ['provider_review_url'].", errors)
            self.assertIn("model_grader_override_receipt.metrics contains unknown field(s): ['provider_override_count'].", errors)
            self.assertIn(
                "model_grader_override_receipt.training_admission contains unknown field(s): ['trainer_dataset_id'].",
                errors,
            )
            self.assertIn(
                "model_grader_override_receipt.execution_boundary contains unknown field(s): ['labels_written_to_dataset'].",
                errors,
            )
            override_receipt.write_text(json.dumps(override_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            stale_override_hash_receipt = root / "stale_override_hash_receipt.json"
            stale_override_hash_payload = json.loads(json.dumps(override_payload))
            stale_override_hash_payload["overrides"][0]["override_sha256"] = "0" * 64
            stale_override_hash_receipt.write_text(
                json.dumps(stale_override_hash_payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            self.assert_validation_error(
                stale_override_hash_receipt,
                "model_grader_override_receipt",
                "override_sha256 does not match override contents",
            )

            try:
                override_rows_link = root / "overrides_link.jsonl"
                override_rows_link.symlink_to(overrides)
                override_receipt_link = root / "override_receipt_link.json"
                override_receipt_link.symlink_to(override_receipt)
            except (NotImplementedError, OSError) as exc:
                self.skipTest(f"symlink unavailable: {exc}")

            symlink_override_receipt = root / "symlink_override_receipt.json"
            symlink_override_payload = dict(override_payload)
            symlink_override_payload["source_artifacts"] = dict(override_payload["source_artifacts"])
            symlink_override_payload["source_artifacts"]["override_rows"] = dict(
                override_payload["source_artifacts"]["override_rows"]
            )
            symlink_override_payload["source_artifacts"]["override_rows"]["path"] = override_rows_link.name
            symlink_override_receipt.write_text(
                json.dumps(symlink_override_payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            self.assert_validation_error(
                symlink_override_receipt,
                "model_grader_override_receipt",
                "override_rows.path must resolve to a regular non-symlink file",
            )

            self.assert_validation_error(
                override_receipt_link,
                "model_grader_override_receipt",
                "model_grader_override_receipt.path must resolve to a regular non-symlink file",
            )
            self.assert_cli_error(
                [
                    "model-grader",
                    "override-receipt",
                    "--dry-run",
                    str(dry_run),
                    "--overrides",
                    str(override_rows_link),
                    "--out",
                    str(root / "override_from_symlinked_rows.json"),
                ],
                "model-grader override rows must resolve to a regular non-symlink file",
            )

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
                        "--override-receipt",
                        str(override_receipt),
                        "--min-calibration-agreement-rate",
                        "1.0",
                        "--max-disagreements",
                        "0",
                        "--created-at",
                        "2026-07-03T00:00:00+00:00",
                        "--out",
                        str(resolved_gate),
                    ]
                ),
                0,
            )
            resolved_payload = json.loads(resolved_gate.read_text(encoding="utf-8"))
            self.assertTrue(resolved_payload["passed"], resolved_payload["blocked_reasons"])
            self.assertTrue(resolved_payload["admission"]["labels_allowed_for_training"])
            self.assertEqual(resolved_payload["admission"]["labels_admitted_count"], 2)
            self.assertTrue(resolved_payload["metrics"]["human_override_receipt_present"])
            self.assertEqual(resolved_payload["metrics"]["human_override_resolved_count"], 1)
            self.assertEqual(resolved_payload["metrics"]["human_override_unresolved_count"], 0)
            self.assert_schema_and_validate(resolved_gate, "model_grader_gate")
            self.assert_cli_error(
                [
                    "model-grader",
                    "gate",
                    "--dry-run",
                    str(dry_run),
                    "--rubric",
                    str(rubric),
                    "--review-calibration",
                    str(calibration),
                    "--override-receipt",
                    str(override_receipt_link),
                    "--out",
                    str(root / "gate_from_symlinked_override.json"),
                ],
                "model-grader override receipt must resolve to a regular non-symlink file",
            )

    def test_schema_names_are_registered(self):
        names = {record["name"] for record in list_schema_records()}
        self.assertIn("rubric_spec", names)
        self.assertIn("model_grader_dry_run", names)
        self.assertIn("model_grader_disagreement_queue", names)
        self.assertIn("model_grader_override_receipt", names)
        self.assertIn("model_grader_gate", names)

    def test_committed_examples_validate_strictly(self):
        for example_dir in (
            ROOT / "examples" / "model_grader",
            ROOT / "examples" / "agentic_training" / "model_grader",
        ):
            with self.subTest(example_dir=str(example_dir.relative_to(ROOT))):
                validation = validate_artifacts(
                    review_export_dir=example_dir / "review",
                    rubric_spec_paths=[example_dir / "rubric.json"],
                    model_grader_dry_run_paths=[example_dir / "dry_run.json"],
                    model_grader_disagreement_queue_paths=[example_dir / "disagreement_queue.json"],
                    model_grader_gate_paths=[
                        example_dir / "blocked_gate.json",
                        example_dir / "passing_gate.json",
                    ],
                    review_calibration_paths=[example_dir / "review_calibration.json"],
                    strict=True,
                )
                self.assertTrue(validation["passed"], validation)

    def assert_schema_and_validate(self, path: Path, schema_name: str) -> None:
        schema = check_schema_file(path)
        self.assertTrue(schema["passed"], schema["errors"])
        kwargs = {f"{schema_name}_paths": [path]}
        validation = validate_artifacts(**kwargs, strict=True)
        self.assertTrue(validation["passed"], validation)

    def assert_validation_error(self, path: Path, schema_name: str, expected: str) -> None:
        kwargs = {f"{schema_name}_paths": [path]}
        validation = validate_artifacts(**kwargs, strict=True)
        self.assertFalse(validation["passed"], validation)
        errors = [
            error
            for target in validation["targets"]
            for error in target["errors"]
        ]
        self.assertTrue(any(expected in error for error in errors), errors)

    def assert_cli_error(self, args: list[str], expected: str) -> None:
        stderr = StringIO()
        with self.assertRaises(SystemExit) as raised, redirect_stdout(StringIO()), redirect_stderr(stderr):
            main(args)
        self.assertEqual(raised.exception.code, 2)
        self.assertIn(expected, stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
