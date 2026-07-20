import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from flightrecorder.cli import main
from flightrecorder.rejection_sampling import build_rejection_sampling_gate, write_rejection_sampling_gate
from flightrecorder.rollout_generation import (
    build_agentic_rollout_plan,
    build_agentic_rollout_receipt,
    write_agentic_rollout_plan,
    write_agentic_rollout_receipt,
)
from flightrecorder.schema_registry import check_schema_file, list_schema_records
from flightrecorder.validation import validate_artifacts


ROOT = Path(__file__).resolve().parents[1]
SCENARIO = ROOT / "scenarios" / "prompt_injection_good.json"


def run_cli(args):
    with redirect_stdout(StringIO()):
        return main(args)


class RejectionSamplingGateTests(unittest.TestCase):
    def test_gate_rejects_schema_invalid_and_symlinked_review_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rollout_receipt = self.write_rollout_receipt(root)
            review_calibration = self.write_json(root / "review_calibration.json", "hfr.review_calibration.v1")
            reviewed_gate = self.write_json(root / "reviewed_gate.json", "hfr.reviewed_gate.v1")
            invalid_gate = root / "invalid_model_grader_gate.json"
            invalid_gate.write_text(
                json.dumps(
                    {
                        "schema_version": "hfr.model_grader_gate.v1",
                        "passed": True,
                        "readiness": "labels_calibrated_for_curated_handoff",
                    }
                ),
                encoding="utf-8",
            )

            gate = build_rejection_sampling_gate(
                rollout_receipt_paths=[rollout_receipt],
                model_grader_gate_paths=[invalid_gate],
                review_calibration_paths=[review_calibration],
                reviewed_gate_paths=[reviewed_gate],
                out_path=root / "invalid_output.json",
                created_at="2026-07-03T00:00:00+00:00",
            )
            self.assertFalse(gate["passed"])
            self.assertFalse(gate["input_artifacts"]["model_grader_gate"][0]["exists"])

            valid_gate = self.write_json(root / "model_grader_gate.json", "hfr.model_grader_gate.v1")
            linked_gate = root / "linked_model_grader_gate.json"
            try:
                linked_gate.symlink_to(valid_gate.name)
            except OSError as exc:
                self.skipTest(f"symlinks unavailable: {exc}")
            symlinked = build_rejection_sampling_gate(
                rollout_receipt_paths=[rollout_receipt],
                model_grader_gate_paths=[linked_gate],
                review_calibration_paths=[review_calibration],
                reviewed_gate_paths=[reviewed_gate],
                out_path=root / "symlinked_output.json",
                created_at="2026-07-03T00:00:00+00:00",
            )
            self.assertFalse(symlinked["passed"])
            self.assertFalse(symlinked["input_artifacts"]["model_grader_gate"][0]["exists"])

    def test_committed_agentic_training_rejection_sampling_gate_replays_inputs(self):
        gate_path = ROOT / "examples" / "agentic_training" / "rejection_sampling_gate.json"
        gate = json.loads(gate_path.read_text(encoding="utf-8"))

        self.assertTrue(gate["passed"], gate["blocked_reasons"])
        self.assertEqual(gate["readiness"], "ready_for_dataset_curation")
        self.assertEqual(gate["rollout_summary"]["mock_rollout_count"], 6)
        self.assertFalse(gate["rollout_summary"]["dataset_rows_created"])
        self.assertFalse(gate["execution_boundary"]["dataset_rows_written"])
        self.assertFalse(gate["execution_boundary"]["weights_updated_by_flight_recorder"])
        input_paths = {
            role: rows[0]["path"]
            for role, rows in gate["input_artifacts"].items()
            if isinstance(rows, list) and rows
        }
        self.assertEqual(input_paths["agentic_rollout_receipt"], "rollouts/rollout_receipt.json")
        self.assertEqual(input_paths["model_grader_gate"], "model_grader/passing_gate.json")
        self.assertEqual(input_paths["review_calibration"], "model_grader/review_calibration.json")
        self.assertEqual(input_paths["reviewed_gate"], "model_grader/reviewed_gate.json")

        schema = check_schema_file(gate_path)
        validation = validate_artifacts(rejection_sampling_gate_paths=[gate_path], strict=True)

        self.assertTrue(schema["passed"], schema["errors"])
        self.assertTrue(validation["passed"], validation)

    def test_gate_admits_calibrated_mock_rollouts_without_writing_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rollout_receipt = self.write_rollout_receipt(root)
            model_grader_gate = self.write_json(root / "model_grader_gate.json", "hfr.model_grader_gate.v1")
            review_calibration = self.write_json(root / "review_calibration.json", "hfr.review_calibration.v1")
            reviewed_gate = self.write_json(root / "reviewed_gate.json", "hfr.reviewed_gate.v1")
            out = root / "rejection_sampling_gate.json"

            gate = build_rejection_sampling_gate(
                rollout_receipt_paths=[rollout_receipt],
                model_grader_gate_paths=[model_grader_gate],
                review_calibration_paths=[review_calibration],
                reviewed_gate_paths=[reviewed_gate],
                out_path=out,
                created_at="2026-07-03T00:00:00+00:00",
            )
            write_rejection_sampling_gate(out, gate)

            schema = check_schema_file(out)
            self.assertTrue(schema["passed"], schema["errors"])
            validation = validate_artifacts(rejection_sampling_gate_paths=[out], strict=True)
            self.assertTrue(validation["passed"], validation)
            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertTrue(payload["passed"], payload["blocked_reasons"])
            self.assertEqual(payload["gate_path"], "rejection_sampling_gate.json")
            self.assertEqual(payload["readiness"], "ready_for_dataset_curation")
            self.assertEqual(payload["rollout_summary"]["mock_rollout_count"], 1)
            self.assertFalse(payload["execution_boundary"]["dataset_rows_written"])
            self.assertFalse(payload["admission_policy"]["accepts_uncalibrated_labels"])

            preserved_out = root / "preserved_rejection_sampling_gate.json"
            preserved_gate = build_rejection_sampling_gate(
                rollout_receipt_paths=[rollout_receipt],
                model_grader_gate_paths=[model_grader_gate],
                review_calibration_paths=[review_calibration],
                reviewed_gate_paths=[reviewed_gate],
                out_path=preserved_out,
                preserve_paths=True,
                created_at="2026-07-03T00:00:00+00:00",
            )
            write_rejection_sampling_gate(preserved_out, preserved_gate)
            preserved_payload = json.loads(preserved_out.read_text(encoding="utf-8"))
            self.assertNotIn(str(root), json.dumps(preserved_payload, sort_keys=True))
            self.assertEqual(preserved_payload["input_artifacts"]["model_grader_gate"][0]["path"], "model_grader_gate.json")
            preserved_validation = validate_artifacts(rejection_sampling_gate_paths=[preserved_out], strict=True)
            self.assertTrue(preserved_validation["passed"], preserved_validation)

    def test_cli_writes_valid_rejection_sampling_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rollout_receipt = self.write_rollout_receipt(root)
            model_grader_gate = self.write_json(root / "model_grader_gate.json", "hfr.model_grader_gate.v1")
            review_calibration = self.write_json(root / "review_calibration.json", "hfr.review_calibration.v1")
            reviewed_gate = self.write_json(root / "reviewed_gate.json", "hfr.reviewed_gate.v1")
            out = root / "gate.json"

            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "flightrecorder",
                    "rejection-sampling-gate",
                    "--rollout-receipt",
                    str(rollout_receipt),
                    "--model-grader-gate",
                    str(model_grader_gate),
                    "--review-calibration",
                    str(review_calibration),
                    "--reviewed-gate",
                    str(reviewed_gate),
                    "--created-at",
                    "2026-07-03T00:00:00+00:00",
                    "--out",
                    str(out),
                ],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr + completed.stdout)
            validation = validate_artifacts(rejection_sampling_gate_paths=[out], strict=True)
            self.assertTrue(validation["passed"], validation)

    def test_cli_rejects_rejection_sampling_output_that_aliases_an_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rollout_receipt = self.write_rollout_receipt(root)
            model_grader_gate = self.write_json(root / "model_grader_gate.json", "hfr.model_grader_gate.v1")
            review_calibration = self.write_json(root / "review_calibration.json", "hfr.review_calibration.v1")
            reviewed_gate = self.write_json(root / "reviewed_gate.json", "hfr.reviewed_gate.v1")
            original = model_grader_gate.read_bytes()

            with self.assertRaises(SystemExit) as raised:
                run_cli(
                    [
                        "rejection-sampling-gate",
                        "--rollout-receipt",
                        str(rollout_receipt),
                        "--model-grader-gate",
                        str(model_grader_gate),
                        "--review-calibration",
                        str(review_calibration),
                        "--reviewed-gate",
                        str(reviewed_gate),
                        "--out",
                        str(model_grader_gate),
                    ]
                )

            self.assertEqual(raised.exception.code, 2)
            self.assertEqual(model_grader_gate.read_bytes(), original)

    def test_validation_rejects_absolute_rejection_sampling_gate_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rollout_receipt = self.write_rollout_receipt(root)
            model_grader_gate = self.write_json(root / "model_grader_gate.json", "hfr.model_grader_gate.v1")
            review_calibration = self.write_json(root / "review_calibration.json", "hfr.review_calibration.v1")
            reviewed_gate = self.write_json(root / "reviewed_gate.json", "hfr.reviewed_gate.v1")
            out = root / "rejection_sampling_gate.json"
            gate = build_rejection_sampling_gate(
                rollout_receipt_paths=[rollout_receipt],
                model_grader_gate_paths=[model_grader_gate],
                review_calibration_paths=[review_calibration],
                reviewed_gate_paths=[reviewed_gate],
                out_path=out,
                created_at="2026-07-03T00:00:00+00:00",
            )
            write_rejection_sampling_gate(out, gate)
            payload = json.loads(out.read_text(encoding="utf-8"))
            payload["gate_path"] = str(out)
            payload["input_artifacts"]["model_grader_gate"][0]["path"] = str(model_grader_gate)
            out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            validation = validate_artifacts(rejection_sampling_gate_paths=[out])

            self.assertFalse(validation["passed"], validation)
            errors = "\n".join(error for target in validation["targets"] for error in target["errors"])
            self.assertIn("rejection_sampling_gate.gate_path must be a safe relative path or redacted placeholder", errors)
            self.assertIn(
                "rejection_sampling_gate.input_artifacts.model_grader_gate[0].path must be a safe relative path or redacted placeholder",
                errors,
            )

    def test_validation_rejects_stale_rejection_sampling_input_ref(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rollout_receipt = self.write_rollout_receipt(root)
            model_grader_gate = self.write_json(root / "model_grader_gate.json", "hfr.model_grader_gate.v1")
            review_calibration = self.write_json(root / "review_calibration.json", "hfr.review_calibration.v1")
            reviewed_gate = self.write_json(root / "reviewed_gate.json", "hfr.reviewed_gate.v1")
            out = root / "rejection_sampling_gate.json"
            gate = build_rejection_sampling_gate(
                rollout_receipt_paths=[rollout_receipt],
                model_grader_gate_paths=[model_grader_gate],
                review_calibration_paths=[review_calibration],
                reviewed_gate_paths=[reviewed_gate],
                out_path=out,
                created_at="2026-07-03T00:00:00+00:00",
            )
            write_rejection_sampling_gate(out, gate)
            self.write_json(model_grader_gate, "hfr.model_grader_gate.v1")
            with model_grader_gate.open("a", encoding="utf-8") as handle:
                handle.write("\n")

            validation = validate_artifacts(rejection_sampling_gate_paths=[out], strict=True)
            self.assertFalse(validation["passed"], validation)
            errors = "\n".join(error for target in validation["targets"] for error in target["errors"])
            self.assertIn("rejection_sampling_gate.input_artifacts.model_grader_gate[0].size_bytes does not match", errors)
            self.assertIn("rejection_sampling_gate.input_artifacts.model_grader_gate[0].sha256 does not match", errors)

    def test_gate_blocks_uncalibrated_or_missing_review_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rollout_receipt = self.write_rollout_receipt(root)
            model_grader_gate = self.write_json(root / "model_grader_gate.json", "hfr.model_grader_gate.v1")
            reviewed_gate = self.write_json(root / "reviewed_gate.json", "hfr.reviewed_gate.v1")

            gate = build_rejection_sampling_gate(
                rollout_receipt_paths=[rollout_receipt],
                model_grader_gate_paths=[model_grader_gate],
                review_calibration_paths=[],
                reviewed_gate_paths=[reviewed_gate],
                created_at="2026-07-03T00:00:00+00:00",
            )

            self.assertFalse(gate["passed"])
            self.assertEqual(gate["readiness"], "blocked")
            self.assertIn("review_calibration_present_and_passing", {check["id"] for check in gate["checks"] if not check["passed"]})
            self.assertFalse(gate["execution_boundary"]["dataset_rows_written"])

    def test_gate_blocks_multiple_review_inputs_from_mixed_dataset_versions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            example = ROOT / "examples" / "agentic_training" / "model_grader"
            first = root / "first"
            second = root / "second"
            shutil.copytree(example, first)
            shutil.copytree(example, second)
            self.regenerate_review_fixture(first)

            completed_labels = second / "review" / "completed_labels.jsonl"
            rows = [json.loads(line) for line in completed_labels.read_text(encoding="utf-8").splitlines() if line]
            rows[0]["notes"] = "Equivalent adjudication with distinct reviewed-dataset provenance."
            completed_labels.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
                encoding="utf-8",
            )
            shutil.rmtree(second / "reviewed")
            self.assertEqual(
                run_cli(
                    [
                        "apply-review",
                        "--review-export",
                        str(second / "review"),
                        "--labels",
                        str(completed_labels),
                        "--out",
                        str(second / "reviewed"),
                    ]
                ),
                0,
            )
            self.assertEqual(
                run_cli(
                    [
                        "review-calibration",
                        "--reviewed-export",
                        str(second / "reviewed"),
                        "--min-comparable-labels",
                        "2",
                        "--min-agreement-rate",
                        "1.0",
                        "--max-disagreements",
                        "0",
                        "--out",
                        str(second / "review_calibration.json"),
                    ]
                ),
                0,
            )
            self.assertEqual(
                run_cli(
                    [
                        "gate-reviewed",
                        "--reviewed-export",
                        str(second / "reviewed"),
                        "--min-reviewed-labels",
                        "2",
                        "--min-accepted",
                        "1",
                        "--min-rejected",
                        "1",
                        "--min-sft",
                        "1",
                        "--min-reward-model",
                        "2",
                        "--min-preferences",
                        "1",
                        "--min-dpo",
                        "1",
                        "--min-medium-or-high-confidence-labels",
                        "2",
                        "--max-needs-review",
                        "0",
                        "--max-low-confidence-labels",
                        "0",
                        "--max-unknown-confidence-labels",
                        "0",
                        "--out",
                        str(second / "reviewed_gate.json"),
                    ]
                ),
                0,
            )
            self.assertEqual(
                run_cli(
                    [
                        "model-grader",
                        "gate",
                        "--dry-run",
                        str(second / "dry_run.json"),
                        "--rubric",
                        str(second / "rubric.json"),
                        "--review-calibration",
                        str(second / "review_calibration.json"),
                        "--min-calibration-agreement-rate",
                        "1.0",
                        "--max-disagreements",
                        "0",
                        "--out",
                        str(second / "passing_gate.json"),
                    ]
                ),
                0,
            )

            gate = build_rejection_sampling_gate(
                rollout_receipt_paths=[self.write_rollout_receipt(root)],
                model_grader_gate_paths=[first / "passing_gate.json", second / "passing_gate.json"],
                review_calibration_paths=[first / "review_calibration.json", second / "review_calibration.json"],
                reviewed_gate_paths=[first / "reviewed_gate.json", second / "reviewed_gate.json"],
                out_path=root / "rejection_sampling_gate.json",
                created_at="2026-07-03T00:00:00+00:00",
            )

            lineage_check = next(check for check in gate["checks"] if check["id"] == "review_dataset_lineage_converges")
            self.assertFalse(gate["passed"])
            self.assertEqual(gate["readiness"], "blocked")
            self.assertFalse(lineage_check["passed"])
            self.assertEqual(len(lineage_check["actual"]["dataset_versions"]), 2)

    def test_gate_rejects_schema_valid_semantically_forged_review_calibration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rollout_receipt = self.write_rollout_receipt(root)
            model_grader_gate = self.write_json(root / "model_grader_gate.json", "hfr.model_grader_gate.v1")
            review_calibration = self.write_json(root / "review_calibration.json", "hfr.review_calibration.v1")
            reviewed_gate = self.write_json(root / "reviewed_gate.json", "hfr.reviewed_gate.v1")
            forged = json.loads(review_calibration.read_text(encoding="utf-8"))
            forged["metrics"]["agreement_count"] = 1
            review_calibration.write_text(json.dumps(forged, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            self.assertTrue(check_schema_file(review_calibration)["passed"])

            gate = build_rejection_sampling_gate(
                rollout_receipt_paths=[rollout_receipt],
                model_grader_gate_paths=[model_grader_gate],
                review_calibration_paths=[review_calibration],
                reviewed_gate_paths=[reviewed_gate],
                out_path=root / "gate.json",
                created_at="2026-07-03T00:00:00+00:00",
            )

            self.assertFalse(gate["passed"])
            self.assertFalse(gate["input_artifacts"]["review_calibration"][0]["exists"])
            self.assertIn(
                "review_calibration_present_and_passing",
                {check["id"] for check in gate["checks"] if not check["passed"]},
            )

    def test_validation_rejects_dataset_side_effect_claims(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rollout_receipt = self.write_rollout_receipt(root)
            model_grader_gate = self.write_json(root / "model_grader_gate.json", "hfr.model_grader_gate.v1")
            review_calibration = self.write_json(root / "review_calibration.json", "hfr.review_calibration.v1")
            reviewed_gate = self.write_json(root / "reviewed_gate.json", "hfr.reviewed_gate.v1")
            out = root / "gate.json"
            gate = build_rejection_sampling_gate(
                rollout_receipt_paths=[rollout_receipt],
                model_grader_gate_paths=[model_grader_gate],
                review_calibration_paths=[review_calibration],
                reviewed_gate_paths=[reviewed_gate],
                out_path=out,
                created_at="2026-07-03T00:00:00+00:00",
            )
            gate["execution_boundary"]["dataset_rows_written"] = True
            write_rejection_sampling_gate(out, gate)

            validation = validate_artifacts(rejection_sampling_gate_paths=[out], strict=True)
            self.assertFalse(validation["passed"], validation)
            errors = "\n".join(error for target in validation["targets"] for error in target["errors"])
            self.assertIn("execution_boundary.dataset_rows_written must be false", errors)

    def test_validation_rejects_forged_provider_and_trainer_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rollout_receipt = self.write_rollout_receipt(root)
            model_grader_gate = self.write_json(root / "model_grader_gate.json", "hfr.model_grader_gate.v1")
            review_calibration = self.write_json(root / "review_calibration.json", "hfr.review_calibration.v1")
            reviewed_gate = self.write_json(root / "reviewed_gate.json", "hfr.reviewed_gate.v1")
            out = root / "gate.json"
            gate = build_rejection_sampling_gate(
                rollout_receipt_paths=[rollout_receipt],
                model_grader_gate_paths=[model_grader_gate],
                review_calibration_paths=[review_calibration],
                reviewed_gate_paths=[reviewed_gate],
                out_path=out,
                created_at="2026-07-03T00:00:00+00:00",
            )
            gate["provider_job_id"] = "job_live"
            gate["checks"][0]["provider_trace_id"] = "trace_live"
            gate["input_artifacts"]["model_grader_gate"][0]["signed_url"] = "https://example.invalid/model-grader-gate.json"
            gate["input_artifacts"]["agentic_rollout_receipt"][0]["provider_rollout_id"] = "rollout_live"
            gate["rollout_summary"]["provider_rollout_count"] = 1
            gate["admission_policy"]["accepts_provider_labels"] = True
            gate["execution_boundary"]["trainer_dataset_written"] = True
            write_rejection_sampling_gate(out, gate)

            schema = check_schema_file(out)
            self.assertFalse(schema["passed"], schema)
            validation = validate_artifacts(rejection_sampling_gate_paths=[out], strict=True)
            self.assertFalse(validation["passed"], validation)
            errors = "\n".join(error for target in validation["targets"] for error in target["errors"])
            self.assertIn("rejection_sampling_gate contains unknown field", errors)
            self.assertIn("rejection_sampling_gate.checks[0] contains unknown field", errors)
            self.assertIn("rejection_sampling_gate.input_artifacts.model_grader_gate[0] contains unknown field", errors)
            self.assertIn("rejection_sampling_gate.admission_policy contains unknown field", errors)

    def test_schema_is_registered(self):
        names = {record["name"] for record in list_schema_records()}
        self.assertIn("rejection_sampling_gate", names)

    def write_rollout_receipt(self, root: Path) -> Path:
        scenario_dir = root / "scenarios"
        scenario_dir.mkdir(parents=True, exist_ok=True)
        scenario = scenario_dir / SCENARIO.name
        shutil.copyfile(SCENARIO, scenario)
        plan_path = root / "rollout_plan.json"
        receipt_path = root / "rollout_receipt.json"
        plan = build_agentic_rollout_plan(
            out_path=plan_path,
            iteration_id="reject-sample",
            scenario_paths=[scenario],
            policies={"baseline": "local/base"},
            max_rollouts=1,
            created_at="2026-07-03T00:00:00+00:00",
        )
        write_agentic_rollout_plan(plan_path, plan)
        receipt = build_agentic_rollout_receipt(
            plan_path=plan_path,
            out_path=receipt_path,
            created_at="2026-07-03T00:00:00+00:00",
        )
        write_agentic_rollout_receipt(receipt_path, receipt)
        return receipt_path

    def write_json(self, path: Path, schema_version: str) -> Path:
        example_dir = ROOT / "examples" / "agentic_training" / "model_grader"
        if not (path.parent / "reviewed" / "reviewed_action_sft.jsonl").is_file():
            shutil.copytree(example_dir / "review", path.parent / "review", dirs_exist_ok=True)
            for filename in ("dry_run.json", "rubric.json"):
                shutil.copyfile(example_dir / filename, path.parent / filename)
            self.regenerate_review_fixture(path.parent)
        sources = {
            "hfr.model_grader_gate.v1": path.parent / "passing_gate.json",
            "hfr.review_calibration.v1": path.parent / "review_calibration.json",
            "hfr.reviewed_gate.v1": path.parent / "reviewed_gate.json",
        }
        source = sources[schema_version]
        if source != path:
            shutil.copyfile(source, path)
        return path

    def regenerate_review_fixture(self, root: Path) -> None:
        reviewed = root / "reviewed"
        if reviewed.exists():
            shutil.rmtree(reviewed)
        self.assertEqual(
            run_cli(
                [
                    "apply-review",
                    "--review-export",
                    str(root / "review"),
                    "--labels",
                    str(root / "review" / "completed_labels.jsonl"),
                    "--out",
                    str(reviewed),
                ]
            ),
            0,
        )
        self.assertEqual(
            run_cli(
                [
                    "review-calibration",
                    "--reviewed-export",
                    str(reviewed),
                    "--min-comparable-labels",
                    "2",
                    "--min-agreement-rate",
                    "1.0",
                    "--max-disagreements",
                    "0",
                    "--out",
                    str(root / "review_calibration.json"),
                ]
            ),
            0,
        )
        self.assertEqual(
            run_cli(
                [
                    "gate-reviewed",
                    "--reviewed-export",
                    str(reviewed),
                    "--min-reviewed-labels",
                    "2",
                    "--min-accepted",
                    "1",
                    "--min-rejected",
                    "1",
                    "--min-sft",
                    "1",
                    "--min-reward-model",
                    "2",
                    "--min-preferences",
                    "1",
                    "--min-dpo",
                    "1",
                    "--min-medium-or-high-confidence-labels",
                    "2",
                    "--max-needs-review",
                    "0",
                    "--max-low-confidence-labels",
                    "0",
                    "--max-unknown-confidence-labels",
                    "0",
                    "--out",
                    str(root / "reviewed_gate.json"),
                ]
            ),
            0,
        )
        self.assertEqual(
            run_cli(
                [
                    "model-grader",
                    "gate",
                    "--dry-run",
                    str(root / "dry_run.json"),
                    "--rubric",
                    str(root / "rubric.json"),
                    "--review-calibration",
                    str(root / "review_calibration.json"),
                    "--min-calibration-agreement-rate",
                    "1.0",
                    "--max-disagreements",
                    "0",
                    "--out",
                    str(root / "passing_gate.json"),
                ]
            ),
            0,
        )


if __name__ == "__main__":
    unittest.main()
