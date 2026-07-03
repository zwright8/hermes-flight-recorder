import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from flightrecorder.dataset_curation import build_dataset_curation_receipt, write_dataset_curation_receipt
from flightrecorder.schema_registry import check_schema_file, list_schema_records
from flightrecorder.validation import validate_artifacts


ROOT = Path(__file__).resolve().parents[1]


class DatasetCurationReceiptTests(unittest.TestCase):
    def test_receipt_admits_training_exports_without_writing_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gate = self.write_rejection_sampling_gate(root / "rejection_sampling_gate.json")
            export_dir = self.write_training_export(root / "training_export")
            out = root / "dataset_curation_receipt.json"

            receipt = build_dataset_curation_receipt(
                rejection_sampling_gate_paths=[gate],
                training_export_paths=[export_dir],
                out_path=out,
                created_at="2026-07-03T00:00:00+00:00",
            )
            write_dataset_curation_receipt(out, receipt)

            schema = check_schema_file(out)
            self.assertTrue(schema["passed"], schema["errors"])
            validation = validate_artifacts(dataset_curation_receipt_paths=[out], strict=True)
            self.assertTrue(validation["passed"], validation)
            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertTrue(payload["passed"], payload["blocked_reasons"])
            self.assertEqual(payload["readiness"], "ready_for_external_trainer_handoff")
            self.assertEqual(payload["curation_summary"]["training_export_count"], 1)
            self.assertEqual(payload["curation_summary"]["curated_rows_written"], 0)
            self.assertFalse(payload["execution_boundary"]["dataset_rows_written"])

    def test_cli_writes_valid_dataset_curation_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gate = self.write_rejection_sampling_gate(root / "rejection_sampling_gate.json")
            export_dir = self.write_training_export(root / "training_export")
            out = root / "receipt.json"

            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "flightrecorder",
                    "dataset-curation-receipt",
                    "--rejection-sampling-gate",
                    str(gate),
                    "--training-export",
                    str(export_dir),
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
            validation = validate_artifacts(dataset_curation_receipt_paths=[out], strict=True)
            self.assertTrue(validation["passed"], validation)

    def test_receipt_blocks_missing_training_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gate = self.write_rejection_sampling_gate(root / "rejection_sampling_gate.json")

            receipt = build_dataset_curation_receipt(
                rejection_sampling_gate_paths=[gate],
                training_export_paths=[],
                created_at="2026-07-03T00:00:00+00:00",
            )

            self.assertFalse(receipt["passed"])
            self.assertEqual(receipt["readiness"], "blocked")
            self.assertIn("training_exports_present", {check["id"] for check in receipt["checks"] if not check["passed"]})
            self.assertFalse(receipt["execution_boundary"]["dataset_rows_written"])

    def test_validation_rejects_dataset_write_claims(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gate = self.write_rejection_sampling_gate(root / "rejection_sampling_gate.json")
            export_dir = self.write_training_export(root / "training_export")
            out = root / "receipt.json"
            receipt = build_dataset_curation_receipt(
                rejection_sampling_gate_paths=[gate],
                training_export_paths=[export_dir],
                out_path=out,
                created_at="2026-07-03T00:00:00+00:00",
            )
            receipt["execution_boundary"]["dataset_rows_written"] = True
            write_dataset_curation_receipt(out, receipt)

            validation = validate_artifacts(dataset_curation_receipt_paths=[out], strict=True)
            self.assertFalse(validation["passed"], validation)
            errors = "\n".join(error for target in validation["targets"] for error in target["errors"])
            self.assertIn("execution_boundary.dataset_rows_written must be false", errors)

    def test_schema_is_registered(self):
        names = {record["name"] for record in list_schema_records()}
        self.assertIn("dataset_curation_receipt", names)

    def write_rejection_sampling_gate(self, path: Path) -> Path:
        path.write_text(
            json.dumps(
                {
                    "schema_version": "hfr.rejection_sampling_gate.v1",
                    "passed": True,
                    "readiness": "ready_for_dataset_curation",
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return path

    def write_training_export(self, path: Path) -> Path:
        path.mkdir(parents=True)
        (path / "manifest.json").write_text(
            json.dumps({"schema_version": "hfr.training_manifest.v1", "passed": True}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return path


if __name__ == "__main__":
    unittest.main()
