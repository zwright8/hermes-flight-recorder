import json
import shutil
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
    def test_receipt_rejects_invalid_and_semantically_failed_training_manifests(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gate = self.write_rejection_sampling_gate(root / "rejection_sampling_gate.json")
            export_dir = self.write_training_export(root / "training_export")
            manifest = export_dir / "manifest.json"
            manifest.write_text('{"schema_version":"hfr.rl.manifest.v1"}\n', encoding="utf-8")

            invalid = build_dataset_curation_receipt(
                rejection_sampling_gate_paths=[gate],
                training_export_paths=[export_dir],
                out_path=root / "invalid_receipt.json",
                created_at="2026-07-03T00:00:00+00:00",
            )
            self.assertFalse(invalid["passed"])
            self.assertFalse(invalid["input_artifacts"]["training_export"][0]["exists"])

            shutil.rmtree(export_dir)
            export_dir = self.write_training_export(root / "training_export")
            payload = json.loads((export_dir / "manifest.json").read_text(encoding="utf-8"))
            payload["redaction_status"]["passed"] = False
            payload["redaction_status"]["unredacted_secret_like_finding_count"] = 1
            (export_dir / "manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            failed = build_dataset_curation_receipt(
                rejection_sampling_gate_paths=[gate],
                training_export_paths=[export_dir],
                out_path=root / "failed_receipt.json",
                created_at="2026-07-03T00:00:00+00:00",
            )
            self.assertFalse(failed["passed"])
            self.assertFalse(failed["input_artifacts"]["training_export"][0]["exists"])

    def test_receipt_rejects_symlinked_training_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gate = self.write_rejection_sampling_gate(root / "rejection_sampling_gate.json")
            export_dir = self.write_training_export(root / "training_export")
            linked = root / "linked_export"
            try:
                linked.symlink_to(export_dir.name, target_is_directory=True)
            except OSError as exc:
                self.skipTest(f"symlinks unavailable: {exc}")

            receipt = build_dataset_curation_receipt(
                rejection_sampling_gate_paths=[gate],
                training_export_paths=[linked],
                out_path=root / "symlinked_receipt.json",
                created_at="2026-07-03T00:00:00+00:00",
            )
            self.assertFalse(receipt["passed"])
            self.assertFalse(receipt["input_artifacts"]["training_export"][0]["exists"])

    def test_committed_agentic_training_curation_receipt_binds_real_export_without_writes(self):
        example_root = ROOT / "examples" / "agentic_training"
        receipt_path = example_root / "dataset_curation_receipt.json"
        export_dir = example_root / "training_export"
        manifest = json.loads((export_dir / "manifest.json").read_text(encoding="utf-8"))
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))

        self.assertEqual(manifest["schema_version"], "hfr.rl.manifest.v1")
        self.assertEqual(manifest["episode_count"], 7)
        self.assertEqual(manifest["sft_count"], 2)
        self.assertEqual(manifest["dpo_count"], 2)
        self.assertTrue(receipt["passed"], receipt["blocked_reasons"])
        self.assertEqual(receipt["readiness"], "ready_for_external_trainer_handoff")
        self.assertEqual(receipt["input_artifacts"]["rejection_sampling_gate"][0]["path"], "rejection_sampling_gate.json")
        self.assertEqual(receipt["input_artifacts"]["training_export"][0]["path"], "training_export")
        self.assertEqual(receipt["input_artifacts"]["training_export"][0]["manifest_path"], "training_export/manifest.json")
        self.assertEqual(receipt["curation_summary"]["curated_rows_written"], 0)
        self.assertFalse(receipt["curation_summary"]["dataset_registry_updated"])
        self.assertFalse(receipt["execution_boundary"]["dataset_rows_written"])
        self.assertFalse(receipt["execution_boundary"]["weights_updated_by_flight_recorder"])

        schema = check_schema_file(receipt_path)
        validation = validate_artifacts(
            training_export_dir=export_dir,
            dataset_curation_receipt_paths=[receipt_path],
            strict=True,
        )

        self.assertTrue(schema["passed"], schema["errors"])
        self.assertTrue(validation["passed"], validation)

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
            self.assertEqual(payload["receipt_path"], "dataset_curation_receipt.json")
            self.assertTrue(payload["passed"], payload["blocked_reasons"])
            self.assertEqual(payload["readiness"], "ready_for_external_trainer_handoff")
            self.assertEqual(payload["curation_summary"]["training_export_count"], 1)
            self.assertEqual(payload["curation_summary"]["curated_rows_written"], 0)
            self.assertFalse(payload["execution_boundary"]["dataset_rows_written"])

            preserved_out = root / "preserved_dataset_curation_receipt.json"
            preserved_receipt = build_dataset_curation_receipt(
                rejection_sampling_gate_paths=[gate],
                training_export_paths=[export_dir],
                out_path=preserved_out,
                preserve_paths=True,
                created_at="2026-07-03T00:00:00+00:00",
            )
            write_dataset_curation_receipt(preserved_out, preserved_receipt)
            preserved_payload = json.loads(preserved_out.read_text(encoding="utf-8"))
            self.assertNotIn(str(root), json.dumps(preserved_payload, sort_keys=True))
            self.assertEqual(preserved_payload["input_artifacts"]["rejection_sampling_gate"][0]["path"], "rejection_sampling_gate.json")
            self.assertEqual(preserved_payload["input_artifacts"]["training_export"][0]["path"], "training_export")
            self.assertEqual(preserved_payload["input_artifacts"]["training_export"][0]["manifest_path"], "training_export/manifest.json")
            preserved_validation = validate_artifacts(dataset_curation_receipt_paths=[preserved_out], strict=True)
            self.assertTrue(preserved_validation["passed"], preserved_validation)

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

    def test_receipt_redacts_unreplayable_external_refs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            external_root = root / "external"
            public_root = root / "public"
            external_root.mkdir(parents=True)
            gate = self.write_rejection_sampling_gate(external_root / "rejection_sampling_gate.json")
            export_dir = self.write_training_export(external_root / "training_export")
            out = public_root / "dataset_curation_receipt.json"
            receipt = build_dataset_curation_receipt(
                rejection_sampling_gate_paths=[gate],
                training_export_paths=[export_dir],
                out_path=out,
                preserve_paths=True,
                created_at="2026-07-03T00:00:00+00:00",
            )
            write_dataset_curation_receipt(out, receipt)

            payload = json.loads(out.read_text(encoding="utf-8"))
            serialized = json.dumps(payload, sort_keys=True)
            self.assertNotIn(str(external_root), serialized)
            self.assertFalse(payload["passed"])
            self.assertEqual(payload["input_artifacts"]["rejection_sampling_gate"][0]["path"], "<redacted:rejection_sampling_gate.json>")
            self.assertFalse(payload["input_artifacts"]["rejection_sampling_gate"][0]["exists"])
            self.assertEqual(payload["input_artifacts"]["training_export"][0]["path"], "<redacted:training_export>")
            self.assertFalse(payload["input_artifacts"]["training_export"][0]["exists"])

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

    def test_validation_rejects_forged_trainer_handoff_fields(self):
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
            receipt["provider_job_id"] = "job_live"
            receipt["checks"][0]["provider_trace_id"] = "trace_live"
            receipt["input_artifacts"]["cloud_training_launch_receipt"] = []
            receipt["input_artifacts"]["training_export"][0]["signed_url"] = "https://example.invalid/training-export"
            receipt["curation_summary"]["provider_dataset_rows_written"] = 12
            receipt["trainer_handoff"]["provider_dataset_uri"] = "s3://example-bucket/training.jsonl"
            receipt["execution_boundary"]["cloud_job_id"] = "job_live"
            write_dataset_curation_receipt(out, receipt)

            schema = check_schema_file(out)
            self.assertFalse(schema["passed"], schema)
            validation = validate_artifacts(dataset_curation_receipt_paths=[out], strict=True)
            self.assertFalse(validation["passed"], validation)
            errors = "\n".join(error for target in validation["targets"] for error in target["errors"])
            self.assertIn("dataset_curation_receipt contains unknown field", errors)
            self.assertIn("dataset_curation_receipt.checks[0] contains unknown field", errors)
            self.assertIn("dataset_curation_receipt.input_artifacts contains unknown field", errors)
            self.assertIn("dataset_curation_receipt.trainer_handoff contains unknown field", errors)

    def test_validation_rejects_absolute_input_artifact_refs(self):
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
            write_dataset_curation_receipt(out, receipt)
            payload = json.loads(out.read_text(encoding="utf-8"))
            payload["receipt_path"] = str(out.resolve())
            payload["input_artifacts"]["rejection_sampling_gate"][0]["path"] = str(gate.resolve())
            payload["input_artifacts"]["training_export"][0]["path"] = str(export_dir.resolve())
            payload["input_artifacts"]["training_export"][0]["manifest_path"] = str((export_dir / "manifest.json").resolve())
            out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            validation = validate_artifacts(dataset_curation_receipt_paths=[out], strict=True)

            self.assertFalse(validation["passed"], validation)
            errors = "\n".join(error for target in validation["targets"] for error in target["errors"])
            self.assertIn("dataset_curation_receipt.receipt_path must be a safe relative path or redacted placeholder", errors)
            self.assertIn(
                "dataset_curation_receipt.input_artifacts.rejection_sampling_gate[0].path must be a safe relative path or redacted placeholder",
                errors,
            )
            self.assertIn(
                "dataset_curation_receipt.input_artifacts.training_export[0].path must be a safe relative path or redacted placeholder",
                errors,
            )
            self.assertIn(
                "dataset_curation_receipt.input_artifacts.training_export[0].manifest_path must be a safe relative path or redacted placeholder",
                errors,
            )

    def test_validation_rejects_stale_rejection_sampling_gate_ref(self):
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
            write_dataset_curation_receipt(out, receipt)
            self.write_rejection_sampling_gate(gate, passed=False)

            validation = validate_artifacts(dataset_curation_receipt_paths=[out], strict=True)

            self.assertFalse(validation["passed"], validation)
            errors = "\n".join(error for target in validation["targets"] for error in target["errors"])
            self.assertIn("dataset_curation_receipt.input_artifacts.rejection_sampling_gate[0].size_bytes does not match", errors)
            self.assertIn("dataset_curation_receipt.input_artifacts.rejection_sampling_gate[0].sha256 does not match", errors)

    def test_validation_rejects_stale_training_export_manifest_ref(self):
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
            write_dataset_curation_receipt(out, receipt)
            (export_dir / "manifest.json").write_text(
                json.dumps({"schema_version": "hfr.training_manifest.v1", "passed": False, "reason": "changed"}, indent=2, sort_keys=True)
                + "\n",
                encoding="utf-8",
            )

            validation = validate_artifacts(dataset_curation_receipt_paths=[out], strict=True)

            self.assertFalse(validation["passed"], validation)
            errors = "\n".join(error for target in validation["targets"] for error in target["errors"])
            self.assertIn("dataset_curation_receipt.input_artifacts.training_export[0].manifest_size_bytes does not match", errors)
            self.assertIn("dataset_curation_receipt.input_artifacts.training_export[0].manifest_sha256 does not match", errors)

    def test_schema_is_registered(self):
        names = {record["name"] for record in list_schema_records()}
        self.assertIn("dataset_curation_receipt", names)

    def write_rejection_sampling_gate(self, path: Path, *, passed: bool = True) -> Path:
        payload = json.loads(
            (ROOT / "examples" / "agentic_training" / "rejection_sampling_gate.json").read_text(encoding="utf-8")
        )
        payload["passed"] = passed
        payload["readiness"] = "ready_for_dataset_curation" if passed else "blocked"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return path

    def write_training_export(self, path: Path) -> Path:
        shutil.copytree(ROOT / "examples" / "agentic_training" / "training_export", path)
        return path


if __name__ == "__main__":
    unittest.main()
