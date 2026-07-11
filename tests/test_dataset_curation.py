import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from flightrecorder.dataset_curation import (
    build_dataset_curation_receipt,
    training_export_lineage_status,
    write_dataset_curation_receipt,
)
from flightrecorder.review import review_item_sha256
from flightrecorder.schema_registry import check_schema_file, list_schema_records
from flightrecorder.training import episode_events_sha256
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

    def test_receipt_blocks_training_export_from_unrelated_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gate = self.write_rejection_sampling_gate(
                root / "rejection_sampling_gate.json"
            )
            runs = root / "unrelated_runs"
            export_dir = root / "unrelated_training_export"
            scenario = ROOT / "scenarios" / "email_reply_completion_good.json"
            for command in (
                [
                    sys.executable,
                    "-m",
                    "flightrecorder",
                    "run",
                    "--scenario",
                    str(scenario),
                    "--out",
                    str(runs / "email"),
                ],
                [
                    sys.executable,
                    "-m",
                    "flightrecorder",
                    "export-rl",
                    "--runs",
                    str(runs),
                    "--out",
                    str(export_dir),
                ],
            ):
                completed = subprocess.run(
                    command,
                    cwd=ROOT,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                )
                self.assertEqual(
                    completed.returncode,
                    0,
                    completed.stderr + completed.stdout,
                )

            out = root / "receipt.json"
            receipt = build_dataset_curation_receipt(
                rejection_sampling_gate_paths=[gate],
                training_export_paths=[export_dir],
                out_path=out,
                created_at="2026-07-03T00:00:00+00:00",
            )
            lineage = next(
                check
                for check in receipt["checks"]
                if check["id"] == "training_exports_cover_admitted_lineage"
            )
            self.assertFalse(receipt["passed"])
            self.assertFalse(lineage["passed"])
            self.assertGreater(lineage["actual"]["reviewed_item_missing_count"], 0)

            lineage["passed"] = True
            lineage["summary"] = (
                "training_exports_cover_admitted_lineage: passed=True"
            )
            receipt["passed"] = True
            receipt["failed_check_count"] = 0
            receipt["blocked_reasons"] = []
            receipt["readiness"] = "ready_for_external_trainer_handoff"
            receipt["recommendation"] = "run_training_gate_and_trainer_preflight"
            write_dataset_curation_receipt(out, receipt)

            validation = validate_artifacts(
                dataset_curation_receipt_paths=[out],
                strict=True,
            )
            self.assertFalse(validation["passed"], validation)
            errors = "\n".join(
                error
                for target in validation["targets"]
                for error in target["errors"]
            )
            self.assertIn(
                "training_exports_cover_admitted_lineage check must match",
                errors,
            )

    def test_lineage_rejects_same_count_mutated_episode_events(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gate = self.write_rejection_sampling_gate(
                root / "rejection_sampling_gate.json"
            )
            export_dir = self.write_training_export(root / "training_export")
            self.prepare_lineage_fixture(root, export_dir)
            episodes_path = export_dir / "episodes.jsonl"
            episodes = self.read_jsonl(episodes_path)
            episode = next(
                row
                for row in episodes
                if row["episode_id"] == "prompt_injection_good"
            )
            episode["events"][0]["text"] = "mutated behavior with unchanged event count"
            self.write_jsonl(episodes_path, episodes)

            passed, actual = training_export_lineage_status([gate], [export_dir])
            replay_passed, replay_actual = training_export_lineage_status(
                [gate], [export_dir]
            )

            self.assertFalse(passed)
            self.assertEqual((passed, actual), (replay_passed, replay_actual))
            self.assertIn(
                "prompt_injection_good",
                actual["mismatched_reviewed_item_ids"],
            )
            self.assertEqual(actual["training_exports"][0]["export_index"], 0)
            self.assertGreater(
                actual["training_exports"][0]["reviewed_item_mismatch_count"],
                0,
            )

    def test_lineage_preserves_human_rejection_of_scorecard_positive_episode(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gate = self.write_rejection_sampling_gate(
                root / "rejection_sampling_gate.json"
            )
            export_dir = self.write_training_export(root / "training_export")
            self.prepare_lineage_fixture(root, export_dir)
            labels_path = root / "model_grader" / "reviewed" / "reviewed_labels.jsonl"
            labels = self.read_jsonl(labels_path)
            accepted = next(
                row
                for row in labels
                if row["review_item_id"] == "prompt_injection_good"
            )
            self.assertTrue(accepted["scorecard"]["passed"])
            accepted["human_label"] = "reject"
            self.write_jsonl(labels_path, labels)

            passed, actual = training_export_lineage_status([gate], [export_dir])

            self.assertFalse(passed)
            export_actual = actual["training_exports"][0]
            mismatch = next(
                row
                for row in export_actual["reviewed_label_mismatches"]
                if row["review_item_id"] == "prompt_injection_good"
            )
            self.assertEqual(mismatch["human_label"], "reject")
            self.assertIn("negative_label_forbids_sft_rows", mismatch["reasons"])
            self.assertIn(
                "negative_label_requires_one_negative_reward_model_row",
                mismatch["reasons"],
            )
            self.assertIn(
                "negative_label_forbids_chosen_dpo_role",
                mismatch["reasons"],
            )

    def test_lineage_requires_every_export_to_cover_every_admitted_item(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gate = self.write_rejection_sampling_gate(
                root / "rejection_sampling_gate.json"
            )
            first = self.write_training_export(root / "training_export_a")
            second = self.write_training_export(root / "training_export_b")
            self.prepare_lineage_fixture(root, first)
            first_episodes = self.read_jsonl(first / "episodes.jsonl")
            second_episodes = self.read_jsonl(second / "episodes.jsonl")
            self.write_jsonl(
                first / "episodes.jsonl",
                [
                    row
                    for row in first_episodes
                    if row["episode_id"] != "prompt_injection_bad"
                ],
            )
            self.write_jsonl(
                second / "episodes.jsonl",
                [
                    row
                    for row in second_episodes
                    if row["episode_id"] != "prompt_injection_good"
                ],
            )

            passed, actual = training_export_lineage_status(
                [gate], [first, second]
            )

            self.assertFalse(passed)
            self.assertEqual(actual["training_export_count"], 2)
            self.assertEqual(
                actual["training_exports"][0]["missing_reviewed_item_ids"],
                ["prompt_injection_bad"],
            )
            self.assertEqual(
                actual["training_exports"][1]["missing_reviewed_item_ids"],
                ["prompt_injection_good"],
            )
            self.assertEqual(actual["reviewed_item_missing_count"], 2)

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

    def test_receipt_rejects_schema_valid_semantically_forged_sampling_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gate = self.write_rejection_sampling_gate(root / "rejection_sampling_gate.json")
            forged = json.loads(gate.read_text(encoding="utf-8"))
            forged["checks"][0]["passed"] = False
            gate.write_text(json.dumps(forged, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            self.assertTrue(check_schema_file(gate)["passed"])
            export_dir = self.write_training_export(root / "training_export")

            receipt = build_dataset_curation_receipt(
                rejection_sampling_gate_paths=[gate],
                training_export_paths=[export_dir],
                out_path=root / "receipt.json",
                created_at="2026-07-03T00:00:00+00:00",
            )

            self.assertFalse(receipt["passed"])
            self.assertFalse(receipt["input_artifacts"]["rejection_sampling_gate"][0]["exists"])
            self.assertIn("rejection_sampling_gate_ready", {check["id"] for check in receipt["checks"] if not check["passed"]})

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

    def test_validation_rejects_transitively_stale_reviewed_source(self):
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
            reviewed_sft = root / "model_grader" / "reviewed" / "reviewed_sft.jsonl"
            reviewed_sft.write_bytes(reviewed_sft.read_bytes() + b"\n")

            validation = validate_artifacts(dataset_curation_receipt_paths=[out], strict=True)

            self.assertFalse(validation["passed"], validation)
            errors = "\n".join(error for target in validation["targets"] for error in target["errors"])
            self.assertIn("must remain semantically ready for role 'rejection_sampling_gate'", errors)

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
        example_root = ROOT / "examples" / "agentic_training"
        shutil.copytree(example_root / "model_grader", path.parent / "model_grader", dirs_exist_ok=True)
        shutil.copytree(example_root / "rollouts", path.parent / "rollouts", dirs_exist_ok=True)
        payload = json.loads((example_root / "rejection_sampling_gate.json").read_text(encoding="utf-8"))
        payload["passed"] = passed
        payload["readiness"] = "ready_for_dataset_curation" if passed else "blocked"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return path

    def write_training_export(self, path: Path) -> Path:
        shutil.copytree(ROOT / "examples" / "agentic_training" / "training_export", path)
        return path

    def prepare_lineage_fixture(self, root: Path, export_dir: Path) -> None:
        episodes = {
            row["episode_id"]: row
            for row in self.read_jsonl(export_dir / "episodes.jsonl")
        }
        items_path = root / "model_grader" / "reviewed" / "provenance" / "review_items.jsonl"
        labels_path = root / "model_grader" / "reviewed" / "reviewed_labels.jsonl"
        items = self.read_jsonl(items_path)
        item_hashes: dict[str, str] = {}
        for item in items:
            episode = episodes[item["episode_id"]]
            item["episode_events_sha256"] = episode_events_sha256(episode["events"])
            item["review_item_sha256"] = review_item_sha256(item)
            item_hashes[item["review_item_id"]] = item["review_item_sha256"]
        labels = self.read_jsonl(labels_path)
        for label in labels:
            label["review_item_sha256"] = item_hashes[label["review_item_id"]]
        self.write_jsonl(items_path, items)
        self.write_jsonl(labels_path, labels)

    @staticmethod
    def read_jsonl(path: Path) -> list[dict]:
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    @staticmethod
    def write_jsonl(path: Path, rows: list[dict]) -> None:
        path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )


if __name__ == "__main__":
    unittest.main()
