from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

from flightrecorder.cli import main
from flightrecorder import cloud_training_completion as completion_module
from flightrecorder.cloud_training_completion import (
    CloudTrainingCompletionError,
    build_cloud_training_completion_receipt,
    cloud_training_completion_digest,
    write_cloud_training_completion_receipt,
)
from flightrecorder.schema_registry import check_schema_contract, check_schema_file
from flightrecorder.source_contract import inspect_artifact_source
from flightrecorder.validation import (
    validate_agentic_training_result,
    validate_cloud_training_completion_receipt,
)
from tests.agentic_loop_fixtures import (
    _write_candidate_training_result,
    write_cloud_completion_fixture,
)


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_ID = "local/mock-candidate"


def run_cli(args: list[str]) -> int:
    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        return main(args)


class CloudTrainingCompletionTests(unittest.TestCase):
    def test_committed_completion_receipt_replays_and_is_schema_valid(self):
        path = (
            ROOT
            / "examples"
            / "agentic_training"
            / "cloud_training_completion_receipt.json"
        )
        receipt = json.loads(path.read_text(encoding="utf-8"))

        self.assertTrue(check_schema_file(path)["passed"])
        validation = validate_cloud_training_completion_receipt(path)
        self.assertEqual(validation.errors, [])
        self.assertEqual(validation.warnings, [])
        self.assertTrue(receipt["passed"])
        self.assertEqual(receipt["execution"]["status"], "completed")
        self.assertTrue(
            receipt["governance"]["cloud_training_completion_claims_allowed"]
        )
        self.assertEqual(
            receipt["identity"]["digest_sha256"],
            cloud_training_completion_digest(receipt),
        )
        self.assertTrue(receipt["execution_boundary"]["import_only"])
        for key, value in receipt["execution_boundary"].items():
            if key.endswith("_by_flight_recorder"):
                self.assertFalse(value, key)

    def test_cli_imports_completion_without_provider_side_effects(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result_path = _write_candidate_training_result(root, CANDIDATE_ID)
            completion_path = write_cloud_completion_fixture(
                root,
                result_path,
                CANDIDATE_ID,
            )
            completion_path.unlink()
            cloud_root = root / "cloud_training"

            code = run_cli(
                [
                    "cloud-training",
                    "import-completion",
                    "--launch-plan",
                    str(cloud_root / "launch_plan.json"),
                    "--launch-receipt",
                    str(cloud_root / "launch_receipt.json"),
                    "--status-receipt",
                    str(cloud_root / "status_receipt.json"),
                    "--runner-metadata",
                    str(cloud_root / "runner_metadata.json"),
                    "--raw-provider-result",
                    str(cloud_root / "raw_provider_result.json"),
                    "--output-artifact-manifest",
                    str(result_path),
                    "--created-at",
                    "2026-07-03T00:30:00+00:00",
                    "--out",
                    str(completion_path),
                ]
            )

            self.assertEqual(code, 0)
            payload = json.loads(completion_path.read_text(encoding="utf-8"))
            self.assertTrue(payload["passed"])
            self.assertFalse(
                payload["execution_boundary"][
                    "provider_api_called_by_flight_recorder"
                ]
            )
            self.assertFalse(
                payload["execution_boundary"][
                    "cloud_job_started_by_flight_recorder"
                ]
            )

    def test_cli_reports_unreplayable_inputs_without_a_traceback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out_dir = root / "receipts"
            out_dir.mkdir()
            stderr = StringIO()

            with redirect_stdout(StringIO()), redirect_stderr(stderr):
                with self.assertRaises(SystemExit) as raised:
                    main(
                        [
                            "cloud-training",
                            "import-completion",
                            "--launch-plan",
                            str(root / "missing-launch-plan.json"),
                            "--launch-receipt",
                            str(root / "missing-launch-receipt.json"),
                            "--status-receipt",
                            str(root / "missing-status-receipt.json"),
                            "--runner-metadata",
                            str(root / "missing-runner-metadata.json"),
                            "--raw-provider-result",
                            str(root / "missing-provider-result.json"),
                            "--output-artifact-manifest",
                            str(root / "missing-output-manifest.json"),
                            "--out",
                            str(out_dir / "completion.json"),
                        ]
                    )

            self.assertEqual(raised.exception.code, 2)
            self.assertIn("flightrecorder: error:", stderr.getvalue())
            self.assertNotIn("Traceback", stderr.getvalue())

    def test_valid_failed_execution_remains_auditable_but_blocks_claims(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result_path = _write_candidate_training_result(root, CANDIDATE_ID)
            completion_path = write_cloud_completion_fixture(
                root,
                result_path,
                CANDIDATE_ID,
                status="failed",
            )
            receipt = json.loads(completion_path.read_text(encoding="utf-8"))

            self.assertTrue(receipt["passed"])
            self.assertTrue(receipt["integrity"]["passed"])
            self.assertEqual(receipt["execution"]["status"], "failed")
            self.assertEqual(receipt["governance"]["readiness"], "blocked")
            self.assertFalse(
                receipt["governance"]["cloud_training_completion_claims_allowed"]
            )
            self.assertEqual(
                validate_cloud_training_completion_receipt(completion_path).errors,
                [],
            )
            inspection = inspect_artifact_source(
                completion_path,
                "cloud_training_completion_receipt",
            )
            self.assertTrue(inspection["ready"])

    def test_current_source_mutation_invalidates_deterministic_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result_path = _write_candidate_training_result(root, CANDIDATE_ID)
            completion_path = write_cloud_completion_fixture(
                root,
                result_path,
                CANDIDATE_ID,
            )
            raw_path = root / "cloud_training" / "raw_provider_result.json"
            raw_path.write_text('{"mutated":true}\n', encoding="utf-8")

            validation = validate_cloud_training_completion_receipt(completion_path)

            self.assertTrue(validation.errors)
            self.assertTrue(
                any(
                    "raw_provider_result.sha256" in error
                    or "replay imported sources" in error
                    for error in validation.errors
                ),
                validation.errors,
            )

    def test_completed_candidate_mismatch_fails_integrity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result_path = _write_candidate_training_result(root, CANDIDATE_ID)
            write_cloud_completion_fixture(
                root,
                result_path,
                CANDIDATE_ID,
            )
            metadata_path = root / "cloud_training" / "runner_metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["candidate_model_id"] = "local/wrong-candidate"
            metadata_path.write_text(
                json.dumps(metadata, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            receipt = build_cloud_training_completion_receipt(
                launch_plan_path=root / "cloud_training" / "launch_plan.json",
                launch_receipt_path=root / "cloud_training" / "launch_receipt.json",
                status_receipt_path=root / "cloud_training" / "status_receipt.json",
                runner_metadata_path=metadata_path,
                raw_provider_result_path=root
                / "cloud_training"
                / "raw_provider_result.json",
                output_artifact_manifest_path=result_path,
                out_path=root / "mismatched_completion.json",
                created_at="2026-07-03T00:30:00+00:00",
            )

            self.assertFalse(receipt["passed"])
            failed = {
                check["id"]
                for check in receipt["integrity"]["checks"]
                if not check["passed"]
            }
            self.assertIn("completed_candidate_matches", failed)
            self.assertFalse(
                receipt["governance"]["cloud_training_completion_claims_allowed"]
            )
            self.assertTrue(
                check_schema_contract(
                    receipt,
                    name_or_id="cloud_training_completion_receipt",
                )["passed"]
            )

    def test_runner_finish_before_start_is_rejected_before_publication(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result_path = _write_candidate_training_result(root, CANDIDATE_ID)
            write_cloud_completion_fixture(root, result_path, CANDIDATE_ID)
            metadata_path = root / "cloud_training" / "runner_metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["finished_at"] = "2026-07-03T00:09:59+00:00"
            metadata_path.write_text(
                json.dumps(metadata, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                CloudTrainingCompletionError,
                "finished_at must not precede started_at",
            ):
                build_cloud_training_completion_receipt(
                    launch_plan_path=root / "cloud_training" / "launch_plan.json",
                    launch_receipt_path=root / "cloud_training" / "launch_receipt.json",
                    status_receipt_path=root / "cloud_training" / "status_receipt.json",
                    runner_metadata_path=metadata_path,
                    raw_provider_result_path=root
                    / "cloud_training"
                    / "raw_provider_result.json",
                    output_artifact_manifest_path=result_path,
                    out_path=root / "invalid-time-completion.json",
                    created_at="2026-07-03T00:30:00+00:00",
                )

    def test_unsafe_runner_metadata_is_not_written_or_echoed(self):
        unsafe_updates = (
            {"provider_job_id": "sk-abcdefgh123456"},
            {"provider_job_id": "C:\\Users\\example\\private-job"},
            {"provider_job_id": "bad\x00job"},
            {"provider_id": "", "status": "bogus"},
            {
                "status": "failed",
                "terminal": True,
                "exit_code": 1,
                "failure": {
                    "class": "provider",
                    "message": "/Users/example/private/provider failure",
                },
            },
        )
        for updates in unsafe_updates:
            with self.subTest(updates=updates), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                result_path = _write_candidate_training_result(root, CANDIDATE_ID)
                completion_path = write_cloud_completion_fixture(
                    root,
                    result_path,
                    CANDIDATE_ID,
                )
                completion_path.unlink()
                cloud_root = root / "cloud_training"
                metadata_path = cloud_root / "runner_metadata.json"
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                metadata.update(updates)
                metadata_path.write_text(
                    json.dumps(metadata, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                stderr = StringIO()

                with redirect_stdout(StringIO()), redirect_stderr(stderr):
                    with self.assertRaises(SystemExit) as raised:
                        main(
                            [
                                "cloud-training",
                                "import-completion",
                                "--launch-plan",
                                str(cloud_root / "launch_plan.json"),
                                "--launch-receipt",
                                str(cloud_root / "launch_receipt.json"),
                                "--status-receipt",
                                str(cloud_root / "status_receipt.json"),
                                "--runner-metadata",
                                str(metadata_path),
                                "--raw-provider-result",
                                str(cloud_root / "raw_provider_result.json"),
                                "--output-artifact-manifest",
                                str(result_path),
                                "--out",
                                str(completion_path),
                            ]
                        )

                self.assertEqual(raised.exception.code, 2)
                self.assertFalse(completion_path.exists())
                self.assertNotIn("sk-abcdefgh123456", stderr.getvalue())
                self.assertNotIn("/Users/example", stderr.getvalue())
                self.assertNotIn("Traceback", stderr.getvalue())

    def test_secret_like_source_basenames_are_not_public_safe(self):
        self.assertFalse(
            completion_module._safe_relative_path("api_key=private-value.json")
        )
        self.assertEqual(
            completion_module._public_basename(
                Path("api_key=private-value.json")
            ),
            "artifact.json",
        )

    def test_completed_execution_requires_external_provider_activity(self):
        required_observations = (
            ("external_provider_api_called", "completed_external_provider_api_called"),
            ("external_cloud_job_started", "completed_external_cloud_job_started"),
            (
                "external_artifacts_downloaded",
                "completed_external_artifacts_downloaded",
            ),
        )
        for field, check_id in required_observations:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                result_path = _write_candidate_training_result(root, CANDIDATE_ID)
                write_cloud_completion_fixture(root, result_path, CANDIDATE_ID)
                cloud_root = root / "cloud_training"
                metadata_path = cloud_root / "runner_metadata.json"
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                metadata["side_effects"][field] = False
                metadata_path.write_text(
                    json.dumps(metadata, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )

                receipt = build_cloud_training_completion_receipt(
                    launch_plan_path=cloud_root / "launch_plan.json",
                    launch_receipt_path=cloud_root / "launch_receipt.json",
                    status_receipt_path=cloud_root / "status_receipt.json",
                    runner_metadata_path=metadata_path,
                    raw_provider_result_path=cloud_root / "raw_provider_result.json",
                    output_artifact_manifest_path=result_path,
                    out_path=root / "missing-external-activity-completion.json",
                    created_at="2026-07-03T00:30:00+00:00",
                )

                failed = {
                    check["id"]
                    for check in receipt["integrity"]["checks"]
                    if not check["passed"]
                }
                self.assertFalse(receipt["passed"])
                self.assertIn(check_id, failed)
                self.assertFalse(
                    receipt["governance"][
                        "cloud_training_completion_claims_allowed"
                    ]
                )
                self.assertTrue(
                    check_schema_contract(
                        receipt,
                        name_or_id="cloud_training_completion_receipt",
                    )["passed"]
                )

    def test_cli_rejects_invalid_or_pre_execution_receipt_timestamps(self):
        for created_at in (
            "not-a-timestamp",
            "2026-07-03T00:19:59+00:00",
            "2026-07-03T00:30:00",
        ):
            with self.subTest(created_at=created_at), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                result_path = _write_candidate_training_result(root, CANDIDATE_ID)
                completion_path = write_cloud_completion_fixture(
                    root,
                    result_path,
                    CANDIDATE_ID,
                )
                completion_path.unlink()
                cloud_root = root / "cloud_training"
                stderr = StringIO()

                with redirect_stdout(StringIO()), redirect_stderr(stderr):
                    with self.assertRaises(SystemExit) as raised:
                        main(
                            [
                                "cloud-training",
                                "import-completion",
                                "--launch-plan",
                                str(cloud_root / "launch_plan.json"),
                                "--launch-receipt",
                                str(cloud_root / "launch_receipt.json"),
                                "--status-receipt",
                                str(cloud_root / "status_receipt.json"),
                                "--runner-metadata",
                                str(cloud_root / "runner_metadata.json"),
                                "--raw-provider-result",
                                str(cloud_root / "raw_provider_result.json"),
                                "--output-artifact-manifest",
                                str(result_path),
                                "--created-at",
                                created_at,
                                "--out",
                                str(completion_path),
                            ]
                        )

                self.assertEqual(raised.exception.code, 2)
                self.assertFalse(completion_path.exists())
                self.assertIn("flightrecorder: error:", stderr.getvalue())
                self.assertNotIn("Traceback", stderr.getvalue())

    def test_runner_execution_cannot_predate_launch_handoff(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result_path = _write_candidate_training_result(root, CANDIDATE_ID)
            write_cloud_completion_fixture(root, result_path, CANDIDATE_ID)
            cloud_root = root / "cloud_training"
            metadata_path = cloud_root / "runner_metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["started_at"] = "2020-01-01T00:00:00+00:00"
            metadata["finished_at"] = "2020-01-01T00:10:00+00:00"
            metadata_path.write_text(
                json.dumps(metadata, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            receipt = build_cloud_training_completion_receipt(
                launch_plan_path=cloud_root / "launch_plan.json",
                launch_receipt_path=cloud_root / "launch_receipt.json",
                status_receipt_path=cloud_root / "status_receipt.json",
                runner_metadata_path=metadata_path,
                raw_provider_result_path=cloud_root / "raw_provider_result.json",
                output_artifact_manifest_path=result_path,
                out_path=root / "pre-launch-run-completion.json",
                created_at="2026-07-03T00:30:00+00:00",
            )

            failed = {
                check["id"]
                for check in receipt["integrity"]["checks"]
                if not check["passed"]
            }
            self.assertFalse(receipt["passed"])
            self.assertIn("runner_started_after_launch_handoff", failed)
            self.assertFalse(
                receipt["governance"]["cloud_training_completion_claims_allowed"]
            )

    def test_cli_rejects_secret_like_output_basename_without_echoing_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result_path = _write_candidate_training_result(root, CANDIDATE_ID)
            write_cloud_completion_fixture(root, result_path, CANDIDATE_ID)
            cloud_root = root / "cloud_training"
            unsafe_name = "api_key=sk-abcdefgh123456.json"
            unsafe_path = root / unsafe_name
            stderr = StringIO()

            with redirect_stdout(StringIO()), redirect_stderr(stderr):
                with self.assertRaises(SystemExit) as raised:
                    main(
                        [
                            "cloud-training",
                            "import-completion",
                            "--launch-plan",
                            str(cloud_root / "launch_plan.json"),
                            "--launch-receipt",
                            str(cloud_root / "launch_receipt.json"),
                            "--status-receipt",
                            str(cloud_root / "status_receipt.json"),
                            "--runner-metadata",
                            str(cloud_root / "runner_metadata.json"),
                            "--raw-provider-result",
                            str(cloud_root / "raw_provider_result.json"),
                            "--output-artifact-manifest",
                            str(result_path),
                            "--out",
                            str(unsafe_path),
                        ]
                    )

            self.assertEqual(raised.exception.code, 2)
            self.assertFalse(unsafe_path.exists())
            self.assertNotIn(unsafe_name, stderr.getvalue())
            self.assertNotIn("Traceback", stderr.getvalue())

    def test_control_evidence_cannot_masquerade_as_model_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result_path = _write_candidate_training_result(root, CANDIDATE_ID)
            write_cloud_completion_fixture(root, result_path, CANDIDATE_ID)
            cloud_root = root / "cloud_training"
            raw_result_path = cloud_root / "raw_provider_result.json"
            raw_bytes = raw_result_path.read_bytes()
            result = json.loads(result_path.read_text(encoding="utf-8"))
            forged = {
                "path": "cloud_training/raw_provider_result.json",
                "sha256": hashlib.sha256(raw_bytes).hexdigest(),
                "size_bytes": len(raw_bytes),
            }
            result["artifacts"][0].update(forged)
            result["registry_update"]["links"][1].update(forged)
            result_path.write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            metadata_path = cloud_root / "runner_metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["source_sha256"]["output_artifact_manifest"] = hashlib.sha256(
                result_path.read_bytes()
            ).hexdigest()
            metadata_path.write_text(
                json.dumps(metadata, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            receipt = build_cloud_training_completion_receipt(
                launch_plan_path=cloud_root / "launch_plan.json",
                launch_receipt_path=cloud_root / "launch_receipt.json",
                status_receipt_path=cloud_root / "status_receipt.json",
                runner_metadata_path=metadata_path,
                raw_provider_result_path=raw_result_path,
                output_artifact_manifest_path=result_path,
                out_path=root / "forged-output-completion.json",
                created_at="2026-07-03T00:30:00+00:00",
            )

            failed = {
                check["id"]
                for check in receipt["integrity"]["checks"]
                if not check["passed"]
            }
            self.assertFalse(receipt["passed"])
            self.assertIn("output_artifacts_current", failed)
            self.assertIn("output_artifacts_disjoint_from_evidence", failed)
            self.assertFalse(
                receipt["governance"]["cloud_training_completion_claims_allowed"]
            )

    def test_zero_byte_adapter_cannot_authorize_completion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result_path = _write_candidate_training_result(root, CANDIDATE_ID)
            write_cloud_completion_fixture(root, result_path, CANDIDATE_ID)
            cloud_root = root / "cloud_training"
            result = json.loads(result_path.read_text(encoding="utf-8"))
            adapter_path = root / result["artifacts"][0]["path"]
            adapter_path.write_bytes(b"")
            empty_fingerprint = {
                "sha256": hashlib.sha256(b"").hexdigest(),
                "size_bytes": 0,
            }
            result["artifacts"][0].update(empty_fingerprint)
            result["registry_update"]["links"][1].update(empty_fingerprint)
            result_path.write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            metadata_path = cloud_root / "runner_metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["source_sha256"]["output_artifact_manifest"] = hashlib.sha256(
                result_path.read_bytes()
            ).hexdigest()
            metadata_path.write_text(
                json.dumps(metadata, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            receipt = build_cloud_training_completion_receipt(
                launch_plan_path=cloud_root / "launch_plan.json",
                launch_receipt_path=cloud_root / "launch_receipt.json",
                status_receipt_path=cloud_root / "status_receipt.json",
                runner_metadata_path=metadata_path,
                raw_provider_result_path=cloud_root / "raw_provider_result.json",
                output_artifact_manifest_path=result_path,
                out_path=root / "empty-output-completion.json",
                created_at="2026-07-03T00:30:00+00:00",
            )

            failed = {
                check["id"]
                for check in receipt["integrity"]["checks"]
                if not check["passed"]
            }
            self.assertFalse(receipt["passed"])
            self.assertIn("output_artifacts_current", failed)
            self.assertFalse(
                receipt["governance"]["cloud_training_completion_claims_allowed"]
            )

    def test_large_opaque_outputs_and_provider_result_remain_replayable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result_path = _write_candidate_training_result(root, CANDIDATE_ID)
            write_cloud_completion_fixture(root, result_path, CANDIDATE_ID)
            cloud_root = root / "cloud_training"
            result = json.loads(result_path.read_text(encoding="utf-8"))
            adapter_path = root / result["artifacts"][0]["path"]
            adapter_payload = b"a" * (5 * 1024 * 1024)
            adapter_path.write_bytes(adapter_payload)
            adapter_fingerprint = {
                "sha256": hashlib.sha256(adapter_payload).hexdigest(),
                "size_bytes": len(adapter_payload),
            }
            result["artifacts"][0].update(adapter_fingerprint)
            result["registry_update"]["links"][1].update(adapter_fingerprint)
            result_path.write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            raw_result_path = cloud_root / "raw_provider_result.json"
            raw_payload = b"r" * (5 * 1024 * 1024)
            raw_result_path.write_bytes(raw_payload)
            metadata_path = cloud_root / "runner_metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["source_sha256"].update(
                {
                    "output_artifact_manifest": hashlib.sha256(
                        result_path.read_bytes()
                    ).hexdigest(),
                    "raw_provider_result": hashlib.sha256(raw_payload).hexdigest(),
                }
            )
            metadata_path.write_text(
                json.dumps(metadata, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            completion_path = root / "large-cloud-completion.json"

            receipt = build_cloud_training_completion_receipt(
                launch_plan_path=cloud_root / "launch_plan.json",
                launch_receipt_path=cloud_root / "launch_receipt.json",
                status_receipt_path=cloud_root / "status_receipt.json",
                runner_metadata_path=metadata_path,
                raw_provider_result_path=raw_result_path,
                output_artifact_manifest_path=result_path,
                out_path=completion_path,
                created_at="2026-07-03T00:30:00+00:00",
            )
            write_cloud_training_completion_receipt(receipt, completion_path)

            self.assertTrue(receipt["passed"])
            self.assertEqual(validate_agentic_training_result(result_path).errors, [])
            self.assertEqual(
                validate_cloud_training_completion_receipt(completion_path).errors,
                [],
            )
            self.assertTrue(
                inspect_artifact_source(
                    result_path,
                    "agentic_training_result",
                )["ready"]
            )
            self.assertTrue(
                inspect_artifact_source(
                    completion_path,
                    "cloud_training_completion_receipt",
                )["ready"]
            )

    def test_deep_linked_preflight_is_blocked_instead_of_crashing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result_path = _write_candidate_training_result(root, CANDIDATE_ID)
            write_cloud_completion_fixture(root, result_path, CANDIDATE_ID)
            cloud_root = root / "cloud_training"
            preflight_path = cloud_root / "preflight.json"
            preflight_path.write_text(
                '{"nested":' * 2_000 + "0" + "}" * 2_000,
                encoding="utf-8",
            )
            launch_plan_path = cloud_root / "launch_plan.json"
            launch_plan = json.loads(launch_plan_path.read_text(encoding="utf-8"))
            preflight_bytes = preflight_path.read_bytes()
            launch_plan["source_artifacts"]["preflight"].update(
                {
                    "sha256": hashlib.sha256(preflight_bytes).hexdigest(),
                    "size_bytes": len(preflight_bytes),
                }
            )
            launch_plan_path.write_text(
                json.dumps(launch_plan, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            receipt = build_cloud_training_completion_receipt(
                launch_plan_path=launch_plan_path,
                launch_receipt_path=cloud_root / "launch_receipt.json",
                status_receipt_path=cloud_root / "status_receipt.json",
                runner_metadata_path=cloud_root / "runner_metadata.json",
                raw_provider_result_path=cloud_root / "raw_provider_result.json",
                output_artifact_manifest_path=result_path,
                out_path=root / "deep-preflight-completion.json",
                created_at="2026-07-03T00:30:00+00:00",
            )

            self.assertFalse(receipt["passed"])
            failed = {
                check["id"]
                for check in receipt["integrity"]["checks"]
                if not check["passed"]
            }
            self.assertIn("training_plan_lineage_converges", failed)


if __name__ == "__main__":
    unittest.main()
