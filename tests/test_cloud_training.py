import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from flightrecorder.cloud_training import (
    PROVIDER_ADAPTER_RECEIPT_TYPES,
    build_cloud_training_artifact_manifest,
    build_cloud_training_provider_registry,
    provider_choices,
)
from flightrecorder.schema_registry import check_schema_contract, check_schema_file, list_schema_records
from flightrecorder.validation import validate_artifacts


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_PLAN = ROOT / "examples" / "agentic_training" / "plans" / "sft_then_dpo_plan.json"


class CloudTrainingTests(unittest.TestCase):
    def test_provider_registry_lists_fail_closed_partner_contracts(self):
        registry = build_cloud_training_provider_registry(created_at="2026-07-03T00:00:00+00:00")

        provider_ids = {provider["id"] for provider in registry["providers"]}
        self.assertGreaterEqual(len(provider_ids), 12)
        self.assertIn("huggingface_jobs", provider_ids)
        self.assertIn("modal", provider_ids)
        self.assertIn("runpod", provider_ids)
        self.assertIn("aws_sagemaker", provider_ids)
        self.assertIn("gcp_vertex_ai", provider_ids)
        self.assertIn("azure_ml", provider_ids)
        self.assertIn("databricks_mosaic", provider_ids)
        self.assertIn("nvidia_dgx_cloud", provider_ids)
        self.assertIn("brev", provider_ids)
        self.assertTrue(all(provider["default_live_execution_allowed"] is False for provider in registry["providers"]))
        self.assertTrue(all(isinstance(provider["client_import_names"], list) for provider in registry["providers"]))
        for provider in registry["providers"]:
            contract = provider["adapter_contract"]
            self.assertEqual(contract["provider_id"], provider["id"])
            self.assertEqual(contract["dry_run_transport"], "mock_receipts")
            self.assertEqual(contract["live_preflight_transport"], "metadata_only")
            self.assertFalse(contract["live_launch_supported"])
            self.assertFalse(contract["provider_api_called_by_flight_recorder"])
            self.assertFalse(contract["credential_values_recorded"])
            self.assertTrue(set(PROVIDER_ADAPTER_RECEIPT_TYPES).issubset(set(contract["receipt_types"])))
        self.assertFalse(registry["execution_boundary"]["provider_api_called"])
        schema = check_schema_contract(registry)
        self.assertTrue(schema["passed"], schema["errors"])

    def test_validate_rejects_forged_provider_adapter_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            registry_path = root / "providers.json"
            registry = build_cloud_training_provider_registry(["modal"], created_at="2026-07-03T00:00:00+00:00")
            registry["providers"][0]["adapter_contract"]["live_launch_supported"] = True
            registry["providers"][0]["adapter_contract"]["provider_api_called_by_flight_recorder"] = True
            registry_path.write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            validation = validate_artifacts(cloud_training_provider_registry_paths=[registry_path], strict=True)

            self.assertFalse(validation["passed"])
            errors = "\n".join(error for target in validation["targets"] for error in target["errors"])
            self.assertIn("adapter_contract.live_launch_supported must be false", errors)
            self.assertIn("adapter_contract.provider_api_called_by_flight_recorder must be false", errors)

    def test_cli_emits_schema_checkable_fail_closed_provider_flow(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            registry = root / "providers.json"
            artifact_manifest = root / "artifacts.json"
            preflight = root / "preflight.json"
            launch_plan = root / "launch_plan.json"
            launch_receipt = root / "launch_receipt.json"
            live_receipt = root / "live_receipt.json"
            status_receipt = root / "status_receipt.json"

            self.assertEqual(
                run_cli(
                    [
                        "cloud-training",
                        "providers",
                        "--provider",
                        "modal",
                        "--created-at",
                        "2026-07-03T00:00:00+00:00",
                        "--out",
                        str(registry),
                    ]
                ),
                0,
            )
            self.assert_schema_and_validate(registry, "cloud_training_provider_registry")

            self.assertEqual(
                run_cli(
                    [
                        "cloud-training",
                        "artifacts",
                        "--provider",
                        "modal",
                        "--upload",
                        str(EXAMPLE_PLAN),
                        "--download",
                        "adapters/candidate/adapter_model.safetensors",
                        "--created-at",
                        "2026-07-03T00:00:00+00:00",
                        "--out",
                        str(artifact_manifest),
                    ]
                ),
                0,
            )
            self.assert_schema_and_validate(artifact_manifest, "cloud_training_artifact_manifest")
            artifact_payload = json.loads(artifact_manifest.read_text(encoding="utf-8"))
            transfer_plan = artifact_payload["transfer_plan"]
            self.assertEqual(transfer_plan["mode"], "dry_run_manifest_only")
            self.assertEqual(transfer_plan["upload_count"], 1)
            self.assertEqual(transfer_plan["expected_download_count"], 1)
            self.assertEqual(transfer_plan["artifact_protocols"], artifact_payload["artifact_protocols"])
            self.assertTrue(transfer_plan["requires_external_runner_upload"])
            self.assertTrue(transfer_plan["requires_external_runner_download"])
            self.assertFalse(transfer_plan["download_artifacts_expected_to_exist_before_launch"])
            self.assertFalse(transfer_plan["flight_recorder_uploaded_artifacts"])
            self.assertFalse(transfer_plan["flight_recorder_downloaded_artifacts"])
            self.assertFalse(transfer_plan["provider_api_called"])

            preflight_code = run_cli(
                [
                    "cloud-training",
                    "preflight",
                    "--provider",
                    "modal",
                    "--agentic-training-plan",
                    str(EXAMPLE_PLAN),
                    "--region",
                    "provider_default",
                    "--gpu-class",
                    "a100",
                    "--max-cost-usd",
                    "0",
                    "--live-preflight",
                    "--created-at",
                    "2026-07-03T00:00:00+00:00",
                    "--out",
                    str(preflight),
                ]
            )
            self.assertEqual(preflight_code, 1)
            preflight_payload = json.loads(preflight.read_text(encoding="utf-8"))
            self.assertEqual(preflight_payload["recommendation"], "block_cloud_training_launch")
            self.assertFalse(preflight_payload["execution_boundary"]["provider_api_called"])
            self.assertTrue(preflight_payload["execution_boundary"]["live_preflight_requested"])
            self.assertTrue(preflight_payload["live_preflight"]["requested"])
            self.assertEqual(preflight_payload["live_preflight"]["transport"], "metadata_only")
            self.assertFalse(preflight_payload["live_preflight"]["provider_api_called"])
            self.assertFalse(preflight_payload["live_preflight"]["client_modules_imported"])
            self.assertFalse(preflight_payload["live_preflight"]["credential_values_recorded"])
            self.assertTrue(all(check["value_recorded"] is False for check in preflight_payload["credential_checks"]))
            self.assert_schema_and_validate(preflight, "cloud_training_preflight")

            forged = json.loads(preflight.read_text(encoding="utf-8"))
            forged["live_preflight"]["provider_api_called"] = True
            preflight.write_text(json.dumps(forged, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            validation = validate_artifacts(cloud_training_preflight_paths=[preflight], strict=True)
            self.assertFalse(validation["passed"])
            errors = "\n".join(error for target in validation["targets"] for error in target["errors"])
            self.assertIn("cloud_training_preflight.live_preflight.provider_api_called must be false.", errors)
            preflight_payload["live_preflight"]["provider_api_called"] = False
            preflight.write_text(json.dumps(preflight_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            self.assertEqual(
                run_cli(
                    [
                        "cloud-training",
                        "plan",
                        "--preflight",
                        str(preflight),
                        "--artifact-manifest",
                        str(artifact_manifest),
                        "--created-at",
                        "2026-07-03T00:00:00+00:00",
                        "--out",
                        str(launch_plan),
                    ]
                ),
                1,
            )
            self.assert_schema_and_validate(launch_plan, "cloud_training_launch_plan")
            launch_plan_payload = json.loads(launch_plan.read_text(encoding="utf-8"))
            self.assertEqual(launch_plan_payload["provider_chain"]["preflight_provider_id"], "modal")
            self.assertEqual(
                launch_plan_payload["provider_chain"]["artifact_manifest_provider_id"],
                "modal",
            )
            self.assertTrue(launch_plan_payload["provider_chain"]["provider_consistent"])
            provider_drift_payload = json.loads(launch_plan.read_text(encoding="utf-8"))
            provider_drift_payload["provider"] = build_cloud_training_provider_registry(["huggingface_jobs"])["providers"][0]
            launch_plan.write_text(json.dumps(provider_drift_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            validation = validate_artifacts(cloud_training_launch_plan_paths=[launch_plan], strict=True)
            self.assertFalse(validation["passed"])
            errors = "\n".join(error for target in validation["targets"] for error in target["errors"])
            self.assertIn("cloud_training_launch_plan.provider.id must match provider_chain.preflight_provider_id.", errors)
            launch_plan.write_text(json.dumps(launch_plan_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            forged_ready_payload = json.loads(launch_plan.read_text(encoding="utf-8"))
            forged_ready_payload["source_artifacts"]["preflight"]["source_passed"] = True
            for check in forged_ready_payload["checks"]:
                if check["id"] == "preflight_ready":
                    check["passed"] = True
                    check["summary"] = "preflight_ready: passed=True"
                    check["actual"]["artifact"]["source_passed"] = True
            forged_ready_payload["blocked_reasons"] = []
            forged_ready_payload["failed_check_count"] = 0
            forged_ready_payload["passed"] = True
            forged_ready_payload["readiness"] = "ready_for_dry_run_launch"
            forged_ready_payload["recommendation"] = "emit_dry_run_launch_receipt"
            launch_plan.write_text(json.dumps(forged_ready_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            validation = validate_artifacts(cloud_training_launch_plan_paths=[launch_plan], strict=True)
            self.assertFalse(validation["passed"])
            errors = "\n".join(error for target in validation["targets"] for error in target["errors"])
            self.assertIn(
                "cloud_training_launch_plan.source_artifacts.preflight.source_passed must match the referenced source artifact.",
                errors,
            )
            self.assertIn("cloud_training_launch_plan.checks.preflight_ready.passed must match source readiness.", errors)
            self.assertIn("cloud_training_launch_plan.failed_check_count expected 1.", errors)
            launch_plan.write_text(json.dumps(launch_plan_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            self.assertEqual(
                run_cli(
                    [
                        "cloud-training",
                        "launch",
                        "--launch-plan",
                        str(launch_plan),
                        "--created-at",
                        "2026-07-03T00:00:00+00:00",
                        "--out",
                        str(launch_receipt),
                    ]
                ),
                1,
            )
            self.assert_schema_and_validate(launch_receipt, "cloud_training_launch_receipt")
            launch_receipt_payload = json.loads(launch_receipt.read_text(encoding="utf-8"))
            forged_launch_receipt = json.loads(launch_receipt.read_text(encoding="utf-8"))
            forged_launch_receipt["source_artifacts"]["launch_plan"]["source_passed"] = True
            for check in forged_launch_receipt["checks"]:
                if check["id"] == "launch_plan_ready":
                    check["passed"] = True
                    check["summary"] = "launch_plan_ready: passed=True"
                    check["actual"]["artifact"]["source_passed"] = True
            forged_launch_receipt["blocked_reasons"] = []
            forged_launch_receipt["failed_check_count"] = 0
            forged_launch_receipt["passed"] = True
            forged_launch_receipt["readiness"] = "dry_run_recorded"
            forged_launch_receipt["recommendation"] = "safe_to_archive_dry_run_receipt"
            launch_receipt.write_text(json.dumps(forged_launch_receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            validation = validate_artifacts(cloud_training_launch_receipt_paths=[launch_receipt], strict=True)
            self.assertFalse(validation["passed"])
            errors = "\n".join(error for target in validation["targets"] for error in target["errors"])
            self.assertIn(
                "cloud_training_launch_receipt.source_artifacts.launch_plan.source_passed must match the referenced source artifact.",
                errors,
            )
            self.assertIn("cloud_training_launch_receipt.checks.launch_plan_ready.passed must match source readiness.", errors)
            self.assertIn("cloud_training_launch_receipt.failed_check_count expected 1.", errors)
            launch_receipt.write_text(json.dumps(launch_receipt_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            forged_launch_receipt = json.loads(launch_receipt.read_text(encoding="utf-8"))
            forged_launch_receipt["launch"]["cloud_job_started"] = True
            forged_launch_receipt["launch"]["provider_job_id"] = "job-forged"
            forged_launch_receipt["launch"]["cost_incurred_usd"] = 1
            launch_receipt.write_text(json.dumps(forged_launch_receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            validation = validate_artifacts(cloud_training_launch_receipt_paths=[launch_receipt], strict=True)
            self.assertFalse(validation["passed"])
            errors = "\n".join(error for target in validation["targets"] for error in target["errors"])
            self.assertIn("cloud_training_launch_receipt.launch.cloud_job_started must be false.", errors)
            self.assertIn("cloud_training_launch_receipt.launch.provider_job_id must be null.", errors)
            self.assertIn("cloud_training_launch_receipt.launch.cost_incurred_usd must be 0.", errors)
            launch_receipt.write_text(json.dumps(launch_receipt_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            self.assertEqual(
                run_cli(
                    [
                        "cloud-training",
                        "launch",
                        "--launch-plan",
                        str(launch_plan),
                        "--live",
                        "--created-at",
                        "2026-07-03T00:00:00+00:00",
                        "--out",
                        str(live_receipt),
                    ]
                ),
                1,
            )
            live_payload = json.loads(live_receipt.read_text(encoding="utf-8"))
            self.assertEqual(live_payload["recommendation"], "block_live_cloud_launch")
            self.assertFalse(live_payload["launch"]["cloud_job_started"])
            self.assert_schema_and_validate(live_receipt, "cloud_training_launch_receipt")

            self.assertEqual(
                run_cli(
                    [
                        "cloud-training",
                        "status",
                        "--launch-receipt",
                        str(launch_receipt),
                        "--cancel",
                        "--created-at",
                        "2026-07-03T00:00:00+00:00",
                        "--out",
                        str(status_receipt),
                    ]
                ),
                0,
            )
            status_payload = json.loads(status_receipt.read_text(encoding="utf-8"))
            self.assertEqual(status_payload["status"]["provider_status"], "not_started")
            self.assertFalse(status_payload["status"]["provider_cancel_called"])
            self.assert_schema_and_validate(status_receipt, "cloud_training_status_receipt")
            forged_status_receipt = json.loads(status_receipt.read_text(encoding="utf-8"))
            forged_status_receipt["source_artifacts"]["launch_receipt"]["source_passed"] = True
            status_receipt.write_text(json.dumps(forged_status_receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            validation = validate_artifacts(cloud_training_status_receipt_paths=[status_receipt], strict=True)
            self.assertFalse(validation["passed"])
            errors = "\n".join(error for target in validation["targets"] for error in target["errors"])
            self.assertIn(
                "cloud_training_status_receipt.source_artifacts.launch_receipt.source_passed must match the referenced source artifact.",
                errors,
            )
            status_receipt.write_text(json.dumps(status_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            forged_status_receipt = json.loads(status_receipt.read_text(encoding="utf-8"))
            forged_status_receipt["status"]["provider_api_called"] = True
            status_receipt.write_text(json.dumps(forged_status_receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            validation = validate_artifacts(cloud_training_status_receipt_paths=[status_receipt], strict=True)
            self.assertFalse(validation["passed"])
            errors = "\n".join(error for target in validation["targets"] for error in target["errors"])
            self.assertIn("cloud_training_status_receipt.status.provider_api_called must be false.", errors)
            self.assertIn("cloud_training_status_receipt.checks.status_check_did_not_call_provider.passed must match source readiness.", errors)
            status_receipt.write_text(json.dumps(status_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def test_launch_and_status_receipts_reject_unknown_side_effect_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan_source = root / "agentic_training_plan.json"
            plan_source.write_text(EXAMPLE_PLAN.read_text(encoding="utf-8"), encoding="utf-8")
            preflight = root / "preflight.json"
            launch_plan = root / "launch_plan.json"
            launch_receipt = root / "launch_receipt.json"
            status_receipt = root / "status_receipt.json"

            self.assertEqual(
                run_cli(
                    [
                        "cloud-training",
                        "preflight",
                        "--provider",
                        "modal",
                        "--agentic-training-plan",
                        str(plan_source),
                        "--region",
                        "provider_default",
                        "--gpu-class",
                        "a100",
                        "--max-cost-usd",
                        "0",
                        "--out",
                        str(preflight),
                    ]
                ),
                1,
            )
            self.assertEqual(run_cli(["cloud-training", "plan", "--preflight", str(preflight), "--out", str(launch_plan)]), 1)
            self.assertEqual(run_cli(["cloud-training", "launch", "--launch-plan", str(launch_plan), "--out", str(launch_receipt)]), 1)
            self.assertEqual(run_cli(["cloud-training", "status", "--launch-receipt", str(launch_receipt), "--out", str(status_receipt)]), 0)
            self.assert_schema_and_validate(launch_receipt, "cloud_training_launch_receipt")
            self.assert_schema_and_validate(status_receipt, "cloud_training_status_receipt")

            launch_payload = json.loads(launch_receipt.read_text(encoding="utf-8"))
            status_payload = json.loads(status_receipt.read_text(encoding="utf-8"))

            forged_launch = json.loads(json.dumps(launch_payload))
            forged_launch["provider_console_url"] = "redacted-provider-console"
            forged_launch["checks"][0]["provider_call"] = "forged"
            forged_launch["source_artifacts"]["launch_plan"]["credential_hint"] = "redacted"
            forged_launch["launch"]["provider_console_url"] = "redacted-provider-console"
            forged_launch["execution_boundary"]["cloud_scheduler_receipt"] = "not-created"
            launch_receipt.write_text(json.dumps(forged_launch, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            schema = check_schema_file(launch_receipt)
            self.assertFalse(schema["passed"], schema)
            validation = validate_artifacts(cloud_training_launch_receipt_paths=[launch_receipt], strict=True)
            self.assertFalse(validation["passed"])
            errors = "\n".join(error for target in validation["targets"] for error in target["errors"])
            self.assertIn("cloud_training_launch_receipt contains unknown field(s): ['provider_console_url'].", errors)
            self.assertIn("cloud_training_launch_receipt.checks[0] contains unknown field(s): ['provider_call'].", errors)
            self.assertIn(
                "cloud_training_launch_receipt.source_artifacts.launch_plan contains unknown field(s): ['credential_hint'].",
                errors,
            )
            self.assertIn("cloud_training_launch_receipt.launch contains unknown field(s): ['provider_console_url'].", errors)
            self.assertIn(
                "cloud_training_launch_receipt.execution_boundary contains unknown field(s): ['cloud_scheduler_receipt'].",
                errors,
            )
            launch_receipt.write_text(json.dumps(launch_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            forged_status = json.loads(json.dumps(status_payload))
            forged_status["provider_poll_url"] = "redacted-provider-poll"
            forged_status["checks"][0]["provider_poll"] = "forged"
            forged_status["source_artifacts"]["launch_receipt"]["credential_hint"] = "redacted"
            forged_status["status"]["provider_status_url"] = "redacted-provider-poll"
            forged_status["execution_boundary"]["cancel_receipt"] = "not-created"
            status_receipt.write_text(json.dumps(forged_status, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            schema = check_schema_file(status_receipt)
            self.assertFalse(schema["passed"], schema)
            validation = validate_artifacts(cloud_training_status_receipt_paths=[status_receipt], strict=True)
            self.assertFalse(validation["passed"])
            errors = "\n".join(error for target in validation["targets"] for error in target["errors"])
            self.assertIn("cloud_training_status_receipt contains unknown field(s): ['provider_poll_url'].", errors)
            self.assertIn("cloud_training_status_receipt.checks[0] contains unknown field(s): ['provider_poll'].", errors)
            self.assertIn(
                "cloud_training_status_receipt.source_artifacts.launch_receipt contains unknown field(s): ['credential_hint'].",
                errors,
            )
            self.assertIn("cloud_training_status_receipt.status contains unknown field(s): ['provider_status_url'].", errors)
            self.assertIn(
                "cloud_training_status_receipt.execution_boundary contains unknown field(s): ['cancel_receipt'].",
                errors,
            )

    def test_cloud_training_source_refs_are_output_relative_and_reject_stale_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sources = root / "sources"
            reports = root / "reports"
            sources.mkdir()
            reports.mkdir()
            plan_source = sources / "agentic_training_plan.json"
            plan_source.write_text(EXAMPLE_PLAN.read_text(encoding="utf-8"), encoding="utf-8")
            artifact_manifest = reports / "artifacts.json"
            preflight = reports / "preflight.json"
            launch_plan = reports / "launch_plan.json"
            launch_receipt = reports / "launch_receipt.json"
            status_receipt = reports / "status_receipt.json"

            self.assertEqual(
                run_cli(["cloud-training", "artifacts", "--provider", "modal", "--upload", str(plan_source), "--out", str(artifact_manifest)]),
                0,
            )
            self.assertEqual(
                run_cli(
                    [
                        "cloud-training",
                        "preflight",
                        "--provider",
                        "modal",
                        "--agentic-training-plan",
                        str(plan_source),
                        "--region",
                        "provider_default",
                        "--gpu-class",
                        "a100",
                        "--max-cost-usd",
                        "0",
                        "--out",
                        str(preflight),
                    ]
                ),
                1,
            )
            self.assertEqual(
                run_cli(
                    [
                        "cloud-training",
                        "plan",
                        "--preflight",
                        str(preflight),
                        "--artifact-manifest",
                        str(artifact_manifest),
                        "--out",
                        str(launch_plan),
                    ]
                ),
                1,
            )
            self.assertEqual(run_cli(["cloud-training", "launch", "--launch-plan", str(launch_plan), "--out", str(launch_receipt)]), 1)
            self.assertEqual(run_cli(["cloud-training", "status", "--launch-receipt", str(launch_receipt), "--out", str(status_receipt)]), 0)

            self.assert_schema_and_validate(artifact_manifest, "cloud_training_artifact_manifest")
            self.assert_schema_and_validate(preflight, "cloud_training_preflight")
            self.assert_schema_and_validate(launch_plan, "cloud_training_launch_plan")
            self.assert_schema_and_validate(launch_receipt, "cloud_training_launch_receipt")
            self.assert_schema_and_validate(status_receipt, "cloud_training_status_receipt")
            preflight_payload = json.loads(preflight.read_text(encoding="utf-8"))
            artifact_payload = json.loads(artifact_manifest.read_text(encoding="utf-8"))
            self.assertEqual(preflight_payload["source_artifacts"]["agentic_training_plan"]["path"], "../sources/agentic_training_plan.json")
            self.assertEqual(artifact_payload["upload_artifacts"][0]["path"], "../sources/agentic_training_plan.json")

            _mutate_json(plan_source, "plan_source")
            _mutate_json(preflight, "preflight")
            _mutate_json(launch_plan, "launch_plan")
            _mutate_json(launch_receipt, "launch_receipt")

            validation = validate_artifacts(
                cloud_training_preflight_paths=[preflight],
                cloud_training_artifact_manifest_paths=[artifact_manifest],
                cloud_training_launch_plan_paths=[launch_plan],
                cloud_training_launch_receipt_paths=[launch_receipt],
                cloud_training_status_receipt_paths=[status_receipt],
                strict=True,
            )
            self.assertFalse(validation["passed"])
            errors = "\n".join(error for target in validation["targets"] for error in target["errors"])
            self.assertIn("cloud_training_preflight.source_artifacts.agentic_training_plan.sha256 does not match the current file.", errors)
            self.assertIn("cloud_training_artifact_manifest.upload_artifacts[0].sha256 does not match the current file.", errors)
            self.assertIn("cloud_training_launch_plan.source_artifacts.preflight.sha256 does not match the current file.", errors)
            self.assertIn("cloud_training_launch_receipt.source_artifacts.launch_plan.sha256 does not match the current file.", errors)
            self.assertIn("cloud_training_status_receipt.source_artifacts.launch_receipt.sha256 does not match the current file.", errors)

    def test_validate_rejects_cloud_training_symlink_source_and_upload_refs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sources = root / "sources"
            reports = root / "reports"
            sources.mkdir()
            reports.mkdir()
            plan_source = sources / "agentic_training_plan.json"
            plan_source.write_text(EXAMPLE_PLAN.read_text(encoding="utf-8"), encoding="utf-8")
            artifact_manifest = reports / "artifacts.json"
            preflight = reports / "preflight.json"

            self.assertEqual(
                run_cli(["cloud-training", "artifacts", "--provider", "modal", "--upload", str(plan_source), "--out", str(artifact_manifest)]),
                0,
            )
            self.assertEqual(
                run_cli(
                    [
                        "cloud-training",
                        "preflight",
                        "--provider",
                        "modal",
                        "--agentic-training-plan",
                        str(plan_source),
                        "--region",
                        "provider_default",
                        "--gpu-class",
                        "a100",
                        "--max-cost-usd",
                        "0",
                        "--out",
                        str(preflight),
                    ]
                ),
                1,
            )

            direct_link = reports / "agentic_training_plan_link.json"
            try:
                direct_link.symlink_to(plan_source)
            except (NotImplementedError, OSError) as exc:
                self.skipTest(f"symlink unavailable: {exc}")
            _set_cloud_training_ref_path(artifact_manifest, ("upload_artifacts", 0), direct_link.name)
            _set_cloud_training_ref_path(preflight, ("source_artifacts", "agentic_training_plan"), direct_link.name)
            validation = validate_artifacts(
                cloud_training_artifact_manifest_paths=[artifact_manifest],
                cloud_training_preflight_paths=[preflight],
                strict=True,
            )
            self.assertFalse(validation["passed"], validation)
            errors = "\n".join(error for target in validation["targets"] for error in target["errors"])
            self.assertIn(
                "cloud_training_artifact_manifest.upload_artifacts[0].path must resolve to a regular non-symlink file when exists is true.",
                errors,
            )
            self.assertIn(
                "cloud_training_preflight.source_artifacts.agentic_training_plan.path must resolve to a regular non-symlink file when exists is true.",
                errors,
            )

            broken_link = reports / "broken_agentic_training_plan_link.json"
            broken_link.symlink_to(reports / "missing_agentic_training_plan.json")
            _set_cloud_training_ref_path(preflight, ("source_artifacts", "agentic_training_plan"), broken_link.name)
            validation = validate_artifacts(cloud_training_preflight_paths=[preflight], strict=True)
            self.assertFalse(validation["passed"], validation)
            errors = "\n".join(error for target in validation["targets"] for error in target["errors"])
            self.assertIn(
                "cloud_training_preflight.source_artifacts.agentic_training_plan.path must resolve to a regular non-symlink file when exists is true.",
                errors,
            )

            linked_target = reports / "linked_target"
            linked_target.mkdir()
            (linked_target / plan_source.name).write_text(plan_source.read_text(encoding="utf-8"), encoding="utf-8")
            linked_parent = reports / "linked_artifacts"
            linked_parent.symlink_to(linked_target, target_is_directory=True)
            _set_cloud_training_ref_path(artifact_manifest, ("upload_artifacts", 0), str(Path(linked_parent.name) / plan_source.name))
            validation = validate_artifacts(cloud_training_artifact_manifest_paths=[artifact_manifest], strict=True)
            self.assertFalse(validation["passed"], validation)
            errors = "\n".join(error for target in validation["targets"] for error in target["errors"])
            self.assertIn(
                "cloud_training_artifact_manifest.upload_artifacts[0].path must resolve to a regular non-symlink file when exists is true.",
                errors,
            )

    def test_validate_rejects_cloud_training_artifacts_missing_required_source_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan_source = root / "agentic_training_plan.json"
            plan_source.write_text(EXAMPLE_PLAN.read_text(encoding="utf-8"), encoding="utf-8")
            preflight = root / "preflight.json"
            launch_plan = root / "launch_plan.json"
            launch_receipt = root / "launch_receipt.json"
            status_receipt = root / "status_receipt.json"

            self.assertEqual(
                run_cli(
                    [
                        "cloud-training",
                        "preflight",
                        "--provider",
                        "modal",
                        "--agentic-training-plan",
                        str(plan_source),
                        "--region",
                        "provider_default",
                        "--gpu-class",
                        "a100",
                        "--max-cost-usd",
                        "0",
                        "--out",
                        str(preflight),
                    ]
                ),
                1,
            )
            self.assertEqual(run_cli(["cloud-training", "plan", "--preflight", str(preflight), "--out", str(launch_plan)]), 1)
            self.assertEqual(run_cli(["cloud-training", "launch", "--launch-plan", str(launch_plan), "--out", str(launch_receipt)]), 1)
            self.assertEqual(run_cli(["cloud-training", "status", "--launch-receipt", str(launch_receipt), "--out", str(status_receipt)]), 0)
            launch_payload = json.loads(launch_plan.read_text(encoding="utf-8"))
            self.assertTrue(launch_payload["provider_chain"]["artifact_manifest_required"])
            self.assertFalse(launch_payload["provider_chain"]["provider_consistent"])
            self.assertIn(
                "artifact_manifest_ready",
                {check["id"] for check in launch_payload["checks"] if not check["passed"]},
            )
            self.assertIn(
                "provider_chain_consistent",
                {check["id"] for check in launch_payload["checks"] if not check["passed"]},
            )
            self.assert_schema_and_validate(launch_plan, "cloud_training_launch_plan")
            for path in (preflight, launch_plan, launch_receipt, status_receipt):
                payload = json.loads(path.read_text(encoding="utf-8"))
                payload.pop("source_artifacts")
                path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            validation = validate_artifacts(
                cloud_training_preflight_paths=[preflight],
                cloud_training_launch_plan_paths=[launch_plan],
                cloud_training_launch_receipt_paths=[launch_receipt],
                cloud_training_status_receipt_paths=[status_receipt],
                strict=True,
            )

            self.assertFalse(validation["passed"])
            errors = "\n".join(error for target in validation["targets"] for error in target["errors"])
            self.assertIn("cloud_training_preflight.source_artifacts must be an object.", errors)
            self.assertIn("cloud_training_launch_plan.source_artifacts must be an object.", errors)
            self.assertIn("cloud_training_launch_receipt.source_artifacts must be an object.", errors)
            self.assertIn("cloud_training_status_receipt.source_artifacts must be an object.", errors)

    def test_validate_rejects_cloud_training_missing_required_source_refs_and_uploads(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan_source = root / "agentic_training_plan.json"
            plan_source.write_text(EXAMPLE_PLAN.read_text(encoding="utf-8"), encoding="utf-8")
            artifact_manifest = root / "artifacts.json"
            preflight = root / "preflight.json"
            launch_plan = root / "launch_plan.json"
            launch_receipt = root / "launch_receipt.json"
            status_receipt = root / "status_receipt.json"

            self.assertEqual(
                run_cli(["cloud-training", "artifacts", "--provider", "modal", "--upload", str(plan_source), "--out", str(artifact_manifest)]),
                0,
            )
            self.assertEqual(
                run_cli(
                    [
                        "cloud-training",
                        "preflight",
                        "--provider",
                        "modal",
                        "--agentic-training-plan",
                        str(plan_source),
                        "--region",
                        "provider_default",
                        "--gpu-class",
                        "a100",
                        "--max-cost-usd",
                        "0",
                        "--out",
                        str(preflight),
                    ]
                ),
                1,
            )
            self.assertEqual(
                run_cli(["cloud-training", "plan", "--preflight", str(preflight), "--artifact-manifest", str(artifact_manifest), "--out", str(launch_plan)]),
                1,
            )
            self.assertEqual(run_cli(["cloud-training", "launch", "--launch-plan", str(launch_plan), "--out", str(launch_receipt)]), 1)
            self.assertEqual(run_cli(["cloud-training", "status", "--launch-receipt", str(launch_receipt), "--out", str(status_receipt)]), 0)

            artifact_payload = json.loads(artifact_manifest.read_text(encoding="utf-8"))
            artifact_payload["upload_artifacts"] = []
            artifact_manifest.write_text(json.dumps(artifact_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            _remove_source_artifact(preflight, "agentic_training_plan")
            _remove_source_artifact(launch_plan, "preflight")
            _remove_source_artifact(launch_receipt, "launch_plan")
            _remove_source_artifact(status_receipt, "launch_receipt")

            validation = validate_artifacts(
                cloud_training_preflight_paths=[preflight],
                cloud_training_artifact_manifest_paths=[artifact_manifest],
                cloud_training_launch_plan_paths=[launch_plan],
                cloud_training_launch_receipt_paths=[launch_receipt],
                cloud_training_status_receipt_paths=[status_receipt],
                strict=True,
            )

            self.assertFalse(validation["passed"])
            errors = "\n".join(error for target in validation["targets"] for error in target["errors"])
            self.assertIn("cloud_training_artifact_manifest.upload_artifacts must include at least one artifact.", errors)
            self.assertIn("cloud_training_preflight.source_artifacts.agentic_training_plan is required.", errors)
            self.assertIn("cloud_training_launch_plan.source_artifacts.preflight is required.", errors)
            self.assertIn("cloud_training_launch_receipt.source_artifacts.launch_plan is required.", errors)
            self.assertIn("cloud_training_status_receipt.source_artifacts.launch_receipt is required.", errors)

    def test_validate_rejects_cloud_training_absolute_source_and_upload_refs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan_source = root / "agentic_training_plan.json"
            plan_source.write_text(EXAMPLE_PLAN.read_text(encoding="utf-8"), encoding="utf-8")
            artifact_manifest = root / "artifacts.json"
            preflight = root / "preflight.json"

            self.assertEqual(
                run_cli(["cloud-training", "artifacts", "--provider", "modal", "--upload", str(plan_source), "--out", str(artifact_manifest)]),
                0,
            )
            self.assertEqual(
                run_cli(
                    [
                        "cloud-training",
                        "preflight",
                        "--provider",
                        "modal",
                        "--agentic-training-plan",
                        str(plan_source),
                        "--region",
                        "provider_default",
                        "--gpu-class",
                        "a100",
                        "--max-cost-usd",
                        "0",
                        "--out",
                        str(preflight),
                    ]
                ),
                1,
            )
            artifact_payload = json.loads(artifact_manifest.read_text(encoding="utf-8"))
            artifact_payload["upload_artifacts"][0]["path"] = str(plan_source.resolve())
            artifact_manifest.write_text(json.dumps(artifact_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            preflight_payload = json.loads(preflight.read_text(encoding="utf-8"))
            preflight_payload["source_artifacts"]["agentic_training_plan"]["path"] = str(plan_source.resolve())
            preflight.write_text(json.dumps(preflight_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            validation = validate_artifacts(
                cloud_training_preflight_paths=[preflight],
                cloud_training_artifact_manifest_paths=[artifact_manifest],
                strict=True,
            )

            self.assertFalse(validation["passed"])
            errors = "\n".join(error for target in validation["targets"] for error in target["errors"])
            self.assertIn("cloud_training_artifact_manifest.upload_artifacts[0].path must be relative to the cloud-training artifact.", errors)
            self.assertIn(
                "cloud_training_preflight.source_artifacts.agentic_training_plan.path must be relative to the cloud-training artifact.",
                errors,
            )

    def test_validate_rejects_forged_cloud_training_transfer_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = root / "artifacts.json"
            manifest = build_cloud_training_artifact_manifest(
                provider_id="modal",
                upload_paths=[EXAMPLE_PLAN],
                expected_downloads=["adapters/candidate/adapter_model.safetensors"],
                output_base_dir=root,
                created_at="2026-07-03T00:00:00+00:00",
            )
            manifest["transfer_plan"]["upload_count"] = 0
            manifest["transfer_plan"]["expected_download_count"] = 0
            manifest["transfer_plan"]["artifact_protocols"] = []
            manifest["transfer_plan"]["flight_recorder_uploaded_artifacts"] = True
            manifest["transfer_plan"]["flight_recorder_downloaded_artifacts"] = True
            manifest["transfer_plan"]["provider_api_called"] = True
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            validation = validate_artifacts(cloud_training_artifact_manifest_paths=[manifest_path], strict=True)

            self.assertFalse(validation["passed"])
            errors = "\n".join(error for target in validation["targets"] for error in target["errors"])
            self.assertIn("cloud_training_artifact_manifest.transfer_plan.upload_count must match cloud training artifact rows.", errors)
            self.assertIn("cloud_training_artifact_manifest.transfer_plan.expected_download_count must match cloud training artifact rows.", errors)
            self.assertIn("cloud_training_artifact_manifest.transfer_plan.artifact_protocols must match cloud training artifact rows.", errors)
            self.assertIn("cloud_training_artifact_manifest.transfer_plan.flight_recorder_uploaded_artifacts must be false.", errors)
            self.assertIn("cloud_training_artifact_manifest.transfer_plan.flight_recorder_downloaded_artifacts must be false.", errors)
            self.assertIn("cloud_training_artifact_manifest.transfer_plan.provider_api_called must be false.", errors)

    def test_launch_plan_blocks_mismatched_preflight_and_artifact_manifest_providers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact_manifest = root / "artifacts.json"
            preflight = root / "preflight.json"
            launch_plan = root / "launch_plan.json"

            self.assertEqual(
                run_cli(
                    [
                        "cloud-training",
                        "artifacts",
                        "--provider",
                        "huggingface_jobs",
                        "--upload",
                        str(EXAMPLE_PLAN),
                        "--out",
                        str(artifact_manifest),
                    ]
                ),
                0,
            )
            self.assertEqual(
                run_cli(
                    [
                        "cloud-training",
                        "preflight",
                        "--provider",
                        "modal",
                        "--agentic-training-plan",
                        str(EXAMPLE_PLAN),
                        "--region",
                        "provider_default",
                        "--gpu-class",
                        "a100",
                        "--max-cost-usd",
                        "0",
                        "--out",
                        str(preflight),
                    ]
                ),
                1,
            )
            self.assertEqual(
                run_cli(
                    [
                        "cloud-training",
                        "plan",
                        "--preflight",
                        str(preflight),
                        "--artifact-manifest",
                        str(artifact_manifest),
                        "--out",
                        str(launch_plan),
                    ]
                ),
                1,
            )
            payload = json.loads(launch_plan.read_text(encoding="utf-8"))
            self.assertFalse(payload["provider_chain"]["provider_consistent"])
            self.assertEqual(payload["provider_chain"]["mismatched_provider_ids"], ["modal", "huggingface_jobs"])
            self.assertIn(
                "provider_chain_consistent",
                {check["id"] for check in payload["checks"] if not check["passed"]},
            )
            self.assert_schema_and_validate(launch_plan, "cloud_training_launch_plan")

            payload["provider_chain"]["provider_consistent"] = True
            payload["provider_chain"]["mismatched_provider_ids"] = []
            launch_plan.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            validation = validate_artifacts(cloud_training_launch_plan_paths=[launch_plan], strict=True)
            self.assertFalse(validation["passed"])
            errors = "\n".join(error for target in validation["targets"] for error in target["errors"])
            self.assertIn(
                "cloud_training_launch_plan.provider_chain.provider_consistent must match launch-plan source providers.",
                errors,
            )
            self.assertIn(
                "cloud_training_launch_plan.provider_chain.mismatched_provider_ids must match launch-plan source providers.",
                errors,
            )

            payload["source_artifacts"]["artifact_manifest"]["exists"] = False
            payload["source_artifacts"]["artifact_manifest"]["path"] = "internal/private_artifacts.json"
            payload["source_artifacts"]["artifact_manifest"]["sha256"] = None
            payload["source_artifacts"]["artifact_manifest"]["size_bytes"] = None
            payload["provider_chain"] = {
                "artifact_manifest_provider_id": "",
                "artifact_manifest_required": False,
                "mismatched_provider_ids": [],
                "preflight_provider_id": "modal",
                "provider_consistent": True,
            }
            launch_plan.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            validation = validate_artifacts(cloud_training_launch_plan_paths=[launch_plan], strict=True)
            self.assertFalse(validation["passed"])
            errors = "\n".join(error for target in validation["targets"] for error in target["errors"])
            self.assertIn(
                "cloud_training_launch_plan.source_artifacts.artifact_manifest.path must be empty when artifact_manifest is missing.",
                errors,
            )
            self.assertIn(
                "cloud_training_launch_plan.checks must include failed artifact_manifest_ready when artifact_manifest is missing.",
                errors,
            )
            self.assertIn("cloud_training_launch_plan.provider_chain.artifact_manifest_required must be true.", errors)
            self.assertIn(
                "cloud_training_launch_plan.provider_chain.provider_consistent must be false when artifact_manifest is missing.",
                errors,
            )

    def test_schema_names_are_registered(self):
        names = {record["name"] for record in list_schema_records()}
        self.assertIn("cloud_training_provider_registry", names)
        self.assertIn("cloud_training_preflight", names)
        self.assertIn("cloud_training_artifact_manifest", names)
        self.assertIn("cloud_training_launch_plan", names)
        self.assertIn("cloud_training_launch_receipt", names)
        self.assertIn("cloud_training_status_receipt", names)

    def test_provider_choices_are_stable(self):
        choices = provider_choices()
        self.assertEqual(choices, sorted(choices))
        self.assertIn("fireworks", choices)
        self.assertIn("together", choices)
        self.assertIn("brev", choices)

    def assert_schema_and_validate(self, path: Path, schema_name: str) -> None:
        schema = check_schema_file(path)
        self.assertTrue(schema["passed"], schema["errors"])
        kwargs = {f"{schema_name}_paths": [path]}
        validation = validate_artifacts(**kwargs, strict=True)
        self.assertTrue(validation["passed"], validation)


def run_cli(args: list[str]) -> int:
    completed = subprocess.run(
        [sys.executable, "-m", "flightrecorder", *args],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return completed.returncode


def _mutate_json(path: Path, marker: str) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload[f"test_mutation_{marker}"] = True
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _remove_source_artifact(path: Path, name: str) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["source_artifacts"].pop(name)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _set_cloud_training_ref_path(path: Path, ref_path: tuple[str, int] | tuple[str, str], value: str) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    container = payload[ref_path[0]]
    container[ref_path[1]]["path"] = value
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
