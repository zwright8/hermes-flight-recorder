import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from flightrecorder.cloud_training import (
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
        self.assertTrue(all(provider["default_live_execution_allowed"] is False for provider in registry["providers"]))
        self.assertTrue(all(isinstance(provider["client_import_names"], list) for provider in registry["providers"]))
        self.assertFalse(registry["execution_boundary"]["provider_api_called"])
        schema = check_schema_contract(registry)
        self.assertTrue(schema["passed"], schema["errors"])

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


if __name__ == "__main__":
    unittest.main()
