import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from flightrecorder.cli import main
from flightrecorder.external_eval import build_external_eval_plan, write_external_eval_plan
from flightrecorder.schema_registry import check_schema_file


def run_cli(args):
    with redirect_stdout(StringIO()):
        return main(args)


class ExternalEvalReceiptTests(unittest.TestCase):
    def test_external_eval_receipt_blocks_unready_plan_but_validates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = _scenario_manifest(root / "heldout.json")
            plan_path = root / "external_eval_plan.json"
            receipt_path = root / "external_eval_receipt.json"

            plan_code = run_cli(["external-eval-plan", "--scenario-manifest", str(manifest), "--out", str(plan_path)])
            receipt_code = run_cli(
                [
                    "external-eval-receipt",
                    "--plan",
                    str(plan_path),
                    "--created-at",
                    "2026-07-03T00:00:00+00:00",
                    "--out",
                    str(receipt_path),
                ]
            )
            validate_code = run_cli(["validate", "--external-eval-receipt", str(receipt_path), "--strict"])
            schema_result = check_schema_file(receipt_path)

            self.assertEqual(plan_code, 1)
            self.assertEqual(receipt_code, 1)
            self.assertEqual(validate_code, 0)
            self.assertTrue(schema_result["passed"], schema_result["errors"])
            receipt = _read_json(receipt_path)
            self.assertFalse(receipt["passed"])
            self.assertEqual(receipt["readiness"], "blocked")
            self.assertEqual(receipt["launch"]["mode"], "dry_run")
            self.assertFalse(receipt["launch"]["live_benchmarks_started"])
            self.assertFalse(receipt["launch"]["provider_api_called"])
            self.assertEqual(receipt["launch"]["cost_incurred_usd"], 0)
            self.assertEqual(receipt["source_plan"]["path"], plan_path.name)

    def test_external_eval_receipt_ready_dry_run_uses_mocked_dependency(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = _scenario_manifest(root / "heldout.json")
            plan_path = root / "external_eval_plan.json"
            receipt_path = root / "external_eval_receipt.json"

            with patch(
                "flightrecorder.external_eval._dependency_status",
                return_value={"available": True, "imports": {"inspect_ai": True}, "commands": {"inspect": True}},
            ):
                plan = build_external_eval_plan(
                    adapters=["inspect_ai"],
                    scenario_manifest=manifest,
                    model_endpoint="http://127.0.0.1:8000/v1",
                    inspect_task_set="heldout-inspect",
                    sandbox_policy="locked-network",
                    allow_installed=True,
                )
            write_external_eval_plan(plan, plan_path)

            receipt_code = run_cli(
                [
                    "external-eval-receipt",
                    "--plan",
                    str(plan_path),
                    "--created-at",
                    "2026-07-03T00:00:00+00:00",
                    "--out",
                    str(receipt_path),
                ]
            )
            validate_code = run_cli(["validate", "--external-eval-receipt", str(receipt_path), "--strict"])

            self.assertEqual(receipt_code, 0)
            self.assertEqual(validate_code, 0)
            receipt = _read_json(receipt_path)
            self.assertTrue(receipt["passed"])
            self.assertEqual(receipt["readiness"], "dry_run_recorded")
            self.assertEqual(receipt["adapter_count"], 1)
            self.assertEqual(receipt["ready_adapter_count"], 1)
            self.assertTrue(receipt["adapter_receipts"][0]["ready"])
            self.assertFalse(receipt["adapter_receipts"][0]["live_benchmark_started"])
            self.assertFalse(receipt["adapter_receipts"][0]["model_downloads_started"])
            self.assertFalse(receipt["adapter_receipts"][0]["credential_values_recorded"])
            self.assertFalse(receipt["adapter_receipts"][0]["adapter_contract"]["live_benchmark_supported"])
            self.assertFalse(receipt["adapter_receipts"][0]["adapter_contract"]["provider_api_called_by_flight_recorder"])
            self.assertFalse(receipt["execution_boundary"]["provider_api_called"])
            self.assertFalse(receipt["execution_boundary"]["weights_updated_by_flight_recorder"])

    def test_validate_rejects_subset_receipt_for_multi_adapter_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = _scenario_manifest(root / "heldout.json")
            plan_path = root / "external_eval_plan.json"
            receipt_path = root / "external_eval_receipt.json"
            validation = root / "validation.json"

            with patch(
                "flightrecorder.external_eval._dependency_status",
                return_value={"available": True, "imports": {"mock_adapter": True}, "commands": {"mock-adapter": True}},
            ):
                plan = build_external_eval_plan(
                    adapters=["bfcl", "inspect_ai"],
                    scenario_manifest=manifest,
                    model_endpoint="http://127.0.0.1:8000/v1",
                    tool_schema_set="heldout-tools",
                    inspect_task_set="heldout-inspect",
                    sandbox_policy="locked-network",
                    allow_installed=True,
                )
            write_external_eval_plan(plan, plan_path)
            receipt_code = run_cli(
                [
                    "external-eval-receipt",
                    "--plan",
                    str(plan_path),
                    "--adapter",
                    "bfcl",
                    "--created-at",
                    "2026-07-03T00:00:00+00:00",
                    "--out",
                    str(receipt_path),
                ]
            )

            code = run_cli(["validate", "--external-eval-receipt", str(receipt_path), "--out", str(validation), "--strict"])

            self.assertEqual(receipt_code, 0)
            self.assertEqual(code, 1)
            errors = "\n".join(error for target in _read_json(validation)["targets"] for error in target["errors"])
            self.assertIn("external_eval_receipt.adapter_receipts must match current source plan selected_adapters.", errors)
            self.assertIn("external_eval_receipt.adapter_count must match current source plan replay.", errors)

    def test_strict_validate_warns_on_absolute_source_plan_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = _scenario_manifest(root / "heldout.json")
            plan_path = root / "external_eval_plan.json"
            receipt_path = root / "external_eval_receipt.json"
            validation = root / "validation.json"
            strict_validation = root / "strict_validation.json"
            run_cli(["external-eval-plan", "--scenario-manifest", str(manifest), "--out", str(plan_path)])

            receipt_code = run_cli(["external-eval-receipt", "--plan", str(plan_path), "--preserve-paths", "--out", str(receipt_path)])
            non_strict_code = run_cli(["validate", "--external-eval-receipt", str(receipt_path), "--out", str(validation)])
            strict_code = run_cli(["validate", "--external-eval-receipt", str(receipt_path), "--strict", "--out", str(strict_validation)])

            self.assertEqual(receipt_code, 1)
            self.assertEqual(non_strict_code, 0)
            self.assertEqual(strict_code, 1)
            warnings = "\n".join(warning for target in _read_json(validation)["targets"] for warning in target["warnings"])
            strict_warnings = "\n".join(
                warning for target in _read_json(strict_validation)["targets"] for warning in target["warnings"]
            )
            expected = "external_eval_receipt.source_plan.path is absolute"
            self.assertIn(expected, warnings)
            self.assertIn(expected, strict_warnings)

    def test_external_eval_receipt_live_request_is_recorded_and_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = _scenario_manifest(root / "heldout.json")
            plan_path = root / "external_eval_plan.json"
            receipt_path = root / "external_eval_receipt.json"
            run_cli(["external-eval-plan", "--scenario-manifest", str(manifest), "--out", str(plan_path)])

            receipt_code = run_cli(
                [
                    "external-eval-receipt",
                    "--plan",
                    str(plan_path),
                    "--live",
                    "--created-at",
                    "2026-07-03T00:00:00+00:00",
                    "--out",
                    str(receipt_path),
                ]
            )
            validate_code = run_cli(["validate", "--external-eval-receipt", str(receipt_path), "--strict"])

            self.assertEqual(receipt_code, 1)
            self.assertEqual(validate_code, 0)
            receipt = _read_json(receipt_path)
            self.assertEqual(receipt["launch"]["mode"], "live")
            self.assertFalse(receipt["launch"]["live_benchmarks_started"])
            self.assertIn("live_benchmark_not_requested: failed", receipt["blocked_reasons"])
            self.assertIn("live_external_eval_blocked_by_default", receipt["adapter_receipts"][0]["blocking_reasons"])

    def test_validate_rejects_forged_adapter_receipt_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = _scenario_manifest(root / "heldout.json")
            plan_path = root / "external_eval_plan.json"
            receipt_path = root / "external_eval_receipt.json"
            validation = root / "validation.json"
            run_cli(["external-eval-plan", "--scenario-manifest", str(manifest), "--out", str(plan_path)])
            run_cli(["external-eval-receipt", "--plan", str(plan_path), "--out", str(receipt_path)])
            receipt = _read_json(receipt_path)
            receipt["adapter_receipts"][0]["adapter_contract"]["provider_api_called_by_flight_recorder"] = True
            receipt["adapter_receipts"][0]["model_downloads_started"] = True
            receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            code = run_cli(["validate", "--external-eval-receipt", str(receipt_path), "--out", str(validation), "--strict"])

            self.assertEqual(code, 1)
            errors = "\n".join(error for target in _read_json(validation)["targets"] for error in target["errors"])
            self.assertIn("adapter_contract.provider_api_called_by_flight_recorder must be false", errors)
            self.assertIn("model_downloads_started must be false", errors)

    def test_validate_rejects_forged_passing_receipt_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = _scenario_manifest(root / "heldout.json")
            plan_path = root / "external_eval_plan.json"
            receipt_path = root / "external_eval_receipt.json"
            validation = root / "validation.json"
            run_cli(["external-eval-plan", "--scenario-manifest", str(manifest), "--out", str(plan_path)])
            run_cli(["external-eval-receipt", "--plan", str(plan_path), "--out", str(receipt_path)])
            receipt = _read_json(receipt_path)
            receipt["passed"] = True
            receipt["readiness"] = "dry_run_recorded"
            receipt["recommendation"] = "archive_external_eval_dry_run"
            receipt["failed_check_count"] = 0
            receipt["blocked_reasons"] = []
            for check in receipt["checks"]:
                check["passed"] = True
                check["summary"] = f"{check['id']}: passed"
            for row in receipt["adapter_receipts"]:
                row["ready"] = True
                row["planned_ready"] = True
                row["blocking_reasons"] = []
            receipt["ready_adapter_count"] = receipt["adapter_count"]
            receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            code = run_cli(["validate", "--external-eval-receipt", str(receipt_path), "--out", str(validation), "--strict"])

            self.assertEqual(code, 1)
            errors = "\n".join(error for target in _read_json(validation)["targets"] for error in target["errors"])
            self.assertIn("external_eval_receipt.passed must match current source plan replay.", errors)
            self.assertIn("external_eval_receipt.checks must match current source plan replay.", errors)
            self.assertIn("external_eval_receipt.adapter_receipts must match current source plan replay.", errors)

    def test_validate_and_schema_reject_unknown_external_eval_receipt_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = _scenario_manifest(root / "heldout.json")
            plan_path = root / "external_eval_plan.json"
            receipt_path = root / "external_eval_receipt.json"
            validation = root / "validation.json"
            run_cli(["external-eval-plan", "--scenario-manifest", str(manifest), "--out", str(plan_path)])
            run_cli(["external-eval-receipt", "--plan", str(plan_path), "--out", str(receipt_path)])
            receipt = _read_json(receipt_path)
            receipt["live_runner_receipt"] = "not-yet-validated"
            receipt["checks"][0]["provider_call"] = "forged"
            receipt["source_plan"]["absolute_source"] = "<redacted:external_eval_plan.json>"
            receipt["adapter_receipts"][0]["adapter_contract"]["provider_job_id"] = "job-123"
            receipt["launch"]["benchmark_url"] = "https://example.invalid/job"
            receipt["execution_boundary"]["live_side_effects"] = True
            receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            code = run_cli(["validate", "--external-eval-receipt", str(receipt_path), "--out", str(validation), "--strict"])
            schema_result = check_schema_file(receipt_path)

            self.assertEqual(code, 1)
            self.assertFalse(schema_result["passed"], schema_result["errors"])
            errors = "\n".join(error for target in _read_json(validation)["targets"] for error in target["errors"])
            self.assertIn("external_eval_receipt contains unknown field(s): ['live_runner_receipt'].", errors)
            self.assertIn("external_eval_receipt.checks[0] contains unknown field(s): ['provider_call'].", errors)
            self.assertIn("external_eval_receipt.source_plan contains unknown field(s): ['absolute_source'].", errors)
            self.assertIn(
                "external_eval_receipt.adapter_receipts[0].adapter_contract contains unknown field(s): ['provider_job_id'].",
                errors,
            )
            self.assertIn("external_eval_receipt.launch contains unknown field(s): ['benchmark_url'].", errors)
            self.assertIn("external_eval_receipt.execution_boundary contains unknown field(s): ['live_side_effects'].", errors)


def _scenario_manifest(path: Path) -> Path:
    payload = {
        "schema_version": "hfr.heldout_scenario_manifest.v1",
        "ready": True,
        "scenario_count": 1,
        "scenario_ids": ["email_reply_completion"],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
