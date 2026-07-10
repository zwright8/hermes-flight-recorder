import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from flightrecorder.rollout_generation import (
    build_agentic_rollout_plan,
    build_agentic_rollout_receipt,
    write_agentic_rollout_plan,
    write_agentic_rollout_receipt,
)
from flightrecorder.schema_registry import check_schema_contract, check_schema_file, list_schema_records
from flightrecorder.validation import validate_artifacts


ROOT = Path(__file__).resolve().parents[1]
SCENARIO = ROOT / "scenarios" / "prompt_injection_good.json"
VERIFIER = ROOT / "examples" / "external_verification" / "sqlite_task_state.verifier.json"


def copy_rollout_inputs(root: Path, *scenario_sources: Path) -> tuple[list[Path], Path]:
    scenario_dir = root / "scenarios"
    verifier_dir = root / "verifiers"
    scenario_dir.mkdir(parents=True, exist_ok=True)
    verifier_dir.mkdir(parents=True, exist_ok=True)
    scenarios = []
    for source in scenario_sources or (SCENARIO,):
        copied = scenario_dir / source.name
        shutil.copyfile(source, copied)
        scenarios.append(copied)
    verifier = verifier_dir / VERIFIER.name
    shutil.copyfile(VERIFIER, verifier)
    return scenarios, verifier


class RolloutGenerationTests(unittest.TestCase):
    def test_rollout_plan_rejects_untrusted_scenario_and_verifier_contracts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scenario = root / "scenario.json"
            verifier = root / "verifier.json"
            scenario.write_text('{"schema_version":"hfr.scenario.contract.v1"}\n', encoding="utf-8")
            verifier.write_text('{"schema_version":"hfr.verifier_config.v1"}\n', encoding="utf-8")

            plan = build_agentic_rollout_plan(
                out_path=root / "plan.json",
                iteration_id="untrusted-inputs",
                scenario_paths=[scenario],
                verifier_paths=[verifier],
                policies={"baseline": "local/base"},
                max_rollouts=1,
                created_at="2026-07-03T00:00:00+00:00",
            )

            self.assertFalse(plan["passed"])
            self.assertEqual(plan["readiness"], "blocked")
            self.assertFalse(plan["scenarios"][0]["exists"])
            self.assertFalse(plan["environment"]["external_state_verifiers"][0]["exists"])

    def test_rollout_plan_rejects_symlinked_scenario_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scenario = root / "scenario.json"
            shutil.copyfile(SCENARIO, scenario)
            linked = root / "linked.json"
            try:
                linked.symlink_to(scenario.name)
            except OSError as exc:
                self.skipTest(f"symlinks unavailable: {exc}")

            plan = build_agentic_rollout_plan(
                out_path=root / "plan.json",
                iteration_id="symlinked-input",
                scenario_paths=[linked],
                policies={"baseline": "local/base"},
                max_rollouts=1,
                created_at="2026-07-03T00:00:00+00:00",
            )

            self.assertFalse(plan["passed"])
            self.assertFalse(plan["scenarios"][0]["exists"])

    def test_rollout_receipt_rejects_schema_invalid_source_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scenarios, _ = copy_rollout_inputs(root)
            plan_path = root / "plan.json"
            plan = build_agentic_rollout_plan(
                out_path=plan_path,
                iteration_id="invalid-source-plan",
                scenario_paths=scenarios,
                policies={"baseline": "local/base"},
                max_rollouts=1,
                created_at="2026-07-03T00:00:00+00:00",
            )
            plan.pop("notes")
            write_agentic_rollout_plan(plan_path, plan)

            receipt = build_agentic_rollout_receipt(
                plan_path=plan_path,
                out_path=root / "receipt.json",
                created_at="2026-07-03T00:00:00+00:00",
            )

            self.assertFalse(receipt["passed"])
            self.assertEqual(receipt["readiness"], "blocked")
            self.assertFalse(receipt["source_plan"]["exists"])

    def test_committed_example_rollout_plan_is_public_safe_and_valid(self):
        plan_path = ROOT / "examples" / "rollout_generation" / "rollout_plan.json"
        payload = json.loads(plan_path.read_text(encoding="utf-8"))

        self.assertTrue(payload["passed"], payload["blocked_reasons"])
        self.assertEqual(payload["readiness"], "ready_for_harness_batch")
        self.assertEqual(payload["budget"]["planned_rollouts"], 6)
        self.assertFalse(payload["budget"]["live_provider_calls_allowed"])
        self.assertEqual(payload["environment"]["external_state_verifier_gate"]["declared_count"], 1)
        self.assertTrue(payload["environment"]["external_state_verifier_gate"]["all_declared_verifiers_resolved"])
        self.assertFalse(payload["environment"]["external_state_verifier_gate"]["verification_side_effects_started"])
        self.assertFalse(payload["execution_boundary"]["rollouts_started"])
        self.assertFalse(payload["execution_boundary"]["model_provider_calls_started"])
        self.assertFalse(payload["execution_boundary"]["dataset_rows_written"])
        self.assertTrue(payload["rejection_sampling"]["requires_review_calibration_before_training"])
        self.assertTrue(all(not Path(scenario["path"]).is_absolute() for scenario in payload["scenarios"]))

        schema = check_schema_file(plan_path)
        validation = validate_artifacts(agentic_rollout_plan_paths=[plan_path], strict=True)

        self.assertTrue(schema["passed"], schema["errors"])
        self.assertTrue(validation["passed"], validation)

    def test_committed_agentic_training_rollout_bundle_is_public_safe_and_valid(self):
        plan_path = ROOT / "examples" / "agentic_training" / "rollouts" / "rollout_plan.json"
        receipt_path = ROOT / "examples" / "agentic_training" / "rollouts" / "rollout_receipt.json"
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))

        self.assertTrue(plan["passed"], plan["blocked_reasons"])
        self.assertEqual(plan["budget"]["planned_rollouts"], 6)
        self.assertFalse(plan["execution_boundary"]["rollouts_started"])
        self.assertFalse(plan["execution_boundary"]["model_provider_calls_started"])
        self.assertFalse(plan["execution_boundary"]["dataset_rows_written"])
        self.assertTrue(receipt["passed"], receipt["blocked_reasons"])
        self.assertEqual(receipt["source_plan"]["path"], "rollout_plan.json")
        self.assertEqual(receipt["mock_rollout_count"], 6)
        self.assertFalse(receipt["execution_boundary"]["live_rollouts_started"])
        self.assertFalse(receipt["execution_boundary"]["model_provider_calls_started"])
        self.assertFalse(receipt["lineage"]["dataset_rows_created"])
        self.assertTrue(all(not Path(row["path"]).is_absolute() for row in plan["scenarios"]))

        validation = validate_artifacts(
            agentic_rollout_plan_paths=[plan_path],
            agentic_rollout_receipt_paths=[receipt_path],
            strict=True,
        )

        self.assertTrue(check_schema_file(plan_path)["passed"])
        self.assertTrue(check_schema_file(receipt_path)["passed"])
        self.assertTrue(validation["passed"], validation)

    def test_rollout_plan_builds_policy_scenario_matrix_without_running(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scenarios, verifier = copy_rollout_inputs(root, SCENARIO)
            plan = build_agentic_rollout_plan(
                out_path=root / "rollout_plan.json",
                iteration_id="rollout-001",
                scenario_paths=scenarios,
                policies={"baseline": "local/base", "candidate": "local/candidate", "teacher": "local/teacher"},
                max_rollouts=3,
                verifier_paths=[verifier],
                created_at="2026-07-03T00:00:00+00:00",
            )

            self.assertTrue(plan["passed"], plan["blocked_reasons"])
            self.assertEqual(plan["budget"]["planned_rollouts"], 3)
            self.assertEqual(plan["scenarios"][0]["path"], "scenarios/prompt_injection_good.json")
            verifier_gate = plan["environment"]["external_state_verifier_gate"]
            self.assertEqual(verifier_gate["declared_count"], 1)
            self.assertEqual(verifier_gate["resolved_count"], 1)
            self.assertTrue(verifier_gate["all_declared_verifiers_resolved"])
            self.assertFalse(verifier_gate["verification_side_effects_started"])
            self.assertFalse(plan["execution_boundary"]["rollouts_started"])
            self.assertFalse(plan["execution_boundary"]["dataset_rows_written"])
            self.assertTrue(plan["rejection_sampling"]["requires_review_calibration_before_training"])
            schema = check_schema_contract(plan)
            self.assertTrue(schema["passed"], schema["errors"])

    def test_cli_writes_schema_checkable_validatable_rollout_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scenarios, verifier = copy_rollout_inputs(root, SCENARIO)
            out = root / "rollout_plan.json"
            receipt_out = root / "rollout_receipt.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "flightrecorder",
                    "agentic-rollout-plan",
                    "--iteration-id",
                    "rollout-cli",
                    "--scenario",
                    str(scenarios[0]),
                    "--policy",
                    "baseline=local/base",
                    "--policy",
                    "candidate=local/candidate",
                    "--max-rollouts",
                    "2",
                    "--verifier",
                    str(verifier),
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
            schema = check_schema_file(out)
            self.assertTrue(schema["passed"], schema["errors"])
            validation = validate_artifacts(agentic_rollout_plan_paths=[out], strict=True)
            self.assertTrue(validation["passed"], validation)
            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(payload["budget"]["planned_rollouts"], 2)
            self.assertTrue(payload["environment"]["external_state_verifier_gate"]["all_declared_verifiers_resolved"])

            receipt_completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "flightrecorder",
                    "agentic-rollout-receipt",
                    "--plan",
                    str(out),
                    "--created-at",
                    "2026-07-03T00:00:00+00:00",
                    "--out",
                    str(receipt_out),
                ],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertEqual(receipt_completed.returncode, 0, receipt_completed.stderr + receipt_completed.stdout)
            receipt_schema = check_schema_file(receipt_out)
            self.assertTrue(receipt_schema["passed"], receipt_schema["errors"])
            receipt_validation = validate_artifacts(agentic_rollout_receipt_paths=[receipt_out], strict=True)
            self.assertTrue(receipt_validation["passed"], receipt_validation)
            receipt = json.loads(receipt_out.read_text(encoding="utf-8"))
            self.assertEqual(receipt["mock_rollout_count"], 2)
            self.assertFalse(receipt["execution_boundary"]["model_provider_calls_started"])
            self.assertFalse(receipt["lineage"]["dataset_rows_created"])

    def test_rollout_plan_redacts_unreplayable_source_refs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs = root / "inputs"
            reports = root / "reports"
            inputs.mkdir()
            reports.mkdir()
            scenario_path = inputs / "private_scenario.json"
            verifier_path = inputs / "private.verifier.json"
            scenario_path.write_text(
                json.dumps({"id": "private-scenario", "schema_version": "PRIVATE_SENTINEL"}, indent=2) + "\n",
                encoding="utf-8",
            )
            verifier_path.write_text(
                json.dumps({"schema_version": "hfr.verifier_config.v1", "private": "PRIVATE_SENTINEL"}, indent=2) + "\n",
                encoding="utf-8",
            )
            plan_path = reports / "rollout_plan.json"
            plan = build_agentic_rollout_plan(
                out_path=plan_path,
                iteration_id="rollout-redacted-plan",
                scenario_paths=[scenario_path],
                policies={"baseline": "local/base"},
                max_rollouts=1,
                verifier_paths=[verifier_path],
                preserve_paths=True,
                created_at="2026-07-03T00:00:00+00:00",
            )
            write_agentic_rollout_plan(plan_path, plan)

            payload = json.loads(plan_path.read_text(encoding="utf-8"))
            rendered = json.dumps(payload, sort_keys=True)
            self.assertFalse(payload["passed"])
            self.assertEqual(payload["budget"]["planned_rollouts"], 0)
            self.assertEqual(payload["scenarios"][0]["path"], "<redacted:private_scenario.json>")
            self.assertFalse(payload["scenarios"][0]["exists"])
            self.assertEqual(payload["environment"]["external_state_verifiers"][0]["path"], "<redacted:private.verifier.json>")
            self.assertFalse(payload["environment"]["external_state_verifier_gate"]["all_declared_verifiers_resolved"])
            self.assertNotIn(str(inputs), rendered)
            self.assertNotIn("PRIVATE_SENTINEL", rendered)

            validation = validate_artifacts(agentic_rollout_plan_paths=[plan_path], strict=True)
            self.assertTrue(validation["passed"], validation)

    def test_rollout_plan_validation_rejects_symlink_source_refs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scenarios, verifier = copy_rollout_inputs(root, SCENARIO)
            scenario_link = scenarios[0].with_name("prompt_injection_link.json")
            verifier_link = verifier.with_name("sqlite_task_state-link.verifier.json")
            try:
                scenario_link.symlink_to(scenarios[0].name)
                verifier_link.symlink_to(verifier.name)
            except (NotImplementedError, OSError) as exc:
                self.skipTest(f"symlinks unavailable: {exc}")
            plan_path = root / "rollout_plan.json"
            plan = build_agentic_rollout_plan(
                out_path=plan_path,
                iteration_id="rollout-symlink-refs",
                scenario_paths=scenarios,
                policies={"baseline": "local/base"},
                max_rollouts=1,
                verifier_paths=[verifier],
                created_at="2026-07-03T00:00:00+00:00",
            )
            write_agentic_rollout_plan(plan_path, plan)
            forged = json.loads(plan_path.read_text(encoding="utf-8"))
            forged["scenarios"][0]["path"] = f"scenarios/{scenario_link.name}"
            forged["environment"]["external_state_verifiers"][0]["path"] = f"verifiers/{verifier_link.name}"
            plan_path.write_text(json.dumps(forged, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            validation = validate_artifacts(agentic_rollout_plan_paths=[plan_path], strict=True)

            self.assertFalse(validation["passed"], validation)
            errors = "\n".join(error for target in validation["targets"] for error in target["errors"])
            self.assertIn("agentic_rollout_plan.scenarios[0].path must resolve to a regular non-symlink scenario file.", errors)
            self.assertIn(
                "agentic_rollout_plan.environment.external_state_verifiers[0].path must resolve to a regular non-symlink verifier config file.",
                errors,
            )

    def test_rollout_receipt_records_mock_rows_without_side_effects(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scenarios, verifier = copy_rollout_inputs(root, SCENARIO)
            plan_path = root / "rollout_plan.json"
            receipt_path = root / "rollout_receipt.json"
            plan = build_agentic_rollout_plan(
                out_path=plan_path,
                iteration_id="rollout-receipt",
                scenario_paths=scenarios,
                policies={"baseline": "local/base", "candidate": "local/candidate"},
                max_rollouts=2,
                verifier_paths=[verifier],
                created_at="2026-07-03T00:00:00+00:00",
            )
            write_agentic_rollout_plan(plan_path, plan)

            receipt = build_agentic_rollout_receipt(
                plan_path=plan_path,
                out_path=receipt_path,
                created_at="2026-07-03T00:00:00+00:00",
            )
            write_agentic_rollout_receipt(receipt_path, receipt)

            schema = check_schema_file(receipt_path)
            self.assertTrue(schema["passed"], schema["errors"])
            validation = validate_artifacts(agentic_rollout_receipt_paths=[receipt_path], strict=True)
            self.assertTrue(validation["passed"], validation)
            payload = json.loads(receipt_path.read_text(encoding="utf-8"))
            self.assertTrue(payload["passed"], payload["blocked_reasons"])
            self.assertEqual(payload["source_plan"]["path"], "rollout_plan.json")
            self.assertEqual(payload["mock_rollout_count"], 2)
            self.assertTrue(all(row["status"] == "mock_recorded" for row in payload["mock_rollouts"]))
            self.assertFalse(any(row["model_provider_called"] for row in payload["mock_rollouts"]))
            self.assertFalse(any(row["dataset_row_written"] for row in payload["mock_rollouts"]))
            self.assertTrue(payload["environment"]["external_state_verifier_gate"]["all_declared_verifiers_resolved"])

    def test_rollout_receipt_writer_redacts_unreplayable_source_plan_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scenarios, _ = copy_rollout_inputs(root, SCENARIO)
            plan_path = root / "rollout_plan.json"
            reports = root / "reports"
            reports.mkdir()
            receipt_path = reports / "rollout_receipt.json"
            plan = build_agentic_rollout_plan(
                out_path=plan_path,
                iteration_id="rollout-redacted-source",
                scenario_paths=scenarios,
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
            receipt["source_plan"]["path"] = str(plan_path)
            write_agentic_rollout_receipt(receipt_path, receipt)

            payload = json.loads(receipt_path.read_text(encoding="utf-8"))
            rendered = json.dumps(payload, sort_keys=True)
            self.assertFalse(payload["passed"])
            self.assertEqual(payload["source_plan"]["path"], "<redacted:rollout_plan.json>")
            self.assertFalse(payload["source_plan"]["exists"])
            self.assertIsNone(payload["source_plan"]["sha256"])
            self.assertEqual(payload["mock_rollout_count"], 0)
            self.assertNotIn(str(root), rendered)

            validation = validate_artifacts(agentic_rollout_receipt_paths=[receipt_path], strict=True)
            self.assertTrue(validation["passed"], validation)

    def test_rollout_receipt_redacts_external_source_plan_without_reading(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs = root / "inputs"
            reports = root / "reports"
            inputs.mkdir()
            reports.mkdir()
            plan_path = inputs / "private_rollout_plan.json"
            receipt_path = reports / "rollout_receipt.json"
            plan_path.write_text(
                json.dumps({"schema_version": "PRIVATE_SENTINEL", "iteration_id": "PRIVATE_SENTINEL"}, indent=2) + "\n",
                encoding="utf-8",
            )

            receipt = build_agentic_rollout_receipt(
                plan_path=plan_path,
                out_path=receipt_path,
                preserve_paths=True,
                output_base_dir=reports,
                created_at="2026-07-03T00:00:00+00:00",
            )
            write_agentic_rollout_receipt(receipt_path, receipt)

            payload = json.loads(receipt_path.read_text(encoding="utf-8"))
            rendered = json.dumps(payload, sort_keys=True)
            self.assertFalse(payload["passed"])
            self.assertEqual(payload["source_plan"]["path"], "<redacted:private_rollout_plan.json>")
            self.assertFalse(payload["source_plan"]["exists"])
            self.assertIsNone(payload["source_plan"]["passed"])
            self.assertEqual(payload["source_plan"]["readiness"], "")
            self.assertEqual(payload["environment"]["id"], "offline_mock")
            self.assertNotIn(str(inputs), rendered)
            self.assertNotIn("PRIVATE_SENTINEL", rendered)

            schema = check_schema_file(receipt_path)
            self.assertTrue(schema["passed"], schema["errors"])
            validation = validate_artifacts(agentic_rollout_receipt_paths=[receipt_path], strict=True)
            self.assertTrue(validation["passed"], validation)

    def test_rollout_receipt_validation_rejects_symlink_source_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scenarios, _ = copy_rollout_inputs(root, SCENARIO)
            plan_path = root / "rollout_plan.json"
            plan_link = root / "rollout_plan-link.json"
            receipt_path = root / "rollout_receipt.json"
            plan = build_agentic_rollout_plan(
                out_path=plan_path,
                iteration_id="rollout-symlink-source-plan",
                scenario_paths=scenarios,
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
            try:
                plan_link.symlink_to(plan_path.name)
            except (NotImplementedError, OSError) as exc:
                self.skipTest(f"symlinks unavailable: {exc}")
            forged = json.loads(receipt_path.read_text(encoding="utf-8"))
            forged["source_plan"]["path"] = plan_link.name
            receipt_path.write_text(json.dumps(forged, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            validation = validate_artifacts(agentic_rollout_receipt_paths=[receipt_path], strict=True)

            self.assertFalse(validation["passed"], validation)
            errors = "\n".join(error for target in validation["targets"] for error in target["errors"])
            self.assertIn(
                "agentic_rollout_receipt.source_plan.path must resolve to a regular non-symlink agentic rollout plan file.",
                errors,
            )

    def test_rollout_receipt_validation_rejects_live_side_effect_claims(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scenarios, _ = copy_rollout_inputs(root, SCENARIO)
            plan_path = root / "rollout_plan.json"
            receipt_path = root / "rollout_receipt.json"
            plan = build_agentic_rollout_plan(
                out_path=plan_path,
                iteration_id="rollout-forged",
                scenario_paths=scenarios,
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
            receipt["execution_boundary"]["model_provider_calls_started"] = True
            receipt["environment"]["external_state_verifier_gate"]["verification_side_effects_started"] = True
            write_agentic_rollout_receipt(receipt_path, receipt)

            validation = validate_artifacts(agentic_rollout_receipt_paths=[receipt_path], strict=True)
            self.assertFalse(validation["passed"], validation)
            errors = "\n".join(error for target in validation["targets"] for error in target["errors"])
            self.assertIn("execution_boundary.model_provider_calls_started must be false", errors)
            self.assertIn("external_state_verifier_gate.verification_side_effects_started must be false", errors)

    def test_rollout_plan_validation_rejects_forged_live_provider_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scenarios, verifier = copy_rollout_inputs(root, SCENARIO)
            plan_path = root / "rollout_plan.json"
            plan = build_agentic_rollout_plan(
                out_path=plan_path,
                iteration_id="rollout-forged-provider",
                scenario_paths=scenarios,
                policies={"baseline": "local/base", "candidate": "local/candidate"},
                max_rollouts=2,
                verifier_paths=[verifier],
                created_at="2026-07-03T00:00:00+00:00",
            )
            plan["provider_job_id"] = "job_live"
            plan["checks"][0]["provider_trace_id"] = "trace_live"
            plan["budget"]["cloud_budget_usd"] = 20
            plan["environment"]["provider_region"] = "us-east-1"
            plan["environment"]["external_state_verifiers"][0]["credential_value"] = "redacted"
            plan["environment"]["external_state_verifier_gate"]["credential_secret_ref"] = "HFR_VERIFIER_TOKEN"
            plan["policies"][0]["provider_api_key_env"] = "MODEL_PROVIDER_KEY"
            plan["scenarios"][0]["signed_url"] = "https://example.invalid/scenario.json"
            plan["harness_batches"][0]["live_provider_job_id"] = "job_live"
            plan["rejection_sampling"]["provider_dataset_uri"] = "s3://example-bucket/rollouts.jsonl"
            plan["lineage"]["trace_signed_url"] = "https://example.invalid/trace.jsonl"
            plan["execution_boundary"]["live_endpoint_url"] = "https://example.invalid/run"
            write_agentic_rollout_plan(plan_path, plan)

            schema = check_schema_file(plan_path)
            self.assertFalse(schema["passed"], schema)
            validation = validate_artifacts(agentic_rollout_plan_paths=[plan_path], strict=True)
            self.assertFalse(validation["passed"], validation)
            errors = "\n".join(error for target in validation["targets"] for error in target["errors"])
            self.assertIn("agentic_rollout_plan contains unknown field", errors)
            self.assertIn("agentic_rollout_plan.checks[0] contains unknown field", errors)
            self.assertIn("agentic_rollout_plan.environment contains unknown field", errors)
            self.assertIn("agentic_rollout_plan.harness_batches[0] contains unknown field", errors)
            self.assertIn("agentic_rollout_plan.execution_boundary contains unknown field", errors)

    def test_rollout_receipt_validation_rejects_forged_live_provider_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scenarios, verifier = copy_rollout_inputs(root, SCENARIO)
            plan_path = root / "rollout_plan.json"
            receipt_path = root / "rollout_receipt.json"
            plan = build_agentic_rollout_plan(
                out_path=plan_path,
                iteration_id="rollout-receipt-forged-provider",
                scenario_paths=scenarios,
                policies={"baseline": "local/base"},
                max_rollouts=1,
                verifier_paths=[verifier],
                created_at="2026-07-03T00:00:00+00:00",
            )
            write_agentic_rollout_plan(plan_path, plan)
            receipt = build_agentic_rollout_receipt(
                plan_path=plan_path,
                out_path=receipt_path,
                created_at="2026-07-03T00:00:00+00:00",
            )
            receipt["provider_job_id"] = "job_live"
            receipt["checks"][0]["provider_trace_id"] = "trace_live"
            receipt["source_plan"]["signed_url"] = "https://example.invalid/rollout-plan.json"
            receipt["environment"]["provider_region"] = "us-east-1"
            receipt["environment"]["external_state_verifiers"][0]["credential_value"] = "redacted"
            receipt["environment"]["external_state_verifier_gate"]["credential_secret_ref"] = "HFR_VERIFIER_TOKEN"
            receipt["mock_rollouts"][0]["provider_completion_id"] = "completion_live"
            receipt["lineage"]["trace_path"] = "traces/live.jsonl"
            receipt["execution_boundary"]["live_endpoint_url"] = "https://example.invalid/run"
            write_agentic_rollout_receipt(receipt_path, receipt)

            schema = check_schema_file(receipt_path)
            self.assertFalse(schema["passed"], schema)
            validation = validate_artifacts(agentic_rollout_receipt_paths=[receipt_path], strict=True)
            self.assertFalse(validation["passed"], validation)
            errors = "\n".join(error for target in validation["targets"] for error in target["errors"])
            self.assertIn("agentic_rollout_receipt contains unknown field", errors)
            self.assertIn("agentic_rollout_receipt.checks[0] contains unknown field", errors)
            self.assertIn("agentic_rollout_receipt.source_plan contains unknown field", errors)
            self.assertIn("agentic_rollout_receipt.mock_rollouts[0] contains unknown field", errors)
            self.assertIn("agentic_rollout_receipt.execution_boundary contains unknown field", errors)

    def test_rollout_plan_blocks_missing_verifier_refs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scenarios, _ = copy_rollout_inputs(root, SCENARIO)
            missing_verifier = root / "missing.verifier.json"
            plan = build_agentic_rollout_plan(
                out_path=root / "rollout_plan.json",
                iteration_id="rollout-missing-verifier",
                scenario_paths=scenarios,
                policies={"baseline": "local/base"},
                max_rollouts=1,
                verifier_paths=[missing_verifier],
                created_at="2026-07-03T00:00:00+00:00",
            )

            self.assertFalse(plan["passed"])
            self.assertEqual(plan["readiness"], "blocked")
            self.assertEqual(plan["environment"]["external_state_verifier_gate"]["declared_count"], 1)
            self.assertEqual(plan["environment"]["external_state_verifier_gate"]["resolved_count"], 0)
            self.assertFalse(plan["environment"]["external_state_verifier_gate"]["all_declared_verifiers_resolved"])
            self.assertIn("external_state_verifiers_resolved: passed=False", plan["blocked_reasons"])

    def test_schema_is_registered(self):
        names = {record["name"] for record in list_schema_records()}
        self.assertIn("agentic_rollout_plan", names)
        self.assertIn("agentic_rollout_receipt", names)


if __name__ == "__main__":
    unittest.main()
