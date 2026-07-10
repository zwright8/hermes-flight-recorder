import hashlib
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from flightrecorder.cli import main
from flightrecorder.external_eval import (
    EXTERNAL_EVAL_ADAPTER_RECEIPT_TYPES,
    adapter_choices,
    build_external_eval_plan,
    write_external_eval_plan,
)
from flightrecorder.schema_registry import check_schema_file
from flightrecorder.validation import validate_artifacts


ROOT = Path(__file__).resolve().parents[1]


def run_cli(args):
    with redirect_stdout(StringIO()):
        return main(args)


class ExternalEvalPlanTests(unittest.TestCase):
    def test_committed_example_external_eval_plan_covers_fail_closed_adapters(self):
        plan_path = ROOT / "examples" / "external_eval" / "external_eval_plan.json"
        plan = _read_json(plan_path)
        adapter_ids = {adapter["id"] for adapter in plan["adapters"]}

        self.assertEqual(adapter_ids, set(adapter_choices()))
        self.assertEqual(set(plan["selected_adapters"]), set(adapter_choices()))
        self.assertEqual(plan["adapter_count"], len(adapter_choices()))
        self.assertEqual(plan["ready_adapter_count"], 0)
        self.assertFalse(plan["ready"])
        self.assertFalse(plan["governance_handoff"]["external_eval_claims_allowed"])
        for adapter in plan["adapters"]:
            self.assertFalse(adapter["ready"])
            self.assertFalse(adapter["adapter_contract"]["live_benchmark_supported"])
            self.assertFalse(adapter["adapter_contract"]["provider_api_called_by_flight_recorder"])
            self.assertEqual(sorted(adapter["adapter_contract"]["receipt_types"]), sorted(EXTERNAL_EVAL_ADAPTER_RECEIPT_TYPES))

        schema_result = check_schema_file(plan_path)
        validate_code = run_cli(["validate", "--external-eval-plan", str(plan_path), "--strict"])

        self.assertTrue(schema_result["passed"], schema_result["errors"])
        self.assertEqual(validate_code, 0)

    def test_committed_example_heldout_chain_is_self_contained(self):
        eval_root = ROOT / "examples" / "external_eval"
        suite_path = eval_root / "heldout_suite_summary.json"
        heldout_path = eval_root / "heldout_scenarios.json"
        plan_path = eval_root / "external_eval_plan.json"
        receipt_path = eval_root / "external_eval_receipt.json"
        suite = _read_json(suite_path)

        validation = validate_artifacts(
            runs_dir=eval_root / "heldout_runs",
            suite_summary_paths=[suite_path],
            heldout_manifest_paths=[heldout_path],
            external_eval_plan_paths=[plan_path],
            external_eval_receipt_paths=[receipt_path],
            strict=True,
        )
        self.assertTrue(validation["passed"], validation)
        self.assertEqual(suite["total"], 1)
        self.assertEqual(suite["passed"], 1)
        self.assertEqual(suite["error_count"], 0)

        run = suite["runs"][0]
        for field_name in (
            "scenario_path",
            "trace_path",
            "before_state_path",
            "state_path",
            "run_dir",
            "report",
            "scorecard",
            "run_digest",
            "lineage",
        ):
            relative_path = Path(run[field_name])
            self.assertFalse(relative_path.is_absolute(), field_name)
            self.assertNotIn("..", relative_path.parts, field_name)
            resolved = (eval_root / relative_path).resolve()
            self.assertTrue(resolved.is_relative_to(eval_root.resolve()), field_name)
            self.assertTrue(resolved.exists(), field_name)

        for field_name in ("scenario", "trace", "before_state", "state"):
            source_path = eval_root / run[f"{field_name}_path"]
            self.assertEqual(run[f"{field_name}_sha256"], hashlib.sha256(source_path.read_bytes()).hexdigest())

        lineage = _read_json(eval_root / run["lineage"])
        self.assertTrue(lineage["replay"]["self_contained"])
        for record in lineage["inputs"]:
            source_path = eval_root / record["path"]
            self.assertTrue(source_path.is_file(), record["name"])
            self.assertEqual(record["size_bytes"], source_path.stat().st_size, record["name"])
            self.assertEqual(record["sha256"], hashlib.sha256(source_path.read_bytes()).hexdigest(), record["name"])

        heldout = _read_json(heldout_path)
        plan = _read_json(plan_path)
        receipt = _read_json(receipt_path)
        self.assertEqual(heldout["sources"][0]["path"], suite_path.name)
        self.assertEqual(plan["inputs"]["scenario_manifest"]["path"], heldout_path.name)
        self.assertEqual(receipt["source_plan"]["path"], plan_path.name)

    def test_external_eval_plan_cli_fails_closed_but_validates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = _scenario_manifest(root / "heldout.json")
            out = root / "external_eval_plan.json"

            code = run_cli(["external-eval-plan", "--scenario-manifest", str(manifest), "--out", str(out)])
            validate_code = run_cli(["validate", "--external-eval-plan", str(out), "--strict"])
            schema_result = check_schema_file(out)

            self.assertEqual(code, 1)
            self.assertEqual(validate_code, 0)
            self.assertTrue(schema_result["passed"], schema_result["errors"])
            plan = _read_json(out)
            self.assertFalse(plan["ready"])
            self.assertEqual(plan["adapter_count"], len(adapter_choices()))
            self.assertEqual(plan["ready_adapter_count"], 0)
            self.assertIn("no_ready_external_adapters", plan["blocking_reasons"])
            self.assertIn("adapter_disabled_until_allow_installed", plan["blocking_reasons"])
            self.assertEqual(len(plan["inputs"]["scenario_manifest"]["sha256"]), 64)
            self.assertEqual(plan["inputs"]["scenario_manifest"]["path"], manifest.name)
            self.assertEqual(plan["inputs"]["scenario_manifest"]["size_bytes"], manifest.stat().st_size)
            self.assertFalse(plan["governance_handoff"]["external_eval_claims_allowed"])
            contract = plan["adapters"][0]["adapter_contract"]
            self.assertEqual(contract["dry_run_transport"], "plan_and_receipt_only")
            self.assertFalse(contract["live_benchmark_supported"])
            self.assertFalse(contract["provider_api_called_by_flight_recorder"])
            self.assertEqual(sorted(contract["receipt_types"]), sorted(EXTERNAL_EVAL_ADAPTER_RECEIPT_TYPES))

    def test_external_eval_plan_ready_path_with_mocked_dependency(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = _scenario_manifest(root / "heldout.json")
            out = root / "external_eval_plan.json"

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
            write_external_eval_plan(plan, out)
            validate_code = run_cli(["validate", "--external-eval-plan", str(out), "--strict"])
            schema_result = check_schema_file(out)

            self.assertEqual(validate_code, 0)
            self.assertTrue(schema_result["passed"], schema_result["errors"])
            self.assertTrue(plan["ready"])
            self.assertEqual(plan["ready_adapter_count"], 1)
            self.assertEqual(plan["blocking_reasons"], [])
            self.assertTrue(plan["adapters"][0]["ready"])
            self.assertEqual(plan["adapters"][0]["blocking_reasons"], [])
            self.assertTrue(plan["governance_handoff"]["external_eval_claims_allowed"])
            written = _read_json(out)
            self.assertEqual(written["inputs"]["scenario_manifest"]["path"], manifest.name)
            self.assertEqual(written["inputs"]["scenario_manifest"]["size_bytes"], manifest.stat().st_size)

    def test_local_mock_external_eval_plan_ready_path_is_keyless(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = _scenario_manifest(root / "heldout.json")
            out = root / "external_eval_plan.json"

            code = run_cli(
                [
                    "external-eval-plan",
                    "--adapter",
                    "local_mock",
                    "--scenario-manifest",
                    str(manifest),
                    "--model-endpoint",
                    "local/mock-candidate",
                    "--allow-installed",
                    "--out",
                    str(out),
                ]
            )
            validate_code = run_cli(["validate", "--external-eval-plan", str(out), "--strict"])
            schema_result = check_schema_file(out)

            self.assertEqual(code, 0)
            self.assertEqual(validate_code, 0)
            self.assertTrue(schema_result["passed"], schema_result["errors"])
            plan = _read_json(out)
            self.assertTrue(plan["ready"])
            self.assertEqual(plan["selected_adapters"], ["local_mock"])
            self.assertEqual(plan["ready_adapter_count"], 1)
            self.assertEqual(plan["blocking_reasons"], [])
            adapter = plan["adapters"][0]
            self.assertEqual(adapter["dependency_status"], {"available": True, "commands": {}, "imports": {}})
            self.assertFalse(adapter["adapter_contract"]["live_benchmark_supported"])
            self.assertFalse(adapter["adapter_contract"]["provider_api_called_by_flight_recorder"])

    def test_preserve_paths_keeps_plan_manifest_output_relative(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = _scenario_manifest(root / "heldout.json")
            out = root / "external_eval_plan.json"
            validation = root / "validation.json"
            strict_validation = root / "strict_validation.json"

            code = run_cli(["external-eval-plan", "--scenario-manifest", str(manifest), "--preserve-paths", "--out", str(out)])
            non_strict_code = run_cli(["validate", "--external-eval-plan", str(out), "--out", str(validation)])
            strict_code = run_cli(["validate", "--external-eval-plan", str(out), "--strict", "--out", str(strict_validation)])

            self.assertEqual(code, 1)
            self.assertEqual(non_strict_code, 0)
            self.assertEqual(strict_code, 0)
            plan = _read_json(out)
            self.assertEqual(plan["inputs"]["scenario_manifest"]["path"], manifest.name)
            warnings = "\n".join(warning for target in _read_json(validation)["targets"] for warning in target["warnings"])
            strict_warnings = "\n".join(
                warning for target in _read_json(strict_validation)["targets"] for warning in target["warnings"]
            )
            self.assertNotIn("external_eval_plan.inputs.scenario_manifest.path is absolute", warnings)
            self.assertNotIn("external_eval_plan.inputs.scenario_manifest.path is absolute", strict_warnings)

    def test_external_eval_plan_redacts_unreplayable_manifest_without_reading_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = root / "sources"
            report_dir = root / "reports"
            source_dir.mkdir()
            report_dir.mkdir()
            manifest = _scenario_manifest(source_dir / "heldout.json", scenario_ids=["LOCAL_SECRET_SCENARIO"])
            out = report_dir / "external_eval_plan.json"

            code = run_cli(["external-eval-plan", "--scenario-manifest", str(manifest), "--preserve-paths", "--out", str(out)])
            validate_code = run_cli(["validate", "--external-eval-plan", str(out), "--strict"])
            schema_result = check_schema_file(out)

            self.assertEqual(code, 1)
            self.assertEqual(validate_code, 0)
            self.assertTrue(schema_result["passed"], schema_result["errors"])
            rendered = out.read_text(encoding="utf-8")
            self.assertNotIn(str(source_dir), rendered)
            self.assertNotIn("../sources", rendered)
            self.assertNotIn("LOCAL_SECRET_SCENARIO", rendered)
            plan = _read_json(out)
            scenario_manifest = plan["inputs"]["scenario_manifest"]
            self.assertEqual(scenario_manifest["path"], "<redacted:heldout.json>")
            self.assertFalse(scenario_manifest["exists"])
            self.assertIsNone(scenario_manifest["sha256"])
            self.assertIsNone(scenario_manifest["size_bytes"])
            self.assertIsNone(scenario_manifest["schema_version"])
            self.assertIsNone(scenario_manifest["ready"])
            self.assertIsNone(scenario_manifest["scenario_count"])
            self.assertIn("missing_scenario_manifest", plan["blocking_reasons"])
            self.assertFalse(plan["governance_handoff"]["external_eval_claims_allowed"])

    def test_external_eval_plan_writer_refreshes_direct_api_external_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = root / "sources"
            report_dir = root / "reports"
            source_dir.mkdir()
            report_dir.mkdir()
            manifest = _scenario_manifest(source_dir / "heldout.json")
            out = report_dir / "external_eval_plan.json"
            with patch(
                "flightrecorder.external_eval._dependency_status",
                return_value={"available": True, "imports": {"bfcl_eval": True}, "commands": {"bfcl": True}},
            ):
                plan = build_external_eval_plan(
                    adapters=["bfcl"],
                    scenario_manifest=manifest,
                    model_endpoint="http://127.0.0.1:8000/v1",
                    tool_schema_set="tools-v1",
                    allow_installed=True,
                )
            self.assertTrue(plan["ready"])

            write_external_eval_plan(plan, out)
            validate_code = run_cli(["validate", "--external-eval-plan", str(out), "--strict"])
            written = _read_json(out)

            self.assertEqual(validate_code, 0)
            self.assertFalse(written["ready"])
            self.assertEqual(written["inputs"]["scenario_manifest"]["path"], "<redacted:heldout.json>")
            self.assertIsNone(written["adapters"][0]["execution_contract"]["scenario_manifest_sha256"])
            self.assertIn("missing_scenario_manifest", written["adapters"][0]["blocking_reasons"])
            self.assertFalse(written["governance_handoff"]["external_eval_claims_allowed"])

    def test_external_eval_plan_writer_redacts_relative_manifest_outside_output_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = root / "sources"
            report_dir = root / "reports"
            source_dir.mkdir()
            report_dir.mkdir()
            _scenario_manifest(source_dir / "heldout.json")
            out = report_dir / "external_eval_plan.json"
            previous_cwd = Path.cwd()
            try:
                os.chdir(root)
                with patch(
                    "flightrecorder.external_eval._dependency_status",
                    return_value={"available": True, "imports": {"bfcl_eval": True}, "commands": {"bfcl": True}},
                ):
                    plan = build_external_eval_plan(
                        adapters=["bfcl"],
                        scenario_manifest=Path("sources") / "heldout.json",
                        model_endpoint="http://127.0.0.1:8000/v1",
                        tool_schema_set="tools-v1",
                        allow_installed=True,
                    )
                self.assertTrue(plan["ready"])
                write_external_eval_plan(plan, out)
            finally:
                os.chdir(previous_cwd)

            validate_code = run_cli(["validate", "--external-eval-plan", str(out), "--strict"])
            written = _read_json(out)

            self.assertEqual(validate_code, 0)
            self.assertFalse(written["ready"])
            self.assertEqual(written["inputs"]["scenario_manifest"]["path"], "<redacted:heldout.json>")
            self.assertNotIn("../sources", out.read_text(encoding="utf-8"))
            self.assertIsNone(written["adapters"][0]["execution_contract"]["scenario_manifest_sha256"])
            self.assertIn("missing_scenario_manifest", written["adapters"][0]["blocking_reasons"])

    def test_eval_summary_surfaces_external_adapter_blockers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = _scenario_manifest(root / "heldout.json")
            plan_path = root / "external_eval_plan.json"
            out = root / "eval_summary.json"
            run_cli(["external-eval-plan", "--scenario-manifest", str(manifest), "--out", str(plan_path)])

            code = run_cli(["eval-summary", "--external-adapter-plan", f"external={plan_path}", "--out", str(out)])
            validate_code = run_cli(["validate", "--eval-summary", str(out), "--strict"])

            self.assertEqual(code, 1)
            self.assertEqual(validate_code, 0)
            summary = _read_json(out)
            self.assertFalse(summary["passed"])
            self.assertEqual(summary["external_adapter_plan_count"], 1)
            self.assertFalse(summary["external_adapter_plans"][0]["ready"])
            self.assertTrue(any(risk["source"] == "external_adapter_plan" for risk in summary["risks"]))

    def test_validate_rejects_ready_plan_with_missing_adapter_blockers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = _scenario_manifest(root / "heldout.json")
            out = root / "external_eval_plan.json"
            validation = root / "validation.json"
            plan = build_external_eval_plan(adapters=["bfcl"], scenario_manifest=manifest)
            plan["ready"] = True
            plan["blocking_reasons"] = []
            plan["adapters"][0]["ready"] = True
            plan["adapters"][0]["blocking_reasons"] = []
            write_external_eval_plan(plan, out)

            code = run_cli(["validate", "--external-eval-plan", str(out), "--out", str(validation)])

            self.assertEqual(code, 1)
            errors = "\n".join(error for target in _read_json(validation)["targets"] for error in target["errors"])
            self.assertIn("external_eval_plan.ready_adapter_count expected 1", errors)
            self.assertIn("blocking_reasons must include dependencies_missing", errors)
            self.assertIn("ready cannot be true while blockers remain", errors)

    def test_validate_rejects_manifest_fingerprint_without_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = _scenario_manifest(root / "heldout.json")
            out = root / "external_eval_plan.json"
            validation = root / "validation.json"
            run_cli(["external-eval-plan", "--scenario-manifest", str(manifest), "--out", str(out)])
            plan = _read_json(out)
            plan["inputs"]["scenario_manifest"].pop("size_bytes")
            out.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            code = run_cli(["validate", "--external-eval-plan", str(out), "--out", str(validation), "--strict"])

            self.assertEqual(code, 1)
            errors = "\n".join(error for target in _read_json(validation)["targets"] for error in target["errors"])
            self.assertIn("external_eval_plan.inputs.scenario_manifest.size_bytes must be a non-negative integer", errors)

    def test_validate_rejects_manifest_fingerprint_size_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = _scenario_manifest(root / "heldout.json")
            out = root / "external_eval_plan.json"
            validation = root / "validation.json"
            run_cli(["external-eval-plan", "--scenario-manifest", str(manifest), "--out", str(out)])
            plan = _read_json(out)
            plan["inputs"]["scenario_manifest"]["size_bytes"] += 1
            out.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            code = run_cli(["validate", "--external-eval-plan", str(out), "--out", str(validation), "--strict"])

            self.assertEqual(code, 1)
            errors = "\n".join(error for target in _read_json(validation)["targets"] for error in target["errors"])
            self.assertIn("external_eval_plan.inputs.scenario_manifest.size_bytes does not match the current file.", errors)

    def test_validate_rejects_manifest_fingerprint_cwd_spoof(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            receipt_dir = root / "receipt"
            receipt_dir.mkdir()
            manifest = _scenario_manifest(root / "heldout.json")
            out = receipt_dir / "external_eval_plan.json"
            validation = root / "validation.json"
            run_cli(["external-eval-plan", "--scenario-manifest", str(manifest), "--out", str(out)])
            plan = _read_json(out)
            scenario_manifest = plan["inputs"]["scenario_manifest"]
            scenario_manifest["path"] = manifest.name
            scenario_manifest["exists"] = True
            scenario_manifest["schema_version"] = "hfr.heldout_scenario_manifest.v1"
            scenario_manifest["sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
            scenario_manifest["size_bytes"] = manifest.stat().st_size
            out.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            previous_cwd = Path.cwd()
            try:
                os.chdir(root)
                code = run_cli(["validate", "--external-eval-plan", str(out), "--out", str(validation), "--strict"])
            finally:
                os.chdir(previous_cwd)

            self.assertEqual(code, 1)
            errors = "\n".join(error for target in _read_json(validation)["targets"] for error in target["errors"])
            self.assertIn("external_eval_plan.inputs.scenario_manifest.path does not resolve to a manifest file.", errors)

    def test_validate_rejects_symlink_scenario_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = _scenario_manifest(root / "heldout.json")
            symlink = root / "heldout-link.json"
            try:
                symlink.symlink_to(manifest.name)
            except (NotImplementedError, OSError) as exc:
                self.skipTest(f"symlinks unavailable: {exc}")
            out = root / "external_eval_plan.json"
            validation = root / "validation.json"
            run_cli(["external-eval-plan", "--scenario-manifest", str(manifest), "--out", str(out)])
            plan = _read_json(out)
            plan["inputs"]["scenario_manifest"]["path"] = symlink.name
            out.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            code = run_cli(["validate", "--external-eval-plan", str(out), "--out", str(validation), "--strict"])

            self.assertEqual(code, 1)
            errors = "\n".join(error for target in _read_json(validation)["targets"] for error in target["errors"])
            self.assertIn(
                "external_eval_plan.inputs.scenario_manifest.path must resolve to a regular non-symlink manifest file.",
                errors,
            )

    def test_validate_rejects_wrong_scenario_manifest_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = _scenario_manifest(root / "heldout.json")
            out = root / "external_eval_plan.json"
            validation = root / "validation.json"
            run_cli(["external-eval-plan", "--scenario-manifest", str(manifest), "--out", str(out)])
            manifest_payload = _read_json(manifest)
            manifest_payload["schema_version"] = "hfr.not_heldout_manifest.v1"
            manifest.write_text(json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            plan = _read_json(out)
            scenario_manifest = plan["inputs"]["scenario_manifest"]
            scenario_manifest["schema_version"] = manifest_payload["schema_version"]
            scenario_manifest["sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
            scenario_manifest["size_bytes"] = manifest.stat().st_size
            for adapter in plan["adapters"]:
                adapter["execution_contract"]["scenario_manifest_sha256"] = scenario_manifest["sha256"]
            out.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            code = run_cli(["validate", "--external-eval-plan", str(out), "--out", str(validation), "--strict"])

            self.assertEqual(code, 1)
            errors = "\n".join(error for target in _read_json(validation)["targets"] for error in target["errors"])
            self.assertIn("external_eval_plan.inputs.scenario_manifest.path must reference", errors)

    def test_validate_rejects_forged_scenario_manifest_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "heldout.json"
            manifest.write_text(
                json.dumps(
                    {
                        "schema_version": "hfr.heldout_scenario_manifest.v1",
                        "ready": False,
                        "scenario_count": 1,
                        "scenario_ids": ["email_reply_completion"],
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            out = root / "external_eval_plan.json"
            validation = root / "validation.json"
            run_cli(["external-eval-plan", "--scenario-manifest", str(manifest), "--out", str(out)])
            plan = _read_json(out)
            scenario_manifest = plan["inputs"]["scenario_manifest"]
            scenario_manifest["ready"] = True
            scenario_manifest["scenario_count"] = 2
            scenario_manifest["sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
            scenario_manifest["size_bytes"] = manifest.stat().st_size
            for adapter in plan["adapters"]:
                adapter["execution_contract"]["scenario_manifest_sha256"] = scenario_manifest["sha256"]
            out.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            code = run_cli(["validate", "--external-eval-plan", str(out), "--out", str(validation), "--strict"])

            self.assertEqual(code, 1)
            errors = "\n".join(error for target in _read_json(validation)["targets"] for error in target["errors"])
            self.assertIn("external_eval_plan.inputs.scenario_manifest.ready must match the current file.", errors)
            self.assertIn("external_eval_plan.inputs.scenario_manifest.scenario_count must match the current file.", errors)

    def test_validate_rejects_forged_adapter_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = _scenario_manifest(root / "heldout.json")
            out = root / "external_eval_plan.json"
            validation = root / "validation.json"
            plan = build_external_eval_plan(adapters=["bfcl"], scenario_manifest=manifest)
            plan["adapters"][0]["adapter_contract"]["receipt_types"].append("hfr.external_eval_live_provider_receipt.v1")
            plan["adapters"][0]["adapter_contract"]["live_benchmark_supported"] = True
            plan["adapters"][0]["adapter_contract"]["provider_api_called_by_flight_recorder"] = True
            write_external_eval_plan(plan, out)

            schema_result = check_schema_file(out)
            code = run_cli(["validate", "--external-eval-plan", str(out), "--out", str(validation), "--strict"])

            self.assertFalse(schema_result["passed"], schema_result)
            self.assertEqual(code, 1)
            errors = "\n".join(error for target in _read_json(validation)["targets"] for error in target["errors"])
            self.assertIn("adapter_contract.receipt_types contains unsupported receipt types", errors)
            self.assertIn("adapter_contract.live_benchmark_supported must be false", errors)
            self.assertIn("adapter_contract.provider_api_called_by_flight_recorder must be false", errors)

    def test_validate_and_schema_reject_unknown_external_eval_plan_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = _scenario_manifest(root / "heldout.json")
            out = root / "external_eval_plan.json"
            validation = root / "validation.json"
            run_cli(["external-eval-plan", "--scenario-manifest", str(manifest), "--out", str(out)])
            plan = _read_json(out)
            plan["provider_job_id"] = "not-owned-by-flight-recorder"
            plan["inputs"]["scenario_manifest"]["absolute_source"] = "<redacted:heldout.json>"
            plan["adapters"][0]["dependency_status"]["live_probe_token"] = True
            plan["adapters"][0]["execution_contract"]["live_benchmark_url"] = "https://example.invalid/job"
            plan["adapters"][0]["adapter_contract"]["credential_hint"] = "redacted"
            plan["governance_handoff"]["live_claims_exported"] = True
            out.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            code = run_cli(["validate", "--external-eval-plan", str(out), "--out", str(validation), "--strict"])
            schema_result = check_schema_file(out)

            self.assertEqual(code, 1)
            self.assertFalse(schema_result["passed"], schema_result["errors"])
            errors = "\n".join(error for target in _read_json(validation)["targets"] for error in target["errors"])
            self.assertIn("external_eval_plan contains unknown field(s): ['provider_job_id'].", errors)
            self.assertIn("external_eval_plan.inputs.scenario_manifest contains unknown field(s): ['absolute_source'].", errors)
            self.assertIn(
                "external_eval_plan.adapters[0].dependency_status contains unknown field(s): ['live_probe_token'].",
                errors,
            )
            self.assertIn(
                "external_eval_plan.adapters[0].execution_contract contains unknown field(s): ['live_benchmark_url'].",
                errors,
            )
            self.assertIn(
                "external_eval_plan.adapters[0].adapter_contract contains unknown field(s): ['credential_hint'].",
                errors,
            )
            self.assertIn(
                "external_eval_plan.governance_handoff contains unknown field(s): ['live_claims_exported'].",
                errors,
            )


def _scenario_manifest(path: Path, *, scenario_ids: list[str] | None = None) -> Path:
    payload = {"schema_version": "hfr.heldout_scenario_manifest.v1", "scenario_ids": scenario_ids or ["email_reply_completion"]}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
