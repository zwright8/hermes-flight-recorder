import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from flightrecorder.agentic_training_result import build_agentic_training_result, write_agentic_training_result
from flightrecorder.agentic_training_runtime import (
    build_agentic_training_runtime_preflight,
    write_agentic_training_runtime_preflight,
)
from flightrecorder.cli import main
from flightrecorder.schema_registry import check_schema_contract, check_schema_file, list_schema_records


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_PLAN = ROOT / "examples" / "agentic_training" / "plans" / "sft_then_dpo_plan.json"


def run_cli(args):
    with redirect_stdout(StringIO()):
        return main(args)


class AgenticTrainingResultTests(unittest.TestCase):
    def write_runtime_preflight(self, path: Path, *, ready: bool) -> None:
        preflight = build_agentic_training_runtime_preflight(
            plan_path=EXAMPLE_PLAN,
            out_path=path,
            require_modules=["json"] if ready else ["hfr_missing_dependency_for_test"],
            skip_default_modules=True,
            created_at="2026-07-02T00:00:00+00:00",
        )
        write_agentic_training_runtime_preflight(path, preflight)

    def test_completed_result_requires_ready_runtime_and_output_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime = root / "runtime_preflight.json"
            adapter = root / "adapter.safetensors"
            metrics = root / "metrics.json"
            self.write_runtime_preflight(runtime, ready=True)
            adapter.write_bytes(b"tiny adapter bytes")
            metrics.write_text(json.dumps({"loss": 0.0}) + "\n", encoding="utf-8")

            result = build_agentic_training_result(
                plan_path=EXAMPLE_PLAN,
                runtime_preflight_path=runtime,
                out_path=root / "result.json",
                status="completed",
                artifacts={"adapter": [adapter], "metrics": [metrics]},
                created_at="2026-07-02T00:00:00+00:00",
            )

            self.assertTrue(result["passed"], result["blocked_reasons"])
            self.assertEqual(result["artifact_path"], "result.json")
            self.assertEqual(result["recommendation"], "register_training_result")
            self.assertEqual(result["training_result"]["status"], "completed")
            self.assertEqual(result["failure"]["class"], "none")
            self.assertEqual(result["metrics"]["adapter_count"], 1)
            self.assertEqual(result["metrics"]["metrics_file_count"], 1)
            self.assertFalse(result["registry_update"]["applied"])
            self.assertTrue(result["registry_update"]["ready_to_apply"])
            self.assertEqual(result["registry_update"]["links"][0]["collection"], "training_runs")
            self.assertEqual(result["registry_update"]["links"][0]["path"], "result.json")
            self.assertIn("adapters", {link["collection"] for link in result["registry_update"]["links"]})
            adapter_link = next(link for link in result["registry_update"]["links"] if link["collection"] == "adapters")
            self.assertEqual(adapter_link["path"], "adapter.safetensors")
            self.assertEqual(adapter_link["size_bytes"], adapter.stat().st_size)
            self.assertFalse(result["execution_boundary"]["flight_recorder_launched_training"])
            self.assertEqual(result["lineage"]["plan"]["size_bytes"], EXAMPLE_PLAN.stat().st_size)
            self.assertEqual(result["lineage"]["runtime_preflight"]["size_bytes"], runtime.stat().st_size)
            plan = json.loads(EXAMPLE_PLAN.read_text(encoding="utf-8"))
            self.assertEqual(result["lineage"]["model"]["size_bytes"], plan["input_manifests"]["model"]["size_bytes"])
            self.assertEqual(result["lineage"]["dataset"]["size_bytes"], plan["input_manifests"]["dataset"]["size_bytes"])
            schema = check_schema_contract(result)
            self.assertTrue(schema["passed"], schema["errors"])

    def test_classified_failure_receipt_can_register_blocked_runtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime = root / "runtime_preflight.json"
            failure_log = root / "runtime.log"
            self.write_runtime_preflight(runtime, ready=False)
            failure_log.write_text("missing dependency hfr_missing_dependency_for_test\n", encoding="utf-8")

            result = build_agentic_training_result(
                plan_path=EXAMPLE_PLAN,
                runtime_preflight_path=runtime,
                out_path=root / "result.json",
                status="blocked",
                failure_class="dependency_missing",
                failure_message="Runtime dependency probe failed before tiny smoke launch.",
                artifacts={"log": [failure_log]},
                created_at="2026-07-02T00:00:00+00:00",
            )

            self.assertTrue(result["passed"], result["blocked_reasons"])
            self.assertEqual(result["recommendation"], "register_training_failure")
            self.assertEqual(result["failure"]["class"], "dependency_missing")
            self.assertEqual(result["failure"]["source"], "runtime_preflight")
            self.assertEqual(result["registry_update"]["links"][0]["status"], "classified_blocked")
            self.assertEqual(len(result["registry_update"]["links"]), 1)
            schema = check_schema_contract(result)
            self.assertTrue(schema["passed"], schema["errors"])

    def test_completed_result_blocks_when_runtime_preflight_was_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime = root / "runtime_preflight.json"
            adapter = root / "adapter.safetensors"
            self.write_runtime_preflight(runtime, ready=False)
            adapter.write_bytes(b"tiny adapter bytes")

            result = build_agentic_training_result(
                plan_path=EXAMPLE_PLAN,
                runtime_preflight_path=runtime,
                out_path=root / "result.json",
                status="completed",
                artifacts={"adapter": [adapter]},
                created_at="2026-07-02T00:00:00+00:00",
            )

            self.assertFalse(result["passed"])
            self.assertEqual(result["recommendation"], "block_training_result_registration")
            self.assertIn("runtime_ready_for_completed_result", {check["id"] for check in result["checks"] if not check["passed"]})
            schema = check_schema_contract(result)
            self.assertTrue(schema["passed"], schema["errors"])

    def test_cli_writes_schema_checkable_result_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime = root / "runtime_preflight.json"
            adapter = root / "adapter.safetensors"
            out = root / "agentic_training_result.json"
            self.write_runtime_preflight(runtime, ready=True)
            adapter.write_bytes(b"tiny adapter bytes")

            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "archive_agentic_training_result.py"),
                    "--plan",
                    str(EXAMPLE_PLAN),
                    "--runtime-preflight",
                    str(runtime),
                    "--status",
                    "completed",
                    "--adapter",
                    str(adapter),
                    "--created-at",
                    "2026-07-02T00:00:00+00:00",
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
            result = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(result["recommendation"], "register_training_result")
            schema = check_schema_file(out)
            self.assertTrue(schema["passed"], schema["errors"])

    def test_validate_accepts_agentic_training_result_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime = root / "runtime_preflight.json"
            adapter = root / "adapter.safetensors"
            out = root / "agentic_training_result.json"
            summary_path = root / "validation.json"
            local_plan = root / EXAMPLE_PLAN.name
            local_plan.write_bytes(EXAMPLE_PLAN.read_bytes())
            self.write_runtime_preflight(runtime, ready=True)
            adapter.write_bytes(b"tiny adapter bytes")

            result = build_agentic_training_result(
                plan_path=local_plan,
                runtime_preflight_path=runtime,
                out_path=out,
                status="completed",
                artifacts={"adapter": [adapter]},
                created_at="2026-07-02T00:00:00+00:00",
            )
            write_agentic_training_result(out, result)

            code = run_cli(
                [
                    "validate",
                    "--agentic-training-result",
                    str(out),
                    "--out",
                    str(summary_path),
                    "--strict",
                ]
            )

            self.assertEqual(code, 0)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertTrue(summary["passed"])
            self.assertEqual(summary["target_count"], 1)
            self.assertEqual(summary["targets"][0]["type"], "agentic_training_result")
            self.assertEqual(summary["targets"][0]["details"]["status"], "completed")
            self.assertEqual(summary["targets"][0]["details"]["output_artifact_count"], 1)

    def test_validate_rejects_agentic_training_result_metric_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime = root / "runtime_preflight.json"
            adapter = root / "adapter.safetensors"
            out = root / "agentic_training_result.json"
            summary_path = root / "validation.json"
            local_plan = root / EXAMPLE_PLAN.name
            local_plan.write_bytes(EXAMPLE_PLAN.read_bytes())
            self.write_runtime_preflight(runtime, ready=True)
            adapter.write_bytes(b"tiny adapter bytes")

            result = build_agentic_training_result(
                plan_path=local_plan,
                runtime_preflight_path=runtime,
                out_path=out,
                status="completed",
                artifacts={"adapter": [adapter]},
                created_at="2026-07-02T00:00:00+00:00",
            )
            result["metrics"]["adapter_count"] = 0
            result["registry_update"]["applied"] = True
            write_agentic_training_result(out, result)

            code = run_cli(
                [
                    "validate",
                    "--agentic-training-result",
                    str(out),
                    "--out",
                    str(summary_path),
                    "--strict",
                ]
            )

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in summary["targets"] for error in target["errors"])
            self.assertIn("agentic_training_result.metrics.adapter_count expected 1", errors)
            self.assertIn("agentic_training_result.registry_update.applied expected False", errors)

    def test_validate_rejects_agentic_training_result_lineage_without_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime = root / "runtime_preflight.json"
            adapter = root / "adapter.safetensors"
            out = root / "agentic_training_result.json"
            summary_path = root / "validation.json"
            local_plan = root / EXAMPLE_PLAN.name
            local_plan.write_bytes(EXAMPLE_PLAN.read_bytes())
            self.write_runtime_preflight(runtime, ready=True)
            adapter.write_bytes(b"tiny adapter bytes")

            result = build_agentic_training_result(
                plan_path=local_plan,
                runtime_preflight_path=runtime,
                out_path=out,
                status="completed",
                artifacts={"adapter": [adapter]},
                created_at="2026-07-02T00:00:00+00:00",
            )
            result["lineage"]["plan"].pop("size_bytes")
            write_agentic_training_result(out, result)

            code = run_cli(
                [
                    "validate",
                    "--agentic-training-result",
                    str(out),
                    "--out",
                    str(summary_path),
                    "--strict",
                ]
            )

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in summary["targets"] for error in target["errors"])
            self.assertIn("agentic_training_result.lineage.plan.size_bytes must be a non-negative integer.", errors)

    def test_validate_rejects_agentic_training_result_lineage_size_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime = root / "runtime_preflight.json"
            adapter = root / "adapter.safetensors"
            out = root / "agentic_training_result.json"
            summary_path = root / "validation.json"
            self.write_runtime_preflight(runtime, ready=True)
            adapter.write_bytes(b"tiny adapter bytes")

            result = build_agentic_training_result(
                plan_path=EXAMPLE_PLAN,
                runtime_preflight_path=runtime,
                out_path=out,
                status="completed",
                artifacts={"adapter": [adapter]},
                created_at="2026-07-02T00:00:00+00:00",
            )
            result["lineage"]["runtime_preflight"]["size_bytes"] += 1
            write_agentic_training_result(out, result)

            code = run_cli(
                [
                    "validate",
                    "--agentic-training-result",
                    str(out),
                    "--out",
                    str(summary_path),
                    "--strict",
                ]
            )

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in summary["targets"] for error in target["errors"])
            self.assertIn("agentic_training_result.lineage.runtime_preflight.size_bytes does not match the current file.", errors)

    def test_validate_rejects_agentic_training_result_artifact_size_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime = root / "runtime_preflight.json"
            adapter = root / "adapter.safetensors"
            out = root / "agentic_training_result.json"
            summary_path = root / "validation.json"
            self.write_runtime_preflight(runtime, ready=True)
            adapter.write_bytes(b"tiny adapter bytes")

            result = build_agentic_training_result(
                plan_path=EXAMPLE_PLAN,
                runtime_preflight_path=runtime,
                out_path=out,
                status="completed",
                artifacts={"adapter": [adapter]},
                created_at="2026-07-02T00:00:00+00:00",
            )
            forged_size = adapter.stat().st_size + 1
            result["artifacts"][0]["size_bytes"] = forged_size
            adapter_link = next(link for link in result["registry_update"]["links"] if link["collection"] == "adapters")
            adapter_link["size_bytes"] = forged_size
            write_agentic_training_result(out, result)

            code = run_cli(
                [
                    "validate",
                    "--agentic-training-result",
                    str(out),
                    "--out",
                    str(summary_path),
                    "--strict",
                ]
            )

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in summary["targets"] for error in target["errors"])
            self.assertIn("agentic_training_result.artifacts[0].size_bytes does not match the current file.", errors)

    def test_validate_rejects_agentic_training_result_symlink_artifact_ref(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            external_dir = root / "external"
            external_dir.mkdir()
            runtime = root / "runtime_preflight.json"
            adapter = root / "adapter.safetensors"
            external_adapter = external_dir / "adapter.safetensors"
            out = root / "agentic_training_result.json"
            summary_path = root / "validation.json"
            local_plan = root / EXAMPLE_PLAN.name
            local_plan.write_bytes(EXAMPLE_PLAN.read_bytes())
            self.write_runtime_preflight(runtime, ready=True)
            external_adapter.write_bytes(b"tiny adapter bytes")
            adapter.symlink_to(external_adapter)

            result = build_agentic_training_result(
                plan_path=local_plan,
                runtime_preflight_path=runtime,
                out_path=out,
                status="completed",
                artifacts={"adapter": [adapter]},
                created_at="2026-07-02T00:00:00+00:00",
            )
            write_agentic_training_result(out, result)

            code = run_cli(
                [
                    "validate",
                    "--agentic-training-result",
                    str(out),
                    "--out",
                    str(summary_path),
                    "--strict",
                ]
            )

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in summary["targets"] for error in target["errors"])
            self.assertIn("agentic_training_result.artifacts[0].path must not be a symlink.", errors)

    def test_validate_rejects_agentic_training_result_absolute_public_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime = root / "runtime_preflight.json"
            adapter = root / "adapter.safetensors"
            out = root / "agentic_training_result.json"
            summary_path = root / "validation.json"
            self.write_runtime_preflight(runtime, ready=True)
            adapter.write_bytes(b"tiny adapter bytes")

            result = build_agentic_training_result(
                plan_path=EXAMPLE_PLAN,
                runtime_preflight_path=runtime,
                out_path=out,
                status="completed",
                artifacts={"adapter": [adapter]},
                created_at="2026-07-02T00:00:00+00:00",
            )
            result["artifact_path"] = "/example/result.json"
            result["training_result"]["output_dir"] = "/example/output"
            result["lineage"]["plan"]["path"] = str(EXAMPLE_PLAN)
            result["registry_update"]["links"][0]["path"] = "/example/result.json"
            write_agentic_training_result(out, result)

            code = run_cli(
                [
                    "validate",
                    "--agentic-training-result",
                    str(out),
                    "--out",
                    str(summary_path),
                    "--strict",
                ]
            )

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in summary["targets"] for error in target["errors"])
            self.assertIn("agentic_training_result.artifact_path must be a safe relative path without traversal.", errors)
            self.assertIn(
                "agentic_training_result.training_result.output_dir must be a safe relative path without traversal.",
                errors,
            )
            self.assertIn("agentic_training_result.lineage.plan.path must be a safe relative path without traversal.", errors)
            self.assertIn(
                "agentic_training_result.registry_update.links[0].path must be a safe relative path without traversal.",
                errors,
            )

    def test_validate_rejects_agentic_training_result_traversal_artifact_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime = root / "runtime_preflight.json"
            adapter = root / "adapter.safetensors"
            out = root / "agentic_training_result.json"
            summary_path = root / "validation.json"
            self.write_runtime_preflight(runtime, ready=True)
            adapter.write_bytes(b"tiny adapter bytes")

            result = build_agentic_training_result(
                plan_path=EXAMPLE_PLAN,
                runtime_preflight_path=runtime,
                out_path=out,
                status="completed",
                artifacts={"adapter": [adapter]},
                created_at="2026-07-02T00:00:00+00:00",
            )
            result["artifacts"][0]["path"] = "../external/adapter.safetensors"
            result["lineage"]["runtime_preflight"]["path"] = "../external/runtime_preflight.json"
            adapter_link = next(link for link in result["registry_update"]["links"] if link["collection"] == "adapters")
            adapter_link["path"] = "../external/adapter.safetensors"
            write_agentic_training_result(out, result)

            code = run_cli(
                [
                    "validate",
                    "--agentic-training-result",
                    str(out),
                    "--out",
                    str(summary_path),
                    "--strict",
                ]
            )

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in summary["targets"] for error in target["errors"])
            self.assertIn("agentic_training_result.artifacts[0].path must be a safe relative path without traversal.", errors)
            self.assertIn(
                "agentic_training_result.lineage.runtime_preflight.path must be a safe relative path without traversal.",
                errors,
            )
            self.assertIn(
                "agentic_training_result.registry_update.links[1].path must be a safe relative path without traversal.",
                errors,
            )

    def test_builder_redacts_out_of_tree_artifact_paths_to_basename(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            receipt_dir = root / "receipt"
            external_dir = root / "external"
            receipt_dir.mkdir()
            external_dir.mkdir()
            runtime = root / "runtime_preflight.json"
            adapter = external_dir / "adapter.safetensors"
            out = receipt_dir / "agentic_training_result.json"
            summary_path = root / "validation.json"
            self.write_runtime_preflight(runtime, ready=True)
            adapter.write_bytes(b"tiny adapter bytes")

            result = build_agentic_training_result(
                plan_path=EXAMPLE_PLAN,
                runtime_preflight_path=runtime,
                out_path=out,
                status="completed",
                artifacts={"adapter": [adapter]},
                created_at="2026-07-02T00:00:00+00:00",
            )

            self.assertEqual(result["artifact_path"], out.name)
            self.assertEqual(result["artifacts"][0]["path"], adapter.name)
            adapter_link = next(link for link in result["registry_update"]["links"] if link["collection"] == "adapters")
            self.assertEqual(adapter_link["path"], adapter.name)
            write_agentic_training_result(out, result)

            code = run_cli(
                [
                    "validate",
                    "--agentic-training-result",
                    str(out),
                    "--out",
                    str(summary_path),
                    "--strict",
                ]
            )

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in summary["targets"] for error in target["errors"])
            self.assertIn("agentic_training_result.artifacts[0].path does not resolve to the current file.", errors)

    def test_validate_rejects_agentic_training_result_manifest_lineage_without_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime = root / "runtime_preflight.json"
            adapter = root / "adapter.safetensors"
            out = root / "agentic_training_result.json"
            summary_path = root / "validation.json"
            self.write_runtime_preflight(runtime, ready=True)
            adapter.write_bytes(b"tiny adapter bytes")

            result = build_agentic_training_result(
                plan_path=EXAMPLE_PLAN,
                runtime_preflight_path=runtime,
                out_path=out,
                status="completed",
                artifacts={"adapter": [adapter]},
                created_at="2026-07-02T00:00:00+00:00",
            )
            result["lineage"]["model"].pop("size_bytes")
            write_agentic_training_result(out, result)

            code = run_cli(
                [
                    "validate",
                    "--agentic-training-result",
                    str(out),
                    "--out",
                    str(summary_path),
                    "--strict",
                ]
            )

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in summary["targets"] for error in target["errors"])
            self.assertIn("agentic_training_result.lineage.model.size_bytes must be a non-negative integer.", errors)

    def test_builder_redacts_unsafe_manifest_paths_from_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime = root / "runtime_preflight.json"
            adapter = root / "adapter.safetensors"
            plan_path = root / "plan.json"
            out = root / "agentic_training_result.json"
            summary_path = root / "validation.json"
            plan = json.loads(EXAMPLE_PLAN.read_text(encoding="utf-8"))
            plan["input_manifests"]["model"]["path"] = "C:\\private\\model_manifest.json"
            plan["input_manifests"]["dataset"]["path"] = "..\\escape\\dataset_manifest.json"
            plan_path.write_text(json.dumps(plan), encoding="utf-8")
            self.write_runtime_preflight(runtime, ready=True)
            adapter.write_bytes(b"tiny adapter bytes")

            result = build_agentic_training_result(
                plan_path=plan_path,
                runtime_preflight_path=runtime,
                out_path=out,
                status="completed",
                artifacts={"adapter": [adapter]},
                created_at="2026-07-02T00:00:00+00:00",
            )

            self.assertEqual(result["lineage"]["model"]["path"], "model_manifest.json")
            self.assertEqual(result["lineage"]["dataset"]["path"], "dataset_manifest.json")
            write_agentic_training_result(out, result)
            schema = check_schema_file(out)
            self.assertTrue(schema["passed"], schema["errors"])
            code = run_cli(
                [
                    "validate",
                    "--agentic-training-result",
                    str(out),
                    "--out",
                    str(summary_path),
                    "--strict",
                ]
            )
            self.assertEqual(code, 0)

    def test_validate_rejects_agentic_training_result_unsafe_manifest_lineage_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime = root / "runtime_preflight.json"
            adapter = root / "adapter.safetensors"
            out = root / "agentic_training_result.json"
            summary_path = root / "validation.json"
            self.write_runtime_preflight(runtime, ready=True)
            adapter.write_bytes(b"tiny adapter bytes")

            result = build_agentic_training_result(
                plan_path=EXAMPLE_PLAN,
                runtime_preflight_path=runtime,
                out_path=out,
                status="completed",
                artifacts={"adapter": [adapter]},
                created_at="2026-07-02T00:00:00+00:00",
            )
            result["lineage"]["model"]["path"] = "/private/example/model_manifest.json"
            write_agentic_training_result(out, result)

            code = run_cli(
                [
                    "validate",
                    "--agentic-training-result",
                    str(out),
                    "--out",
                    str(summary_path),
                    "--strict",
                ]
            )

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in summary["targets"] for error in target["errors"])
            self.assertIn("agentic_training_result.lineage.model.path must be a safe relative path without traversal.", errors)

    def test_validate_rejects_agentic_training_result_manifest_lineage_size_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime = root / "runtime_preflight.json"
            adapter = root / "adapter.safetensors"
            out = root / "agentic_training_result.json"
            summary_path = root / "validation.json"
            local_plan = root / EXAMPLE_PLAN.name
            local_plan.write_bytes(EXAMPLE_PLAN.read_bytes())
            self.write_runtime_preflight(runtime, ready=True)
            adapter.write_bytes(b"tiny adapter bytes")

            result = build_agentic_training_result(
                plan_path=local_plan,
                runtime_preflight_path=runtime,
                out_path=out,
                status="completed",
                artifacts={"adapter": [adapter]},
                created_at="2026-07-02T00:00:00+00:00",
            )
            result["lineage"]["dataset"]["size_bytes"] += 1
            write_agentic_training_result(out, result)

            code = run_cli(
                [
                    "validate",
                    "--agentic-training-result",
                    str(out),
                    "--out",
                    str(summary_path),
                    "--strict",
                ]
            )

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in summary["targets"] for error in target["errors"])
            self.assertIn(
                "agentic_training_result.lineage.dataset.size_bytes must match agentic_training_plan.input_manifests.dataset.size_bytes.",
                errors,
            )

    def test_validate_rejects_agentic_training_result_registry_link_without_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime = root / "runtime_preflight.json"
            adapter = root / "adapter.safetensors"
            out = root / "agentic_training_result.json"
            summary_path = root / "validation.json"
            self.write_runtime_preflight(runtime, ready=True)
            adapter.write_bytes(b"tiny adapter bytes")

            result = build_agentic_training_result(
                plan_path=EXAMPLE_PLAN,
                runtime_preflight_path=runtime,
                out_path=out,
                status="completed",
                artifacts={"adapter": [adapter]},
                created_at="2026-07-02T00:00:00+00:00",
            )
            adapter_link = next(link for link in result["registry_update"]["links"] if link["collection"] == "adapters")
            adapter_link.pop("size_bytes")
            write_agentic_training_result(out, result)

            code = run_cli(
                [
                    "validate",
                    "--agentic-training-result",
                    str(out),
                    "--out",
                    str(summary_path),
                    "--strict",
                ]
            )

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in summary["targets"] for error in target["errors"])
            self.assertIn(
                "agentic_training_result.registry_update.links[1].size_bytes must be a non-negative integer for artifact registry links.",
                errors,
            )

    def test_validate_rejects_completed_result_missing_output_registry_link(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime = root / "runtime_preflight.json"
            adapter = root / "adapter.safetensors"
            out = root / "agentic_training_result.json"
            summary_path = root / "validation.json"
            self.write_runtime_preflight(runtime, ready=True)
            adapter.write_bytes(b"tiny adapter bytes")

            result = build_agentic_training_result(
                plan_path=EXAMPLE_PLAN,
                runtime_preflight_path=runtime,
                out_path=out,
                status="completed",
                artifacts={"adapter": [adapter]},
                created_at="2026-07-02T00:00:00+00:00",
            )
            result["registry_update"]["links"] = [
                link for link in result["registry_update"]["links"] if link["collection"] == "training_runs"
            ]
            write_agentic_training_result(out, result)

            code = run_cli(
                [
                    "validate",
                    "--agentic-training-result",
                    str(out),
                    "--out",
                    str(summary_path),
                    "--strict",
                ]
            )

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in summary["targets"] for error in target["errors"])
            self.assertIn(
                "agentic_training_result.registry_update.links must include size-bound registry links for completed output artifacts",
                errors,
            )

    def test_validate_rejects_completed_result_duplicate_output_artifact_without_duplicate_link(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime = root / "runtime_preflight.json"
            adapter = root / "adapter.safetensors"
            out = root / "agentic_training_result.json"
            summary_path = root / "validation.json"
            self.write_runtime_preflight(runtime, ready=True)
            adapter.write_bytes(b"tiny adapter bytes")

            result = build_agentic_training_result(
                plan_path=EXAMPLE_PLAN,
                runtime_preflight_path=runtime,
                out_path=out,
                status="completed",
                artifacts={"adapter": [adapter]},
                created_at="2026-07-02T00:00:00+00:00",
            )
            result["artifacts"].append(dict(result["artifacts"][0]))
            result["metrics"]["artifact_count"] += 1
            result["metrics"]["regular_artifact_count"] += 1
            result["metrics"]["output_artifact_count"] += 1
            result["metrics"]["adapter_count"] += 1
            write_agentic_training_result(out, result)

            code = run_cli(
                [
                    "validate",
                    "--agentic-training-result",
                    str(out),
                    "--out",
                    str(summary_path),
                    "--strict",
                ]
            )

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in summary["targets"] for error in target["errors"])
            self.assertIn(
                "agentic_training_result.registry_update.links must include size-bound registry links for completed output artifacts",
                errors,
            )

    def test_validate_rejects_completed_result_without_registry_update(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime = root / "runtime_preflight.json"
            adapter = root / "adapter.safetensors"
            out = root / "agentic_training_result.json"
            summary_path = root / "validation.json"
            self.write_runtime_preflight(runtime, ready=True)
            adapter.write_bytes(b"tiny adapter bytes")

            result = build_agentic_training_result(
                plan_path=EXAMPLE_PLAN,
                runtime_preflight_path=runtime,
                out_path=out,
                status="completed",
                artifacts={"adapter": [adapter]},
                created_at="2026-07-02T00:00:00+00:00",
            )
            result.pop("registry_update")
            write_agentic_training_result(out, result)

            code = run_cli(
                [
                    "validate",
                    "--agentic-training-result",
                    str(out),
                    "--out",
                    str(summary_path),
                    "--strict",
                ]
            )

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in summary["targets"] for error in target["errors"])
            self.assertIn("agentic_training_result.registry_update is required for completed receipts.", errors)

    def test_validate_rejects_completed_result_registry_link_wrong_collection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime = root / "runtime_preflight.json"
            adapter = root / "adapter.safetensors"
            out = root / "agentic_training_result.json"
            summary_path = root / "validation.json"
            self.write_runtime_preflight(runtime, ready=True)
            adapter.write_bytes(b"tiny adapter bytes")

            result = build_agentic_training_result(
                plan_path=EXAMPLE_PLAN,
                runtime_preflight_path=runtime,
                out_path=out,
                status="completed",
                artifacts={"adapter": [adapter]},
                created_at="2026-07-02T00:00:00+00:00",
            )
            adapter_link = next(link for link in result["registry_update"]["links"] if link["collection"] == "adapters")
            adapter_link["collection"] = "datasets"
            write_agentic_training_result(out, result)

            code = run_cli(
                [
                    "validate",
                    "--agentic-training-result",
                    str(out),
                    "--out",
                    str(summary_path),
                    "--strict",
                ]
            )

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in summary["targets"] for error in target["errors"])
            self.assertIn(
                "agentic_training_result.registry_update.links[1].collection must be 'adapters' for output artifact registry links.",
                errors,
            )

    def test_validate_rejects_completed_result_registry_link_wrong_kind(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime = root / "runtime_preflight.json"
            adapter = root / "adapter.safetensors"
            out = root / "agentic_training_result.json"
            summary_path = root / "validation.json"
            self.write_runtime_preflight(runtime, ready=True)
            adapter.write_bytes(b"tiny adapter bytes")

            result = build_agentic_training_result(
                plan_path=EXAMPLE_PLAN,
                runtime_preflight_path=runtime,
                out_path=out,
                status="completed",
                artifacts={"adapter": [adapter]},
                created_at="2026-07-02T00:00:00+00:00",
            )
            adapter_link = next(link for link in result["registry_update"]["links"] if link["collection"] == "adapters")
            adapter_link["kind"] = "checkpoint"
            write_agentic_training_result(out, result)

            code = run_cli(
                [
                    "validate",
                    "--agentic-training-result",
                    str(out),
                    "--out",
                    str(summary_path),
                    "--strict",
                ]
            )

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in summary["targets"] for error in target["errors"])
            self.assertIn(
                "agentic_training_result.registry_update.links[1].kind must match the supplied output artifact role.",
                errors,
            )

    def test_validate_rejects_completed_result_registry_link_with_null_sha(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime = root / "runtime_preflight.json"
            adapter = root / "adapter.safetensors"
            out = root / "agentic_training_result.json"
            summary_path = root / "validation.json"
            self.write_runtime_preflight(runtime, ready=True)
            adapter.write_bytes(b"tiny adapter bytes")

            result = build_agentic_training_result(
                plan_path=EXAMPLE_PLAN,
                runtime_preflight_path=runtime,
                out_path=out,
                status="completed",
                artifacts={"adapter": [adapter]},
                created_at="2026-07-02T00:00:00+00:00",
            )
            adapter_link = next(link for link in result["registry_update"]["links"] if link["collection"] == "adapters")
            adapter_link["sha256"] = None
            write_agentic_training_result(out, result)

            code = run_cli(
                [
                    "validate",
                    "--agentic-training-result",
                    str(out),
                    "--out",
                    str(summary_path),
                    "--strict",
                ]
            )

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in summary["targets"] for error in target["errors"])
            self.assertIn(
                "agentic_training_result.registry_update.links[1].sha256 must be a SHA-256 hex string for artifact registry links.",
                errors,
            )

    def test_validate_rejects_completed_result_registry_link_without_sha(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime = root / "runtime_preflight.json"
            adapter = root / "adapter.safetensors"
            out = root / "agentic_training_result.json"
            summary_path = root / "validation.json"
            self.write_runtime_preflight(runtime, ready=True)
            adapter.write_bytes(b"tiny adapter bytes")

            result = build_agentic_training_result(
                plan_path=EXAMPLE_PLAN,
                runtime_preflight_path=runtime,
                out_path=out,
                status="completed",
                artifacts={"adapter": [adapter]},
                created_at="2026-07-02T00:00:00+00:00",
            )
            adapter_link = next(link for link in result["registry_update"]["links"] if link["collection"] == "adapters")
            adapter_link.pop("sha256")
            write_agentic_training_result(out, result)

            code = run_cli(
                [
                    "validate",
                    "--agentic-training-result",
                    str(out),
                    "--out",
                    str(summary_path),
                    "--strict",
                ]
            )

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in summary["targets"] for error in target["errors"])
            self.assertIn(
                "agentic_training_result.registry_update.links[1].sha256 must be a SHA-256 hex string for artifact registry links.",
                errors,
            )

    def test_validate_rejects_agentic_training_result_lineage_file_claim_bypass(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime = root / "runtime_preflight.json"
            adapter = root / "adapter.safetensors"
            out = root / "agentic_training_result.json"
            summary_path = root / "validation.json"
            self.write_runtime_preflight(runtime, ready=True)
            adapter.write_bytes(b"tiny adapter bytes")

            result = build_agentic_training_result(
                plan_path=EXAMPLE_PLAN,
                runtime_preflight_path=runtime,
                out_path=out,
                status="completed",
                artifacts={"adapter": [adapter]},
                created_at="2026-07-02T00:00:00+00:00",
            )
            result["lineage"]["plan"].update(
                {
                    "exists": False,
                    "regular_file": False,
                    "sha256": None,
                    "size_bytes": 0,
                }
            )
            write_agentic_training_result(out, result)

            code = run_cli(
                [
                    "validate",
                    "--agentic-training-result",
                    str(out),
                    "--out",
                    str(summary_path),
                    "--strict",
                ]
            )

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in summary["targets"] for error in target["errors"])
            self.assertIn("agentic_training_result.lineage.plan.exists must be true.", errors)
            self.assertIn("agentic_training_result.lineage.plan.regular_file must be true.", errors)
            self.assertIn("agentic_training_result.lineage.plan.sha256 must be a SHA-256 hex string.", errors)

    def test_validate_rejects_agentic_training_result_lineage_cwd_spoof(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            receipt_dir = root / "receipt"
            receipt_dir.mkdir()
            runtime = root / "runtime_preflight.json"
            adapter = root / "adapter.safetensors"
            cwd_only_plan = root / "spoof_plan.json"
            out = receipt_dir / "agentic_training_result.json"
            summary_path = root / "validation.json"
            self.write_runtime_preflight(runtime, ready=True)
            adapter.write_bytes(b"tiny adapter bytes")
            cwd_only_plan.write_bytes(EXAMPLE_PLAN.read_bytes())

            result = build_agentic_training_result(
                plan_path=EXAMPLE_PLAN,
                runtime_preflight_path=runtime,
                out_path=out,
                status="completed",
                artifacts={"adapter": [adapter]},
                created_at="2026-07-02T00:00:00+00:00",
            )
            result["lineage"]["plan"].update(
                {
                    "path": cwd_only_plan.name,
                    "sha256": result["lineage"]["plan"]["sha256"],
                    "size_bytes": cwd_only_plan.stat().st_size,
                }
            )
            write_agentic_training_result(out, result)

            previous_cwd = Path.cwd()
            try:
                os.chdir(root)
                code = run_cli(
                    [
                        "validate",
                        "--agentic-training-result",
                        str(out),
                        "--out",
                        str(summary_path),
                        "--strict",
                    ]
                )
            finally:
                os.chdir(previous_cwd)

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in summary["targets"] for error in target["errors"])
            self.assertIn("agentic_training_result.lineage.plan.path does not resolve to the current file.", errors)

    def test_validate_rejects_passed_result_with_irregular_artifact_ref(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime = root / "runtime_preflight.json"
            adapter = root / "adapter.safetensors"
            out = root / "agentic_training_result.json"
            summary_path = root / "validation.json"
            self.write_runtime_preflight(runtime, ready=True)
            adapter.write_bytes(b"tiny adapter bytes")

            result = build_agentic_training_result(
                plan_path=EXAMPLE_PLAN,
                runtime_preflight_path=runtime,
                out_path=out,
                status="completed",
                artifacts={"adapter": [adapter]},
                created_at="2026-07-02T00:00:00+00:00",
            )
            result["artifacts"][0]["regular_file"] = False
            result["artifacts"][0]["sha256"] = None
            result["metrics"]["regular_artifact_count"] = 0
            write_agentic_training_result(out, result)

            code = run_cli(
                [
                    "validate",
                    "--agentic-training-result",
                    str(out),
                    "--out",
                    str(summary_path),
                    "--strict",
                ]
            )

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in summary["targets"] for error in target["errors"])
            self.assertIn("agentic_training_result passed receipts require all artifact refs to be regular files", errors)

    def test_committed_example_result_receipt_validates(self):
        result_path = ROOT / "examples" / "agentic_training" / "completed_result.json"

        code = run_cli(["validate", "--agentic-training-result", str(result_path), "--strict"])

        self.assertEqual(code, 0)
        result = json.loads(result_path.read_text(encoding="utf-8"))
        self.assertEqual(result["recommendation"], "register_training_result")
        self.assertEqual(result["registry_update"]["links"][0]["collection"], "training_runs")
        plan = json.loads((ROOT / "examples" / "agentic_training" / "plans" / "sft_then_dpo_plan.json").read_text(encoding="utf-8"))
        self.assertEqual(result["lineage"]["model"]["size_bytes"], plan["input_manifests"]["model"]["size_bytes"])
        self.assertEqual(result["lineage"]["dataset"]["size_bytes"], plan["input_manifests"]["dataset"]["size_bytes"])
        adapter_link = next(link for link in result["registry_update"]["links"] if link["collection"] == "adapters")
        adapter_path = (result_path.parent / adapter_link["path"]).resolve()
        self.assertEqual(adapter_link["size_bytes"], adapter_path.stat().st_size)
        schema = check_schema_file(result_path)
        self.assertTrue(schema["passed"], schema["errors"])

    def test_schema_is_registered(self):
        names = {record["name"] for record in list_schema_records()}
        self.assertIn("agentic_training_result", names)


if __name__ == "__main__":
    unittest.main()
