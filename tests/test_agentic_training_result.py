import json
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
            self.assertEqual(result["recommendation"], "register_training_result")
            self.assertEqual(result["training_result"]["status"], "completed")
            self.assertEqual(result["failure"]["class"], "none")
            self.assertEqual(result["metrics"]["adapter_count"], 1)
            self.assertEqual(result["metrics"]["metrics_file_count"], 1)
            self.assertFalse(result["registry_update"]["applied"])
            self.assertTrue(result["registry_update"]["ready_to_apply"])
            self.assertEqual(result["registry_update"]["links"][0]["collection"], "training_runs")
            self.assertIn("adapters", {link["collection"] for link in result["registry_update"]["links"]})
            self.assertFalse(result["execution_boundary"]["flight_recorder_launched_training"])
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

    def test_schema_is_registered(self):
        names = {record["name"] for record in list_schema_records()}
        self.assertIn("agentic_training_result", names)


if __name__ == "__main__":
    unittest.main()
