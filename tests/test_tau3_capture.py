import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from flightrecorder.tau3_capture import capture_to_hfr, validate_tau3_capture
from flightrecorder.tau3_training_artifacts import REQUIRED_ARTIFACTS, validate_tau3_training_bundle
from scripts.build_tau3_training_artifacts import (
    Tau3BundleBuildError,
    _finalize_bundle_manifest,
    _public_contract,
    _local_model_identity_matches,
    _rehearsal_captures,
    _rehearsal_config,
    _trainer_command,
    build_bundle,
)


ROOT = Path(__file__).resolve().parents[1]


class Tau3CaptureTests(unittest.TestCase):
    def test_successful_capture_projects_action_training_evidence(self):
        capture = next(row for row in _rehearsal_captures() if row["outcome"]["success"] is True)
        self.assertEqual(validate_tau3_capture(capture), [])

        artifacts = capture_to_hfr(capture)

        self.assertTrue(artifacts["scorecard"]["passed"])
        self.assertTrue(artifacts["task_completion"]["task_evidence_configured"])
        self.assertTrue(artifacts["trajectory_v2"]["action_training"]["eligible"])
        self.assertTrue(artifacts["state_diff"]["comparison_complete"])

    def test_negative_capture_remains_evidence_but_is_not_action_sft_eligible(self):
        capture = next(
            row
            for row in _rehearsal_captures()
            if row["behavior"] == "hallucinated_tool" and row["outcome"]["success"] is False
        )

        artifacts = capture_to_hfr(capture)

        self.assertFalse(artifacts["scorecard"]["passed"])
        self.assertNotIn("trajectory_v2", artifacts)
        self.assertEqual(artifacts["normalized_trace"]["events"][1]["tool_name"], "invented_tool")

    def test_sealed_or_unverifiable_capture_fails_closed(self):
        capture = _rehearsal_captures()[0]
        capture["split"] = "sealed"
        capture["state_transition"]["executable"] = False

        errors = validate_tau3_capture(capture)

        self.assertTrue(any("sealed" in error for error in errors))
        self.assertIn("state_transition.executable must be true", errors)

    def test_rehearsal_builder_emits_complete_non_ready_bundle(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp) / "bundle"
            completed = subprocess.run(
                [
                    str(ROOT / ".venv" / "bin" / "python"),
                    str(ROOT / "scripts" / "build_tau3_training_artifacts.py"),
                    "--mode",
                    "rehearsal",
                    "--out",
                    str(bundle),
                ],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            summary = json.loads(completed.stdout)
            self.assertTrue(summary["passed"])
            self.assertFalse(summary["ready_for_training"])
            self.assertEqual(summary["required_artifact_count"], len(REQUIRED_ARTIFACTS))

            rehearsal = validate_tau3_training_bundle(bundle, strict=True, allow_rehearsal=True)
            production = validate_tau3_training_bundle(bundle, strict=True, allow_rehearsal=False)
            self.assertTrue(rehearsal["passed"], rehearsal)
            self.assertFalse(production["passed"])

            export = json.loads((bundle / "exports" / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(export["sft_count"], 8)
            self.assertEqual(export["action_sft_count"], 8)
            self.assertEqual(export["dpo_count"], 8)
            launch = json.loads((bundle / "training" / "trainer_launch_check.json").read_text(encoding="utf-8"))
            self.assertTrue(launch["passed"])
            self.assertFalse(launch.get("executed", False))

    def test_rehearsal_builder_accepts_an_existing_empty_output_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp) / "bundle"
            bundle.mkdir()

            result = build_bundle(bundle, mode="rehearsal")

            self.assertTrue(result["validation"]["passed"], result["validation"])
            self.assertFalse(result["ready_for_training"])

    def test_final_tau_validation_can_only_remove_readiness(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp) / "bundle"
            build_bundle(bundle, mode="rehearsal")

            ready, validation = _finalize_bundle_manifest(
                bundle,
                mode="production",
                provisional_ready=True,
                created_at="2026-07-22T00:00:00+00:00",
            )

            manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
            self.assertFalse(ready)
            self.assertFalse(validation["passed"])
            self.assertFalse(manifest["ready_for_training"])

    def test_production_builder_requires_frozen_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            completed = subprocess.run(
                [
                    str(ROOT / ".venv" / "bin" / "python"),
                    str(ROOT / "scripts" / "build_tau3_training_artifacts.py"),
                    "--mode",
                    "production",
                    "--out",
                    str(Path(tmp) / "bundle"),
                ],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 2)
            self.assertIn("requires --config and --captures", completed.stderr)

    def test_production_command_must_be_frozen_in_qlora_plan(self):
        config = _rehearsal_config("2026-07-22T00:00:00+00:00")
        with self.assertRaisesRegex(Tau3BundleBuildError, "command_argv"):
            _trainer_command(config, "production")

        config["mlx_qlora_plan"]["command_argv"] = [
            "python",
            "-m",
            "mlx_lm.lora",
            "--model",
            "model_input",
            "--train",
            "--data",
            "input_export",
            "--adapter-path",
            "adapter_output",
        ]
        command = _trainer_command(config, "production")

        self.assertIn("--model model_input", command)
        self.assertNotIn("--iters 1", command)

        for remote_model in ("mlx-community/model", "https://example.test/model"):
            config["mlx_qlora_plan"]["command_argv"][4] = remote_model
            with self.assertRaisesRegex(Tau3BundleBuildError, "--model"):
                _trainer_command(config, "production")
        config["mlx_qlora_plan"]["command_argv"][4] = "model_input"
        config["mlx_qlora_plan"]["command_argv"][7] = "../other-data"
        with self.assertRaisesRegex(Tau3BundleBuildError, "--data"):
            _trainer_command(config, "production")

    def test_local_model_identity_cannot_be_inferred_from_directory_name(self):
        revision = "a" * 64
        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp) / revision
            model_path.mkdir()
            (model_path / "config.json").write_text("{}\n", encoding="utf-8")

            self.assertFalse(
                _local_model_identity_matches(
                    {"name": "local/model", "revision": revision},
                    model_path,
                    revision,
                )
            )

    def test_rehearsal_command_is_tiny(self):
        config = _rehearsal_config("2026-07-22T00:00:00+00:00")
        self.assertIn("--iters 1", _trainer_command(config, "rehearsal"))

    def test_public_protocol_contract_removes_machine_specific_paths(self):
        public = _public_contract({
            "revision": "abc123",
            "local_path": "/Users/example/.cache/model",
            "nested": {"tokenizer_path": "/private/tokenizer", "sha256": "f" * 64},
        })

        self.assertEqual(public, {"revision": "abc123", "nested": {"sha256": "f" * 64}})


if __name__ == "__main__":
    unittest.main()
