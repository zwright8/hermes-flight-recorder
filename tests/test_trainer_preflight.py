import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from flightrecorder.cli import main


ROOT = Path(__file__).resolve().parents[1]


def run_cli(args):
    with redirect_stdout(StringIO()):
        return main(args)


def run_cli_output(args):
    output = StringIO()
    with redirect_stdout(output):
        code = main(args)
    return code, output.getvalue()


class TrainerPreflightTests(unittest.TestCase):
    def test_trainer_preflight_accepts_passed_training_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp) / "runs"
            gate = Path(tmp) / "training_gate.json"
            preflight = Path(tmp) / "trainer_preflight.json"
            run_cli(["run-suite", "--scenarios", str(ROOT / "scenarios"), "--out", str(runs), "--export-rl"])
            self.assertEqual(
                run_cli(
                    [
                        "gate-export",
                        "--training-export",
                        str(runs / "training_export"),
                        "--policy",
                        str(ROOT / "examples" / "training_gate_policy.demo.json"),
                        "--out",
                        str(gate),
                    ]
                ),
                0,
            )

            code = run_cli(
                [
                    "trainer-preflight",
                    "--gate",
                    str(gate),
                    "--training-export",
                    str(runs / "training_export"),
                    "--require-gate",
                    "training_gate",
                    "--trainer-command",
                    "python train.py --dataset runs/training_export",
                    "--metadata",
                    "launcher=dry-run",
                    "--out",
                    str(preflight),
                ]
            )

            self.assertEqual(code, 0)
            result = json.loads(preflight.read_text(encoding="utf-8"))
            self.assertEqual(result["schema_version"], "hfr.trainer_preflight.v1")
            self.assertTrue(result["passed"])
            self.assertEqual(result["recommendation"], "launch_allowed")
            self.assertEqual(result["gate_count"], 1)
            self.assertEqual(result["gates"][0]["id"], "training_gate")
            self.assertTrue(result["gates"][0]["validation"]["passed"])
            self.assertEqual(result["metadata"]["launcher"], "dry-run")
            self.assertEqual(result["trainer_command"]["argv"][:2], ["python", "train.py"])
            self.assertIn("training_export_sft_jsonl", result["artifacts"])
            self.assertEqual(len(result["artifacts"]["training_export_sft_jsonl"]["sha256"]), 64)
            self.assertEqual(run_cli(["validate", "--trainer-preflight", str(preflight), "--strict"]), 0)

            launch_check = Path(tmp) / "trainer_launch_check.json"
            self.assertEqual(
                run_cli(
                    [
                        "trainer-launch-check",
                        "--preflight",
                        str(preflight),
                        "--require-gate",
                        "training_gate",
                        "--require-metadata",
                        "launcher=dry-run",
                        "--out",
                        str(launch_check),
                    ]
                ),
                0,
            )
            launch = json.loads(launch_check.read_text(encoding="utf-8"))
            self.assertEqual(launch["schema_version"], "hfr.trainer_launch_check.v1")
            self.assertTrue(launch["passed"])
            self.assertEqual(launch["recommendation"], "launch_allowed")
            self.assertEqual(launch["approved_command"]["argv"][:2], ["python", "train.py"])
            self.assertTrue(launch["approved_command"]["approved"])
            self.assertEqual(run_cli(["validate", "--trainer-launch-check", str(launch_check), "--strict"]), 0)

            code, output = run_cli_output(["trainer-launch-check", "--preflight", str(preflight), "--print-command"])
            self.assertEqual(code, 0)
            self.assertEqual(output.strip(), "python train.py --dataset runs/training_export")

    def test_trainer_preflight_blocks_failed_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp) / "runs"
            gate = Path(tmp) / "training_gate.json"
            preflight = Path(tmp) / "trainer_preflight.json"
            run_cli(["run", "--scenario", str(ROOT / "scenarios" / "prompt_injection_bad.json"), "--out", str(runs / "bad")])
            run_cli(["export-rl", "--runs", str(runs), "--out", str(runs / "training_export")])
            self.assertEqual(
                run_cli(
                    [
                        "gate-export",
                        "--training-export",
                        str(runs / "training_export"),
                        "--min-pass-rate",
                        "1.0",
                        "--out",
                        str(gate),
                    ]
                ),
                1,
            )

            code = run_cli(
                [
                    "trainer-preflight",
                    "--gate",
                    str(gate),
                    "--training-export",
                    str(runs / "training_export"),
                    "--out",
                    str(preflight),
                ]
            )

            self.assertEqual(code, 1)
            result = json.loads(preflight.read_text(encoding="utf-8"))
            self.assertEqual(result["recommendation"], "block_launch")
            self.assertIn("gate_passed", {check["id"] for check in result["checks"] if not check["passed"]})

            launch_check = Path(tmp) / "trainer_launch_check.json"
            self.assertEqual(
                run_cli(["trainer-launch-check", "--preflight", str(preflight), "--out", str(launch_check)]),
                1,
            )
            launch = json.loads(launch_check.read_text(encoding="utf-8"))
            self.assertEqual(launch["recommendation"], "block_launch")
            self.assertIn("preflight_passed", {check["id"] for check in launch["checks"] if not check["passed"]})

    def test_trainer_preflight_blocks_unvalidated_training_gate_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp) / "runs"
            gate = Path(tmp) / "training_gate.json"
            preflight = Path(tmp) / "trainer_preflight.json"
            run_cli(["run", "--scenario", str(ROOT / "scenarios" / "prompt_injection_good.json"), "--out", str(runs / "good")])
            run_cli(["export-rl", "--runs", str(runs), "--out", str(runs / "training_export")])
            self.assertEqual(
                run_cli(["gate-export", "--training-export", str(runs / "training_export"), "--skip-validation", "--out", str(gate)]),
                0,
            )

            code = run_cli(
                [
                    "trainer-preflight",
                    "--gate",
                    str(gate),
                    "--training-export",
                    str(runs / "training_export"),
                    "--out",
                    str(preflight),
                ]
            )

            self.assertEqual(code, 1)
            result = json.loads(preflight.read_text(encoding="utf-8"))
            self.assertIn("gate_validation_passed", {check["id"] for check in result["checks"] if not check["passed"]})

    def test_validate_rejects_stale_trainer_preflight_gate_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp) / "runs"
            gate = Path(tmp) / "training_gate.json"
            preflight = Path(tmp) / "trainer_preflight.json"
            summary = Path(tmp) / "validation.json"
            run_cli(["run", "--scenario", str(ROOT / "scenarios" / "prompt_injection_good.json"), "--out", str(runs / "good")])
            run_cli(["export-rl", "--runs", str(runs), "--out", str(runs / "training_export")])
            run_cli(["gate-export", "--training-export", str(runs / "training_export"), "--out", str(gate)])
            run_cli(
                [
                    "trainer-preflight",
                    "--gate",
                    str(gate),
                    "--training-export",
                    str(runs / "training_export"),
                    "--preserve-paths",
                    "--out",
                    str(preflight),
                ]
            )
            gate_payload = json.loads(gate.read_text(encoding="utf-8"))
            gate_payload["passed"] = False
            gate.write_text(json.dumps(gate_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            code = run_cli(["validate", "--trainer-preflight", str(preflight), "--out", str(summary)])

            self.assertEqual(code, 1)
            validation = json.loads(summary.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in validation["targets"] for error in target["errors"])
            self.assertIn("trainer_preflight.gates[0].sha256", errors)

            launch_check = Path(tmp) / "trainer_launch_check.json"
            self.assertEqual(
                run_cli(["trainer-launch-check", "--preflight", str(preflight), "--out", str(launch_check)]),
                1,
            )
            launch = json.loads(launch_check.read_text(encoding="utf-8"))
            self.assertIn("preflight_validation_passed", {check["id"] for check in launch["checks"] if not check["passed"]})


if __name__ == "__main__":
    unittest.main()
