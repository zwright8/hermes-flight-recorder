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


class TrainingGateTests(unittest.TestCase):
    def test_gate_export_accepts_demo_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp) / "runs"
            gate = Path(tmp) / "training_gate.json"
            run_cli(["run-suite", "--scenarios", str(ROOT / "scenarios"), "--out", str(runs), "--export-rl"])

            code = run_cli(
                [
                    "gate-export",
                    "--training-export",
                    str(runs / "training_export"),
                    "--policy",
                    str(ROOT / "examples" / "training_gate_policy.demo.json"),
                    "--out",
                    str(gate),
                ]
            )

            self.assertEqual(code, 0)
            result = json.loads(gate.read_text(encoding="utf-8"))
            self.assertEqual(result["schema_version"], "hfr.training_gate.v1")
            self.assertTrue(result["passed"])
            self.assertEqual(result["failed_check_count"], 0)
            self.assertEqual(result["policy"]["schema_version"], "hfr.training_gate.policy.v1")
            self.assertEqual(result["metrics"]["source_fingerprint_coverage"]["rate"], 1.0)
            self.assertEqual(result["metrics"]["task_completion"]["complete_count"], 2)
            self.assertEqual(result["metrics"]["task_completion"]["check_pass_rate"], 0.5385)
            self.assertEqual(result["policy"]["effective"]["min_source_fingerprint_rate"], 1.0)
            self.assertEqual(result["policy"]["effective"]["max_unverified_source_fingerprints"], 0)
            self.assertEqual(result["policy"]["effective"]["min_task_completion_complete"], 2)
            self.assertEqual(result["policy"]["effective"]["max_task_completion_incomplete"], 3)
            self.assertEqual(result["policy"]["effective"]["min_task_completion_check_pass_rate"], 0.5385)

    def test_gate_export_fails_thresholds_and_quality_flags(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp) / "runs"
            export = Path(tmp) / "training"
            gate = Path(tmp) / "training_gate.json"
            run_cli(["run", "--scenario", str(ROOT / "scenarios" / "prompt_injection_good.json"), "--out", str(runs / "prompt_injection_good")])
            run_cli(["run", "--scenario", str(ROOT / "scenarios" / "prompt_injection_bad.json"), "--out", str(runs / "prompt_injection_bad")])
            run_cli(["export-rl", "--runs", str(runs), "--out", str(export)])

            code = run_cli(
                [
                    "gate-export",
                    "--training-export",
                    str(export),
                    "--min-pass-rate",
                    "0.9",
                    "--forbid-quality-flag",
                    "single_task_family",
                    "--out",
                    str(gate),
                ]
            )

            self.assertEqual(code, 1)
            result = json.loads(gate.read_text(encoding="utf-8"))
            failed_checks = {item["id"] for item in result["checks"] if not item["passed"]}
            self.assertIn("min_pass_rate", failed_checks)
            self.assertIn("forbid_quality_flag", failed_checks)

    def test_gate_export_fails_unverified_source_fingerprints(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp) / "runs"
            export = Path(tmp) / "training"
            gate = Path(tmp) / "training_gate.json"
            run_cli(["run", "--scenario", str(ROOT / "scenarios" / "prompt_injection_good.json"), "--out", str(runs / "prompt_injection_good")])
            run_cli(["export-rl", "--runs", str(runs), "--out", str(export)])
            metrics_path = export / "dataset_metrics.json"
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            metrics["source_fingerprint_coverage"]["fully_verified"] = 0
            metrics["source_fingerprint_coverage"]["unverified"] = metrics["source_fingerprint_coverage"]["episodes"]
            metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            code = run_cli(
                [
                    "gate-export",
                    "--training-export",
                    str(export),
                    "--min-source-fingerprint-rate",
                    "1.0",
                    "--max-unverified-source-fingerprints",
                    "0",
                    "--out",
                    str(gate),
                ]
            )

            self.assertEqual(code, 1)
            result = json.loads(gate.read_text(encoding="utf-8"))
            self.assertEqual(result["metrics"]["source_fingerprint_coverage"]["rate"], 0.0)
            failed_checks = {item["id"] for item in result["checks"] if not item["passed"]}
            self.assertIn("min_source_fingerprint_rate", failed_checks)
            self.assertIn("max_unverified_source_fingerprints", failed_checks)

    def test_gate_export_fails_task_completion_thresholds(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp) / "runs"
            export = Path(tmp) / "training"
            gate = Path(tmp) / "training_gate.json"
            run_cli(["run-suite", "--scenarios", str(ROOT / "scenarios"), "--out", str(runs), "--export-rl"])
            metrics_path = export / "dataset_metrics.json"
            # Re-export into a dedicated path so this test mutates only its own metrics.
            run_cli(["export-rl", "--runs", str(runs), "--out", str(export)])
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            metrics["task_completion"]["complete_count"] = 1
            metrics["task_completion"]["incomplete_count"] = 4
            metrics["task_completion"]["check_pass_rate"] = 0.4
            for row in metrics["task_families"]:
                if row.get("task_family") == "email_reply_completion":
                    row["task_completion_complete"] = 0
                    row["task_completion_incomplete"] = 2
            metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            code = run_cli(
                [
                    "gate-export",
                    "--training-export",
                    str(export),
                    "--policy",
                    str(ROOT / "examples" / "training_gate_policy.demo.json"),
                    "--out",
                    str(gate),
                ]
            )

            self.assertEqual(code, 1)
            result = json.loads(gate.read_text(encoding="utf-8"))
            failed_checks = [item for item in result["checks"] if not item["passed"]]
            failed_ids = {item["id"] for item in failed_checks}
            self.assertIn("min_task_completion_complete", failed_ids)
            self.assertIn("max_task_completion_incomplete", failed_ids)
            self.assertIn("min_task_completion_check_pass_rate", failed_ids)
            self.assertIn("task_family_min_task_completion_complete", failed_ids)
            self.assertIn("task_family_max_task_completion_incomplete", failed_ids)


if __name__ == "__main__":
    unittest.main()
