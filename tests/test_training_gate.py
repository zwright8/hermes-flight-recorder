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
            self.assertEqual(result["metrics"]["validation"]["passed"], True)
            self.assertEqual(result["metrics"]["validation"]["error_count"], 0)
            self.assertEqual(result["policy"]["schema_version"], "hfr.training_gate.policy.v1")
            self.assertTrue(result["policy"]["effective"]["require_valid_export"])
            self.assertTrue(result["policy"]["effective"]["strict_validation"])
            self.assertEqual(result["metrics"]["source_fingerprint_coverage"]["rate"], 1.0)
            self.assertEqual(result["metrics"]["trainer_view_source_fingerprint_coverage"]["rows"], 10)
            self.assertEqual(result["metrics"]["trainer_view_source_fingerprint_coverage"]["fully_verified"], 10)
            self.assertEqual(result["metrics"]["trainer_view_source_fingerprint_coverage"]["fully_verified_rate"], 1.0)
            self.assertEqual(result["metrics"]["task_completion"]["complete_count"], 2)
            self.assertEqual(result["metrics"]["task_completion"]["check_pass_rate"], 0.5385)
            self.assertEqual(result["metrics"]["trace_signal"]["average_event_count"], 5.67)
            self.assertEqual(result["metrics"]["trace_signal"]["tool_or_api_episode_rate"], 0.8333)
            self.assertEqual(result["metrics"]["trace_signal"]["risk_count"], 2)
            self.assertEqual(result["metrics"]["dataset_splits"]["task_family_count"], 4)
            self.assertEqual(result["metrics"]["dataset_splits"]["train_episode_count"], 3)
            self.assertEqual(result["metrics"]["dataset_splits"]["validation_episode_count"], 2)
            self.assertEqual(result["metrics"]["dataset_splits"]["test_episode_count"], 1)
            self.assertTrue(result["metrics"]["dataset_splits"]["family_exclusive"])
            self.assertEqual(result["policy"]["effective"]["min_source_fingerprint_rate"], 1.0)
            self.assertEqual(result["policy"]["effective"]["max_unverified_source_fingerprints"], 0)
            self.assertEqual(result["policy"]["effective"]["min_trainer_view_source_fingerprint_rate"], 1.0)
            self.assertEqual(result["policy"]["effective"]["max_unverified_trainer_view_source_fingerprints"], 0)
            self.assertEqual(result["policy"]["effective"]["min_task_completion_complete"], 2)
            self.assertEqual(result["policy"]["effective"]["max_task_completion_incomplete"], 3)
            self.assertEqual(result["policy"]["effective"]["min_task_completion_check_pass_rate"], 0.5385)
            self.assertEqual(result["policy"]["effective"]["min_trace_average_events"], 5.0)
            self.assertEqual(result["policy"]["effective"]["min_trace_event_type_count"], 4)
            self.assertEqual(result["policy"]["effective"]["min_trace_final_answer_rate"], 1.0)
            self.assertEqual(result["policy"]["effective"]["min_trace_tool_or_api_rate"], 0.8)
            self.assertEqual(result["policy"]["effective"]["max_trace_empty_final_answers"], 0)
            self.assertEqual(result["policy"]["effective"]["max_trace_risk_count"], 2)
            self.assertEqual(result["policy"]["effective"]["min_split_task_families"], 4)
            self.assertEqual(result["policy"]["effective"]["min_train_episodes"], 3)
            self.assertEqual(result["policy"]["effective"]["min_validation_episodes"], 2)
            self.assertEqual(result["policy"]["effective"]["min_test_episodes"], 1)
            self.assertTrue(result["policy"]["effective"]["require_family_exclusive_splits"])
            self.assertEqual(result["policy"]["effective"]["require_trace_event_types"], ["assistant_message"])

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

    def test_gate_export_blocks_invalid_export_fingerprints_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp) / "runs"
            export = Path(tmp) / "training"
            gate = Path(tmp) / "training_gate.json"
            run_cli(["run", "--scenario", str(ROOT / "scenarios" / "prompt_injection_good.json"), "--out", str(runs / "prompt_injection_good")])
            run_cli(["export-rl", "--runs", str(runs), "--out", str(export)])
            manifest_path = export / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["artifact_fingerprints"]["episodes"]["sha256"] = "0" * 64
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            code = run_cli(["gate-export", "--training-export", str(export), "--out", str(gate)])

            self.assertEqual(code, 1)
            result = json.loads(gate.read_text(encoding="utf-8"))
            failed_checks = {item["id"] for item in result["checks"] if not item["passed"]}
            self.assertIn("valid_training_export", failed_checks)
            self.assertEqual(result["metrics"]["validation"]["passed"], False)
            self.assertGreater(result["metrics"]["validation"]["error_count"], 0)

    def test_gate_export_can_explicitly_skip_validation_for_legacy_handoffs(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp) / "runs"
            export = Path(tmp) / "training"
            gate = Path(tmp) / "training_gate.json"
            run_cli(["run", "--scenario", str(ROOT / "scenarios" / "prompt_injection_good.json"), "--out", str(runs / "prompt_injection_good")])
            run_cli(["export-rl", "--runs", str(runs), "--out", str(export)])
            manifest_path = export / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["artifact_fingerprints"]["episodes"]["sha256"] = "0" * 64
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            code = run_cli(["gate-export", "--training-export", str(export), "--skip-validation", "--out", str(gate)])

            self.assertEqual(code, 0)
            result = json.loads(gate.read_text(encoding="utf-8"))
            self.assertNotIn("valid_training_export", {item["id"] for item in result["checks"]})
            self.assertEqual(result["metrics"]["validation"]["available"], False)

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

    def test_gate_export_fails_unverified_trainer_view_source_fingerprints(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp) / "runs"
            export = Path(tmp) / "training"
            gate = Path(tmp) / "training_gate.json"
            run_cli(["run", "--scenario", str(ROOT / "scenarios" / "prompt_injection_good.json"), "--out", str(runs / "prompt_injection_good")])
            run_cli(["run", "--scenario", str(ROOT / "scenarios" / "prompt_injection_bad.json"), "--out", str(runs / "prompt_injection_bad")])
            run_cli(["export-rl", "--runs", str(runs), "--out", str(export)])
            metrics_path = export / "dataset_metrics.json"
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            coverage = metrics["trainer_view_source_fingerprint_coverage"]
            coverage["fully_verified"] = 0
            coverage["unverified"] = coverage["rows"]
            coverage["fully_verified_rate"] = 0.0
            metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            code = run_cli(
                [
                    "gate-export",
                    "--training-export",
                    str(export),
                    "--min-trainer-view-source-fingerprint-rate",
                    "1.0",
                    "--max-unverified-trainer-view-source-fingerprints",
                    "0",
                    "--out",
                    str(gate),
                ]
            )

            self.assertEqual(code, 1)
            result = json.loads(gate.read_text(encoding="utf-8"))
            self.assertEqual(result["metrics"]["trainer_view_source_fingerprint_coverage"]["fully_verified_rate"], 0.0)
            failed_checks = {item["id"] for item in result["checks"] if not item["passed"]}
            self.assertIn("min_trainer_view_source_fingerprint_rate", failed_checks)
            self.assertIn("max_unverified_trainer_view_source_fingerprints", failed_checks)

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

    def test_gate_export_fails_trace_signal_thresholds(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp) / "runs"
            export = Path(tmp) / "training"
            gate = Path(tmp) / "training_gate.json"
            run_cli(["run-suite", "--scenarios", str(ROOT / "scenarios"), "--out", str(runs), "--export-rl"])
            run_cli(["export-rl", "--runs", str(runs), "--out", str(export)])
            metrics_path = export / "dataset_metrics.json"
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            metrics["trace_signal"]["average_event_count"] = 1.0
            metrics["trace_signal"]["event_type_count"] = 1
            metrics["trace_signal"]["final_answer_rate"] = 0.5
            metrics["trace_signal"]["tool_or_api_episode_rate"] = 0.0
            metrics["trace_signal"]["empty_final_answer_count"] = 3
            metrics["trace_signal"]["risk_count"] = 99
            metrics["trace_signal"]["event_type_counts"] = [{"id": "user_message", "count": 6}]
            metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            code = run_cli(
                [
                    "gate-export",
                    "--training-export",
                    str(export),
                    "--min-trace-average-events",
                    "5",
                    "--min-trace-event-type-count",
                    "4",
                    "--min-trace-final-answer-rate",
                    "1.0",
                    "--min-trace-tool-or-api-rate",
                    "0.8",
                    "--max-trace-empty-final-answers",
                    "0",
                    "--max-trace-risk-count",
                    "2",
                    "--require-trace-event-type",
                    "assistant_message",
                    "--out",
                    str(gate),
                ]
            )

            self.assertEqual(code, 1)
            result = json.loads(gate.read_text(encoding="utf-8"))
            failed_ids = [item["id"] for item in result["checks"] if not item["passed"]]
            self.assertIn("min_trace_average_events", failed_ids)
            self.assertIn("min_trace_event_type_count", failed_ids)
            self.assertIn("min_trace_final_answer_rate", failed_ids)
            self.assertIn("min_trace_tool_or_api_rate", failed_ids)
            self.assertIn("max_trace_empty_final_answers", failed_ids)
            self.assertIn("max_trace_risk_count", failed_ids)
            self.assertIn("require_trace_event_type", failed_ids)

    def test_gate_export_fails_dataset_split_thresholds(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp) / "runs"
            export = Path(tmp) / "training"
            gate = Path(tmp) / "training_gate.json"
            run_cli(["run-suite", "--scenarios", str(ROOT / "scenarios"), "--out", str(runs), "--export-rl"])
            run_cli(["export-rl", "--runs", str(runs), "--out", str(export)])
            metrics_path = export / "dataset_metrics.json"
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            metrics["dataset_splits"] = {
                "task_family_count": 1,
                "episode_count": 6,
                "train_episode_count": 6,
                "validation_episode_count": 0,
                "test_episode_count": 0,
                "family_exclusive": False,
            }
            metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            code = run_cli(
                [
                    "gate-export",
                    "--training-export",
                    str(export),
                    "--skip-validation",
                    "--min-split-task-families",
                    "4",
                    "--min-validation-episodes",
                    "1",
                    "--min-test-episodes",
                    "1",
                    "--require-family-exclusive-splits",
                    "--out",
                    str(gate),
                ]
            )

            self.assertEqual(code, 1)
            result = json.loads(gate.read_text(encoding="utf-8"))
            failed_ids = [item["id"] for item in result["checks"] if not item["passed"]]
            self.assertIn("min_split_task_families", failed_ids)
            self.assertIn("min_validation_episodes", failed_ids)
            self.assertIn("min_test_episodes", failed_ids)
            self.assertIn("require_family_exclusive_splits", failed_ids)
            self.assertEqual(result["metrics"]["dataset_splits"]["family_exclusive"], False)


if __name__ == "__main__":
    unittest.main()
