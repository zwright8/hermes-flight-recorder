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


class ValidationTests(unittest.TestCase):
    def test_validate_accepts_generated_runs_and_training_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp) / "runs"
            export = runs / "training_export"
            summary_path = Path(tmp) / "validation.json"
            run_cli(["run", "--scenario", str(ROOT / "scenarios" / "prompt_injection_good.json"), "--out", str(runs / "good")])
            run_cli(["run", "--scenario", str(ROOT / "scenarios" / "prompt_injection_bad.json"), "--out", str(runs / "bad")])
            run_cli(["export-rl", "--runs", str(runs), "--out", str(export)])

            code = run_cli(
                [
                    "validate",
                    "--runs",
                    str(runs),
                    "--training-export",
                    str(export),
                    "--out",
                    str(summary_path),
                    "--strict",
                ]
            )

            self.assertEqual(code, 0)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertTrue(summary["passed"])
            self.assertEqual(summary["schema_version"], "hfr.validation.v1")
            self.assertEqual(summary["error_count"], 0)
            self.assertEqual(summary["target_count"], 3)

    def test_validate_rejects_inconsistent_scorecard(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            run_cli(["run", "--scenario", str(ROOT / "scenarios" / "prompt_injection_good.json"), "--out", str(run_dir)])
            score_path = run_dir / "scorecard.json"
            scorecard = json.loads(score_path.read_text(encoding="utf-8"))
            scorecard["passed"] = False
            score_path.write_text(json.dumps(scorecard), encoding="utf-8")

            code = run_cli(["validate", "--run", str(run_dir)])

            self.assertEqual(code, 1)

    def test_validate_rejects_malformed_scorecard_evidence_refs(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            summary_path = Path(tmp) / "validation.json"
            run_cli(["run", "--scenario", str(ROOT / "scenarios" / "prompt_injection_bad.json"), "--out", str(run_dir)])
            score_path = run_dir / "scorecard.json"
            scorecard = json.loads(score_path.read_text(encoding="utf-8"))
            scorecard["rules"][0]["evidence_refs"] = [{"target": "event", "event_index": -1}]
            score_path.write_text(json.dumps(scorecard), encoding="utf-8")

            code = run_cli(["validate", "--run", str(run_dir), "--out", str(summary_path)])

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in summary["targets"] for error in target["errors"])
            self.assertIn("evidence_refs[0].event_index", errors)

    def test_validate_rejects_broken_preference_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp) / "runs"
            export = Path(tmp) / "training"
            summary_path = Path(tmp) / "validation.json"
            run_cli(["run", "--scenario", str(ROOT / "scenarios" / "prompt_injection_good.json"), "--out", str(runs / "prompt_injection_good")])
            run_cli(["run", "--scenario", str(ROOT / "scenarios" / "prompt_injection_bad.json"), "--out", str(runs / "prompt_injection_bad")])
            run_cli(["export-rl", "--runs", str(runs), "--out", str(export)])
            preference_path = export / "preferences.jsonl"
            preference = json.loads(preference_path.read_text(encoding="utf-8").splitlines()[0])
            preference["chosen_episode_id"] = "missing-episode"
            preference_path.write_text(json.dumps(preference) + "\n", encoding="utf-8")

            code = run_cli(["validate", "--training-export", str(export), "--out", str(summary_path)])

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in summary["targets"] for error in target["errors"])
            self.assertIn("does not reference an episode", errors)

    def test_validate_rejects_broken_failure_mode_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp) / "runs"
            export = Path(tmp) / "training"
            summary_path = Path(tmp) / "validation.json"
            run_cli(["run", "--scenario", str(ROOT / "scenarios" / "prompt_injection_bad.json"), "--out", str(runs / "prompt_injection_bad")])
            run_cli(["export-rl", "--runs", str(runs), "--out", str(export)])
            failure_path = export / "failure_modes.jsonl"
            failure = json.loads(failure_path.read_text(encoding="utf-8").splitlines()[0])
            failure["episode_id"] = "missing-episode"
            failure_path.write_text(json.dumps(failure) + "\n", encoding="utf-8")

            code = run_cli(["validate", "--training-export", str(export), "--out", str(summary_path)])

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in summary["targets"] for error in target["errors"])
            self.assertIn("failure_modes[0].episode_id", errors)
            self.assertIn("does not reference an episode", errors)

    def test_validate_rejects_broken_step_reward_event_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp) / "runs"
            export = Path(tmp) / "training"
            summary_path = Path(tmp) / "validation.json"
            run_cli(["run", "--scenario", str(ROOT / "scenarios" / "prompt_injection_bad.json"), "--out", str(runs / "prompt_injection_bad")])
            run_cli(["export-rl", "--runs", str(runs), "--out", str(export)])
            step_reward_path = export / "step_rewards.jsonl"
            rows = step_reward_path.read_text(encoding="utf-8").splitlines()
            step_reward = json.loads(rows[0])
            step_reward["target"] = "event"
            step_reward["event_index"] = 999
            rows[0] = json.dumps(step_reward)
            step_reward_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

            code = run_cli(["validate", "--training-export", str(export), "--out", str(summary_path)])

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in summary["targets"] for error in target["errors"])
            self.assertIn("step_rewards[0].event_index", errors)
            self.assertIn("outside episode", errors)

    def test_validate_rejects_step_reward_delta_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp) / "runs"
            export = Path(tmp) / "training"
            summary_path = Path(tmp) / "validation.json"
            run_cli(["run", "--scenario", str(ROOT / "scenarios" / "prompt_injection_bad.json"), "--out", str(runs / "prompt_injection_bad")])
            run_cli(["export-rl", "--runs", str(runs), "--out", str(export)])
            step_reward_path = export / "step_rewards.jsonl"
            rows = [json.loads(line) for line in step_reward_path.read_text(encoding="utf-8").splitlines()]
            rows[0]["reward_delta"] = 0.0
            step_reward_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

            code = run_cli(["validate", "--training-export", str(export), "--out", str(summary_path)])

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in summary["targets"] for error in target["errors"])
            self.assertIn("step_rewards for episode", errors)
            self.assertIn("expected", errors)

    def test_validate_rejects_reward_model_view_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp) / "runs"
            export = Path(tmp) / "training"
            summary_path = Path(tmp) / "validation.json"
            run_cli(["run", "--scenario", str(ROOT / "scenarios" / "prompt_injection_good.json"), "--out", str(runs / "prompt_injection_good")])
            run_cli(["export-rl", "--runs", str(runs), "--out", str(export)])
            reward_model_path = export / "reward_model.jsonl"
            sample = json.loads(reward_model_path.read_text(encoding="utf-8").splitlines()[0])
            sample["response"] = "drifted response"
            reward_model_path.write_text(json.dumps(sample) + "\n", encoding="utf-8")

            code = run_cli(["validate", "--training-export", str(export), "--out", str(summary_path)])

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in summary["targets"] for error in target["errors"])
            self.assertIn("reward_model[0].response does not match episode", errors)

    def test_validate_rejects_missing_dpo_view_pair(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp) / "runs"
            export = Path(tmp) / "training"
            summary_path = Path(tmp) / "validation.json"
            run_cli(["run", "--scenario", str(ROOT / "scenarios" / "prompt_injection_good.json"), "--out", str(runs / "prompt_injection_good")])
            run_cli(["run", "--scenario", str(ROOT / "scenarios" / "prompt_injection_bad.json"), "--out", str(runs / "prompt_injection_bad")])
            run_cli(["export-rl", "--runs", str(runs), "--out", str(export)])
            (export / "dpo.jsonl").write_text("", encoding="utf-8")

            code = run_cli(["validate", "--training-export", str(export), "--out", str(summary_path)])

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in summary["targets"] for error in target["errors"])
            self.assertIn("manifest.dpo_count", errors)
            self.assertIn("dpo.jsonl missing preference pairs", errors)

    def test_validate_rejects_dataset_metrics_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp) / "runs"
            export = Path(tmp) / "training"
            summary_path = Path(tmp) / "validation.json"
            run_cli(["run", "--scenario", str(ROOT / "scenarios" / "prompt_injection_good.json"), "--out", str(runs / "prompt_injection_good")])
            run_cli(["run", "--scenario", str(ROOT / "scenarios" / "prompt_injection_bad.json"), "--out", str(runs / "prompt_injection_bad")])
            run_cli(["export-rl", "--runs", str(runs), "--out", str(export)])
            metrics_path = export / "dataset_metrics.json"
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            metrics["pass_rate"] = 1.0
            metrics_path.write_text(json.dumps(metrics), encoding="utf-8")

            code = run_cli(["validate", "--training-export", str(export), "--out", str(summary_path)])

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in summary["targets"] for error in target["errors"])
            self.assertIn("dataset_metrics.pass_rate", errors)

    def test_validate_warns_on_legacy_training_export_without_trainer_views(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp) / "runs"
            export = Path(tmp) / "training"
            summary_path = Path(tmp) / "validation.json"
            run_cli(["run", "--scenario", str(ROOT / "scenarios" / "prompt_injection_good.json"), "--out", str(runs / "prompt_injection_good")])
            run_cli(["export-rl", "--runs", str(runs), "--out", str(export)])
            for name in ("sft", "dpo", "reward_model"):
                (export / f"{name}.jsonl").unlink()
            (export / "dataset_metrics.json").unlink()
            (export / "DATASET_CARD.md").unlink()
            manifest_path = export / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for name in ("sft", "dpo", "reward_model", "dataset_metrics", "dataset_card"):
                manifest["outputs"].pop(name, None)
            for name in ("sft_count", "dpo_count", "reward_model_count", "quality_flag_count"):
                manifest.pop(name, None)
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            non_strict = run_cli(["validate", "--training-export", str(export), "--out", str(summary_path)])
            strict = run_cli(["validate", "--training-export", str(export), "--strict"])

            self.assertEqual(non_strict, 0)
            self.assertEqual(strict, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            warnings = "\n".join(warning for target in summary["targets"] for warning in target["warnings"])
            self.assertIn("sft.jsonl is missing", warnings)
            self.assertIn("manifest.sft_count is missing", warnings)
            self.assertIn("dataset_metrics.json is missing", warnings)
            self.assertIn("manifest.quality_flag_count is missing", warnings)

    def test_validate_accepts_suite_summary_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp) / "runs"
            summary_path = Path(tmp) / "validation.json"
            run_cli(["run-suite", "--scenarios", str(ROOT / "scenarios"), "--out", str(runs)])

            code = run_cli(
                [
                    "validate",
                    "--suite-summary",
                    str(runs / "suite_summary.json"),
                    "--out",
                    str(summary_path),
                    "--strict",
                ]
            )

            self.assertEqual(code, 0)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertTrue(summary["passed"])
            self.assertEqual(summary["target_count"], 1)
            self.assertEqual(summary["targets"][0]["type"], "suite_summary")

    def test_validate_rejects_broken_suite_summary_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp) / "runs"
            summary_path = Path(tmp) / "validation.json"
            run_cli(["run-suite", "--scenarios", str(ROOT / "scenarios"), "--out", str(runs)])
            suite_path = runs / "suite_summary.json"
            suite = json.loads(suite_path.read_text(encoding="utf-8"))
            suite["metrics"]["pass_rate"] = 1.0
            suite_path.write_text(json.dumps(suite), encoding="utf-8")

            code = run_cli(
                [
                    "validate",
                    "--suite-summary",
                    str(suite_path),
                    "--out",
                    str(summary_path),
                ]
            )

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in summary["targets"] for error in target["errors"])
            self.assertIn("suite_summary.metrics.pass_rate", errors)

    def test_validate_warns_on_legacy_family_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp) / "runs"
            summary_path = Path(tmp) / "validation.json"
            run_cli(["run-suite", "--scenarios", str(ROOT / "scenarios"), "--out", str(runs)])
            suite_path = runs / "suite_summary.json"
            suite = json.loads(suite_path.read_text(encoding="utf-8"))
            for row in suite["metrics"]["task_families"]:
                row.pop("critical_failure_counts", None)
            suite_path.write_text(json.dumps(suite), encoding="utf-8")

            code = run_cli(
                [
                    "validate",
                    "--suite-summary",
                    str(suite_path),
                    "--out",
                    str(summary_path),
                ]
            )

            self.assertEqual(code, 0)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            warnings = "\n".join(warning for target in summary["targets"] for warning in target["warnings"])
            self.assertIn("critical_failure_counts is missing", warnings)

    def test_validate_strict_fails_on_warnings(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            run_cli(["run", "--scenario", str(ROOT / "scenarios" / "prompt_injection_good.json"), "--out", str(run_dir)])
            (run_dir / "report.html").unlink()

            non_strict = run_cli(["validate", "--run", str(run_dir)])
            strict = run_cli(["validate", "--run", str(run_dir), "--strict"])

            self.assertEqual(non_strict, 0)
            self.assertEqual(strict, 1)

    def test_validate_without_targets_fails(self):
        with redirect_stdout(StringIO()):
            code = main(["validate"])

        self.assertEqual(code, 1)


if __name__ == "__main__":
    unittest.main()
