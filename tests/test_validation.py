import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from flightrecorder.cli import main
from flightrecorder.schema_registry import check_schema_file


ROOT = Path(__file__).resolve().parents[1]


def run_cli(args):
    with redirect_stdout(StringIO()):
        return main(args)


def write_jsonl(path: Path, rows):
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


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
            schema = check_schema_file(summary_path)
            self.assertTrue(schema["passed"], schema["errors"])
            self.assertEqual(schema["schema"]["name"], "validation")

    def test_validate_rejects_verified_source_fingerprint_without_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp) / "runs"
            export = Path(tmp) / "training_export"
            summary_path = Path(tmp) / "validation.json"
            run_cli(["run", "--scenario", str(ROOT / "scenarios" / "prompt_injection_good.json"), "--out", str(runs / "good")])
            run_cli(["export-rl", "--runs", str(runs), "--out", str(export)])
            episodes_path = export / "episodes.jsonl"
            episodes = [json.loads(line) for line in episodes_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            episodes[0]["source_fingerprints"]["scenario"].pop("size_bytes")
            write_jsonl(episodes_path, episodes)

            code = run_cli(["validate", "--training-export", str(export), "--out", str(summary_path)])

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in summary["targets"] for error in target["errors"])
            self.assertIn("episodes[0].source_fingerprint_status verified requires scenario and source_trace SHA-256 and size_bytes values.", errors)

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

    def test_validate_rejects_stale_task_completion_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            summary_path = Path(tmp) / "validation.json"
            run_cli(["run", "--scenario", str(ROOT / "scenarios" / "email_reply_completion_good.json"), "--out", str(run_dir)])
            task_path = run_dir / "task_completion.json"
            task = json.loads(task_path.read_text(encoding="utf-8"))
            task["status"] = "incomplete"
            task["passed"] = False
            task_path.write_text(json.dumps(task), encoding="utf-8")

            code = run_cli(["validate", "--run", str(run_dir), "--out", str(summary_path)])

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in summary["targets"] for error in target["errors"])
            self.assertIn("task_completion.json must match scorecard.task_completion", errors)

    def test_validate_accepts_captured_state_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = root / "artifact.json"
            artifact.write_text(json.dumps({"status": "complete"}), encoding="utf-8")
            snapshot = root / "state.json"
            summary_path = root / "validation.json"
            run_cli(
                [
                    "capture-state",
                    "--file",
                    f"artifact={artifact}",
                    "--json",
                    f"artifact={artifact}",
                    "--set",
                    "task.status=complete",
                    "--preserve-paths",
                    "--out",
                    str(snapshot),
                ]
            )

            code = run_cli(["validate", "--state-snapshot", str(snapshot), "--out", str(summary_path), "--strict"])

            self.assertEqual(code, 0)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertTrue(summary["passed"])
            self.assertEqual(summary["targets"][0]["type"], "state_snapshot")

    def test_validate_rejects_stale_captured_state_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = root / "artifact.txt"
            artifact.write_text("first", encoding="utf-8")
            snapshot = root / "state.json"
            summary_path = root / "validation.json"
            run_cli(
                [
                    "capture-state",
                    "--file",
                    f"artifact={artifact}",
                    "--preserve-paths",
                    "--out",
                    str(snapshot),
                ]
            )
            artifact.write_text("changed", encoding="utf-8")

            code = run_cli(["validate", "--state-snapshot", str(snapshot), "--out", str(summary_path)])

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in summary["targets"] for error in target["errors"])
            self.assertIn("state_snapshot.filesystem.files.artifact.sha256", errors)

    def test_validate_rejects_stale_captured_state_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = root / "artifact.txt"
            artifact.write_text("first", encoding="utf-8")
            snapshot = root / "state.json"
            summary_path = root / "validation.json"
            run_cli(
                [
                    "capture-state",
                    "--file",
                    f"artifact={artifact}",
                    "--preserve-paths",
                    "--out",
                    str(snapshot),
                ]
            )
            payload = json.loads(snapshot.read_text(encoding="utf-8"))
            payload["filesystem"]["files"]["artifact"]["size_bytes"] += 1
            snapshot.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            code = run_cli(["validate", "--state-snapshot", str(snapshot), "--out", str(summary_path)])

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in summary["targets"] for error in target["errors"])
            self.assertIn("state_snapshot.filesystem.files.artifact.size_bytes", errors)

    def test_validate_run_dir_rejects_stale_captured_state_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = root / "artifact.txt"
            artifact.write_text("first", encoding="utf-8")
            state_path = root / "state.json"
            scenario_path = root / "scenario.json"
            run_dir = root / "run"
            summary_path = root / "validation.json"
            run_cli(
                [
                    "capture-state",
                    "--file",
                    f"artifact={artifact}",
                    "--set",
                    "gmail.threads.email-123.sent_replies.0.status=sent",
                    "--set",
                    "gmail.threads.email-123.sent_replies.0.message_id=msg-email-123-001",
                    "--preserve-paths",
                    "--out",
                    str(state_path),
                ]
            )
            scenario = json.loads((ROOT / "scenarios" / "email_reply_completion_good.json").read_text(encoding="utf-8"))
            scenario["trace"]["path"] = str(ROOT / "fixtures" / "email_reply_completion_good.observer.jsonl")
            scenario["state"]["path"] = str(state_path)
            scenario["state"].pop("before_path", None)
            scenario["assertions"]["required_state"][0]["where"] = {
                "observations.gmail.threads.email-123.sent_replies.0.message_id": {"matches": "^msg-email-123-"},
                "observations.gmail.threads.email-123.sent_replies.0.status": "sent",
            }
            scenario["assertions"]["required_state_transitions"] = []
            scenario_path.write_text(json.dumps(scenario), encoding="utf-8")
            run_cli(["run", "--scenario", str(scenario_path), "--out", str(run_dir), "--fail-on-score"])
            artifact.write_text("changed", encoding="utf-8")

            code = run_cli(["validate", "--run", str(run_dir), "--out", str(summary_path)])

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in summary["targets"] for error in target["errors"])
            self.assertIn("state_snapshot.filesystem.files.artifact.sha256", errors)

    def test_validate_run_dir_rejects_stale_captured_state_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = root / "artifact.txt"
            artifact.write_text("first", encoding="utf-8")
            state_path = root / "state.json"
            scenario_path = root / "scenario.json"
            run_dir = root / "run"
            summary_path = root / "validation.json"
            run_cli(
                [
                    "capture-state",
                    "--file",
                    f"artifact={artifact}",
                    "--set",
                    "gmail.threads.email-123.sent_replies.0.status=sent",
                    "--set",
                    "gmail.threads.email-123.sent_replies.0.message_id=msg-email-123-001",
                    "--preserve-paths",
                    "--out",
                    str(state_path),
                ]
            )
            scenario = json.loads((ROOT / "scenarios" / "email_reply_completion_good.json").read_text(encoding="utf-8"))
            scenario["trace"]["path"] = str(ROOT / "fixtures" / "email_reply_completion_good.observer.jsonl")
            scenario["state"]["path"] = str(state_path)
            scenario["state"].pop("before_path", None)
            scenario["assertions"]["required_state"][0]["where"] = {
                "observations.gmail.threads.email-123.sent_replies.0.message_id": {"matches": "^msg-email-123-"},
                "observations.gmail.threads.email-123.sent_replies.0.status": "sent",
            }
            scenario["assertions"]["required_state_transitions"] = []
            scenario_path.write_text(json.dumps(scenario), encoding="utf-8")
            run_cli(["run", "--scenario", str(scenario_path), "--out", str(run_dir), "--fail-on-score"])
            captured = run_dir / "state_snapshot.json"
            payload = json.loads(captured.read_text(encoding="utf-8"))
            payload["filesystem"]["files"]["artifact"]["size_bytes"] += 1
            captured.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            code = run_cli(["validate", "--run", str(run_dir), "--out", str(summary_path)])

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in summary["targets"] for error in target["errors"])
            self.assertIn("state_snapshot.filesystem.files.artifact.size_bytes", errors)

    def test_validate_rejects_stale_lineage_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            summary_path = Path(tmp) / "validation.json"
            run_cli(["run", "--scenario", str(ROOT / "scenarios" / "prompt_injection_good.json"), "--out", str(run_dir)])
            lineage_path = run_dir / "artifact_lineage.json"
            lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
            score_record = next(item for item in lineage["outputs"] if item["name"] == "scorecard")
            score_record["sha256"] = "0" * 64
            lineage_path.write_text(json.dumps(lineage), encoding="utf-8")

            code = run_cli(["validate", "--run", str(run_dir), "--out", str(summary_path)])

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in summary["targets"] for error in target["errors"])
            self.assertIn("artifact_lineage.outputs.scorecard.sha256", errors)

    def test_validate_rejects_stale_lineage_replay_fingerprint(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            summary_path = Path(tmp) / "validation.json"
            run_cli(["run", "--scenario", str(ROOT / "scenarios" / "prompt_injection_good.json"), "--out", str(run_dir)])
            lineage_path = run_dir / "artifact_lineage.json"
            lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
            lineage["replay"]["input_fingerprints"]["scenario"]["sha256"] = "0" * 64
            lineage_path.write_text(json.dumps(lineage), encoding="utf-8")

            code = run_cli(["validate", "--run", str(run_dir), "--out", str(summary_path)])

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in summary["targets"] for error in target["errors"])
            self.assertIn("artifact_lineage.replay.input_fingerprints.scenario.sha256", errors)

    def test_validate_rejects_stale_lineage_replay_input_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            summary_path = Path(tmp) / "validation.json"
            run_cli(["run", "--scenario", str(ROOT / "scenarios" / "prompt_injection_good.json"), "--out", str(run_dir)])
            lineage_path = run_dir / "artifact_lineage.json"
            lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
            lineage["replay"]["input_fingerprints"]["scenario"]["size_bytes"] += 1
            lineage_path.write_text(json.dumps(lineage), encoding="utf-8")

            code = run_cli(["validate", "--run", str(run_dir), "--out", str(summary_path)])

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in summary["targets"] for error in target["errors"])
            self.assertIn("artifact_lineage.replay.input_fingerprints.scenario.size_bytes", errors)

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

    def test_validate_rejects_broken_curriculum_priority_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp) / "runs"
            export = Path(tmp) / "training"
            summary_path = Path(tmp) / "validation.json"
            run_cli(["run", "--scenario", str(ROOT / "scenarios" / "prompt_injection_bad.json"), "--out", str(runs / "prompt_injection_bad")])
            run_cli(["export-rl", "--runs", str(runs), "--out", str(export)])
            curriculum_path = export / "curriculum.json"
            curriculum = json.loads(curriculum_path.read_text(encoding="utf-8"))
            mode = curriculum["task_families"][0]["failure_modes"][0]
            mode["priority_score"] = 999
            mode["priority_band"] = "urgent"
            mode["example_evidence_refs"] = [{"target": "not-a-target"}]
            curriculum_path.write_text(json.dumps(curriculum, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            code = run_cli(["validate", "--training-export", str(export), "--out", str(summary_path)])

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in summary["targets"] for error in target["errors"])
            self.assertIn("priority_score", errors)
            self.assertIn("priority_band", errors)
            self.assertIn("example_evidence_refs", errors)

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
            metrics["trace_signal"]["average_event_count"] = 999.0
            metrics["trainer_view_source_fingerprint_coverage"]["fully_verified"] = 0
            metrics_path.write_text(json.dumps(metrics), encoding="utf-8")

            code = run_cli(["validate", "--training-export", str(export), "--out", str(summary_path)])

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in summary["targets"] for error in target["errors"])
            self.assertIn("dataset_metrics.pass_rate", errors)
            self.assertIn("dataset_metrics.trace_signal.average_event_count", errors)
            self.assertIn("dataset_metrics.trainer_view_source_fingerprint_coverage.fully_verified", errors)
            self.assertIn("manifest.artifact_fingerprints.dataset_metrics.sha256", errors)

    def test_validate_rejects_dataset_split_row_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp) / "runs"
            export = Path(tmp) / "training"
            summary_path = Path(tmp) / "validation.json"
            run_cli(["run", "--scenario", str(ROOT / "scenarios" / "prompt_injection_good.json"), "--out", str(runs / "prompt_injection_good")])
            run_cli(["run", "--scenario", str(ROOT / "scenarios" / "prompt_injection_bad.json"), "--out", str(runs / "prompt_injection_bad")])
            run_cli(["export-rl", "--runs", str(runs), "--out", str(export)])
            train_path = export / "splits" / "train" / "episodes.jsonl"
            validation_path = export / "splits" / "validation" / "episodes.jsonl"
            train_rows = [json.loads(line) for line in train_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            moved = train_rows.pop()
            write_jsonl(train_path, train_rows)
            write_jsonl(validation_path, [moved])

            code = run_cli(["validate", "--training-export", str(export), "--out", str(summary_path)])

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in summary["targets"] for error in target["errors"])
            self.assertIn("splits/validation/episodes.jsonl row 1 belongs in 'train'", errors)
            self.assertIn("dataset_splits.split_counts.validation.episode_count", errors)
            self.assertIn("manifest.artifact_fingerprints.train_episodes.sha256", errors)

    def test_validate_rejects_dataset_split_scenario_leakage_metadata_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp) / "runs"
            export = Path(tmp) / "training"
            summary_path = Path(tmp) / "validation.json"
            run_cli(["run-suite", "--scenarios", str(ROOT / "scenarios"), "--out", str(runs), "--export-rl"])
            export = runs / "training_export"
            splits_path = export / "dataset_splits.json"
            splits = json.loads(splits_path.read_text(encoding="utf-8"))
            splits["leakage_checks"]["heldout_scenario_exclusive"] = False
            splits["leakage_checks"]["cross_split_scenario_ids"] = ["forged_scenario"]
            splits_path.write_text(json.dumps(splits, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            code = run_cli(["validate", "--training-export", str(export), "--out", str(summary_path)])

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in summary["targets"] for error in target["errors"])
            self.assertIn("dataset_splits.leakage_checks.heldout_scenario_exclusive", errors)
            self.assertIn("dataset_splits.leakage_checks.cross_split_scenario_ids", errors)

    def test_validate_rejects_dataset_registry_manifest_hash_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp) / "runs"
            export = Path(tmp) / "training"
            summary_path = Path(tmp) / "validation.json"
            run_cli(["run", "--scenario", str(ROOT / "scenarios" / "prompt_injection_good.json"), "--out", str(runs / "prompt_injection_good")])
            run_cli(["export-rl", "--runs", str(runs), "--out", str(export)])
            manifest_path = export / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["notes"].append("tampered after registry emission")
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            code = run_cli(["validate", "--training-export", str(export), "--out", str(summary_path)])

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in summary["targets"] for error in target["errors"])
            self.assertIn("dataset_registry.manifest_sha256 must match manifest.json contents", errors)

    def test_validate_rejects_training_manifest_artifact_fingerprint_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp) / "runs"
            export = Path(tmp) / "training"
            summary_path = Path(tmp) / "validation.json"
            run_cli(["run", "--scenario", str(ROOT / "scenarios" / "prompt_injection_good.json"), "--out", str(runs / "prompt_injection_good")])
            run_cli(["export-rl", "--runs", str(runs), "--out", str(export)])
            manifest_path = export / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["artifact_fingerprints"]["episodes"]["sha256"] = "0" * 64
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            code = run_cli(["validate", "--training-export", str(export), "--out", str(summary_path)])

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in summary["targets"] for error in target["errors"])
            self.assertIn("manifest.artifact_fingerprints.episodes.sha256", errors)

    def test_validate_rejects_symlinked_training_export_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = root / "runs"
            export = root / "training"
            summary_path = root / "validation.json"
            run_cli(["run", "--scenario", str(ROOT / "scenarios" / "prompt_injection_good.json"), "--out", str(runs / "prompt_injection_good")])
            run_cli(["export-rl", "--runs", str(runs), "--out", str(export)])
            episodes_path = export / "episodes.jsonl"
            external_path = root / "external_episodes.jsonl"
            external_path.write_text(episodes_path.read_text(encoding="utf-8"), encoding="utf-8")
            episodes_path.unlink()
            try:
                episodes_path.symlink_to(external_path)
            except (NotImplementedError, OSError) as exc:
                self.skipTest(f"symlink creation unavailable: {exc}")

            code = run_cli(["validate", "--training-export", str(export), "--out", str(summary_path)])

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in summary["targets"] for error in target["errors"])
            self.assertIn("manifest.artifact_fingerprints.episodes file must not be a symlink", errors)

    def test_validate_warns_on_legacy_training_export_without_trainer_views(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp) / "runs"
            export = Path(tmp) / "training"
            summary_path = Path(tmp) / "validation.json"
            run_cli(["run", "--scenario", str(ROOT / "scenarios" / "prompt_injection_good.json"), "--out", str(runs / "prompt_injection_good")])
            run_cli(["export-rl", "--runs", str(runs), "--out", str(export)])
            for name in ("sft", "dpo", "reward_model"):
                (export / f"{name}.jsonl").unlink()
            for path in (export / "splits").glob("*/*.jsonl"):
                path.unlink()
            (export / "dataset_splits.json").unlink()
            (export / "dataset_metrics.json").unlink()
            (export / "dataset_registry.json").unlink()
            (export / "DATASET_CARD.md").unlink()
            manifest_path = export / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for name in ("sft", "dpo", "reward_model", "dataset_metrics", "dataset_splits", "dataset_registry", "dataset_card"):
                manifest["outputs"].pop(name, None)
            for split_name in ("train", "validation", "test"):
                for artifact_name in ("episodes", "rewards", "step_rewards", "preferences", "failure_modes", "sft", "dpo", "reward_model"):
                    manifest["outputs"].pop(f"{split_name}_{artifact_name}", None)
            for name in ("sft_count", "dpo_count", "reward_model_count", "quality_flag_count"):
                manifest.pop(name, None)
            for name in ("dataset_version", "redaction_status", "label_provenance", "registry"):
                manifest.pop(name, None)
            manifest.pop("dataset_splits", None)
            manifest.pop("artifact_fingerprints", None)
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
            self.assertIn("dataset_splits.json is missing", warnings)
            self.assertIn("manifest.quality_flag_count is missing", warnings)
            self.assertIn("manifest.artifact_fingerprints is missing", warnings)

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

    def test_validate_accepts_suite_trend(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp) / "runs"
            trend_path = Path(tmp) / "suite_trend.json"
            validation_path = Path(tmp) / "validation.json"
            run_cli(["run-suite", "--scenarios", str(ROOT / "scenarios"), "--out", str(runs)])
            run_cli(
                [
                    "trend-suite",
                    "--suite-summary",
                    str(runs / "suite_summary.json"),
                    "--suite-summary",
                    str(runs / "suite_summary.json"),
                    "--out",
                    str(trend_path),
                ]
            )

            code = run_cli(["validate", "--suite-trend", str(trend_path), "--out", str(validation_path), "--strict"])

            self.assertEqual(code, 0)
            summary = json.loads(validation_path.read_text(encoding="utf-8"))
            self.assertTrue(summary["passed"])
            self.assertEqual(summary["target_count"], 1)
            self.assertEqual(summary["targets"][0]["type"], "suite_trend")
            self.assertEqual(summary["targets"][0]["details"]["point_count"], 2)

    def test_validate_rejects_broken_suite_trend_delta(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp) / "runs"
            trend_path = Path(tmp) / "suite_trend.json"
            validation_path = Path(tmp) / "validation.json"
            run_cli(["run-suite", "--scenarios", str(ROOT / "scenarios"), "--out", str(runs)])
            run_cli(
                [
                    "trend-suite",
                    "--suite-summary",
                    str(runs / "suite_summary.json"),
                    "--suite-summary",
                    str(runs / "suite_summary.json"),
                    "--out",
                    str(trend_path),
                ]
            )
            trend = json.loads(trend_path.read_text(encoding="utf-8"))
            trend["points"][1]["delta_from_previous"]["average_score_delta"] = 99.0
            trend_path.write_text(json.dumps(trend), encoding="utf-8")

            code = run_cli(["validate", "--suite-trend", str(trend_path), "--out", str(validation_path)])

            self.assertEqual(code, 1)
            summary = json.loads(validation_path.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in summary["targets"] for error in target["errors"])
            self.assertIn("suite_trend.points[1].delta_from_previous.average_score_delta", errors)

    def test_validate_strict_fails_on_warnings(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            run_cli(["run", "--scenario", str(ROOT / "scenarios" / "prompt_injection_good.json"), "--out", str(run_dir)])
            (run_dir / "report.html").unlink()
            (run_dir / "artifact_lineage.json").unlink()

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
