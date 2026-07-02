import json
import hashlib
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

from flightrecorder.cli import main
from flightrecorder.training import (
    DATASET_SPLIT_ARTIFACTS,
    DATASET_SPLIT_NAMES,
    TrainingExportError,
    export_rl_dataset,
    redaction_scan_artifacts,
)


ROOT = Path(__file__).resolve().parents[1]


def run_cli(args):
    with redirect_stdout(StringIO()):
        return main(args)


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def split_artifact_keys():
    return {f"{split}_{artifact}" for split in DATASET_SPLIT_NAMES for artifact in DATASET_SPLIT_ARTIFACTS}


class TrainingExportTests(unittest.TestCase):
    def test_export_rl_writes_episode_reward_preference_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp) / "runs"
            good = runs / "prompt_injection_good"
            bad = runs / "prompt_injection_bad"
            out = Path(tmp) / "training"
            run_cli(["run", "--scenario", str(ROOT / "scenarios" / "prompt_injection_good.json"), "--out", str(good)])
            run_cli(["run", "--scenario", str(ROOT / "scenarios" / "prompt_injection_bad.json"), "--out", str(bad)])

            code = run_cli(["export-rl", "--runs", str(runs), "--out", str(out)])

            self.assertEqual(code, 0)
            manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
            episodes = read_jsonl(out / "episodes.jsonl")
            rewards = read_jsonl(out / "rewards.jsonl")
            step_rewards = read_jsonl(out / "step_rewards.jsonl")
            preferences = read_jsonl(out / "preferences.jsonl")
            failure_modes = read_jsonl(out / "failure_modes.jsonl")
            sft = read_jsonl(out / "sft.jsonl")
            dpo = read_jsonl(out / "dpo.jsonl")
            reward_model = read_jsonl(out / "reward_model.jsonl")
            curriculum = json.loads((out / "curriculum.json").read_text(encoding="utf-8"))
            dataset_metrics = json.loads((out / "dataset_metrics.json").read_text(encoding="utf-8"))
            dataset_splits = json.loads((out / "dataset_splits.json").read_text(encoding="utf-8"))
            dataset_registry = json.loads((out / "dataset_registry.json").read_text(encoding="utf-8"))
            dataset_card = (out / "DATASET_CARD.md").read_text(encoding="utf-8")

            self.assertEqual(manifest["schema_version"], "hfr.rl.manifest.v1")
            self.assertRegex(manifest["dataset_version"], r"^hfrds-[0-9a-f]+$")
            self.assertEqual(manifest["dataset_version"], dataset_registry["dataset_version"])
            self.assertEqual(manifest["registry"]["selection_key"], manifest["dataset_version"])
            self.assertTrue(manifest["redaction_status"]["passed"])
            self.assertTrue(dataset_metrics["redaction_status"]["passed"])
            self.assertEqual(manifest["label_provenance"], dataset_metrics["label_provenance"])
            trainer_views = manifest["trainer_views"]
            views_by_id = {view["view_id"]: view for view in trainer_views["views"]}
            self.assertEqual(trainer_views["contract_version"], "hfr.rl.trainer_views.v1")
            self.assertEqual(trainer_views, dataset_metrics["trainer_views"])
            self.assertEqual(trainer_views, dataset_registry["trainer_views"])
            self.assertEqual(dataset_registry["selection"]["mode_to_view"], trainer_views["mode_to_view"])
            self.assertEqual(dataset_registry["selection"]["root_views"], trainer_views["root_views"])
            self.assertEqual(trainer_views["mode_to_view"]["sft"], "sft")
            self.assertEqual(trainer_views["mode_to_view"]["action_sft"], "action_sft")
            self.assertEqual(trainer_views["mode_to_view"]["dpo"], "dpo")
            self.assertEqual(trainer_views["mode_to_view"]["reward_model"], "reward_model")
            self.assertEqual(trainer_views["mode_to_view"]["step_reward"], "step_reward")
            self.assertEqual(trainer_views["mode_to_view"]["process_reward"], "process_reward")
            self.assertEqual(trainer_views["mode_to_view"]["curriculum"], "curriculum")
            self.assertEqual(views_by_id["action_sft"]["artifact_path"], "sft.jsonl")
            self.assertEqual(views_by_id["action_sft"]["row_count"], len(sft))
            self.assertEqual(views_by_id["process_reward"]["artifact_path"], "step_rewards.jsonl")
            self.assertEqual(views_by_id["process_reward"]["row_count"], len(step_rewards))
            self.assertIn("train", views_by_id["process_reward"]["split_paths"])
            self.assertEqual(views_by_id["curriculum"]["artifact_path"], "curriculum.json")
            self.assertEqual(views_by_id["curriculum"]["row_count"], len(failure_modes))
            self.assertEqual(dataset_registry["manifest_sha256"], hashlib.sha256((out / "manifest.json").read_bytes()).hexdigest())
            self.assertIn("dataset_registry", manifest["outputs"])
            self.assertEqual(manifest["episode_count"], 2)
            self.assertEqual(manifest["reward_count"], 2)
            self.assertEqual(manifest["step_reward_count"], len(step_rewards))
            self.assertEqual(manifest["preference_count"], 1)
            self.assertEqual(manifest["failure_mode_count"], len(failure_modes))
            self.assertEqual(manifest["sft_count"], len(sft))
            self.assertEqual(manifest["dpo_count"], len(dpo))
            self.assertEqual(manifest["reward_model_count"], len(reward_model))
            self.assertEqual(manifest["quality_flag_count"], len(dataset_metrics["quality_flags"]))
            self.assertIn("step_rewards", manifest["outputs"])
            self.assertIn("failure_modes", manifest["outputs"])
            self.assertIn("curriculum", manifest["outputs"])
            self.assertIn("sft", manifest["outputs"])
            self.assertIn("dpo", manifest["outputs"])
            self.assertIn("reward_model", manifest["outputs"])
            self.assertIn("dataset_metrics", manifest["outputs"])
            self.assertIn("dataset_splits", manifest["outputs"])
            self.assertIn("dataset_card", manifest["outputs"])
            self.assertEqual(
                set(manifest["artifact_fingerprints"]),
                {
                    "curriculum",
                    "dataset_card",
                    "dataset_metrics",
                    "dataset_splits",
                    "dpo",
                    "episodes",
                    "failure_modes",
                    "preferences",
                    "reward_model",
                    "rewards",
                    "sft",
                    "step_rewards",
                }
                | split_artifact_keys(),
            )
            self.assertTrue(all(record["exists"] is True for record in manifest["artifact_fingerprints"].values()))
            self.assertTrue(all(len(record["sha256"]) == 64 for record in manifest["artifact_fingerprints"].values()))
            self.assertNotIn(str(Path(tmp)), json.dumps(manifest["artifact_fingerprints"]))
            self.assertNotIn(str(Path(tmp)), (out / "manifest.json").read_text(encoding="utf-8"))
            self.assertTrue(all(str(Path(tmp)) not in json.dumps(episode) for episode in episodes))
            self.assertEqual({episode["schema_version"] for episode in episodes}, {"hfr.rl.episode.v1"})
            self.assertTrue(all("artifact_lineage.json" in episode["source_lineage"] for episode in episodes))
            self.assertTrue(all(episode["trace_signal"]["event_count"] == len(episode["events"]) for episode in episodes))
            self.assertTrue(all(episode["trace_signal"]["has_final_answer"] for episode in episodes))
            self.assertEqual({episode["source_fingerprint_status"] for episode in episodes}, {"verified"})
            self.assertTrue(all(len(episode["source_fingerprints"]["scenario"]["sha256"]) == 64 for episode in episodes))
            self.assertTrue(all(len(episode["source_fingerprints"]["source_trace"]["sha256"]) == 64 for episode in episodes))
            self.assertEqual({episode["task_completion"]["schema_version"] for episode in episodes}, {"hfr.task_completion.v1"})
            self.assertEqual({episode["task_completion"]["status"] for episode in episodes}, {"complete", "incomplete"})
            self.assertTrue(all(episode["outcome"]["task_completion_status"] == episode["task_completion"]["status"] for episode in episodes))
            completed_episode = next(episode for episode in episodes if episode["task_completion"]["status"] == "complete")
            self.assertFalse(completed_episode["state_diff"]["available"])
            self.assertFalse(completed_episode["state_diff"]["changed"])
            self.assertEqual(completed_episode["state_diff"]["change_count"], 0)
            self.assertFalse(completed_episode["outcome"]["state_changed"])
            self.assertEqual(completed_episode["outcome"]["state_change_count"], 0)
            self.assertEqual({reward["schema_version"] for reward in rewards}, {"hfr.rl.reward.v1"})
            self.assertTrue(all(reward["source_fingerprints"] for reward in rewards))
            self.assertEqual({reward["task_completion_status"] for reward in rewards}, {"complete", "incomplete"})
            completed_reward = next(reward for reward in rewards if reward["task_completion_status"] == "complete")
            self.assertFalse(completed_reward["state_changed"])
            self.assertEqual(completed_reward["state_change_count"], 0)
            self.assertEqual({step_reward["schema_version"] for step_reward in step_rewards}, {"hfr.rl.step_reward.v1"})
            self.assertEqual(preferences[0]["schema_version"], "hfr.rl.preference.v1")
            self.assertEqual({failure["schema_version"] for failure in failure_modes}, {"hfr.rl.failure_mode.v1"})
            self.assertEqual({sample["schema_version"] for sample in sft}, {"hfr.rl.sft.v1"})
            self.assertEqual({pair["schema_version"] for pair in dpo}, {"hfr.rl.dpo.v1"})
            self.assertEqual({sample["schema_version"] for sample in reward_model}, {"hfr.rl.reward_model.v1"})
            self.assertEqual(curriculum["schema_version"], "hfr.rl.curriculum.v1")
            self.assertEqual(dataset_metrics["schema_version"], "hfr.rl.dataset_metrics.v1")
            self.assertEqual(dataset_splits["schema_version"], "hfr.rl.dataset_splits.v1")
            self.assertEqual(curriculum["failure_mode_count"], len(failure_modes))
            self.assertEqual(dataset_metrics["artifact_counts"]["episodes"], 2)
            self.assertEqual(dataset_metrics["artifact_counts"]["dpo"], 1)
            self.assertEqual(dataset_metrics["source_fingerprint_coverage"]["fully_verified"], 2)
            self.assertEqual(dataset_metrics["source_fingerprint_coverage"]["unverified"], 0)
            self.assertEqual(dataset_metrics["trainer_view_source_fingerprint_coverage"]["rows"], 4)
            self.assertEqual(dataset_metrics["trainer_view_source_fingerprint_coverage"]["fully_verified"], 4)
            self.assertEqual(dataset_metrics["trainer_view_source_fingerprint_coverage"]["unverified"], 0)
            self.assertEqual(dataset_metrics["trainer_view_source_fingerprint_coverage"]["fully_verified_rate"], 1.0)
            self.assertEqual(dataset_metrics["task_completion"]["episode_count"], 2)
            self.assertEqual(dataset_metrics["task_completion"]["configured_count"], 2)
            self.assertEqual(dataset_metrics["task_completion"]["complete_count"], 1)
            self.assertEqual(dataset_metrics["task_completion"]["incomplete_count"], 1)
            self.assertEqual(dataset_metrics["trace_signal"]["episode_count"], 2)
            self.assertEqual(dataset_metrics["trace_signal"]["average_event_count"], 5.0)
            self.assertEqual(dataset_metrics["trace_signal"]["event_type_count"], 4)
            self.assertEqual(dataset_metrics["trace_signal"]["final_answer_rate"], 1.0)
            self.assertEqual(dataset_metrics["trace_signal"]["tool_or_api_episode_rate"], 1.0)
            self.assertEqual(dataset_metrics["dataset_splits"], dataset_splits["summary"])
            self.assertEqual(manifest["dataset_splits"], dataset_splits["summary"])
            self.assertEqual(dataset_splits["split_names"], list(DATASET_SPLIT_NAMES))
            self.assertEqual(dataset_splits["artifact_names"], list(DATASET_SPLIT_ARTIFACTS))
            self.assertEqual(dataset_splits["summary"]["episode_count"], 2)
            self.assertEqual(dataset_splits["summary"]["task_family_count"], 1)
            self.assertEqual(dataset_splits["summary"]["train_episode_count"], 2)
            self.assertEqual(dataset_splits["summary"]["validation_episode_count"], 0)
            self.assertEqual(dataset_splits["summary"]["test_episode_count"], 0)
            self.assertTrue(dataset_splits["summary"]["family_exclusive"])
            self.assertTrue(dataset_splits["summary"]["heldout_scenario_exclusive"])
            self.assertTrue(dataset_splits["leakage_checks"]["family_exclusive"])
            self.assertTrue(dataset_splits["leakage_checks"]["heldout_scenario_exclusive"])
            self.assertEqual(dataset_splits["leakage_checks"]["cross_split_task_families"], [])
            self.assertEqual(dataset_splits["leakage_checks"]["cross_split_scenario_ids"], [])
            self.assertEqual(dataset_registry["leakage_checks"], dataset_splits["leakage_checks"])
            self.assertEqual(dataset_splits["assignments"][0]["task_family"], "prompt_injection")
            self.assertEqual(dataset_splits["assignments"][0]["split"], "train")
            self.assertEqual(dataset_splits["assignments"][0]["episode_ids"], ["prompt_injection_bad", "prompt_injection_good"])
            for artifact_name in DATASET_SPLIT_ARTIFACTS:
                train_rows = read_jsonl(out / "splits" / "train" / f"{artifact_name}.jsonl")
                validation_rows = read_jsonl(out / "splits" / "validation" / f"{artifact_name}.jsonl")
                test_rows = read_jsonl(out / "splits" / "test" / f"{artifact_name}.jsonl")
                self.assertEqual(len(validation_rows), 0)
                self.assertEqual(len(test_rows), 0)
                self.assertEqual(dataset_splits["split_counts"]["train"]["artifacts"][artifact_name], len(train_rows))
            self.assertEqual(dataset_metrics["pass_rate"], 0.5)
            self.assertEqual(dataset_metrics["task_families"][0]["task_family"], "prompt_injection")
            self.assertEqual(dataset_metrics["task_families"][0]["task_completion_complete"], 1)
            self.assertEqual(dataset_metrics["task_families"][0]["task_completion_incomplete"], 1)
            self.assertEqual(dataset_metrics["task_families"][0]["trace_average_event_count"], 5.0)
            self.assertEqual(dataset_metrics["task_families"][0]["trace_tool_or_api_episode_rate"], 1.0)
            self.assertIn("single_task_family", {flag["id"] for flag in dataset_metrics["quality_flags"]})
            self.assertIn("empty_validation_split", {flag["id"] for flag in dataset_metrics["quality_flags"]})
            self.assertIn("empty_test_split", {flag["id"] for flag in dataset_metrics["quality_flags"]})
            self.assertIn("# Flight Recorder Dataset Card", dataset_card)
            self.assertIn("## Source Fingerprints", dataset_card)
            self.assertIn("Fully verified trainer-view rows", dataset_card)
            self.assertIn("## Trace Signal", dataset_card)
            self.assertIn("## Dataset Splits", dataset_card)
            self.assertIn("## Redaction", dataset_card)
            self.assertIn("## Label Provenance", dataset_card)
            self.assertIn("## Trainer Views", dataset_card)
            self.assertIn("`action_sft`", dataset_card)
            self.assertIn("`process_reward`", dataset_card)
            self.assertIn("## Task Families", dataset_card)
            self.assertEqual([sample["episode_id"] for sample in sft], ["prompt_injection_good"])
            self.assertEqual(sft[0]["source_fingerprint_status"], "verified")
            self.assertEqual(sft[0]["task_completion_status"], "complete")
            self.assertEqual(preferences[0]["chosen_episode_id"], "prompt_injection_good")
            self.assertEqual(preferences[0]["rejected_episode_id"], "prompt_injection_bad")
            self.assertEqual(preferences[0]["task_family"], "prompt_injection")
            self.assertEqual(preferences[0]["chosen_score"], 100)
            self.assertEqual(preferences[0]["rejected_score"], 0)
            self.assertEqual(preferences[0]["chosen"]["task_completion"]["status"], "complete")
            self.assertEqual(preferences[0]["rejected"]["task_completion"]["status"], "incomplete")
            self.assertEqual(dpo[0]["preference_id"], preferences[0]["preference_id"])
            self.assertEqual(dpo[0]["chosen"], preferences[0]["chosen"]["final_answer"])
            self.assertEqual(dpo[0]["rejected"], preferences[0]["rejected"]["final_answer"])
            self.assertEqual(dpo[0]["chosen_source_fingerprint_status"], "verified")
            self.assertEqual(dpo[0]["rejected_source_fingerprint_status"], "verified")
            self.assertEqual({sample["episode_id"] for sample in reward_model}, {"prompt_injection_good", "prompt_injection_bad"})
            self.assertEqual({sample["source_fingerprint_status"] for sample in reward_model}, {"verified"})
            self.assertEqual({sample["task_completion_status"] for sample in reward_model}, {"complete", "incomplete"})

    def test_export_rl_blocks_unredacted_secret_like_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp) / "runs"
            good = runs / "prompt_injection_good"
            out = Path(tmp) / "training"
            run_cli(["run", "--scenario", str(ROOT / "scenarios" / "prompt_injection_good.json"), "--out", str(good)])

            with self.assertRaises(TrainingExportError):
                export_rl_dataset(runs, out, metadata={"api_key": "sk-test-unredacted"})

    def test_redaction_scan_detects_standalone_credential_literals(self):
        rows = {
            "sft": [
                {
                    "messages": [
                        {
                            "role": "assistant",
                            "content": "fixture placeholder sk-exampleStandaloneToken000000 for scanner coverage",
                        }
                    ],
                    "metadata": {
                        "github": "ghp_exampleToken000000000000000000000000",
                        "slack": "xoxb-example-token-000000000000",
                        "slack_workflow": "xwfp-example-token-000000000000",
                        "slack_app": "xapp-example-token-000000000000",
                        "slack_rotated": "xoxe-example-token-000000000000",
                    },
                }
            ]
        }

        findings = redaction_scan_artifacts(rows)

        self.assertEqual(len(findings), 6)
        self.assertEqual({finding["artifact"] for finding in findings}, {"sft"})
        self.assertEqual({finding["kind"] for finding in findings}, {"secret_like_literal"})
        self.assertTrue(all(finding["preview"] == "secret-like literal value omitted" for finding in findings))
        self.assertNotIn("sk-exampleStandaloneToken000000", json.dumps(findings))
        self.assertNotIn("ghp_exampleToken000000000000000000000000", json.dumps(findings))
        self.assertNotIn("xoxb-example-token-000000000000", json.dumps(findings))
        self.assertNotIn("xwfp-example-token-000000000000", json.dumps(findings))
        self.assertNotIn("xapp-example-token-000000000000", json.dumps(findings))
        self.assertNotIn("xoxe-example-token-000000000000", json.dumps(findings))

    def test_redaction_scan_allows_hashes_dataset_ids_and_fingerprints(self):
        sha256 = "a" * 64
        rows = {
            "manifest": [
                {
                    "dataset_version": "hfrds-0123456789abcdef",
                    "manifest_sha256": sha256,
                    "artifact_fingerprints": {
                        "manifest": {
                            "path": "manifest.json",
                            "sha256": sha256,
                        }
                    },
                    "notes": [
                        f"source fingerprint {sha256}",
                        "dataset hfrds-0123456789abcdef is selectable",
                        "short examples like xapp-demo and xwfp-short are documentation labels, not credentials",
                    ],
                }
            ]
        }

        self.assertEqual(redaction_scan_artifacts(rows), [])

    def test_export_rl_blocks_standalone_credential_like_trace_value(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp) / "runs"
            good = runs / "prompt_injection_good"
            out = Path(tmp) / "training"
            run_cli(["run", "--scenario", str(ROOT / "scenarios" / "prompt_injection_good.json"), "--out", str(good)])
            trace_path = good / "normalized_trace.json"
            trace = json.loads(trace_path.read_text(encoding="utf-8"))
            trace["final_answer"] = "fixture placeholder sk-exampleStandaloneToken000000"
            trace_path.write_text(json.dumps(trace, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            with self.assertRaises(TrainingExportError):
                export_rl_dataset(runs, out)

    def test_export_rl_excludes_final_answer_only_positive_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp) / "runs"
            good = runs / "prompt_injection_good"
            out = Path(tmp) / "training"
            run_cli(["run", "--scenario", str(ROOT / "scenarios" / "prompt_injection_good.json"), "--out", str(good)])
            scorecard_path = good / "scorecard.json"
            scorecard = json.loads(scorecard_path.read_text(encoding="utf-8"))
            scorecard.pop("task_completion", None)
            scorecard_path.write_text(json.dumps(scorecard, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            export_rl_dataset(runs, out)

            manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
            dataset_metrics = json.loads((out / "dataset_metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(read_jsonl(out / "sft.jsonl"), [])
            self.assertEqual(read_jsonl(out / "reward_model.jsonl"), [])
            self.assertEqual(manifest["label_provenance"]["final_answer_only_excluded_count"], 1)
            self.assertIn(
                "final_answer_only_success_excluded",
                {flag["id"] for flag in dataset_metrics["quality_flags"]},
            )

    def test_export_rl_includes_failed_rule_attribution(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp) / "runs"
            bad = runs / "prompt_injection_bad"
            out = Path(tmp) / "training"
            run_cli(["run", "--scenario", str(ROOT / "scenarios" / "prompt_injection_bad.json"), "--out", str(bad)])

            code = run_cli(["export-rl", "--runs", str(runs), "--out", str(out)])

            self.assertEqual(code, 0)
            reward = read_jsonl(out / "rewards.jsonl")[0]
            step_rewards = read_jsonl(out / "step_rewards.jsonl")
            failure_modes = read_jsonl(out / "failure_modes.jsonl")
            curriculum = json.loads((out / "curriculum.json").read_text(encoding="utf-8"))
            failed_rules = {item["rule_id"] for item in reward["rule_rewards"] if not item["passed"]}
            attribution = reward["attribution"]
            failure_rule_ids = {item["rule_id"] for item in failure_modes}
            forbidden_failure = next(item for item in failure_modes if item["rule_id"] == "forbidden_actions")
            prompt_curriculum = next(family for family in curriculum["task_families"] if family["task_family"] == "prompt_injection")
            forbidden_mode = next(item for item in prompt_curriculum["failure_modes"] if item["rule_id"] == "forbidden_actions")
            curriculum_rule_ids = {
                mode["rule_id"]
                for family in curriculum["task_families"]
                for mode in family["failure_modes"]
            }
            self.assertIn("forbidden_actions", failed_rules)
            self.assertIn("forbidden_actions", failure_rule_ids)
            self.assertIn("forbidden_actions", curriculum_rule_ids)
            self.assertTrue(any(item["critical"] for item in failure_modes))
            self.assertTrue(any(item["target"] == "event" for item in attribution))
            self.assertTrue(any(item["target"] == "final_answer" for item in attribution))
            self.assertTrue(any("evidence_ref" in item for item in attribution))
            self.assertTrue(any(item["target"] == "event" and "event_index" in item for item in step_rewards))
            self.assertTrue(any(item["target"] == "final_answer" for item in step_rewards))
            self.assertTrue(any(item.get("evidence_ref") for item in step_rewards))
            self.assertTrue(all(item["episode_id"] == "prompt_injection_bad" for item in step_rewards))
            self.assertEqual(
                [mode["priority_score"] for mode in prompt_curriculum["failure_modes"]],
                sorted((mode["priority_score"] for mode in prompt_curriculum["failure_modes"]), reverse=True),
            )
            self.assertEqual(forbidden_mode["priority_band"], "high")
            self.assertEqual(forbidden_mode["priority_score"], 145)
            self.assertEqual(forbidden_mode["max_penalty"], 35)
            self.assertEqual(forbidden_mode["average_penalty"], 35.0)
            self.assertEqual(forbidden_mode["scenario_ids"], ["prompt_injection_bad"])
            self.assertEqual(forbidden_mode["failure_ids"], ["prompt_injection_bad:forbidden_actions"])
            self.assertTrue(any(ref["target"] == "event" for ref in forbidden_mode["example_evidence_refs"]))
            for rule_id in failed_rules:
                rule_reward = next(item for item in reward["rule_rewards"] if item["rule_id"] == rule_id)
                step_delta = sum(item["reward_delta"] for item in step_rewards if item["rule_id"] == rule_id)
                self.assertAlmostEqual(step_delta, rule_reward["reward_delta"], places=6)
            self.assertTrue(forbidden_failure["evidence_refs"])
            self.assertTrue(any(ref["target"] == "event" for ref in forbidden_failure["evidence_refs"]))

    def test_export_rl_supports_binary_rewards_and_pair_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp) / "runs"
            good = runs / "prompt_injection_good"
            bad = runs / "prompt_injection_bad"
            out = Path(tmp) / "training"
            run_cli(["run", "--scenario", str(ROOT / "scenarios" / "prompt_injection_good.json"), "--out", str(good)])
            run_cli(["run", "--scenario", str(ROOT / "scenarios" / "prompt_injection_bad.json"), "--out", str(bad)])

            code = run_cli(
                [
                    "export-rl",
                    "--runs",
                    str(runs),
                    "--out",
                    str(out),
                    "--reward-scale",
                    "binary",
                    "--max-pairs-per-family",
                    "1",
                ]
            )

            self.assertEqual(code, 0)
            rewards = {item["episode_id"]: item for item in read_jsonl(out / "rewards.jsonl")}
            preferences = read_jsonl(out / "preferences.jsonl")
            self.assertEqual(rewards["prompt_injection_good"]["reward"], 1.0)
            self.assertEqual(rewards["prompt_injection_bad"]["reward"], 0.0)
            self.assertEqual(len(preferences), 1)

    def test_export_rl_preserves_experiment_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp) / "runs"
            out = Path(tmp) / "training"
            run_cli(["run", "--scenario", str(ROOT / "scenarios" / "prompt_injection_good.json"), "--out", str(runs / "good")])

            code = run_cli(
                [
                    "export-rl",
                    "--runs",
                    str(runs),
                    "--out",
                    str(out),
                    "--metadata",
                    "agent=hermes",
                    "--metadata",
                    "model=fixture-model",
                    "--metadata",
                    "skill_rev=abc123",
                ]
            )

            self.assertEqual(code, 0)
            manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
            dataset_metrics = json.loads((out / "dataset_metrics.json").read_text(encoding="utf-8"))
            dataset_card = (out / "DATASET_CARD.md").read_text(encoding="utf-8")
            expected = {"agent": "hermes", "model": "fixture-model", "skill_rev": "abc123"}
            self.assertEqual(manifest["metadata"], expected)
            self.assertEqual(dataset_metrics["metadata"], expected)
            self.assertIn("## Experiment Metadata", dataset_card)
            self.assertIn("`agent`", dataset_card)
            self.assertEqual(run_cli(["validate", "--training-export", str(out), "--strict"]), 0)

    def test_run_suite_metadata_flows_to_summary_and_training_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp) / "runs"

            code = run_cli(
                [
                    "run-suite",
                    "--scenarios",
                    str(ROOT / "scenarios"),
                    "--out",
                    str(runs),
                    "--export-rl",
                    "--metadata",
                    "candidate=baseline",
                    "--metadata",
                    "policy_rev=demo",
                ]
            )

            self.assertEqual(code, 0)
            summary = json.loads((runs / "suite_summary.json").read_text(encoding="utf-8"))
            manifest = json.loads((runs / "training_export" / "manifest.json").read_text(encoding="utf-8"))
            dataset_metrics = json.loads((runs / "training_export" / "dataset_metrics.json").read_text(encoding="utf-8"))
            episodes = read_jsonl(runs / "training_export" / "episodes.jsonl")
            expected = {"candidate": "baseline", "policy_rev": "demo"}
            self.assertEqual(summary["metadata"], expected)
            self.assertEqual(manifest["metadata"], expected)
            self.assertEqual(dataset_metrics["metadata"], expected)
            completed_email = next(episode for episode in episodes if episode["scenario_id"] == "email_reply_completion_good")
            self.assertEqual(
                len(completed_email["source_fingerprints"]["source_before_state_snapshot"]["sha256"]),
                64,
            )
            self.assertEqual(len(completed_email["source_fingerprints"]["source_state_snapshot"]["sha256"]), 64)
            self.assertTrue(completed_email["state_diff"]["changed"])
            self.assertEqual(completed_email["state_diff"]["change_count"], 2)
            self.assertEqual(run_cli(["validate", "--suite-summary", str(runs / "suite_summary.json"), "--strict"]), 0)

    def test_metadata_requires_key_value_pairs(self):
        with self.assertRaises(SystemExit) as raised, redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            main(["export-rl", "--runs", "runs", "--out", "out", "--metadata", "agent"])

        self.assertEqual(raised.exception.code, 2)

    def test_export_rl_missing_completed_runs_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp) / "runs"
            runs.mkdir()
            with self.assertRaises(SystemExit) as raised, redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                main(["export-rl", "--runs", str(runs), "--out", str(Path(tmp) / "training")])

            self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
