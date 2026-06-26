import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

from flightrecorder.cli import main


ROOT = Path(__file__).resolve().parents[1]


def run_cli(args):
    with redirect_stdout(StringIO()):
        return main(args)


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


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
            dataset_card = (out / "DATASET_CARD.md").read_text(encoding="utf-8")

            self.assertEqual(manifest["schema_version"], "hfr.rl.manifest.v1")
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
            self.assertIn("dataset_card", manifest["outputs"])
            self.assertNotIn(str(Path(tmp)), (out / "manifest.json").read_text(encoding="utf-8"))
            self.assertTrue(all(str(Path(tmp)) not in json.dumps(episode) for episode in episodes))
            self.assertEqual({episode["schema_version"] for episode in episodes}, {"hfr.rl.episode.v1"})
            self.assertEqual({reward["schema_version"] for reward in rewards}, {"hfr.rl.reward.v1"})
            self.assertEqual({step_reward["schema_version"] for step_reward in step_rewards}, {"hfr.rl.step_reward.v1"})
            self.assertEqual(preferences[0]["schema_version"], "hfr.rl.preference.v1")
            self.assertEqual({failure["schema_version"] for failure in failure_modes}, {"hfr.rl.failure_mode.v1"})
            self.assertEqual({sample["schema_version"] for sample in sft}, {"hfr.rl.sft.v1"})
            self.assertEqual({pair["schema_version"] for pair in dpo}, {"hfr.rl.dpo.v1"})
            self.assertEqual({sample["schema_version"] for sample in reward_model}, {"hfr.rl.reward_model.v1"})
            self.assertEqual(curriculum["schema_version"], "hfr.rl.curriculum.v1")
            self.assertEqual(dataset_metrics["schema_version"], "hfr.rl.dataset_metrics.v1")
            self.assertEqual(curriculum["failure_mode_count"], len(failure_modes))
            self.assertEqual(dataset_metrics["artifact_counts"]["episodes"], 2)
            self.assertEqual(dataset_metrics["artifact_counts"]["dpo"], 1)
            self.assertEqual(dataset_metrics["pass_rate"], 0.5)
            self.assertEqual(dataset_metrics["task_families"][0]["task_family"], "prompt_injection")
            self.assertIn("single_task_family", {flag["id"] for flag in dataset_metrics["quality_flags"]})
            self.assertIn("# Flight Recorder Dataset Card", dataset_card)
            self.assertIn("## Task Families", dataset_card)
            self.assertEqual([sample["episode_id"] for sample in sft], ["prompt_injection_good"])
            self.assertEqual(preferences[0]["chosen_episode_id"], "prompt_injection_good")
            self.assertEqual(preferences[0]["rejected_episode_id"], "prompt_injection_bad")
            self.assertEqual(preferences[0]["task_family"], "prompt_injection")
            self.assertEqual(preferences[0]["chosen_score"], 100)
            self.assertEqual(preferences[0]["rejected_score"], 0)
            self.assertEqual(dpo[0]["preference_id"], preferences[0]["preference_id"])
            self.assertEqual(dpo[0]["chosen"], preferences[0]["chosen"]["final_answer"])
            self.assertEqual(dpo[0]["rejected"], preferences[0]["rejected"]["final_answer"])
            self.assertEqual({sample["episode_id"] for sample in reward_model}, {"prompt_injection_good", "prompt_injection_bad"})

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

    def test_export_rl_missing_completed_runs_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp) / "runs"
            runs.mkdir()
            with self.assertRaises(SystemExit) as raised, redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                main(["export-rl", "--runs", str(runs), "--out", str(Path(tmp) / "training")])

            self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
