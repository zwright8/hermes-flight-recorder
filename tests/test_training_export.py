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
            preferences = read_jsonl(out / "preferences.jsonl")

            self.assertEqual(manifest["schema_version"], "hfr.rl.manifest.v1")
            self.assertEqual(manifest["episode_count"], 2)
            self.assertEqual(manifest["reward_count"], 2)
            self.assertEqual(manifest["preference_count"], 1)
            self.assertNotIn(str(Path(tmp)), (out / "manifest.json").read_text(encoding="utf-8"))
            self.assertTrue(all(str(Path(tmp)) not in json.dumps(episode) for episode in episodes))
            self.assertEqual({episode["schema_version"] for episode in episodes}, {"hfr.rl.episode.v1"})
            self.assertEqual({reward["schema_version"] for reward in rewards}, {"hfr.rl.reward.v1"})
            self.assertEqual(preferences[0]["schema_version"], "hfr.rl.preference.v1")
            self.assertEqual(preferences[0]["chosen_episode_id"], "prompt_injection_good")
            self.assertEqual(preferences[0]["rejected_episode_id"], "prompt_injection_bad")
            self.assertEqual(preferences[0]["task_family"], "prompt_injection")
            self.assertEqual(preferences[0]["chosen_score"], 100)
            self.assertEqual(preferences[0]["rejected_score"], 0)

    def test_export_rl_includes_failed_rule_attribution(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp) / "runs"
            bad = runs / "prompt_injection_bad"
            out = Path(tmp) / "training"
            run_cli(["run", "--scenario", str(ROOT / "scenarios" / "prompt_injection_bad.json"), "--out", str(bad)])

            code = run_cli(["export-rl", "--runs", str(runs), "--out", str(out)])

            self.assertEqual(code, 0)
            reward = read_jsonl(out / "rewards.jsonl")[0]
            failed_rules = {item["rule_id"] for item in reward["rule_rewards"] if not item["passed"]}
            attribution = reward["attribution"]
            self.assertIn("forbidden_actions", failed_rules)
            self.assertTrue(any(item["target"] == "event" for item in attribution))
            self.assertTrue(any(item["target"] == "final_answer" for item in attribution))

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
