import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from flightrecorder.cli import main
from flightrecorder.digest import build_run_digest, render_run_digest_markdown
from flightrecorder.schema_registry import check_schema_contract
from flightrecorder.state_diff import build_state_diff


ROOT = Path(__file__).resolve().parents[1]


def run_cli(args):
    with redirect_stdout(StringIO()):
        return main(args)


class RunDigestTests(unittest.TestCase):
    def test_run_emits_digest_and_lineage_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"

            code = run_cli(["run", "--scenario", str(ROOT / "scenarios" / "prompt_injection_good.json"), "--out", str(run_dir)])

            self.assertEqual(code, 0)
            digest = json.loads((run_dir / "run_digest.json").read_text(encoding="utf-8"))
            lineage = json.loads((run_dir / "artifact_lineage.json").read_text(encoding="utf-8"))
            self.assertEqual(digest["schema_version"], "hfr.run_digest.v1")
            self.assertEqual(digest["scenario"]["id"], "prompt_injection_good")
            self.assertTrue(digest["outcome"]["passed"])
            self.assertEqual(digest["training_signals"]["binary_reward"], 1)
            self.assertIn("run_digest", {item["name"] for item in lineage["outputs"]})
            self.assertIn(
                {
                    "from": ["scenario", "normalized_trace", "scorecard", "task_completion"],
                    "to": "run_digest",
                    "operation": "summarize_run_evidence",
                },
                lineage["graph"],
            )

    def test_digest_command_regenerates_json_and_markdown(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            digest_out = Path(tmp) / "digest.json"
            markdown_out = Path(tmp) / "digest.md"
            run_cli(["run", "--scenario", str(ROOT / "scenarios" / "prompt_injection_good.json"), "--out", str(run_dir)])

            code = run_cli(["digest", "--run", str(run_dir), "--out", str(digest_out), "--markdown-out", str(markdown_out)])

            self.assertEqual(code, 0)
            digest = json.loads(digest_out.read_text(encoding="utf-8"))
            markdown = markdown_out.read_text(encoding="utf-8")
            self.assertEqual(digest["outcome"]["task_completion_status"], "complete")
            self.assertIn("# Prompt Injection In Tool Output - Passing Run", markdown)
            self.assertIn("## Recommended Actions", markdown)

    def test_failed_digest_contains_repair_signals(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"

            run_cli(["run", "--scenario", str(ROOT / "scenarios" / "prompt_injection_bad.json"), "--out", str(run_dir)])

            digest = json.loads((run_dir / "run_digest.json").read_text(encoding="utf-8"))
            failed_rule_ids = {rule["id"] for rule in digest["rules"]["failed"]}
            action_ids = {action["id"] for action in digest["recommended_actions"]}
            self.assertFalse(digest["outcome"]["passed"])
            self.assertEqual(digest["training_signals"]["binary_reward"], 0)
            self.assertIn("forbidden_actions", failed_rule_ids)
            self.assertIn("repair_failed_rules", action_ids)
            self.assertIn("block_promotion", action_ids)

    def test_digest_carries_state_change_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"

            run_cli(["run", "--scenario", str(ROOT / "scenarios" / "email_reply_completion_good.json"), "--out", str(run_dir)])

            digest = json.loads((run_dir / "run_digest.json").read_text(encoding="utf-8"))
            self.assertTrue(digest["state_changes"]["available"])
            self.assertTrue(digest["state_changes"]["changed"])
            self.assertEqual(digest["state_changes"]["change_count"], 2)
            self.assertEqual(digest["training_signals"]["state_change_count"], 2)
            self.assertEqual(digest["training_signals"]["task_completion_status"], "complete")
            self.assertIn("stateful_success_reward", {action["id"] for action in digest["recommended_actions"]})

    def test_digest_preserves_unknown_state_comparison(self):
        state_diff = build_state_diff(
            {"outer": {"items": [{"value": "before"}] * 20}},
            {"outer": {"items": [{"value": "after"}] * 20}},
            max_depth=1,
        )
        digest = build_run_digest(
            {"id": "uncertain_state"},
            {"events": []},
            {
                "passed": False,
                "score": 0,
                "pass_threshold": 90,
                "summary": "State comparison is incomplete.",
                "rules": [],
            },
            state_diff=state_diff,
        )

        self.assertFalse(digest["state_changes"]["comparison_complete"])
        self.assertEqual(digest["state_changes"]["change_status"], "unknown")
        self.assertTrue(check_schema_contract(digest)["passed"])
        markdown = render_run_digest_markdown(digest)
        self.assertIn("- Status: `UNKNOWN`", markdown)
        self.assertIn("- Changed: `unknown`", markdown)
        self.assertNotIn("- Changed: `false`", markdown)

    def test_digest_without_state_diff_is_unknown_not_unchanged(self):
        digest = build_run_digest(
            {"id": "missing_state"},
            {"events": []},
            {
                "passed": False,
                "score": 0,
                "pass_threshold": 90,
                "summary": "No state evidence.",
                "rules": [],
            },
        )

        self.assertFalse(digest["state_changes"]["available"])
        self.assertFalse(digest["state_changes"]["comparison_complete"])
        self.assertEqual(digest["state_changes"]["change_status"], "unknown")

    def test_validate_rejects_stale_digest(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            validation_path = Path(tmp) / "validation.json"
            run_cli(["run", "--scenario", str(ROOT / "scenarios" / "prompt_injection_good.json"), "--out", str(run_dir)])
            digest_path = run_dir / "run_digest.json"
            digest = json.loads(digest_path.read_text(encoding="utf-8"))
            digest["outcome"]["score"] = 0
            digest_path.write_text(json.dumps(digest, sort_keys=True) + "\n", encoding="utf-8")

            code = run_cli(["validate", "--run", str(run_dir), "--out", str(validation_path)])

            self.assertEqual(code, 1)
            summary = json.loads(validation_path.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in summary["targets"] for error in target["errors"])
            self.assertIn("run_digest.outcome.score must match scorecard.score", errors)


if __name__ == "__main__":
    unittest.main()
