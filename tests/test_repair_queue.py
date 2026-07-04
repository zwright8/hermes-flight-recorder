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


class RepairQueueTests(unittest.TestCase):
    def test_repair_queue_exports_failed_rule_tasks(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp) / "runs"
            queue_path = Path(tmp) / "repair_queue.json"
            self.assertEqual(run_cli(["run-suite", "--scenarios", str(ROOT / "scenarios"), "--out", str(runs)]), 0)

            code = run_cli(["repair-queue", "--runs", str(runs), "--out", str(queue_path)])

            self.assertEqual(code, 0)
            queue = json.loads(queue_path.read_text(encoding="utf-8"))
            self.assertEqual(queue["schema_version"], "hfr.repair_queue.v1")
            self.assertTrue(queue["passed"])
            self.assertEqual(queue["item_count"], 14)
            self.assertEqual(queue["metrics"]["critical_item_count"], 14)
            self.assertEqual(queue["metrics"]["scenario_count"], 5)
            rule_counts = {row["id"]: row["count"] for row in queue["metrics"]["rule_counts"]}
            self.assertEqual(rule_counts["required_evidence"], 3)
            self.assertEqual(rule_counts["required_action_sequences"], 2)
            self.assertEqual(rule_counts["required_event_counts"], 2)
            item = next(row for row in queue["items"] if row["rule_id"] == "forbidden_actions")
            self.assertEqual(item["schema_version"], "hfr.repair_item.v1")
            self.assertEqual(item["priority"], "critical")
            self.assertTrue(item["critical"])
            self.assertIn("regression_scenario", item["source_artifacts"])
            self.assertEqual(len(item["source_artifact_fingerprints"]["scorecard"]["sha256"]), 64)
            self.assertIsInstance(item["source_artifact_fingerprints"]["scorecard"]["size_bytes"], int)
            self.assertIn("python -m flightrecorder run", item["replay"]["command"])
            self.assertTrue(item["evidence_refs"])
            self.assertTrue(item["evidence_snippets"])
            event_snippet = next(snippet for snippet in item["evidence_snippets"] if snippet["target"] == "event")
            self.assertEqual(event_snippet["event_index"], 3)
            self.assertEqual(event_snippet["tool_name"], "terminal")
            self.assertIn("evil.example", event_snippet["text"])
            self.assertLessEqual(len(event_snippet["text"]), 600)

            self.assertEqual(run_cli(["validate", "--repair-queue", str(queue_path), "--strict"]), 0)
            schema = check_schema_file(queue_path)
            self.assertTrue(schema["passed"], schema["errors"])
            self.assertEqual(schema["schema"]["name"], "repair_queue")

    def test_run_suite_evidence_handoff_writes_repair_queue(self):
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
                    "--validate",
                    "--strict",
                    "--evidence-handoff",
                ]
            )

            self.assertEqual(code, 0)
            summary = json.loads((runs / "suite_summary.json").read_text(encoding="utf-8"))
            validation = json.loads((runs / "validation.json").read_text(encoding="utf-8"))
            bundle = json.loads((runs / "evidence_bundle.json").read_text(encoding="utf-8"))
            self.assertTrue((runs / "repair_queue.json").exists())
            self.assertIn("repair_queue", summary["artifacts"])
            self.assertIn("repair_queue", {target["type"] for target in validation["targets"]})
            self.assertEqual(bundle["metrics"]["repair_queue"]["item_count"], 14)
            self.assertEqual(bundle["decision"]["key_metrics"]["repair_queue"]["critical_item_count"], 14)

    def test_validate_rejects_stale_repair_queue_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp) / "runs"
            queue_path = Path(tmp) / "repair_queue.json"
            summary_path = Path(tmp) / "validation.json"
            self.assertEqual(run_cli(["run-suite", "--scenarios", str(ROOT / "scenarios"), "--out", str(runs)]), 0)
            self.assertEqual(run_cli(["repair-queue", "--runs", str(runs), "--out", str(queue_path)]), 0)
            queue = json.loads(queue_path.read_text(encoding="utf-8"))
            queue["metrics"]["item_count"] = 0
            queue["items"][0]["evidence_snippets"][0]["text"] = "x" * 601
            queue_path.write_text(json.dumps(queue, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            code = run_cli(["validate", "--repair-queue", str(queue_path), "--out", str(summary_path)])

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in summary["targets"] for error in target["errors"])
            self.assertIn("repair_queue.metrics.item_count", errors)
            self.assertIn("evidence_snippets", errors)

    def test_validate_rejects_stale_or_moved_repair_queue_source_fingerprints(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = root / "runs"
            queue_path = root / "repair_queue.json"
            summary_path = root / "validation.json"
            self.assertEqual(run_cli(["run-suite", "--scenarios", str(ROOT / "scenarios"), "--out", str(runs)]), 0)
            self.assertEqual(run_cli(["repair-queue", "--runs", str(runs), "--out", str(queue_path)]), 0)
            self.assertEqual(run_cli(["validate", "--repair-queue", str(queue_path), "--strict"]), 0)
            queue = json.loads(queue_path.read_text(encoding="utf-8"))
            scorecard_ref = queue["items"][0]["source_artifact_fingerprints"]["scorecard"]
            scorecard_path = queue_path.parent / scorecard_ref["path"]
            scorecard = json.loads(scorecard_path.read_text(encoding="utf-8"))
            scorecard["stale_after_queue_write"] = True
            scorecard_path.write_text(json.dumps(scorecard, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            code = run_cli(["validate", "--repair-queue", str(queue_path), "--out", str(summary_path)])

            self.assertEqual(code, 1)
            errors = "\n".join(error for target in json.loads(summary_path.read_text(encoding="utf-8"))["targets"] for error in target["errors"])
            self.assertIn("repair_queue.items[0].source_artifact_fingerprints.scorecard.size_bytes does not match the current file.", errors)
            self.assertIn("repair_queue.items[0].source_artifact_fingerprints.scorecard.sha256 does not match the current file.", errors)

            copied_queue = root / "copy" / "repair_queue.json"
            copied_queue.parent.mkdir()
            copied_queue.write_text(queue_path.read_text(encoding="utf-8"), encoding="utf-8")
            code = run_cli(["validate", "--repair-queue", str(copied_queue), "--out", str(summary_path)])

            self.assertEqual(code, 1)
            errors = "\n".join(error for target in json.loads(summary_path.read_text(encoding="utf-8"))["targets"] for error in target["errors"])
            self.assertIn("repair_queue.items[0].source_artifact_fingerprints.scorecard.path does not resolve to an existing file.", errors)

    def test_repair_queue_nested_output_refs_validate_from_queue_location(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = root / "runs"
            queue_path = root / "nested" / "repair_queue.json"
            self.assertEqual(run_cli(["run-suite", "--scenarios", str(ROOT / "scenarios"), "--out", str(runs)]), 0)
            self.assertEqual(run_cli(["repair-queue", "--runs", str(runs), "--out", str(queue_path)]), 0)

            queue = json.loads(queue_path.read_text(encoding="utf-8"))
            scorecard_ref = queue["items"][0]["source_artifact_fingerprints"]["scorecard"]
            self.assertTrue(scorecard_ref["path"].startswith("../runs/"), scorecard_ref)
            self.assertEqual(run_cli(["validate", "--repair-queue", str(queue_path), "--strict"]), 0)

    def test_repair_queue_preserve_paths_warns_on_absolute_sources_and_keeps_fingerprints_queue_relative(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = root / "runs"
            queue_path = root / "nested" / "repair_queue.json"
            summary_path = root / "validation.json"
            self.assertEqual(run_cli(["run-suite", "--scenarios", str(ROOT / "scenarios"), "--out", str(runs)]), 0)
            self.assertEqual(run_cli(["repair-queue", "--runs", str(runs), "--out", str(queue_path), "--preserve-paths"]), 0)

            queue = json.loads(queue_path.read_text(encoding="utf-8"))
            item = queue["items"][0]
            self.assertTrue(Path(item["source_artifacts"]["scorecard"]).is_absolute())
            self.assertFalse(Path(item["source_artifact_fingerprints"]["scorecard"]["path"]).is_absolute())
            self.assertEqual(run_cli(["validate", "--repair-queue", str(queue_path), "--out", str(summary_path)]), 0)
            self.assertEqual(run_cli(["validate", "--repair-queue", str(queue_path), "--strict", "--out", str(summary_path)]), 1)
            warnings = "\n".join(
                warning for target in json.loads(summary_path.read_text(encoding="utf-8"))["targets"] for warning in target["warnings"]
            )
            self.assertIn("repair_queue.runs_dir is absolute", warnings)
            self.assertIn("repair_queue.items[0].source_artifacts.run_dir is absolute", warnings)
            self.assertIn("repair_queue.items[0].source_artifacts.scorecard is absolute", warnings)

    def test_repair_queue_strict_warns_on_absolute_replay_tokens(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = root / "runs"
            queue_path = root / "repair_queue.json"
            summary_path = root / "validation.json"
            self.assertEqual(run_cli(["run-suite", "--scenarios", str(ROOT / "scenarios"), "--out", str(runs)]), 0)
            self.assertEqual(run_cli(["repair-queue", "--runs", str(runs), "--out", str(queue_path)]), 0)
            queue = json.loads(queue_path.read_text(encoding="utf-8"))
            replay = queue["items"][0]["replay"]
            private_scenario = root / "private_scenario.json"
            replay["command"] = f"python -m flightrecorder run --scenario={private_scenario}"
            replay["argv"] = ["python", "-m", "flightrecorder", "run", "--scenario", str(private_scenario)]
            queue_path.write_text(json.dumps(queue, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            self.assertEqual(run_cli(["validate", "--repair-queue", str(queue_path), "--out", str(summary_path)]), 0)
            self.assertEqual(run_cli(["validate", "--repair-queue", str(queue_path), "--strict", "--out", str(summary_path)]), 1)
            warnings = "\n".join(
                warning for target in json.loads(summary_path.read_text(encoding="utf-8"))["targets"] for warning in target["warnings"]
            )
            self.assertIn("repair_queue.items[0].replay.argv[5] is absolute", warnings)
            self.assertIn("repair_queue.items[0].replay.command[4] contains absolute path", warnings)


if __name__ == "__main__":
    unittest.main()
