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
            self.assertEqual(queue["item_count"], 10)
            self.assertEqual(queue["metrics"]["critical_item_count"], 10)
            self.assertEqual(queue["metrics"]["scenario_count"], 4)
            rule_counts = {row["id"]: row["count"] for row in queue["metrics"]["rule_counts"]}
            self.assertEqual(rule_counts["required_evidence"], 2)
            item = next(row for row in queue["items"] if row["rule_id"] == "forbidden_actions")
            self.assertEqual(item["schema_version"], "hfr.repair_item.v1")
            self.assertEqual(item["priority"], "critical")
            self.assertTrue(item["critical"])
            self.assertIn("regression_scenario", item["source_artifacts"])
            self.assertIn("python -m flightrecorder run", item["replay"]["command"])
            self.assertTrue(item["evidence_refs"])
            self.assertTrue(item["evidence_snippets"])
            event_snippet = next(snippet for snippet in item["evidence_snippets"] if snippet["target"] == "event")
            self.assertEqual(event_snippet["event_index"], 3)
            self.assertEqual(event_snippet["tool_name"], "terminal")
            self.assertIn("evil.example", event_snippet["text"])
            self.assertLessEqual(len(event_snippet["text"]), 600)

            self.assertEqual(run_cli(["validate", "--repair-queue", str(queue_path), "--strict"]), 0)

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
            self.assertEqual(bundle["metrics"]["repair_queue"]["item_count"], 10)
            self.assertEqual(bundle["decision"]["key_metrics"]["repair_queue"]["critical_item_count"], 10)

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


if __name__ == "__main__":
    unittest.main()
