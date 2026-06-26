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


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class ReviewExportTests(unittest.TestCase):
    def test_export_review_writes_review_queue_and_label_template(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp) / "runs"
            good = runs / "prompt_injection_good"
            bad = runs / "prompt_injection_bad"
            out = Path(tmp) / "review"
            run_cli(["run", "--scenario", str(ROOT / "scenarios" / "prompt_injection_good.json"), "--out", str(good)])
            run_cli(["run", "--scenario", str(ROOT / "scenarios" / "prompt_injection_bad.json"), "--out", str(bad)])

            code = run_cli(["export-review", "--runs", str(runs), "--out", str(out)])

            self.assertEqual(code, 0)
            manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
            items = read_jsonl(out / "review_items.jsonl")
            labels = read_jsonl(out / "label_template.jsonl")
            instructions = (out / "REVIEW_INSTRUCTIONS.md").read_text(encoding="utf-8")
            self.assertEqual(manifest["schema_version"], "hfr.review.manifest.v1")
            self.assertEqual(manifest["item_count"], 2)
            self.assertEqual(manifest["passed_count"], 1)
            self.assertEqual(manifest["failed_count"], 1)
            self.assertIn("review_items", manifest["outputs"])
            self.assertIn("label_template", manifest["outputs"])
            self.assertEqual({item["schema_version"] for item in items}, {"hfr.review.item.v1"})
            self.assertEqual({label["schema_version"] for label in labels}, {"hfr.review.label.v1"})
            self.assertTrue(all(str(Path(tmp)) not in json.dumps(item) for item in items))
            self.assertTrue(all("lineage" in item["source_artifacts"] for item in items))
            bad_item = next(item for item in items if item["scenario_id"] == "prompt_injection_bad")
            good_item = next(item for item in items if item["scenario_id"] == "prompt_injection_good")
            self.assertEqual(bad_item["suggested_human_label"], "reject")
            self.assertEqual(good_item["suggested_human_label"], "accept")
            self.assertIn("event", bad_item["evidence_target_counts"])
            self.assertTrue(bad_item["rule_summaries"])
            self.assertEqual(labels[0]["human_label"], None)
            self.assertIn("Human labels should be grounded", instructions)
            self.assertEqual(run_cli(["validate", "--review-export", str(out), "--strict"]), 0)

    def test_export_review_only_failed_filters_queue(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp) / "runs"
            out = Path(tmp) / "review"
            run_cli(["run", "--scenario", str(ROOT / "scenarios" / "prompt_injection_good.json"), "--out", str(runs / "good")])
            run_cli(["run", "--scenario", str(ROOT / "scenarios" / "prompt_injection_bad.json"), "--out", str(runs / "bad")])

            code = run_cli(["export-review", "--runs", str(runs), "--out", str(out), "--only-failed"])

            self.assertEqual(code, 0)
            manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
            items = read_jsonl(out / "review_items.jsonl")
            self.assertEqual(manifest["item_count"], 1)
            self.assertEqual(manifest["failed_count"], 1)
            self.assertFalse(items[0]["scorecard"]["passed"])

    def test_validate_review_export_rejects_broken_label_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp) / "runs"
            out = Path(tmp) / "review"
            summary_path = Path(tmp) / "validation.json"
            run_cli(["run", "--scenario", str(ROOT / "scenarios" / "prompt_injection_bad.json"), "--out", str(runs / "bad")])
            run_cli(["export-review", "--runs", str(runs), "--out", str(out)])
            label_path = out / "label_template.jsonl"
            label = read_jsonl(label_path)[0]
            label["review_item_id"] = "missing-item"
            label_path.write_text(json.dumps(label) + "\n", encoding="utf-8")

            code = run_cli(["validate", "--review-export", str(out), "--out", str(summary_path)])

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in summary["targets"] for error in target["errors"])
            self.assertIn("does not reference a review item", errors)


if __name__ == "__main__":
    unittest.main()
