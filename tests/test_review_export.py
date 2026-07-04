import json
import hashlib
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

from flightrecorder.cli import main
from flightrecorder.review import _reviewed_dataset_version_id
from flightrecorder.schema_registry import check_schema_contract, check_schema_file


ROOT = Path(__file__).resolve().parents[1]


def run_cli(args):
    with redirect_stdout(StringIO()):
        return main(args)


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_completed_labels(review_dir: Path, labels_path: Path) -> None:
    rows = read_jsonl(review_dir / "label_template.jsonl")
    for row in rows:
        row["human_label"] = row["suggested_human_label"]
        row["reviewer"] = "test-reviewer"
        row["reviewer_confidence"] = "high"
        row["reviewed_at"] = "2026-06-26T00:00:00Z"
        row["notes"] = "Accepted suggested label for fixture coverage."
    labels_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


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
            self.assertEqual(manifest["confidence_options"], ["high", "medium", "low", "unknown"])
            self.assertIn("review_items", manifest["outputs"])
            self.assertIn("label_template", manifest["outputs"])
            self.assertIn("artifact_fingerprints", manifest)
            self.assertIn("review_items", manifest["artifact_fingerprints"])
            self.assertEqual({item["schema_version"] for item in items}, {"hfr.review.item.v1"})
            self.assertEqual({label["schema_version"] for label in labels}, {"hfr.review.label.v1"})
            self.assertTrue(all(len(item["review_item_sha256"]) == 64 for item in items))
            self.assertEqual(
                {label["review_item_sha256"] for label in labels},
                {item["review_item_sha256"] for item in items},
            )
            self.assertTrue(all(str(Path(tmp)) not in json.dumps(item) for item in items))
            self.assertTrue(all("lineage" in item["source_artifacts"] for item in items))
            bad_item = next(item for item in items if item["scenario_id"] == "prompt_injection_bad")
            good_item = next(item for item in items if item["scenario_id"] == "prompt_injection_good")
            self.assertEqual(bad_item["suggested_human_label"], "reject")
            self.assertEqual(good_item["suggested_human_label"], "accept")
            self.assertIn("event", bad_item["evidence_target_counts"])
            self.assertTrue(bad_item["rule_summaries"])
            self.assertEqual(labels[0]["human_label"], None)
            self.assertEqual(labels[0]["reviewer_confidence"], None)
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

    def test_export_review_rejects_symlinked_output_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp) / "runs"
            real_out = Path(tmp) / "redirected_review"
            linked_out = Path(tmp) / "review_output_link"
            run_cli(["run", "--scenario", str(ROOT / "scenarios" / "prompt_injection_bad.json"), "--out", str(runs / "bad")])
            real_out.mkdir()
            try:
                linked_out.symlink_to(real_out, target_is_directory=True)
            except (NotImplementedError, OSError) as exc:
                self.skipTest(f"symlink unavailable: {exc}")

            stderr = StringIO()
            with self.assertRaises(SystemExit) as raised, redirect_stdout(StringIO()), redirect_stderr(stderr):
                main(["export-review", "--runs", str(runs), "--out", str(linked_out)])

            self.assertEqual(raised.exception.code, 2)
            self.assertIn("review export output must resolve to a regular non-symlink directory", stderr.getvalue())

    def test_export_review_rejects_symlinked_output_leaf(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp) / "runs"
            out = Path(tmp) / "review"
            redirected = Path(tmp) / "redirected_review_items.jsonl"
            run_cli(["run", "--scenario", str(ROOT / "scenarios" / "prompt_injection_bad.json"), "--out", str(runs / "bad")])
            out.mkdir()
            redirected.write_text("", encoding="utf-8")
            try:
                (out / "review_items.jsonl").symlink_to(redirected)
            except (NotImplementedError, OSError) as exc:
                self.skipTest(f"symlink unavailable: {exc}")

            stderr = StringIO()
            with self.assertRaises(SystemExit) as raised, redirect_stdout(StringIO()), redirect_stderr(stderr):
                main(["export-review", "--runs", str(runs), "--out", str(out)])

            self.assertEqual(raised.exception.code, 2)
            self.assertIn("review output file must resolve to a regular non-symlink file", stderr.getvalue())

    def test_export_review_rejects_symlinked_runs_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp) / "runs"
            linked_runs = Path(tmp) / "runs_link"
            out = Path(tmp) / "review"
            run_cli(["run", "--scenario", str(ROOT / "scenarios" / "prompt_injection_bad.json"), "--out", str(runs / "bad")])
            try:
                linked_runs.symlink_to(runs, target_is_directory=True)
            except (NotImplementedError, OSError) as exc:
                self.skipTest(f"symlink unavailable: {exc}")

            stderr = StringIO()
            with self.assertRaises(SystemExit) as raised, redirect_stdout(StringIO()), redirect_stderr(stderr):
                main(["export-review", "--runs", str(linked_runs), "--out", str(out)])

            self.assertEqual(raised.exception.code, 2)
            self.assertIn("Runs directory must resolve to a regular non-symlink directory", stderr.getvalue())

    def test_export_review_rejects_symlinked_child_run_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp) / "runs"
            out = Path(tmp) / "review"
            run_cli(["run", "--scenario", str(ROOT / "scenarios" / "prompt_injection_bad.json"), "--out", str(runs / "bad")])
            try:
                (runs / "linked_bad").symlink_to(runs / "bad", target_is_directory=True)
            except (NotImplementedError, OSError) as exc:
                self.skipTest(f"symlink unavailable: {exc}")

            stderr = StringIO()
            with self.assertRaises(SystemExit) as raised, redirect_stdout(StringIO()), redirect_stderr(stderr):
                main(["export-review", "--runs", str(runs), "--out", str(out)])

            self.assertEqual(raised.exception.code, 2)
            self.assertIn("Run directory linked_bad must resolve to a regular non-symlink directory", stderr.getvalue())

    def test_export_review_rejects_symlinked_run_source_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp) / "runs"
            out = Path(tmp) / "review"
            run_dir = runs / "bad"
            run_cli(["run", "--scenario", str(ROOT / "scenarios" / "prompt_injection_bad.json"), "--out", str(run_dir)])
            trace_path = run_dir / "normalized_trace.json"
            redirected_trace = Path(tmp) / "redirected_trace.json"
            redirected_trace.write_text(trace_path.read_text(encoding="utf-8"), encoding="utf-8")
            trace_path.unlink()
            try:
                trace_path.symlink_to(redirected_trace)
            except (NotImplementedError, OSError) as exc:
                self.skipTest(f"symlink unavailable: {exc}")

            stderr = StringIO()
            with self.assertRaises(SystemExit) as raised, redirect_stdout(StringIO()), redirect_stderr(stderr):
                main(["export-review", "--runs", str(runs), "--out", str(out)])

            self.assertEqual(raised.exception.code, 2)
            self.assertIn("normalized_trace.json must resolve to a regular non-symlink file", stderr.getvalue())

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

    def test_validate_review_export_rejects_stale_label_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp) / "runs"
            out = Path(tmp) / "review"
            summary_path = Path(tmp) / "validation.json"
            run_cli(["run", "--scenario", str(ROOT / "scenarios" / "prompt_injection_bad.json"), "--out", str(runs / "bad")])
            run_cli(["export-review", "--runs", str(runs), "--out", str(out)])
            item_path = out / "review_items.jsonl"
            item = read_jsonl(item_path)[0]
            item["final_answer"] = "tampered after label export"
            item_path.write_text(json.dumps(item, sort_keys=True) + "\n", encoding="utf-8")

            code = run_cli(["validate", "--review-export", str(out), "--out", str(summary_path)])

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in summary["targets"] for error in target["errors"])
            self.assertIn("review_item_sha256 does not match review item contents", errors)
            self.assertIn("review_item_sha256 does not match referenced review item", errors)

    def test_validate_review_export_rejects_missing_confidence_options(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp) / "runs"
            out = Path(tmp) / "review"
            summary_path = Path(tmp) / "validation.json"
            run_cli(["run", "--scenario", str(ROOT / "scenarios" / "prompt_injection_bad.json"), "--out", str(runs / "bad")])
            run_cli(["export-review", "--runs", str(runs), "--out", str(out)])
            manifest_path = out / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest.pop("confidence_options", None)
            manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")

            code = run_cli(["validate", "--review-export", str(out), "--out", str(summary_path)])

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in summary["targets"] for error in target["errors"])
            self.assertIn("manifest.confidence_options must be", errors)

    def test_validate_review_export_rejects_symlinked_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp) / "runs"
            out = Path(tmp) / "review"
            linked = Path(tmp) / "review_link"
            summary_path = Path(tmp) / "validation.json"
            run_cli(["run", "--scenario", str(ROOT / "scenarios" / "prompt_injection_bad.json"), "--out", str(runs / "bad")])
            run_cli(["export-review", "--runs", str(runs), "--out", str(out)])
            try:
                linked.symlink_to(out, target_is_directory=True)
            except (NotImplementedError, OSError) as exc:
                self.skipTest(f"symlink unavailable: {exc}")

            code = run_cli(["validate", "--review-export", str(linked), "--out", str(summary_path)])

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in summary["targets"] for error in target["errors"])
            self.assertIn("Review export path must resolve to a regular non-symlink directory", errors)

    def test_validate_review_export_rejects_symlinked_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp) / "runs"
            out = Path(tmp) / "review"
            summary_path = Path(tmp) / "validation.json"
            run_cli(["run", "--scenario", str(ROOT / "scenarios" / "prompt_injection_bad.json"), "--out", str(runs / "bad")])
            run_cli(["export-review", "--runs", str(runs), "--out", str(out)])
            manifest_path = out / "manifest.json"
            manifest_target = Path(tmp) / "manifest_target.json"
            manifest_target.write_text(manifest_path.read_text(encoding="utf-8"), encoding="utf-8")
            manifest_path.unlink()
            try:
                manifest_path.symlink_to(manifest_target)
            except (NotImplementedError, OSError) as exc:
                self.skipTest(f"symlink unavailable: {exc}")

            code = run_cli(["validate", "--review-export", str(out), "--out", str(summary_path)])

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in summary["targets"] for error in target["errors"])
            self.assertIn("manifest.json must resolve to a regular non-symlink file", errors)

    def test_validate_review_export_rejects_missing_artifact_fingerprints(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp) / "runs"
            out = Path(tmp) / "review"
            summary_path = Path(tmp) / "validation.json"
            run_cli(["run", "--scenario", str(ROOT / "scenarios" / "prompt_injection_bad.json"), "--out", str(runs / "bad")])
            run_cli(["export-review", "--runs", str(runs), "--out", str(out)])
            manifest_path = out / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest.pop("artifact_fingerprints", None)
            manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")

            code = run_cli(["validate", "--review-export", str(out), "--out", str(summary_path)])

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in summary["targets"] for error in target["errors"])
            self.assertIn("manifest.artifact_fingerprints is missing", errors)

    def test_validate_review_export_rejects_symlinked_review_items(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp) / "runs"
            out = Path(tmp) / "review"
            summary_path = Path(tmp) / "validation.json"
            run_cli(["run", "--scenario", str(ROOT / "scenarios" / "prompt_injection_bad.json"), "--out", str(runs / "bad")])
            run_cli(["export-review", "--runs", str(runs), "--out", str(out)])
            item_path = out / "review_items.jsonl"
            item_target = Path(tmp) / "review_items_target.jsonl"
            item_target.write_text(item_path.read_text(encoding="utf-8"), encoding="utf-8")
            item_path.unlink()
            try:
                item_path.symlink_to(item_target)
            except (NotImplementedError, OSError) as exc:
                self.skipTest(f"symlink unavailable: {exc}")

            code = run_cli(["validate", "--review-export", str(out), "--out", str(summary_path)])

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in summary["targets"] for error in target["errors"])
            self.assertIn("review_items.jsonl must resolve to a regular non-symlink file", errors)

    def test_validate_review_export_rejects_symlinked_label_template(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp) / "runs"
            out = Path(tmp) / "review"
            summary_path = Path(tmp) / "validation.json"
            run_cli(["run", "--scenario", str(ROOT / "scenarios" / "prompt_injection_bad.json"), "--out", str(runs / "bad")])
            run_cli(["export-review", "--runs", str(runs), "--out", str(out)])
            label_path = out / "label_template.jsonl"
            label_target = Path(tmp) / "label_template_target.jsonl"
            label_target.write_text(label_path.read_text(encoding="utf-8"), encoding="utf-8")
            label_path.unlink()
            try:
                label_path.symlink_to(label_target)
            except (NotImplementedError, OSError) as exc:
                self.skipTest(f"symlink unavailable: {exc}")

            code = run_cli(["validate", "--review-export", str(out), "--out", str(summary_path)])

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in summary["targets"] for error in target["errors"])
            self.assertIn("label_template.jsonl must resolve to a regular non-symlink file", errors)

    def test_apply_review_writes_reviewed_training_views(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp) / "runs"
            review = Path(tmp) / "review"
            labels_path = Path(tmp) / "completed_labels.jsonl"
            out = Path(tmp) / "reviewed"
            run_cli(["run", "--scenario", str(ROOT / "scenarios" / "prompt_injection_good.json"), "--out", str(runs / "prompt_injection_good")])
            run_cli(["run", "--scenario", str(ROOT / "scenarios" / "prompt_injection_bad.json"), "--out", str(runs / "prompt_injection_bad")])
            run_cli(["export-review", "--runs", str(runs), "--out", str(review)])
            write_completed_labels(review, labels_path)

            code = run_cli(["apply-review", "--review-export", str(review), "--labels", str(labels_path), "--out", str(out)])

            self.assertEqual(code, 0)
            manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
            reviewed_labels = read_jsonl(out / "reviewed_labels.jsonl")
            sft = read_jsonl(out / "reviewed_sft.jsonl")
            reward_model = read_jsonl(out / "reviewed_reward_model.jsonl")
            preferences = read_jsonl(out / "reviewed_preferences.jsonl")
            dpo = read_jsonl(out / "reviewed_dpo.jsonl")
            dataset_registry = json.loads((out / "dataset_registry.json").read_text(encoding="utf-8"))
            dataset_registry_schema = check_schema_file(out / "dataset_registry.json")
            self.assertTrue(dataset_registry_schema["passed"], dataset_registry_schema["errors"])
            self.assertEqual(manifest["schema_version"], "hfr.reviewed.manifest.v1")
            self.assertRegex(manifest["dataset_version"], r"^hfrds-[0-9a-f]+$")
            self.assertEqual(dataset_registry["dataset_version"], manifest["dataset_version"])
            self.assertEqual(dataset_registry["manifest_sha256"], hashlib.sha256((out / "manifest.json").read_bytes()).hexdigest())
            self.assertTrue(manifest["redaction_status"]["passed"])
            self.assertEqual(manifest["label_provenance"], dataset_registry["label_provenance"])
            self.assertIn("dataset_registry", manifest["outputs"])
            trainer_views = manifest["trainer_views"]
            self.assertEqual(trainer_views["contract_version"], "hfr.rl.trainer_views.v1")
            self.assertEqual(trainer_views, dataset_registry["trainer_views"])
            self.assertEqual(manifest["registry"]["mode_to_view"], trainer_views["mode_to_view"])
            self.assertEqual(dataset_registry["selection"]["mode_to_view"], trainer_views["mode_to_view"])
            self.assertEqual(dataset_registry["selection"]["root_views"], trainer_views["root_views"])
            self.assertEqual(trainer_views["mode_to_view"]["sft"], "reviewed_sft")
            self.assertEqual(trainer_views["mode_to_view"]["action_sft"], "reviewed_sft")
            self.assertEqual(trainer_views["mode_to_view"]["dpo"], "reviewed_dpo")
            self.assertEqual(trainer_views["mode_to_view"]["reward_model"], "reviewed_reward_model")
            views_by_id = {view["view_id"]: view for view in trainer_views["views"]}
            self.assertEqual(views_by_id["reviewed_sft"]["row_count"], 1)
            self.assertEqual(views_by_id["reviewed_dpo"]["row_count"], 1)
            self.assertEqual(views_by_id["reviewed_reward_model"]["row_count"], 2)
            self.assertEqual(manifest["reviewed_label_count"], 2)
            self.assertEqual(manifest["sft_count"], 1)
            self.assertEqual(manifest["reward_model_count"], 2)
            self.assertEqual(manifest["preference_count"], 1)
            self.assertEqual(manifest["dpo_count"], 1)
            self.assertEqual(manifest["confidence_counts"], {"high": 2, "medium": 0, "low": 0, "unknown": 0})
            self.assertEqual(manifest["medium_or_high_confidence_label_count"], 2)
            self.assertEqual({row["schema_version"] for row in reviewed_labels}, {"hfr.reviewed.label.v1"})
            self.assertEqual({row["schema_version"] for row in sft}, {"hfr.reviewed.sft.v1"})
            self.assertEqual({row["schema_version"] for row in reward_model}, {"hfr.reviewed.reward_model.v1"})
            self.assertEqual({row["schema_version"] for row in preferences}, {"hfr.reviewed.preference.v1"})
            self.assertEqual({row["schema_version"] for row in dpo}, {"hfr.reviewed.dpo.v1"})
            self.assertIn("artifact_fingerprints", manifest)
            self.assertIn("reviewed_labels", manifest["artifact_fingerprints"])
            self.assertIn("source_review_artifacts", manifest)
            self.assertIn("labels_artifact", manifest)
            self.assertTrue(all(len(row["review_item_sha256"]) == 64 for row in reviewed_labels))
            self.assertTrue(all(len(row["source_label_sha256"]) == 64 for row in reviewed_labels))
            self.assertEqual({row["reviewer_confidence"] for row in reviewed_labels}, {"high"})
            accepted_label = next(row for row in reviewed_labels if row["human_label"] == "accept")
            self.assertEqual(sft[0]["review_item_sha256"], accepted_label["review_item_sha256"])
            self.assertEqual(sft[0]["reviewer_confidence"], "high")
            self.assertEqual(reward_model[0]["reviewer_confidence"], "high")
            self.assertEqual(sft[0]["episode_id"], "prompt_injection_good")
            self.assertEqual(preferences[0]["chosen_episode_id"], "prompt_injection_good")
            self.assertEqual(preferences[0]["rejected_episode_id"], "prompt_injection_bad")
            self.assertEqual(preferences[0]["chosen_reviewer_confidence"], "high")
            self.assertEqual(preferences[0]["rejected_reviewer_confidence"], "high")
            self.assertIn("chosen_review_item_sha256", preferences[0])
            self.assertIn("rejected_review_item_sha256", dpo[0])
            self.assertEqual(dpo[0]["chosen_reviewer_confidence"], "high")
            self.assertEqual(dpo[0]["chosen"], sft[0]["response"])
            self.assertEqual(run_cli(["validate", "--reviewed-export", str(out), "--strict"]), 0)
            bad_registry = json.loads(json.dumps(dataset_registry))
            bad_registry.pop("labels_artifact")
            bad_schema = check_schema_contract(bad_registry)
            self.assertFalse(bad_schema["passed"])
            self.assertIn("expected exactly one matching schema from oneOf, got 0", "\n".join(bad_schema["errors"]))

    def test_reviewed_dataset_version_ignores_fingerprint_display_paths(self):
        artifacts = {
            "reviewed_labels": {
                "path": "reviewed/reviewed_labels.jsonl",
                "exists": True,
                "regular_file": True,
                "symlink": False,
                "size_bytes": 17,
                "sha256": "a" * 64,
            }
        }
        source = {
            "review_items": {
                "path": "review/review_items.jsonl",
                "exists": True,
                "regular_file": True,
                "symlink": False,
                "size_bytes": 31,
                "sha256": "b" * 64,
            }
        }
        labels = {
            "path": "completed_labels.jsonl",
            "exists": True,
            "regular_file": True,
            "symlink": False,
            "size_bytes": 23,
            "sha256": "c" * 64,
        }
        moved_artifacts = json.loads(json.dumps(artifacts))
        moved_source = json.loads(json.dumps(source))
        moved_labels = json.loads(json.dumps(labels))
        moved_artifacts["reviewed_labels"]["path"] = "moved/reviewed_labels.jsonl"
        moved_source["review_items"]["path"] = "<redacted:review_items.jsonl>"
        moved_labels["path"] = "labels/completed_labels.jsonl"

        self.assertEqual(
            _reviewed_dataset_version_id(artifacts, source, labels),
            _reviewed_dataset_version_id(moved_artifacts, moved_source, moved_labels),
        )

    def test_apply_review_rejects_symlinked_review_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp) / "runs"
            review = Path(tmp) / "review"
            linked = Path(tmp) / "review_link"
            labels_path = Path(tmp) / "completed_labels.jsonl"
            out = Path(tmp) / "reviewed"
            run_cli(["run", "--scenario", str(ROOT / "scenarios" / "prompt_injection_bad.json"), "--out", str(runs / "bad")])
            run_cli(["export-review", "--runs", str(runs), "--out", str(review)])
            write_completed_labels(review, labels_path)
            try:
                linked.symlink_to(review, target_is_directory=True)
            except (NotImplementedError, OSError) as exc:
                self.skipTest(f"symlink unavailable: {exc}")

            stderr = StringIO()
            with self.assertRaises(SystemExit) as raised, redirect_stdout(StringIO()), redirect_stderr(stderr):
                main(["apply-review", "--review-export", str(linked), "--labels", str(labels_path), "--out", str(out)])

            self.assertEqual(raised.exception.code, 2)
            self.assertIn("review export must resolve to a regular non-symlink directory", stderr.getvalue())

    def test_apply_review_rejects_symlinked_output_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp) / "runs"
            review = Path(tmp) / "review"
            labels_path = Path(tmp) / "completed_labels.jsonl"
            real_out = Path(tmp) / "redirected_reviewed"
            linked_out = Path(tmp) / "reviewed_output_link"
            run_cli(["run", "--scenario", str(ROOT / "scenarios" / "prompt_injection_bad.json"), "--out", str(runs / "bad")])
            run_cli(["export-review", "--runs", str(runs), "--out", str(review)])
            write_completed_labels(review, labels_path)
            real_out.mkdir()
            try:
                linked_out.symlink_to(real_out, target_is_directory=True)
            except (NotImplementedError, OSError) as exc:
                self.skipTest(f"symlink unavailable: {exc}")

            stderr = StringIO()
            with self.assertRaises(SystemExit) as raised, redirect_stdout(StringIO()), redirect_stderr(stderr):
                main(["apply-review", "--review-export", str(review), "--labels", str(labels_path), "--out", str(linked_out)])

            self.assertEqual(raised.exception.code, 2)
            self.assertIn("reviewed export output must resolve to a regular non-symlink directory", stderr.getvalue())

    def test_apply_review_rejects_symlinked_output_leaf(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp) / "runs"
            review = Path(tmp) / "review"
            labels_path = Path(tmp) / "completed_labels.jsonl"
            out = Path(tmp) / "reviewed"
            redirected = Path(tmp) / "redirected_reviewed_labels.jsonl"
            run_cli(["run", "--scenario", str(ROOT / "scenarios" / "prompt_injection_bad.json"), "--out", str(runs / "bad")])
            run_cli(["export-review", "--runs", str(runs), "--out", str(review)])
            write_completed_labels(review, labels_path)
            out.mkdir()
            redirected.write_text("", encoding="utf-8")
            try:
                (out / "reviewed_labels.jsonl").symlink_to(redirected)
            except (NotImplementedError, OSError) as exc:
                self.skipTest(f"symlink unavailable: {exc}")

            stderr = StringIO()
            with self.assertRaises(SystemExit) as raised, redirect_stdout(StringIO()), redirect_stderr(stderr):
                main(["apply-review", "--review-export", str(review), "--labels", str(labels_path), "--out", str(out)])

            self.assertEqual(raised.exception.code, 2)
            self.assertIn("review output file must resolve to a regular non-symlink file", stderr.getvalue())

    def test_validate_reviewed_export_rejects_dataset_registry_hash_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp) / "runs"
            review = Path(tmp) / "review"
            labels_path = Path(tmp) / "completed_labels.jsonl"
            out = Path(tmp) / "reviewed"
            summary_path = Path(tmp) / "validation.json"
            run_cli(["run", "--scenario", str(ROOT / "scenarios" / "prompt_injection_good.json"), "--out", str(runs / "prompt_injection_good")])
            run_cli(["export-review", "--runs", str(runs), "--out", str(review)])
            write_completed_labels(review, labels_path)
            run_cli(["apply-review", "--review-export", str(review), "--labels", str(labels_path), "--out", str(out)])
            manifest_path = out / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["notes"].append("tampered after registry emission")
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            code = run_cli(["validate", "--reviewed-export", str(out), "--out", str(summary_path)])

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in summary["targets"] for error in target["errors"])
            self.assertIn("dataset_registry.manifest_sha256 must match manifest.json contents", errors)

    def test_validate_reviewed_export_rejects_symlinked_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp) / "runs"
            review = Path(tmp) / "review"
            labels_path = Path(tmp) / "completed_labels.jsonl"
            out = Path(tmp) / "reviewed"
            linked = Path(tmp) / "reviewed_link"
            summary_path = Path(tmp) / "validation.json"
            run_cli(["run", "--scenario", str(ROOT / "scenarios" / "prompt_injection_good.json"), "--out", str(runs / "prompt_injection_good")])
            run_cli(["export-review", "--runs", str(runs), "--out", str(review)])
            write_completed_labels(review, labels_path)
            run_cli(["apply-review", "--review-export", str(review), "--labels", str(labels_path), "--out", str(out)])
            try:
                linked.symlink_to(out, target_is_directory=True)
            except (NotImplementedError, OSError) as exc:
                self.skipTest(f"symlink unavailable: {exc}")

            code = run_cli(["validate", "--reviewed-export", str(linked), "--out", str(summary_path)])

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in summary["targets"] for error in target["errors"])
            self.assertIn("Reviewed export path must resolve to a regular non-symlink directory", errors)

    def test_validate_reviewed_export_rejects_symlinked_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp) / "runs"
            review = Path(tmp) / "review"
            labels_path = Path(tmp) / "completed_labels.jsonl"
            out = Path(tmp) / "reviewed"
            summary_path = Path(tmp) / "validation.json"
            run_cli(["run", "--scenario", str(ROOT / "scenarios" / "prompt_injection_good.json"), "--out", str(runs / "prompt_injection_good")])
            run_cli(["export-review", "--runs", str(runs), "--out", str(review)])
            write_completed_labels(review, labels_path)
            run_cli(["apply-review", "--review-export", str(review), "--labels", str(labels_path), "--out", str(out)])
            manifest_path = out / "manifest.json"
            manifest_target = Path(tmp) / "reviewed_manifest_target.json"
            manifest_target.write_text(manifest_path.read_text(encoding="utf-8"), encoding="utf-8")
            manifest_path.unlink()
            try:
                manifest_path.symlink_to(manifest_target)
            except (NotImplementedError, OSError) as exc:
                self.skipTest(f"symlink unavailable: {exc}")

            code = run_cli(["validate", "--reviewed-export", str(out), "--out", str(summary_path)])

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in summary["targets"] for error in target["errors"])
            self.assertIn("manifest.json must resolve to a regular non-symlink file", errors)

    def test_validate_reviewed_export_rejects_symlinked_reviewed_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp) / "runs"
            review = Path(tmp) / "review"
            labels_path = Path(tmp) / "completed_labels.jsonl"
            out = Path(tmp) / "reviewed"
            summary_path = Path(tmp) / "validation.json"
            run_cli(["run", "--scenario", str(ROOT / "scenarios" / "prompt_injection_good.json"), "--out", str(runs / "prompt_injection_good")])
            run_cli(["export-review", "--runs", str(runs), "--out", str(review)])
            write_completed_labels(review, labels_path)
            run_cli(["apply-review", "--review-export", str(review), "--labels", str(labels_path), "--out", str(out)])
            reviewed_labels_path = out / "reviewed_labels.jsonl"
            reviewed_labels_target = Path(tmp) / "reviewed_labels_target.jsonl"
            reviewed_labels_target.write_text(reviewed_labels_path.read_text(encoding="utf-8"), encoding="utf-8")
            reviewed_labels_path.unlink()
            try:
                reviewed_labels_path.symlink_to(reviewed_labels_target)
            except (NotImplementedError, OSError) as exc:
                self.skipTest(f"symlink unavailable: {exc}")

            code = run_cli(["validate", "--reviewed-export", str(out), "--out", str(summary_path)])

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in summary["targets"] for error in target["errors"])
            self.assertIn("reviewed_labels.jsonl must resolve to a regular non-symlink file", errors)

    def test_validate_reviewed_export_rejects_symlinked_dataset_registry(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp) / "runs"
            review = Path(tmp) / "review"
            labels_path = Path(tmp) / "completed_labels.jsonl"
            out = Path(tmp) / "reviewed"
            summary_path = Path(tmp) / "validation.json"
            run_cli(["run", "--scenario", str(ROOT / "scenarios" / "prompt_injection_good.json"), "--out", str(runs / "prompt_injection_good")])
            run_cli(["export-review", "--runs", str(runs), "--out", str(review)])
            write_completed_labels(review, labels_path)
            run_cli(["apply-review", "--review-export", str(review), "--labels", str(labels_path), "--out", str(out)])
            registry_path = out / "dataset_registry.json"
            registry_target = Path(tmp) / "dataset_registry_target.json"
            registry_target.write_text(registry_path.read_text(encoding="utf-8"), encoding="utf-8")
            registry_path.unlink()
            try:
                registry_path.symlink_to(registry_target)
            except (NotImplementedError, OSError) as exc:
                self.skipTest(f"symlink unavailable: {exc}")

            code = run_cli(["validate", "--reviewed-export", str(out), "--out", str(summary_path)])

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in summary["targets"] for error in target["errors"])
            self.assertIn("dataset_registry.json must resolve to a regular non-symlink file", errors)

    def test_validate_reviewed_export_rejects_trainer_view_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp) / "runs"
            review = Path(tmp) / "review"
            labels_path = Path(tmp) / "completed_labels.jsonl"
            out = Path(tmp) / "reviewed"
            summary_path = Path(tmp) / "validation.json"
            run_cli(["run", "--scenario", str(ROOT / "scenarios" / "prompt_injection_good.json"), "--out", str(runs / "prompt_injection_good")])
            run_cli(["export-review", "--runs", str(runs), "--out", str(review)])
            write_completed_labels(review, labels_path)
            run_cli(["apply-review", "--review-export", str(review), "--labels", str(labels_path), "--out", str(out)])
            registry_path = out / "dataset_registry.json"
            registry = json.loads(registry_path.read_text(encoding="utf-8"))
            registry["selection"]["mode_to_view"]["dpo"] = "reviewed_sft"
            registry_path.write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            code = run_cli(["validate", "--reviewed-export", str(out), "--out", str(summary_path)])

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in summary["targets"] for error in target["errors"])
            self.assertIn("dataset_registry.selection.mode_to_view must match manifest.trainer_views.mode_to_view", errors)

    def test_apply_review_rejects_invalid_reviewer_confidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp) / "runs"
            review = Path(tmp) / "review"
            labels_path = Path(tmp) / "completed_labels.jsonl"
            out = Path(tmp) / "reviewed"
            run_cli(["run", "--scenario", str(ROOT / "scenarios" / "prompt_injection_bad.json"), "--out", str(runs / "bad")])
            run_cli(["export-review", "--runs", str(runs), "--out", str(review)])
            write_completed_labels(review, labels_path)
            rows = read_jsonl(labels_path)
            rows[0]["reviewer_confidence"] = "maybe"
            labels_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")

            with self.assertRaises(SystemExit) as raised, redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                main(["apply-review", "--review-export", str(review), "--labels", str(labels_path), "--out", str(out)])

            self.assertEqual(raised.exception.code, 2)

    def test_apply_review_requires_reviewer_confidence_for_completed_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp) / "runs"
            review = Path(tmp) / "review"
            labels_path = Path(tmp) / "completed_labels.jsonl"
            out = Path(tmp) / "reviewed"
            run_cli(["run", "--scenario", str(ROOT / "scenarios" / "prompt_injection_bad.json"), "--out", str(runs / "bad")])
            run_cli(["export-review", "--runs", str(runs), "--out", str(review)])
            write_completed_labels(review, labels_path)
            rows = read_jsonl(labels_path)
            rows[0].pop("reviewer_confidence", None)
            labels_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")

            with self.assertRaises(SystemExit) as raised, redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                main(["apply-review", "--review-export", str(review), "--labels", str(labels_path), "--out", str(out)])

            self.assertEqual(raised.exception.code, 2)

    def test_validate_reviewed_export_rejects_stripped_confidence_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp) / "runs"
            review = Path(tmp) / "review"
            labels_path = Path(tmp) / "completed_labels.jsonl"
            out = Path(tmp) / "reviewed"
            summary_path = Path(tmp) / "validation.json"
            run_cli(["run", "--scenario", str(ROOT / "scenarios" / "prompt_injection_good.json"), "--out", str(runs / "good")])
            run_cli(["run", "--scenario", str(ROOT / "scenarios" / "prompt_injection_bad.json"), "--out", str(runs / "bad")])
            run_cli(["export-review", "--runs", str(runs), "--out", str(review)])
            write_completed_labels(review, labels_path)
            run_cli(["apply-review", "--review-export", str(review), "--labels", str(labels_path), "--out", str(out)])

            reviewed_labels_path = out / "reviewed_labels.jsonl"
            reviewed_labels = read_jsonl(reviewed_labels_path)
            reviewed_labels[0].pop("reviewer_confidence", None)
            reviewed_labels_path.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in reviewed_labels),
                encoding="utf-8",
            )
            sft_path = out / "reviewed_sft.jsonl"
            sft = read_jsonl(sft_path)
            sft[0].pop("reviewer_confidence", None)
            sft_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in sft), encoding="utf-8")
            preferences_path = out / "reviewed_preferences.jsonl"
            preferences = read_jsonl(preferences_path)
            preferences[0].pop("chosen_reviewer_confidence", None)
            preferences[0]["chosen"].pop("reviewer_confidence", None)
            preferences_path.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in preferences),
                encoding="utf-8",
            )

            code = run_cli(["validate", "--reviewed-export", str(out), "--out", str(summary_path)])

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in summary["targets"] for error in target["errors"])
            self.assertIn("reviewed_labels[0].reviewer_confidence is required", errors)
            self.assertIn("reviewed_sft[0].reviewer_confidence is required", errors)
            self.assertIn("reviewed_preferences[0].chosen_reviewer_confidence is required", errors)
            self.assertIn("reviewed_preferences[0].chosen.reviewer_confidence is required", errors)

    def test_apply_review_rejects_stale_review_item_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp) / "runs"
            review = Path(tmp) / "review"
            labels_path = Path(tmp) / "completed_labels.jsonl"
            out = Path(tmp) / "reviewed"
            run_cli(["run", "--scenario", str(ROOT / "scenarios" / "prompt_injection_bad.json"), "--out", str(runs / "bad")])
            run_cli(["export-review", "--runs", str(runs), "--out", str(review)])
            write_completed_labels(review, labels_path)
            item_path = review / "review_items.jsonl"
            item = read_jsonl(item_path)[0]
            item["prompt"] = "tampered prompt after human label export"
            item_path.write_text(json.dumps(item, sort_keys=True) + "\n", encoding="utf-8")

            with self.assertRaises(SystemExit) as raised, redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                main(["apply-review", "--review-export", str(review), "--labels", str(labels_path), "--out", str(out)])

            self.assertEqual(raised.exception.code, 2)

    def test_apply_review_rejects_unlabeled_templates(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp) / "runs"
            review = Path(tmp) / "review"
            out = Path(tmp) / "reviewed"
            run_cli(["run", "--scenario", str(ROOT / "scenarios" / "prompt_injection_good.json"), "--out", str(runs / "good")])
            run_cli(["export-review", "--runs", str(runs), "--out", str(review)])

            with self.assertRaises(SystemExit) as raised, redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                main(["apply-review", "--review-export", str(review), "--out", str(out)])

            self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
