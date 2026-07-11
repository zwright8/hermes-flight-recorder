import errno
import json
import hashlib
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import flightrecorder.review as review_module
import flightrecorder.path_safety as path_safety_module
from flightrecorder.cli import main
from flightrecorder.path_safety import (
    AtomicNamespaceMutationError,
    DirectoryCleanupStatus,
    output_directory_lock_is_held,
)
from flightrecorder.review import (
    ReviewExportError,
    _reviewed_dataset_version_id,
    apply_review_labels,
    review_item_sha256,
)
from flightrecorder.schema_registry import check_schema_contract, check_schema_file
from flightrecorder.training import episode_events_sha256
from flightrecorder.validation import validate_reviewed_export


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


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def refresh_reviewed_export_metadata(reviewed: Path) -> None:
    """Make a tampered trainer view internally self-consistent for adversarial tests."""
    manifest_path = reviewed / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact_names = (
        "reviewed_labels",
        "reviewed_sft",
        "reviewed_reward_model",
        "reviewed_preferences",
        "reviewed_dpo",
    )
    row_counts: dict[str, int] = {}
    for artifact_name in artifact_names:
        artifact_path = reviewed / f"{artifact_name}.jsonl"
        content = artifact_path.read_bytes()
        row_counts[artifact_name] = len(read_jsonl(artifact_path))
        fingerprint = manifest["artifact_fingerprints"][artifact_name]
        fingerprint["sha256"] = hashlib.sha256(content).hexdigest()
        fingerprint["size_bytes"] = len(content)

    for field_name, artifact_name in (
        ("reviewed_label_count", "reviewed_labels"),
        ("sft_count", "reviewed_sft"),
        ("reward_model_count", "reviewed_reward_model"),
        ("preference_count", "reviewed_preferences"),
        ("dpo_count", "reviewed_dpo"),
    ):
        manifest[field_name] = row_counts[artifact_name]
    manifest["label_provenance"]["trainer_view_counts"] = {
        artifact_name: row_counts[artifact_name]
        for artifact_name in (
            "reviewed_sft",
            "reviewed_reward_model",
            "reviewed_preferences",
            "reviewed_dpo",
        )
    }
    for view in manifest["trainer_views"]["views"]:
        view["row_count"] = row_counts[view["artifact"]]

    dataset_version = _reviewed_dataset_version_id(
        manifest["artifact_fingerprints"],
        manifest["source_review_artifacts"],
        manifest["labels_artifact"],
    )
    manifest["dataset_version"] = dataset_version
    manifest["registry"]["selection_key"] = dataset_version
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    registry_path = reviewed / "dataset_registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    registry["dataset_version"] = dataset_version
    registry["selection"]["key"] = dataset_version
    registry["selection"]["trainer_preflight_arg"] = (
        f"--require-dataset-version {dataset_version}"
    )
    registry["trainer_views"] = json.loads(
        json.dumps(manifest["trainer_views"])
    )
    registry["label_provenance"] = json.loads(
        json.dumps(manifest["label_provenance"])
    )
    registry["artifact_fingerprints"] = json.loads(
        json.dumps(manifest["artifact_fingerprints"])
    )
    registry["manifest_sha256"] = hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()
    registry_path.write_text(
        json.dumps(registry, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def make_reviewed_export(tmp: str, *, include_bad: bool = False) -> Path:
    root = Path(tmp)
    runs = root / "runs"
    review = root / "review"
    labels = root / "completed_labels.jsonl"
    reviewed = root / "reviewed"
    run_cli(["run", "--scenario", str(ROOT / "scenarios" / "prompt_injection_good.json"), "--out", str(runs / "good")])
    if include_bad:
        run_cli(
            [
                "run",
                "--scenario",
                str(ROOT / "scenarios" / "prompt_injection_bad.json"),
                "--out",
                str(runs / "bad"),
            ]
        )
    run_cli(["export-review", "--runs", str(runs), "--out", str(review)])
    write_completed_labels(review, labels)
    run_cli(["apply-review", "--review-export", str(review), "--labels", str(labels), "--out", str(reviewed)])
    return reviewed


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
            self.assertTrue(all(len(item["episode_events_sha256"]) == 64 for item in items))
            self.assertTrue(
                all(
                    set(item["source_artifact_fingerprints"])
                    == {"normalized_trace", "scorecard"}
                    for item in items
                )
            )
            self.assertTrue(
                all(
                    fingerprint["algorithm"] == "sha256-canonical-json-v1"
                    and len(fingerprint["sha256"]) == 64
                    and fingerprint["sha256"] == fingerprint["sha256"].lower()
                    and fingerprint["size_bytes"] > 0
                    for item in items
                    for fingerprint in item["source_artifact_fingerprints"].values()
                )
            )
            for item in items:
                run_dir = runs / item["episode_id"]
                trace = json.loads(
                    (run_dir / "normalized_trace.json").read_text(encoding="utf-8")
                )
                scorecard = json.loads(
                    (run_dir / "scorecard.json").read_text(encoding="utf-8")
                )
                self.assertEqual(
                    item["episode_events_sha256"],
                    episode_events_sha256(trace["events"]),
                )
                for artifact_name, source in (
                    ("normalized_trace", trace),
                    ("scorecard", scorecard),
                ):
                    encoded = json.dumps(
                        source,
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=False,
                        allow_nan=False,
                    ).encode("utf-8")
                    fingerprint = item["source_artifact_fingerprints"][artifact_name]
                    self.assertEqual(fingerprint["sha256"], hashlib.sha256(encoded).hexdigest())
                    self.assertEqual(fingerprint["size_bytes"], len(encoded))
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

    def test_preserve_paths_redacts_absolute_review_item_source_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp) / "runs"
            out = Path(tmp) / "review"
            strict_validation = Path(tmp) / "strict_validation.json"
            run_cli(["run", "--scenario", str(ROOT / "scenarios" / "prompt_injection_good.json"), "--out", str(runs / "good")])
            run_cli(["run", "--scenario", str(ROOT / "scenarios" / "prompt_injection_bad.json"), "--out", str(runs / "bad")])

            code = run_cli(["export-review", "--runs", str(runs), "--out", str(out), "--preserve-paths"])

            self.assertEqual(code, 0)
            manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
            items = read_jsonl(out / "review_items.jsonl")
            self.assertTrue(manifest["source_runs_dir"].startswith("<redacted:"))
            self.assertTrue(manifest["output_dir"].startswith("<redacted:"))
            self.assertTrue(items[0]["source_artifacts"]["run_dir"].startswith("<redacted:"))
            self.assertTrue(items[0]["source_artifacts"]["report"].startswith("<redacted:"))
            self.assertFalse(Path(items[0]["source_artifacts"]["run_dir"]).is_absolute())
            self.assertFalse(Path(items[0]["source_artifacts"]["report"]).is_absolute())
            self.assertNotIn(str(Path(tmp)), json.dumps({"manifest": manifest, "items": items}))
            self.assertEqual(run_cli(["validate", "--review-export", str(out), "--strict", "--out", str(strict_validation)]), 0)

    def test_validate_review_export_rejects_absolute_review_item_source_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp) / "runs"
            out = Path(tmp) / "review"
            validation = Path(tmp) / "validation.json"
            run_cli(["run", "--scenario", str(ROOT / "scenarios" / "prompt_injection_good.json"), "--out", str(runs / "good")])
            run_cli(["export-review", "--runs", str(runs), "--out", str(out)])
            item_path = out / "review_items.jsonl"
            items = read_jsonl(item_path)
            items[0]["source_artifacts"]["report"] = str(Path(tmp) / "private-report.html")
            item_path.write_text("".join(json.dumps(item, sort_keys=True) + "\n" for item in items), encoding="utf-8")

            self.assertEqual(run_cli(["validate", "--review-export", str(out), "--out", str(validation)]), 1)
            errors = [
                error
                for target in json.loads(validation.read_text(encoding="utf-8"))["targets"]
                for error in target["errors"]
            ]
            self.assertIn(
                "review_items[0].source_artifacts.report must be a safe relative path or redacted placeholder.",
                errors,
            )

    def test_preserve_paths_redacts_absolute_reviewed_label_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp) / "runs"
            review = Path(tmp) / "review"
            labels_path = Path(tmp) / "completed_labels.jsonl"
            out = Path(tmp) / "reviewed"
            strict_validation = Path(tmp) / "strict_validation.json"
            run_cli(["run", "--scenario", str(ROOT / "scenarios" / "prompt_injection_good.json"), "--out", str(runs / "good")])
            run_cli(["run", "--scenario", str(ROOT / "scenarios" / "prompt_injection_bad.json"), "--out", str(runs / "bad")])
            run_cli(["export-review", "--runs", str(runs), "--out", str(review), "--preserve-paths"])
            write_completed_labels(review, labels_path)

            code = run_cli(
                [
                    "apply-review",
                    "--review-export",
                    str(review),
                    "--labels",
                    str(labels_path),
                    "--out",
                    str(out),
                    "--preserve-paths",
                ]
            )

            self.assertEqual(code, 0)
            manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
            reviewed_labels = read_jsonl(out / "reviewed_labels.jsonl")
            self.assertEqual(manifest["source_review_export"], "provenance")
            self.assertEqual(manifest["labels_path"], "provenance/completed_labels.jsonl")
            self.assertEqual(reviewed_labels[0]["source_label_file"], "provenance/completed_labels.jsonl")
            self.assertTrue(reviewed_labels[0]["source_artifacts"]["run_dir"].startswith("<redacted:"))
            self.assertTrue(reviewed_labels[0]["source_artifacts"]["report"].startswith("<redacted:"))
            self.assertFalse(Path(reviewed_labels[0]["source_label_file"]).is_absolute())
            self.assertFalse(Path(reviewed_labels[0]["source_artifacts"]["run_dir"]).is_absolute())
            self.assertFalse(Path(reviewed_labels[0]["source_artifacts"]["report"]).is_absolute())
            self.assertNotIn(str(Path(tmp)), json.dumps({"manifest": manifest, "reviewed_labels": reviewed_labels}))
            self.assertEqual(run_cli(["validate", "--reviewed-export", str(out), "--strict", "--out", str(strict_validation)]), 0)

    def test_validate_reviewed_export_rejects_absolute_reviewed_label_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp) / "runs"
            review = Path(tmp) / "review"
            labels_path = Path(tmp) / "completed_labels.jsonl"
            out = Path(tmp) / "reviewed"
            validation = Path(tmp) / "validation.json"
            run_cli(["run", "--scenario", str(ROOT / "scenarios" / "prompt_injection_good.json"), "--out", str(runs / "good")])
            run_cli(["export-review", "--runs", str(runs), "--out", str(review)])
            write_completed_labels(review, labels_path)
            run_cli(["apply-review", "--review-export", str(review), "--labels", str(labels_path), "--out", str(out)])
            reviewed_labels_path = out / "reviewed_labels.jsonl"
            rows = read_jsonl(reviewed_labels_path)
            rows[0]["source_label_file"] = str(labels_path)
            rows[0]["source_artifacts"]["report"] = str(Path(tmp) / "private-report.html")
            reviewed_labels_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")

            self.assertEqual(run_cli(["validate", "--reviewed-export", str(out), "--out", str(validation)]), 1)
            errors = [
                error
                for target in json.loads(validation.read_text(encoding="utf-8"))["targets"]
                for error in target["errors"]
            ]
            self.assertIn(
                "reviewed_labels[0].source_label_file must be a safe relative path or redacted placeholder.",
                errors,
            )
            self.assertIn(
                "reviewed_labels[0].source_artifacts.report must be a safe relative path or redacted placeholder.",
                errors,
            )

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

    def test_validate_review_export_requires_behavior_and_source_fingerprints(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp) / "runs"
            out = Path(tmp) / "review"
            summary_path = Path(tmp) / "validation.json"
            run_cli(["run", "--scenario", str(ROOT / "scenarios" / "prompt_injection_good.json"), "--out", str(runs / "good")])
            run_cli(["export-review", "--runs", str(runs), "--out", str(out)])
            item_path = out / "review_items.jsonl"
            item = read_jsonl(item_path)[0]
            item.pop("episode_events_sha256")
            item["source_artifact_fingerprints"].pop("scorecard")
            item_path.write_text(json.dumps(item, sort_keys=True) + "\n", encoding="utf-8")

            code = run_cli(["validate", "--review-export", str(out), "--out", str(summary_path)])

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in summary["targets"] for error in target["errors"])
            self.assertIn("episode_events_sha256 must be a lowercase SHA-256", errors)
            self.assertIn("source_artifact_fingerprints.scorecard must be an object", errors)

    def test_review_item_sha256_ignores_source_artifact_display_paths(self):
        item = {
            "schema_version": "hfr.review.item.v1",
            "review_item_id": "case-1",
            "source_artifacts": {
                "run_dir": "runs/case-1",
                "normalized_trace": "runs/case-1/normalized_trace.json",
                "scorecard": "runs/case-1/scorecard.json",
            },
            "source_artifact_fingerprints": {
                "normalized_trace": {
                    "algorithm": "sha256-canonical-json-v1",
                    "sha256": "1" * 64,
                    "size_bytes": 10,
                },
                "scorecard": {
                    "algorithm": "sha256-canonical-json-v1",
                    "sha256": "2" * 64,
                    "size_bytes": 20,
                },
            },
            "prompt": "Summarize the issue.",
            "final_answer": "Done.",
        }
        moved = json.loads(json.dumps(item))
        moved["source_artifacts"]["run_dir"] = "<redacted:case-1>"
        moved["source_artifacts"]["normalized_trace"] = "elsewhere/normalized_trace.json"
        moved["source_artifacts"]["scorecard"] = "elsewhere/scorecard.json"
        missing_role = json.loads(json.dumps(item))
        missing_role["source_artifacts"].pop("scorecard")
        changed_content = json.loads(json.dumps(item))
        changed_content["source_artifact_fingerprints"]["scorecard"]["sha256"] = "3" * 64

        self.assertEqual(review_item_sha256(item), review_item_sha256(moved))
        self.assertNotEqual(review_item_sha256(item), review_item_sha256(missing_role))
        self.assertNotEqual(review_item_sha256(item), review_item_sha256(changed_content))

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

    def test_apply_review_rerun_replaces_existing_reviewed_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            reviewed = make_reviewed_export(tmp)
            review = Path(tmp) / "review"
            labels_path = Path(tmp) / "completed_labels.jsonl"
            labels = read_jsonl(labels_path)
            labels[0]["notes"] = "Updated on a reviewed-export rerun."
            labels_path.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in labels),
                encoding="utf-8",
            )

            manifest = apply_review_labels(review, reviewed, labels_path=labels_path)

            self.assertEqual(manifest, json.loads((reviewed / "manifest.json").read_text(encoding="utf-8")))
            self.assertEqual(
                read_jsonl(reviewed / "provenance" / "completed_labels.jsonl")[0]["notes"],
                "Updated on a reviewed-export rerun.",
            )
            self.assertFalse(list(Path(tmp).glob(".reviewed.staging-*")))
            self.assertFalse(list(Path(tmp).glob(".reviewed.backup-*")))

    def test_apply_review_rejects_manifest_only_unowned_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_reviewed_export(tmp)
            review = root / "review"
            labels_path = root / "completed_labels.jsonl"
            target = root / "manifest-only"
            target.mkdir()
            personal_manifest = target / "manifest.json"
            original = b"personal notes, not a reviewed-export manifest\n"
            personal_manifest.write_bytes(original)

            with self.assertRaisesRegex(
                ReviewExportError,
                "not a semantically valid prior reviewed export",
            ):
                apply_review_labels(review, target, labels_path=labels_path)

            self.assertEqual(personal_manifest.read_bytes(), original)
            self.assertEqual(list(target.iterdir()), [personal_manifest])

    def test_apply_review_rejects_unowned_replacement_between_check_and_baseline(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reviewed = make_reviewed_export(tmp)
            review = root / "review"
            labels_path = root / "completed_labels.jsonl"
            saved_reviewed = root / "saved-reviewed"
            sentinel = reviewed / "unrelated-private-data.txt"
            actual_attestation = review_module._reviewed_output_attestation
            replaced = False

            def replace_before_baseline(path: Path):
                nonlocal replaced
                if path == reviewed and not replaced:
                    replaced = True
                    reviewed.rename(saved_reviewed)
                    reviewed.mkdir()
                    sentinel.write_text("must survive\n", encoding="utf-8")
                return actual_attestation(path)

            with (
                patch.object(
                    review_module,
                    "_reviewed_output_attestation",
                    side_effect=replace_before_baseline,
                ),
                self.assertRaisesRegex(
                    ReviewExportError,
                    "not a semantically valid prior reviewed export",
                ),
            ):
                apply_review_labels(review, reviewed, labels_path=labels_path)

            self.assertTrue(replaced)
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "must survive\n")
            self.assertTrue((saved_reviewed / "manifest.json").is_file())

    def test_apply_review_rolls_back_new_target_after_post_rename_fsync_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_reviewed_export(tmp)
            review = root / "review"
            labels_path = root / "completed_labels.jsonl"
            target = root / "new-reviewed"
            actual_rename = review_module.atomic_rename_entry_noreplace

            def rename_then_fail(parent_descriptor, source_name, target_name):
                actual_rename(parent_descriptor, source_name, target_name)
                raise AtomicNamespaceMutationError(
                    "injected post-rename fsync failure"
                )

            with (
                patch.object(
                    review_module,
                    "atomic_rename_entry_noreplace",
                    side_effect=rename_then_fail,
                ),
                self.assertRaisesRegex(
                    ReviewExportError,
                    "publication was rolled back",
                ),
            ):
                apply_review_labels(review, target, labels_path=labels_path)

            self.assertFalse(target.exists())
            self.assertFalse(list(root.glob(".new-reviewed.staging-*")))

    def test_apply_review_reports_target_reinserted_after_noreplace_cleanup(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_reviewed_export(tmp)
            review = root / "review"
            labels_path = root / "completed_labels.jsonl"
            target = root / "new-reviewed"
            actual_rename = review_module.atomic_rename_entry_noreplace
            actual_remove = review_module.remove_directory_entry_tree_if_identity
            outcomes = []

            def rename_then_fail(parent_descriptor, source_name, target_name):
                actual_rename(parent_descriptor, source_name, target_name)
                raise AtomicNamespaceMutationError(
                    "injected post-rename fsync failure"
                )

            def remove_then_reinsert(
                parent_descriptor,
                name,
                expected_identity,
                expected_namespace,
            ):
                outcome = actual_remove(
                    parent_descriptor,
                    name,
                    expected_identity,
                    expected_namespace,
                )
                outcomes.append(outcome)
                target.mkdir()
                (target / "concurrent.txt").write_text(
                    "concurrent\n",
                    encoding="utf-8",
                )
                return outcome

            with (
                patch.object(
                    review_module,
                    "atomic_rename_entry_noreplace",
                    side_effect=rename_then_fail,
                ),
                patch.object(
                    review_module,
                    "remove_directory_entry_tree_if_identity",
                    side_effect=remove_then_reinsert,
                ),
                self.assertRaises(ReviewExportError) as raised,
            ):
                apply_review_labels(review, target, labels_path=labels_path)

            self.assertEqual(len(outcomes), 1)
            self.assertEqual(outcomes[0].status, DirectoryCleanupStatus.COMPLETE)
            self.assertIn(str(target), str(raised.exception))
            self.assertIn("concurrent namespace entries", str(raised.exception))
            self.assertNotIn("was retained for recovery", str(raised.exception))
            self.assertEqual(
                (target / "concurrent.txt").read_text(encoding="utf-8"),
                "concurrent\n",
            )

    def test_apply_review_reports_target_reinserted_after_unconfirmed_cleanup(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_reviewed_export(tmp)
            review = root / "review"
            labels_path = root / "completed_labels.jsonl"
            target = root / "new-reviewed"
            actual_rename = review_module.atomic_rename_entry_noreplace
            actual_remove = review_module.remove_directory_entry_tree_if_identity
            outcomes = []

            def rename_then_fail(parent_descriptor, source_name, target_name):
                actual_rename(parent_descriptor, source_name, target_name)
                raise AtomicNamespaceMutationError(
                    "injected post-rename fsync failure"
                )

            def remove_unconfirmed_then_reinsert(
                parent_descriptor,
                name,
                expected_identity,
                expected_namespace,
            ):
                actual_sync = path_safety_module._sync_directory_descriptor
                sync_count = 0

                def fail_first_sync(descriptor):
                    nonlocal sync_count
                    sync_count += 1
                    if sync_count == 1:
                        raise OSError(
                            errno.EIO,
                            "injected post-root-removal sync failure",
                        )
                    return actual_sync(descriptor)

                with patch.object(
                    path_safety_module,
                    "_sync_directory_descriptor",
                    side_effect=fail_first_sync,
                ):
                    outcome = actual_remove(
                        parent_descriptor,
                        name,
                        expected_identity,
                        expected_namespace,
                    )
                outcomes.append(outcome)
                target.mkdir()
                (target / "concurrent.txt").write_text(
                    "concurrent\n",
                    encoding="utf-8",
                )
                return outcome

            with (
                patch.object(
                    review_module,
                    "atomic_rename_entry_noreplace",
                    side_effect=rename_then_fail,
                ),
                patch.object(
                    review_module,
                    "remove_directory_entry_tree_if_identity",
                    side_effect=remove_unconfirmed_then_reinsert,
                ),
                self.assertRaises(ReviewExportError) as raised,
            ):
                apply_review_labels(review, target, labels_path=labels_path)

            self.assertEqual(len(outcomes), 1)
            self.assertEqual(
                outcomes[0].status,
                DirectoryCleanupStatus.COMPLETE_DURABILITY_UNCONFIRMED,
            )
            self.assertIn(str(target), str(raised.exception))
            self.assertIn("concurrent namespace entries", str(raised.exception))
            self.assertNotIn("was retained for recovery", str(raised.exception))
            self.assertEqual(
                (target / "concurrent.txt").read_text(encoding="utf-8"),
                "concurrent\n",
            )

    def test_apply_review_rolls_back_exchange_after_post_fsync_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reviewed = make_reviewed_export(tmp)
            review = root / "review"
            labels_path = root / "completed_labels.jsonl"
            before = {
                path.relative_to(reviewed): path.read_bytes()
                for path in reviewed.rglob("*")
                if path.is_file()
            }
            actual_exchange = review_module.atomic_exchange_entries
            exchange_count = 0

            def exchange_then_fail_once(parent_descriptor, left_name, right_name):
                nonlocal exchange_count
                exchange_count += 1
                actual_exchange(parent_descriptor, left_name, right_name)
                if exchange_count == 1:
                    raise AtomicNamespaceMutationError(
                        "injected post-exchange fsync failure"
                    )

            with (
                patch.object(
                    review_module,
                    "atomic_exchange_entries",
                    side_effect=exchange_then_fail_once,
                ),
                self.assertRaisesRegex(
                    ReviewExportError,
                    "exchange was rolled back",
                ),
            ):
                apply_review_labels(review, reviewed, labels_path=labels_path)

            after = {
                path.relative_to(reviewed): path.read_bytes()
                for path in reviewed.rglob("*")
                if path.is_file()
            }
            self.assertEqual(exchange_count, 2)
            self.assertEqual(after, before)
            self.assertFalse(list(root.glob(".reviewed.staging-*")))

    def test_apply_review_validation_failure_preserves_existing_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            reviewed = make_reviewed_export(tmp)
            review = Path(tmp) / "review"
            labels_path = Path(tmp) / "completed_labels.jsonl"
            before = {
                path.relative_to(reviewed): path.read_bytes()
                for path in reviewed.rglob("*")
                if path.is_file()
            }

            with patch(
                "flightrecorder.review._validate_staged_reviewed_export",
                side_effect=ReviewExportError("forced staged validation failure"),
            ):
                with self.assertRaisesRegex(ReviewExportError, "forced staged validation failure"):
                    apply_review_labels(review, reviewed, labels_path=labels_path)

            after = {
                path.relative_to(reviewed): path.read_bytes()
                for path in reviewed.rglob("*")
                if path.is_file()
            }
            self.assertEqual(after, before)
            self.assertFalse(list(Path(tmp).glob(".reviewed.staging-*")))

    def test_apply_review_rejects_concurrent_existing_export_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            reviewed = make_reviewed_export(tmp)
            review = Path(tmp) / "review"
            labels_path = Path(tmp) / "completed_labels.jsonl"
            manifest_path = reviewed / "manifest.json"
            original = manifest_path.read_bytes()

            def mutate_before_publish(staging: Path) -> None:
                manifest_path.write_bytes(original + b" ")

            with patch(
                "flightrecorder.review._validate_staged_reviewed_export",
                side_effect=mutate_before_publish,
            ):
                with self.assertRaisesRegex(ReviewExportError, "changed before publication"):
                    apply_review_labels(review, reviewed, labels_path=labels_path)

            self.assertEqual(manifest_path.read_bytes(), original + b" ")
            self.assertFalse(list(Path(tmp).glob(".reviewed.staging-*")))

    def test_apply_review_rolls_back_post_check_atomic_swap_race(self):
        with tempfile.TemporaryDirectory() as tmp:
            reviewed = make_reviewed_export(tmp)
            review = Path(tmp) / "review"
            labels_path = Path(tmp) / "completed_labels.jsonl"
            manifest_path = reviewed / "manifest.json"
            competing = manifest_path.read_bytes() + b" "
            actual_exchange = review_module.atomic_exchange_entries
            exchanged = False

            def compete_then_exchange(parent_descriptor, left_name, right_name):
                nonlocal exchanged
                if not exchanged:
                    manifest_path.write_bytes(competing)
                    exchanged = True
                return actual_exchange(parent_descriptor, left_name, right_name)

            with patch(
                "flightrecorder.review.atomic_exchange_entries",
                side_effect=compete_then_exchange,
            ):
                with self.assertRaisesRegex(
                    ReviewExportError,
                    "changed during atomic publication",
                ):
                    apply_review_labels(
                        review,
                        reviewed,
                        labels_path=labels_path,
                    )

            self.assertEqual(manifest_path.read_bytes(), competing)
            self.assertFalse(list(Path(tmp).glob(".reviewed.staging-*")))

    def test_apply_review_rolls_back_concurrent_empty_directory_and_preserves_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            reviewed = make_reviewed_export(tmp)
            review = Path(tmp) / "review"
            labels_path = Path(tmp) / "completed_labels.jsonl"
            concurrent = reviewed / "concurrent-empty-directory"
            actual_exchange = review_module.atomic_exchange_entries
            exchanged = False

            def compete_then_exchange(parent_descriptor, left_name, right_name):
                nonlocal exchanged
                if not exchanged:
                    concurrent.mkdir()
                    exchanged = True
                return actual_exchange(parent_descriptor, left_name, right_name)

            with patch(
                "flightrecorder.review.atomic_exchange_entries",
                side_effect=compete_then_exchange,
            ):
                with self.assertRaisesRegex(
                    ReviewExportError,
                    "changed during atomic publication",
                ):
                    apply_review_labels(
                        review,
                        reviewed,
                        labels_path=labels_path,
                    )

            self.assertTrue(concurrent.is_dir())
            self.assertEqual(list(concurrent.iterdir()), [])
            self.assertFalse(list(Path(tmp).glob(".reviewed.staging-*")))

    def test_apply_review_cleanup_preserves_file_added_after_ownership_approval(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reviewed = make_reviewed_export(tmp)
            review = root / "review"
            labels_path = root / "completed_labels.jsonl"
            actual_remove = review_module.remove_directory_entry_tree_if_identity
            injected: list[tuple[Path, bool]] = []

            def inject_then_remove(
                parent_descriptor,
                name,
                expected_identity,
                expected_namespace,
            ):
                sentinel = root / name / "concurrent-unowned.txt"
                sentinel.write_text("must survive concurrent cleanup\n", encoding="utf-8")
                removed = actual_remove(
                    parent_descriptor,
                    name,
                    expected_identity,
                    expected_namespace,
                )
                injected.append((sentinel, removed))
                return removed

            with (
                patch.object(
                    review_module,
                    "remove_directory_entry_tree_if_identity",
                    side_effect=inject_then_remove,
                ),
                self.assertRaisesRegex(
                    ReviewExportError,
                    "displaced prior version was retained for recovery",
                ),
            ):
                apply_review_labels(review, reviewed, labels_path=labels_path)

            self.assertEqual(len(injected), 1)
            sentinel, removed = injected[0]
            self.assertFalse(removed)
            self.assertEqual(
                sentinel.read_text(encoding="utf-8"),
                "must survive concurrent cleanup\n",
            )
            self.assertTrue(reviewed.is_dir())

    def test_apply_review_cleanup_preserves_replaced_file_after_ownership_approval(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reviewed = make_reviewed_export(tmp)
            review = root / "review"
            labels_path = root / "completed_labels.jsonl"
            saved_manifest = root / "approved-manifest-before-race.json"
            actual_remove = review_module.remove_directory_entry_tree_if_identity
            replacement_paths: list[Path] = []

            def replace_then_remove(
                parent_descriptor,
                name,
                expected_identity,
                expected_namespace,
            ):
                manifest = root / name / "manifest.json"
                manifest.rename(saved_manifest)
                manifest.write_text("unowned replacement\n", encoding="utf-8")
                replacement_paths.append(manifest)
                return actual_remove(
                    parent_descriptor,
                    name,
                    expected_identity,
                    expected_namespace,
                )

            with (
                patch.object(
                    review_module,
                    "remove_directory_entry_tree_if_identity",
                    side_effect=replace_then_remove,
                ),
                self.assertRaisesRegex(
                    ReviewExportError,
                    "displaced prior version was retained for recovery",
                ),
            ):
                apply_review_labels(review, reviewed, labels_path=labels_path)

            self.assertEqual(len(replacement_paths), 1)
            self.assertEqual(
                replacement_paths[0].read_text(encoding="utf-8"),
                "unowned replacement\n",
            )
            self.assertTrue(saved_manifest.is_file())
            self.assertTrue(reviewed.is_dir())

    def test_apply_review_cleanup_preserves_empty_directory_added_after_approval(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reviewed = make_reviewed_export(tmp)
            review = root / "review"
            labels_path = root / "completed_labels.jsonl"
            actual_remove = review_module.remove_directory_entry_tree_if_identity
            injected: list[tuple[Path, bool]] = []

            def inject_then_remove(
                parent_descriptor,
                name,
                expected_identity,
                expected_namespace,
            ):
                directory = root / name / "concurrent-empty-directory"
                directory.mkdir()
                removed = actual_remove(
                    parent_descriptor,
                    name,
                    expected_identity,
                    expected_namespace,
                )
                injected.append((directory, removed))
                return removed

            with (
                patch.object(
                    review_module,
                    "remove_directory_entry_tree_if_identity",
                    side_effect=inject_then_remove,
                ),
                self.assertRaisesRegex(
                    ReviewExportError,
                    "displaced prior version was retained for recovery",
                ),
            ):
                apply_review_labels(review, reviewed, labels_path=labels_path)

            self.assertEqual(len(injected), 1)
            directory, removed = injected[0]
            self.assertFalse(removed)
            self.assertTrue(directory.is_dir())
            self.assertEqual(list(directory.iterdir()), [])
            self.assertTrue(reviewed.is_dir())

    def test_apply_review_reports_partial_cleanup_after_second_unlink_eio(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reviewed = make_reviewed_export(tmp)
            review = root / "review"
            labels_path = root / "completed_labels.jsonl"
            actual_remove = review_module.remove_directory_entry_tree_if_identity
            outcomes = []

            def remove_with_second_unlink_failure(
                parent_descriptor,
                name,
                expected_identity,
                expected_namespace,
            ):
                actual_unlink = path_safety_module.os.unlink
                unlink_count = 0

                def fail_second_private_unlink(entry_name, *, dir_fd=None):
                    nonlocal unlink_count
                    if str(entry_name).startswith(".hfr-remove-"):
                        unlink_count += 1
                        if unlink_count == 2:
                            raise OSError(
                                errno.EIO,
                                "injected second unlink failure",
                            )
                    return actual_unlink(entry_name, dir_fd=dir_fd)

                with patch.object(
                    path_safety_module.os,
                    "unlink",
                    side_effect=fail_second_private_unlink,
                ):
                    outcome = actual_remove(
                        parent_descriptor,
                        name,
                        expected_identity,
                        expected_namespace,
                    )
                outcomes.append(outcome)
                return outcome

            with (
                patch.object(
                    review_module,
                    "remove_directory_entry_tree_if_identity",
                    side_effect=remove_with_second_unlink_failure,
                ),
                self.assertRaises(ReviewExportError) as raised,
            ):
                apply_review_labels(review, reviewed, labels_path=labels_path)

            self.assertEqual(len(outcomes), 1)
            outcome = outcomes[0]
            self.assertEqual(outcome.status, DirectoryCleanupStatus.PARTIAL)
            self.assertFalse(outcome.durability_confirmed)
            self.assertIn("cleanup of the displaced prior version was partial", str(raised.exception))
            self.assertNotIn("was retained for recovery", str(raised.exception))
            self.assertTrue(outcome.recovery_entries)
            for relative_entry in outcome.recovery_entries:
                recovery_path = root / relative_entry
                self.assertIn(str(recovery_path), str(raised.exception))
                self.assertTrue(recovery_path.exists())
            self.assertTrue(reviewed.is_dir())

    def test_apply_review_reports_complete_cleanup_with_unconfirmed_durability(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reviewed = make_reviewed_export(tmp)
            review = root / "review"
            labels_path = root / "completed_labels.jsonl"
            actual_remove = review_module.remove_directory_entry_tree_if_identity
            outcomes = []

            def remove_with_post_root_sync_failure(
                parent_descriptor,
                name,
                expected_identity,
                expected_namespace,
            ):
                actual_sync = path_safety_module._sync_directory_descriptor
                sync_count = 0

                def fail_first_sync(descriptor):
                    nonlocal sync_count
                    sync_count += 1
                    if sync_count == 1:
                        raise OSError(
                            errno.EIO,
                            "injected post-root-removal sync failure",
                        )
                    return actual_sync(descriptor)

                with patch.object(
                    path_safety_module,
                    "_sync_directory_descriptor",
                    side_effect=fail_first_sync,
                ):
                    outcome = actual_remove(
                        parent_descriptor,
                        name,
                        expected_identity,
                        expected_namespace,
                    )
                outcomes.append(outcome)
                return outcome

            with (
                patch.object(
                    review_module,
                    "remove_directory_entry_tree_if_identity",
                    side_effect=remove_with_post_root_sync_failure,
                ),
                self.assertRaises(ReviewExportError) as raised,
            ):
                apply_review_labels(review, reviewed, labels_path=labels_path)

            self.assertEqual(len(outcomes), 1)
            outcome = outcomes[0]
            self.assertEqual(
                outcome.status,
                DirectoryCleanupStatus.COMPLETE_DURABILITY_UNCONFIRMED,
            )
            self.assertFalse(outcome.durability_confirmed)
            self.assertIn("was completely removed", str(raised.exception))
            self.assertIn("durability could not be confirmed", str(raised.exception))
            self.assertNotIn("was retained for recovery", str(raised.exception))
            self.assertEqual(outcome.recovery_entries, ())
            self.assertTrue(reviewed.is_dir())
            self.assertFalse(list(root.glob(".reviewed.staging-*")))

    def test_apply_review_reports_vault_recovery_path_after_public_reinsertion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reviewed = make_reviewed_export(tmp)
            review = root / "review"
            labels_path = root / "completed_labels.jsonl"
            actual_remove = review_module.remove_directory_entry_tree_if_identity
            outcomes = []
            public_reinsertions = []

            def remove_after_public_reinsertion(
                parent_descriptor,
                name,
                expected_identity,
                expected_namespace,
            ):
                actual_unlink = path_safety_module.os.unlink
                unlink_count = 0
                public_path = root / name

                def reinsert_then_fail(entry_name, *, dir_fd=None):
                    nonlocal unlink_count
                    if str(entry_name).startswith(".hfr-remove-"):
                        unlink_count += 1
                        if unlink_count == 2:
                            public_path.mkdir()
                            (public_path / "concurrent.txt").write_text(
                                "concurrent\n",
                                encoding="utf-8",
                            )
                            public_reinsertions.append(public_path)
                            raise OSError(
                                errno.EIO,
                                "injected second unlink failure",
                            )
                    return actual_unlink(entry_name, dir_fd=dir_fd)

                with patch.object(
                    path_safety_module.os,
                    "unlink",
                    side_effect=reinsert_then_fail,
                ):
                    outcome = actual_remove(
                        parent_descriptor,
                        name,
                        expected_identity,
                        expected_namespace,
                    )
                outcomes.append(outcome)
                return outcome

            with (
                patch.object(
                    review_module,
                    "remove_directory_entry_tree_if_identity",
                    side_effect=remove_after_public_reinsertion,
                ),
                self.assertRaises(ReviewExportError) as raised,
            ):
                apply_review_labels(review, reviewed, labels_path=labels_path)

            self.assertEqual(len(outcomes), 1)
            outcome = outcomes[0]
            self.assertEqual(outcome.status, DirectoryCleanupStatus.PARTIAL)
            self.assertFalse(outcome.durability_confirmed)
            self.assertEqual(len(outcome.cleanup_artifacts), 1)
            self.assertTrue(outcome.recovery_entries)
            self.assertEqual(len(outcome.concurrent_entries), 1)
            self.assertNotIn("was retained for recovery", str(raised.exception))
            for relative_entry in outcome.recovery_entries:
                recovery_path = root / relative_entry
                self.assertIn(str(recovery_path), str(raised.exception))
                self.assertTrue(recovery_path.exists())
            for relative_entry in outcome.concurrent_entries:
                concurrent_path = root / relative_entry
                self.assertIn(str(concurrent_path), str(raised.exception))
                self.assertTrue(concurrent_path.exists())
            self.assertEqual(len(public_reinsertions), 1)
            self.assertEqual(
                (public_reinsertions[0] / "concurrent.txt").read_text(
                    encoding="utf-8"
                ),
                "concurrent\n",
            )

    def test_apply_review_holds_canonical_lock_through_staged_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            reviewed = make_reviewed_export(tmp)
            review = Path(tmp) / "review"
            labels_path = Path(tmp) / "completed_labels.jsonl"
            actual_validate = review_module._validate_staged_reviewed_export

            def assert_locked(staging):
                self.assertTrue(output_directory_lock_is_held(reviewed))
                actual_validate(staging)

            with patch(
                "flightrecorder.review._validate_staged_reviewed_export",
                side_effect=assert_locked,
            ):
                apply_review_labels(review, reviewed, labels_path=labels_path)

    def test_apply_review_rejects_corrupt_source_review_manifest_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            reviewed = make_reviewed_export(tmp)
            review = Path(tmp) / "review"
            labels_path = Path(tmp) / "completed_labels.jsonl"
            manifest_path = review / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["item_count"] = 999
            manifest["artifact_fingerprints"] = {}
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            existing = (reviewed / "manifest.json").read_bytes()

            with self.assertRaisesRegex(
                ReviewExportError,
                "staged reviewed export failed validation",
            ):
                apply_review_labels(review, reviewed, labels_path=labels_path)

            self.assertEqual((reviewed / "manifest.json").read_bytes(), existing)

    def test_apply_review_rejects_unrelated_existing_output_entries(self):
        with tempfile.TemporaryDirectory() as tmp:
            reviewed = make_reviewed_export(tmp)
            review = Path(tmp) / "review"
            labels_path = Path(tmp) / "completed_labels.jsonl"
            unrelated = reviewed / "operator-notes.txt"
            unrelated.write_text("do not replace", encoding="utf-8")
            manifest_before = (reviewed / "manifest.json").read_bytes()

            with self.assertRaisesRegex(ReviewExportError, "unrelated entry"):
                apply_review_labels(review, reviewed, labels_path=labels_path)

            self.assertEqual(unrelated.read_text(encoding="utf-8"), "do not replace")
            self.assertEqual((reviewed / "manifest.json").read_bytes(), manifest_before)

    def test_apply_review_rejects_review_source_output_overlap(self):
        with tempfile.TemporaryDirectory() as tmp:
            make_reviewed_export(tmp)
            review = Path(tmp) / "review"
            labels_path = Path(tmp) / "completed_labels.jsonl"
            review_manifest_before = (review / "manifest.json").read_bytes()

            with self.assertRaisesRegex(ReviewExportError, "must not overlap"):
                apply_review_labels(review, review / "reviewed", labels_path=labels_path)

            self.assertEqual((review / "manifest.json").read_bytes(), review_manifest_before)
            self.assertFalse((review / "reviewed").exists())

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

    def test_validate_reviewed_export_rejects_provenance_label_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            reviewed = make_reviewed_export(tmp)
            provenance_labels = reviewed / "provenance" / "completed_labels.jsonl"
            rows = read_jsonl(provenance_labels)
            rows[0]["human_label"] = "reject"
            provenance_labels.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
                encoding="utf-8",
            )
            summary_path = Path(tmp) / "validation.json"

            code = run_cli(["validate", "--reviewed-export", str(reviewed), "--out", str(summary_path)])

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in summary["targets"] for error in target["errors"])
            self.assertIn(
                "reviewed_labels.jsonl must match deterministic replay of provenance review_items and completed_labels.",
                errors,
            )

    def test_validate_reviewed_export_rejects_self_consistent_poisoned_sft(self):
        with tempfile.TemporaryDirectory() as tmp:
            reviewed = make_reviewed_export(tmp)
            sft_path = reviewed / "reviewed_sft.jsonl"
            rows = read_jsonl(sft_path)
            rows[0]["response"] = "Unapproved response injected after human review."
            write_jsonl(sft_path, rows)
            refresh_reviewed_export_metadata(reviewed)

            validation = validate_reviewed_export(reviewed)

            self.assertIn(
                "reviewed_sft.jsonl must match deterministic replay of reviewed_labels.jsonl.",
                validation.errors,
            )

    def test_validate_reviewed_export_rejects_missing_and_duplicate_trainer_rows(self):
        cases = (
            ("reviewed_sft", "missing"),
            ("reviewed_sft", "duplicate"),
            ("reviewed_reward_model", "missing"),
            ("reviewed_reward_model", "duplicate"),
            ("reviewed_preferences", "missing"),
            ("reviewed_preferences", "duplicate"),
            ("reviewed_dpo", "missing"),
            ("reviewed_dpo", "duplicate"),
        )
        for artifact_name, mutation in cases:
            with self.subTest(artifact=artifact_name, mutation=mutation):
                with tempfile.TemporaryDirectory() as tmp:
                    reviewed = make_reviewed_export(tmp, include_bad=True)
                    artifact_path = reviewed / f"{artifact_name}.jsonl"
                    rows = read_jsonl(artifact_path)
                    self.assertTrue(rows)
                    mutated_rows = [] if mutation == "missing" else [*rows, dict(rows[0])]
                    write_jsonl(artifact_path, mutated_rows)
                    refresh_reviewed_export_metadata(reviewed)

                    validation = validate_reviewed_export(reviewed)

                    self.assertIn(
                        f"{artifact_name}.jsonl must match deterministic replay of reviewed_labels.jsonl.",
                        validation.errors,
                    )

    def test_validate_reviewed_export_requires_replay_pair_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            reviewed = make_reviewed_export(tmp)
            manifest_path = reviewed / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest.pop("max_pairs_per_family")
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            validation = validate_reviewed_export(reviewed)

            self.assertIn(
                "manifest.max_pairs_per_family must be a non-negative integer for reviewed trainer-view replay.",
                validation.errors,
            )

    def test_validate_reviewed_export_rejects_secret_like_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            reviewed = make_reviewed_export(tmp)
            provenance_manifest = reviewed / "provenance" / "review_manifest.json"
            payload = json.loads(provenance_manifest.read_text(encoding="utf-8"))
            payload["api_token"] = "unredacted-test-credential"
            provenance_manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            summary_path = Path(tmp) / "validation.json"

            code = run_cli(["validate", "--reviewed-export", str(reviewed), "--out", str(summary_path)])

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in summary["targets"] for error in target["errors"])
            self.assertIn("reviewed export contains unredacted secret-like values.", errors)

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

    def test_apply_review_requires_reviewer_identity_and_timestamp(self):
        with tempfile.TemporaryDirectory() as tmp:
            reviewed = make_reviewed_export(tmp)
            review = Path(tmp) / "review"
            labels_path = Path(tmp) / "completed_labels.jsonl"
            rows = read_jsonl(labels_path)
            rows[0]["reviewer"] = ""
            labels_path.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ReviewExportError, "missing reviewer identity"):
                apply_review_labels(review, reviewed, labels_path=labels_path)

            rows[0]["reviewer"] = "test-reviewer"
            rows[0]["reviewed_at"] = ""
            labels_path.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ReviewExportError, "missing reviewed_at timestamp"):
                apply_review_labels(review, reviewed, labels_path=labels_path)

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
