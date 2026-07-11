import hashlib
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from flightrecorder.cli import main
from flightrecorder.reviewed_gate import ReviewedGateError, build_reviewed_export_source_artifact
from flightrecorder.source_contract import inspect_artifact_source
from flightrecorder.validation import validate_artifacts


ROOT = Path(__file__).resolve().parents[1]
CONFIDENCE_MANIFEST_FIELDS = (
    "confidence_counts",
    "high_confidence_label_count",
    "medium_or_high_confidence_label_count",
    "low_confidence_label_count",
    "unknown_confidence_label_count",
)


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
        row["notes"] = "Accepted suggested label for reviewed-gate fixture coverage."
    labels_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def make_reviewed_export(tmp: str):
    root = Path(tmp)
    runs = root / "runs"
    review = root / "review"
    labels = root / "completed_labels.jsonl"
    reviewed = root / "reviewed"
    run_cli(["run-suite", "--scenarios", str(ROOT / "scenarios"), "--out", str(runs)])
    run_cli(["export-review", "--runs", str(runs), "--out", str(review)])
    write_completed_labels(review, labels)
    run_cli(["apply-review", "--review-export", str(review), "--labels", str(labels), "--out", str(reviewed)])
    return reviewed


def rewrite_reviewed_manifest_without_semantic_change(reviewed: Path) -> None:
    manifest_path = reviewed / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_path.write_text(json.dumps(manifest, indent=4, sort_keys=True) + "\n", encoding="utf-8")
    registry_path = reviewed / "dataset_registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    registry["manifest_sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    registry_path.write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n", encoding="utf-8")


class ReviewedGateTests(unittest.TestCase):
    def test_committed_agentic_training_reviewed_gate_is_schema_checkable(self):
        gate_path = ROOT / "examples" / "agentic_training" / "model_grader" / "reviewed_gate.json"
        result = json.loads(gate_path.read_text(encoding="utf-8"))

        self.assertEqual(result["schema_version"], "hfr.reviewed_gate.v1")
        self.assertTrue(result["passed"])
        self.assertEqual(result["failed_check_count"], 0)
        self.assertEqual(result["decision"]["readiness"], "ready")
        self.assertEqual(result["metrics"]["reviewed_label_count"], 2)
        self.assertEqual(result["metrics"]["accepted_count"], 1)
        self.assertEqual(result["metrics"]["rejected_count"], 1)
        self.assertEqual(result["metrics"]["task_families"], ["prompt_injection"])
        self.assertEqual(run_cli(["schemas", "--check", str(gate_path)]), 0)

    def test_gate_reviewed_accepts_demo_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            reviewed = make_reviewed_export(tmp)
            gate = Path(tmp) / "reviewed_gate.json"

            code = run_cli(
                [
                    "gate-reviewed",
                    "--reviewed-export",
                    str(reviewed),
                    "--policy",
                    str(ROOT / "examples" / "reviewed_gate_policy.demo.json"),
                    "--out",
                    str(gate),
                ]
            )

            self.assertEqual(code, 0)
            result = json.loads(gate.read_text(encoding="utf-8"))
            self.assertEqual(result["schema_version"], "hfr.reviewed_gate.v1")
            self.assertTrue(result["passed"])
            self.assertEqual(result["failed_check_count"], 0)
            self.assertEqual(result["decision"]["readiness"], "ready")
            self.assertEqual(result["decision"]["recommendation"], "promote_iteration")
            self.assertGreaterEqual(result["decision"]["key_metrics"]["accepted_count"], 2)
            self.assertEqual(result["policy"]["schema_version"], "hfr.reviewed_gate.policy.v1")
            self.assertGreaterEqual(result["metrics"]["accepted_count"], 2)
            self.assertGreaterEqual(result["metrics"]["rejected_count"], 2)
            self.assertEqual(result["metrics"]["low_confidence_label_count"], 0)
            self.assertEqual(result["metrics"]["unknown_confidence_label_count"], 0)
            source = result["source_artifacts"]["reviewed_export"]
            self.assertEqual(source["path"], "reviewed")
            self.assertEqual(source["kind"], "directory")
            self.assertTrue(source["exists"])
            self.assertFalse(source["contains_symlinks"])
            self.assertNotIn("selected_file_sha256", source)
            self.assertEqual(len(source["sha256"]), 64)
            self.assertGreater(source["file_count"], 0)
            self.assertGreater(source["size_bytes"], 0)
            self.assertEqual(run_cli(["schemas", "--check", str(gate)]), 0)

    def test_reviewed_gate_source_contract_rejects_a_replaced_valid_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            reviewed = make_reviewed_export(tmp)
            gate = Path(tmp) / "reviewed_gate.json"
            self.assertEqual(run_cli(["gate-reviewed", "--reviewed-export", str(reviewed), "--out", str(gate)]), 0)
            self.assertTrue(inspect_artifact_source(gate, "reviewed_gate")["ready"])

            rewrite_reviewed_manifest_without_semantic_change(reviewed)
            reviewed_validation = validate_artifacts(reviewed_export_dir=reviewed, strict=True)

            self.assertTrue(reviewed_validation["passed"], reviewed_validation)
            source = inspect_artifact_source(gate, "reviewed_gate")
            self.assertTrue(source["schema_valid"])
            self.assertFalse(source["semantic_valid"])
            self.assertFalse(source["ready"])

    def test_reviewed_export_source_rejects_manifest_change_between_tree_attestations(self):
        with tempfile.TemporaryDirectory() as tmp:
            reviewed = make_reviewed_export(tmp)
            from flightrecorder.reviewed_gate import _fingerprint_open_directory as fingerprint

            call_count = 0

            def fingerprint_then_mutate(*args, **kwargs):
                nonlocal call_count
                result = fingerprint(*args, **kwargs)
                call_count += 1
                if call_count == 1:
                    with (reviewed / "reviewed_sft.jsonl").open("ab") as handle:
                        handle.write(b"\n")
                return result

            with patch(
                "flightrecorder.reviewed_gate._fingerprint_open_directory",
                side_effect=fingerprint_then_mutate,
            ):
                with self.assertRaisesRegex(
                    ReviewedGateError,
                    "changed while its manifest was being captured",
                ):
                    build_reviewed_export_source_artifact(
                        reviewed,
                        display_path="reviewed",
                    )
            self.assertEqual(call_count, 2)

    def test_reviewed_export_source_rejects_manifest_aba_during_snapshot_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            reviewed = make_reviewed_export(tmp)
            manifest_path = reviewed / "manifest.json"
            original_manifest = manifest_path.read_bytes()
            alternate = json.loads(original_manifest.decode("utf-8"))
            alternate["dataset_version"] = "aba-substitute-dataset"
            alternate_manifest = (
                json.dumps(alternate, indent=2, sort_keys=True) + "\n"
            ).encode("utf-8")
            manifest_read_count = 0

            def read_alternate_then_restore(_reviewed):
                nonlocal manifest_read_count
                manifest_read_count += 1
                manifest_path.write_bytes(alternate_manifest)
                try:
                    return alternate_manifest
                finally:
                    manifest_path.write_bytes(original_manifest)

            with patch(
                "flightrecorder.reviewed_gate._read_reviewed_manifest_bytes",
                side_effect=read_alternate_then_restore,
            ):
                with self.assertRaisesRegex(
                    ReviewedGateError,
                    "manifest bytes did not match the descriptor-bound tree fingerprint",
                ):
                    build_reviewed_export_source_artifact(
                        reviewed,
                        display_path="reviewed",
                    )

            self.assertEqual(manifest_read_count, 1)
            self.assertEqual(manifest_path.read_bytes(), original_manifest)

    def test_gate_reviewed_uses_snapshot_manifest_across_transient_aba(self):
        with tempfile.TemporaryDirectory() as tmp:
            reviewed = make_reviewed_export(tmp)
            manifest_path = reviewed / "manifest.json"
            original_manifest = manifest_path.read_bytes()
            original = json.loads(original_manifest.decode("utf-8"))
            alternate = json.loads(original_manifest.decode("utf-8"))
            alternate["reviewed_label_count"] = 999
            alternate["label_counts"] = {"accept": 999}
            alternate_manifest = (
                json.dumps(alternate, indent=2, sort_keys=True) + "\n"
            ).encode("utf-8")
            from flightrecorder.reviewed_gate import evaluate_reviewed_gate

            for skip_validation in (False, True):
                with self.subTest(skip_validation=skip_validation):
                    gate = Path(tmp) / f"reviewed_gate_{skip_validation}.json"
                    transient_count = 0

                    def evaluate_during_transient_aba(*args, **kwargs):
                        nonlocal transient_count
                        manifest_path.write_bytes(alternate_manifest)
                        transient_count += 1
                        try:
                            return evaluate_reviewed_gate(*args, **kwargs)
                        finally:
                            manifest_path.write_bytes(original_manifest)

                    command = [
                        "gate-reviewed",
                        "--reviewed-export",
                        str(reviewed),
                        "--out",
                        str(gate),
                    ]
                    if skip_validation:
                        command.insert(-2, "--skip-validation")
                    with patch(
                        "flightrecorder.cli.evaluate_reviewed_gate",
                        side_effect=evaluate_during_transient_aba,
                    ):
                        code = run_cli(command)

                    self.assertEqual(code, 1 if skip_validation else 0)
                    self.assertEqual(transient_count, 1)
                    result = json.loads(gate.read_text(encoding="utf-8"))
                    self.assertEqual(
                        result["metrics"]["reviewed_label_count"],
                        original["reviewed_label_count"],
                    )
                    self.assertEqual(
                        result["metrics"]["label_counts"],
                        original["label_counts"],
                    )
                    self.assertEqual(
                        result["source_artifacts"]["reviewed_export"]["manifest_sha256"],
                        hashlib.sha256(original_manifest).hexdigest(),
                    )
                    self.assertEqual(manifest_path.read_bytes(), original_manifest)

    def test_gate_reviewed_preserve_paths_compatibility_stays_relocatable(self):
        with tempfile.TemporaryDirectory() as tmp:
            reviewed = make_reviewed_export(tmp)
            gate = Path(tmp) / "reviewed_gate.json"

            code = run_cli(
                [
                    "gate-reviewed",
                    "--reviewed-export",
                    str(reviewed),
                    "--out",
                    str(gate),
                    "--preserve-paths",
                ]
            )

            self.assertEqual(code, 0)
            result = json.loads(gate.read_text(encoding="utf-8"))
            self.assertEqual(result["reviewed_export"], "reviewed")
            self.assertEqual(result["source_artifacts"]["reviewed_export"]["path"], "reviewed")
            self.assertNotIn(str(Path(tmp).resolve()), json.dumps(result, sort_keys=True))
            validation = validate_artifacts(reviewed_gate_paths=[gate], strict=True)
            self.assertTrue(validation["passed"], validation)

    def test_reviewed_gate_source_contract_rejects_a_deleted_export_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            reviewed = make_reviewed_export(tmp)
            gate = Path(tmp) / "reviewed_gate.json"
            self.assertEqual(run_cli(["gate-reviewed", "--reviewed-export", str(reviewed), "--out", str(gate)]), 0)
            self.assertTrue(inspect_artifact_source(gate, "reviewed_gate")["ready"])

            (reviewed / "reviewed_sft.jsonl").unlink()

            source = inspect_artifact_source(gate, "reviewed_gate")
            self.assertTrue(source["schema_valid"])
            self.assertFalse(source["semantic_valid"])
            self.assertFalse(source["ready"])

    def test_reviewed_gate_source_contract_rejects_an_undeclared_export_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            reviewed = make_reviewed_export(tmp)
            gate = Path(tmp) / "reviewed_gate.json"
            self.assertEqual(run_cli(["gate-reviewed", "--reviewed-export", str(reviewed), "--out", str(gate)]), 0)
            self.assertTrue(inspect_artifact_source(gate, "reviewed_gate")["ready"])

            (reviewed / "operator-notes.txt").write_text("not part of the reviewed contract", encoding="utf-8")

            source = inspect_artifact_source(gate, "reviewed_gate")
            self.assertTrue(source["schema_valid"])
            self.assertFalse(source["semantic_valid"])
            self.assertFalse(source["ready"])

    def test_gate_reviewed_rejects_output_inside_the_reviewed_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            reviewed = make_reviewed_export(tmp)
            output = reviewed / "reviewed_gate.json"

            with self.assertRaises(SystemExit) as raised, redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                main(["gate-reviewed", "--reviewed-export", str(reviewed), "--out", str(output)])

            self.assertEqual(raised.exception.code, 2)
            self.assertFalse(output.exists())

    def test_gate_reviewed_fails_thresholds_and_forbidden_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            reviewed = make_reviewed_export(tmp)
            gate = Path(tmp) / "reviewed_gate.json"

            code = run_cli(
                [
                    "gate-reviewed",
                    "--reviewed-export",
                    str(reviewed),
                    "--min-reviewed-labels",
                    "99",
                    "--forbid-label",
                    "reject",
                    "--out",
                    str(gate),
                ]
            )

            self.assertEqual(code, 1)
            result = json.loads(gate.read_text(encoding="utf-8"))
            self.assertEqual(result["decision"]["readiness"], "blocked")
            self.assertEqual(result["decision"]["blocking_check_count"], 2)
            failed_checks = {item["id"] for item in result["checks"] if not item["passed"]}
            self.assertIn("min_reviewed_labels", failed_checks)
            self.assertIn("forbid_label", failed_checks)
            self.assertEqual(run_cli(["schemas", "--check", str(gate)]), 0)

    def test_gate_reviewed_fails_confidence_thresholds(self):
        with tempfile.TemporaryDirectory() as tmp:
            reviewed = make_reviewed_export(tmp)
            gate = Path(tmp) / "reviewed_gate.json"
            manifest_path = reviewed / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["confidence_counts"] = {"high": 1, "medium": 2, "low": 1, "unknown": 2}
            manifest["high_confidence_label_count"] = 1
            manifest["medium_or_high_confidence_label_count"] = 3
            manifest["low_confidence_label_count"] = 1
            manifest["unknown_confidence_label_count"] = 2
            manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")

            code = run_cli(
                [
                    "gate-reviewed",
                    "--reviewed-export",
                    str(reviewed),
                    "--skip-validation",
                    "--min-high-confidence-labels",
                    "2",
                    "--min-medium-or-high-confidence-labels",
                    "6",
                    "--max-low-confidence-labels",
                    "0",
                    "--max-unknown-confidence-labels",
                    "0",
                    "--out",
                    str(gate),
                ]
            )

            self.assertEqual(code, 1)
            result = json.loads(gate.read_text(encoding="utf-8"))
            failed_checks = {item["id"] for item in result["checks"] if not item["passed"]}
            self.assertIn("min_high_confidence_labels", failed_checks)
            self.assertIn("min_medium_or_high_confidence_labels", failed_checks)
            self.assertIn("max_low_confidence_labels", failed_checks)
            self.assertIn("max_unknown_confidence_labels", failed_checks)

    def test_gate_reviewed_treats_missing_confidence_counts_as_unknown(self):
        with tempfile.TemporaryDirectory() as tmp:
            reviewed = make_reviewed_export(tmp)
            gate = Path(tmp) / "reviewed_gate.json"
            manifest_path = reviewed / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for field_name in CONFIDENCE_MANIFEST_FIELDS:
                manifest.pop(field_name, None)
            manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")

            code = run_cli(
                [
                    "gate-reviewed",
                    "--reviewed-export",
                    str(reviewed),
                    "--skip-validation",
                    "--max-unknown-confidence-labels",
                    "0",
                    "--out",
                    str(gate),
                ]
            )

            self.assertEqual(code, 1)
            result = json.loads(gate.read_text(encoding="utf-8"))
            self.assertEqual(result["metrics"]["unknown_confidence_label_count"], result["metrics"]["reviewed_label_count"])
            failed_checks = {item["id"] for item in result["checks"] if not item["passed"]}
            self.assertIn("max_unknown_confidence_labels", failed_checks)

    def test_gate_reviewed_rejects_missing_confidence_counts_with_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            reviewed = make_reviewed_export(tmp)
            gate = Path(tmp) / "reviewed_gate.json"
            manifest_path = reviewed / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for field_name in CONFIDENCE_MANIFEST_FIELDS:
                manifest.pop(field_name, None)
            manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")

            code = run_cli(
                [
                    "gate-reviewed",
                    "--reviewed-export",
                    str(reviewed),
                    "--max-unknown-confidence-labels",
                    "0",
                    "--out",
                    str(gate),
                ]
            )

            self.assertEqual(code, 1)
            result = json.loads(gate.read_text(encoding="utf-8"))
            failed_checks = {item["id"] for item in result["checks"] if not item["passed"]}
            self.assertIn("valid_reviewed_export", failed_checks)
            self.assertIn("max_unknown_confidence_labels", failed_checks)
            self.assertGreater(result["metrics"]["validation"]["error_count"], 0)

    def test_gate_reviewed_fails_invalid_reviewed_export_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            reviewed = make_reviewed_export(tmp)
            gate = Path(tmp) / "reviewed_gate.json"
            labels_path = reviewed / "reviewed_labels.jsonl"
            rows = read_jsonl(labels_path)
            rows[0]["notes"] = "tampered after reviewed export"
            labels_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")

            code = run_cli(["gate-reviewed", "--reviewed-export", str(reviewed), "--out", str(gate)])

            self.assertEqual(code, 1)
            result = json.loads(gate.read_text(encoding="utf-8"))
            failed_checks = {item["id"] for item in result["checks"] if not item["passed"]}
            self.assertIn("valid_reviewed_export", failed_checks)
            self.assertFalse(result["metrics"]["validation"]["passed"])
            self.assertGreater(result["metrics"]["validation"]["error_count"], 0)

    def test_gate_reviewed_can_explicitly_skip_validation_for_legacy_handoffs(self):
        with tempfile.TemporaryDirectory() as tmp:
            reviewed = make_reviewed_export(tmp)
            gate = Path(tmp) / "reviewed_gate.json"
            labels_path = reviewed / "reviewed_labels.jsonl"
            rows = read_jsonl(labels_path)
            rows[0]["notes"] = "tampered legacy reviewed export"
            labels_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")

            code = run_cli(["gate-reviewed", "--reviewed-export", str(reviewed), "--skip-validation", "--out", str(gate)])

            self.assertEqual(code, 1)
            result = json.loads(gate.read_text(encoding="utf-8"))
            self.assertEqual([check["id"] for check in result["checks"]], ["valid_reviewed_export"])
            self.assertFalse(result["checks"][0]["passed"])
            self.assertFalse(result["passed"])
            self.assertEqual(result["decision"]["readiness"], "blocked")
            self.assertFalse(result["metrics"]["validation"]["available"])
            source = inspect_artifact_source(gate, "reviewed_gate")
            self.assertTrue(source["schema_valid"])
            self.assertFalse(source["semantic_valid"])
            self.assertFalse(source["ready"])

    def test_gate_reviewed_rejects_unknown_policy_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            reviewed = make_reviewed_export(tmp)
            policy = Path(tmp) / "bad_policy.json"
            policy.write_text(
                json.dumps(
                    {
                        "schema_version": "hfr.reviewed_gate.policy.v1",
                        "forbid_labels": ["maybe"],
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaises(SystemExit) as raised, redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                main(
                    [
                        "gate-reviewed",
                        "--reviewed-export",
                        str(reviewed),
                        "--policy",
                        str(policy),
                        "--out",
                        str(Path(tmp) / "gate.json"),
                    ]
                )

            self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
