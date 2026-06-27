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


def run_cli_output(args):
    output = StringIO()
    with redirect_stdout(output):
        code = main(args)
    return code, output.getvalue()


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_completed_labels(review_dir: Path, labels_path: Path) -> None:
    rows = read_jsonl(review_dir / "label_template.jsonl")
    for row in rows:
        row["human_label"] = row["suggested_human_label"]
        row["reviewer"] = "trainer-preflight-test"
        row["reviewed_at"] = "2026-06-26T00:00:00Z"
        row["notes"] = "Accepted suggested label for trainer-preflight coverage."
    labels_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def make_reviewed_export(root: Path) -> Path:
    runs = root / "runs"
    review = root / "review"
    labels = root / "completed_labels.jsonl"
    reviewed = root / "reviewed"
    run_cli(["run-suite", "--scenarios", str(ROOT / "scenarios"), "--out", str(runs)])
    run_cli(["export-review", "--runs", str(runs), "--out", str(review)])
    write_completed_labels(review, labels)
    run_cli(["apply-review", "--review-export", str(review), "--labels", str(labels), "--out", str(reviewed)])
    return reviewed


def write_passed_evidence_bundle(path: Path) -> None:
    bundle = {
        "schema_version": "hfr.evidence_bundle.v1",
        "bundle_path": str(path),
        "passed": True,
        "readiness": "ready",
        "decision": {
            "readiness": "ready",
            "recommendation": "promote_handoff",
            "summary": "Minimal test evidence bundle is ready.",
            "blocking_check_count": 0,
            "next_actions": [],
        },
        "check_count": 0,
        "failed_check_count": 0,
        "checks": [],
        "artifacts": {},
        "metrics": {},
        "notes": [],
    }
    path.write_text(json.dumps(bundle, sort_keys=True) + "\n", encoding="utf-8")


def write_improvement_ledger_gate(path: Path) -> None:
    metrics = {
        "plan_count": 2,
        "unique_work_item_count": 1,
        "open_work_item_count": 1,
        "new_work_item_count": 0,
        "recurring_work_item_count": 1,
        "resolved_work_item_count": 0,
        "critical_open_work_item_count": 0,
        "high_open_work_item_count": 1,
        "open_priority_counts": [{"id": "high", "count": 1}],
        "open_category_counts": [{"id": "repair", "count": 1}],
    }
    gate = {
        "schema_version": "hfr.improvement_ledger_gate.v1",
        "improvement_ledger": "runs/improvement_ledger.json",
        "passed": True,
        "decision": {
            "readiness": "ready",
            "recommendation": "promote_iteration",
            "summary": "Improvement-ledger gate is ready.",
            "blocking_check_count": 0,
            "blocking_checks": [],
            "key_metrics": metrics,
        },
        "check_count": 1,
        "failed_check_count": 0,
        "checks": [
            {
                "id": "max_recurring_work_items",
                "passed": True,
                "actual": 1,
                "expected": {"max": 1},
                "summary": "max_recurring_work_items: actual=1, max=1",
            }
        ],
        "metrics": metrics,
        "policy": {
            "schema_version": "hfr.improvement_ledger_gate.policy.v1",
            "path": "examples/improvement_ledger_gate_policy.demo.json",
            "effective": {"max_recurring_work_items": 1},
        },
    }
    path.write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8")


class TrainerPreflightTests(unittest.TestCase):
    def test_trainer_preflight_accepts_passed_training_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp) / "runs"
            gate = Path(tmp) / "training_gate.json"
            preflight = Path(tmp) / "trainer_preflight.json"
            run_cli(["run-suite", "--scenarios", str(ROOT / "scenarios"), "--out", str(runs), "--export-rl"])
            self.assertEqual(
                run_cli(
                    [
                        "gate-export",
                        "--training-export",
                        str(runs / "training_export"),
                        "--policy",
                        str(ROOT / "examples" / "training_gate_policy.demo.json"),
                        "--out",
                        str(gate),
                    ]
                ),
                0,
            )

            code = run_cli(
                [
                    "trainer-preflight",
                    "--gate",
                    str(gate),
                    "--training-export",
                    str(runs / "training_export"),
                    "--require-gate",
                    "training_gate",
                    "--trainer-command",
                    "python train.py --dataset runs/training_export",
                    "--metadata",
                    "launcher=dry-run",
                    "--preserve-paths",
                    "--out",
                    str(preflight),
                ]
            )

            self.assertEqual(code, 0)
            result = json.loads(preflight.read_text(encoding="utf-8"))
            self.assertEqual(result["schema_version"], "hfr.trainer_preflight.v1")
            self.assertTrue(result["passed"])
            self.assertEqual(result["recommendation"], "launch_allowed")
            self.assertEqual(result["gate_count"], 1)
            self.assertEqual(result["gates"][0]["id"], "training_gate")
            self.assertTrue(result["gates"][0]["validation"]["passed"])
            self.assertEqual(result["metadata"]["launcher"], "dry-run")
            self.assertEqual(result["trainer_command"]["argv"][:2], ["python", "train.py"])
            self.assertIn("training_export_sft_jsonl", result["artifacts"])
            self.assertEqual(len(result["artifacts"]["training_export_sft_jsonl"]["sha256"]), 64)
            self.assertIn("training_export_dataset_splits_json", result["artifacts"])
            self.assertEqual(len(result["artifacts"]["training_export_dataset_splits_json"]["sha256"]), 64)
            self.assertIn("training_export_splits_train_episodes_jsonl", result["artifacts"])
            self.assertEqual(len(result["artifacts"]["training_export_splits_train_episodes_jsonl"]["sha256"]), 64)
            self.assertTrue(result["schema_contracts"]["training_export_manifest_json"]["passed"])
            self.assertTrue(result["schema_contracts"]["training_export_sft_jsonl"]["passed"])
            self.assertEqual(result["schema_contracts"]["training_export_sft_jsonl"]["schema_name"], "rl_sft")
            self.assertGreaterEqual(result["schema_contracts"]["training_export_sft_jsonl"]["row_count"], 1)
            self.assertEqual(run_cli(["validate", "--trainer-preflight", str(preflight), "--strict"]), 0)

            launch_check = Path(tmp) / "trainer_launch_check.json"
            self.assertEqual(
                run_cli(
                    [
                        "trainer-launch-check",
                        "--preflight",
                        str(preflight),
                        "--require-gate",
                        "training_gate",
                        "--require-metadata",
                        "launcher=dry-run",
                        "--out",
                        str(launch_check),
                    ]
                ),
                0,
            )
            launch = json.loads(launch_check.read_text(encoding="utf-8"))
            self.assertEqual(launch["schema_version"], "hfr.trainer_launch_check.v1")
            self.assertTrue(launch["passed"])
            self.assertEqual(launch["recommendation"], "launch_allowed")
            self.assertEqual(launch["approved_command"]["argv"][:2], ["python", "train.py"])
            self.assertTrue(launch["approved_command"]["approved"])
            self.assertEqual(run_cli(["validate", "--trainer-launch-check", str(launch_check), "--strict"]), 0)

            archive = Path(tmp) / "trainer_archive"
            self.assertEqual(
                run_cli(
                    [
                        "trainer-archive",
                        "--preflight",
                        str(preflight),
                        "--launch-check",
                        str(launch_check),
                        "--out",
                        str(archive),
                        "--require-self-contained",
                    ]
                ),
                0,
            )
            manifest_path = archive / "trainer_archive.json"
            result = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(result["schema_version"], "hfr.trainer_archive.v1")
            self.assertTrue(result["passed"])
            self.assertTrue(result["self_contained"])
            self.assertTrue(result["ready_for_training"])
            roles = {artifact["role"] for artifact in result["artifacts"]}
            self.assertIn("trainer_preflight", roles)
            self.assertIn("trainer_launch_check", roles)
            self.assertIn("gate", roles)
            self.assertIn("trainer_artifact", roles)
            self.assertIn("schema_contract", roles)
            self.assertGreater(result["metrics"]["directory_artifact_count"], 0)
            self.assertEqual(run_cli(["validate", "--trainer-archive", str(archive), "--strict"]), 0)
            self.assertEqual(run_cli(["schemas", "--check", str(manifest_path)]), 0)

            code, output = run_cli_output(["trainer-launch-check", "--preflight", str(preflight), "--print-command"])
            self.assertEqual(code, 0)
            self.assertEqual(output.strip(), "python train.py --dataset runs/training_export")

    def test_trainer_preflight_blocks_failed_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp) / "runs"
            gate = Path(tmp) / "training_gate.json"
            preflight = Path(tmp) / "trainer_preflight.json"
            run_cli(["run", "--scenario", str(ROOT / "scenarios" / "prompt_injection_bad.json"), "--out", str(runs / "bad")])
            run_cli(["export-rl", "--runs", str(runs), "--out", str(runs / "training_export")])
            self.assertEqual(
                run_cli(
                    [
                        "gate-export",
                        "--training-export",
                        str(runs / "training_export"),
                        "--min-pass-rate",
                        "1.0",
                        "--out",
                        str(gate),
                    ]
                ),
                1,
            )

            code = run_cli(
                [
                    "trainer-preflight",
                    "--gate",
                    str(gate),
                    "--training-export",
                    str(runs / "training_export"),
                    "--out",
                    str(preflight),
                ]
            )

            self.assertEqual(code, 1)
            result = json.loads(preflight.read_text(encoding="utf-8"))
            self.assertEqual(result["recommendation"], "block_launch")
            self.assertIn("gate_passed", {check["id"] for check in result["checks"] if not check["passed"]})

            launch_check = Path(tmp) / "trainer_launch_check.json"
            self.assertEqual(
                run_cli(["trainer-launch-check", "--preflight", str(preflight), "--out", str(launch_check)]),
                1,
            )
            launch = json.loads(launch_check.read_text(encoding="utf-8"))
            self.assertEqual(launch["recommendation"], "block_launch")
            self.assertIn("preflight_passed", {check["id"] for check in launch["checks"] if not check["passed"]})

    def test_trainer_preflight_blocks_unvalidated_training_gate_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp) / "runs"
            gate = Path(tmp) / "training_gate.json"
            preflight = Path(tmp) / "trainer_preflight.json"
            run_cli(["run", "--scenario", str(ROOT / "scenarios" / "prompt_injection_good.json"), "--out", str(runs / "good")])
            run_cli(["export-rl", "--runs", str(runs), "--out", str(runs / "training_export")])
            self.assertEqual(
                run_cli(["gate-export", "--training-export", str(runs / "training_export"), "--skip-validation", "--out", str(gate)]),
                0,
            )

            code = run_cli(
                [
                    "trainer-preflight",
                    "--gate",
                    str(gate),
                    "--training-export",
                    str(runs / "training_export"),
                    "--out",
                    str(preflight),
                ]
            )

            self.assertEqual(code, 1)
            result = json.loads(preflight.read_text(encoding="utf-8"))
            self.assertIn("gate_validation_passed", {check["id"] for check in result["checks"] if not check["passed"]})

    def test_trainer_preflight_blocks_unvalidated_reviewed_gate_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reviewed = make_reviewed_export(root)
            gate = root / "reviewed_gate.json"
            preflight = root / "trainer_preflight.json"
            self.assertEqual(
                run_cli(["gate-reviewed", "--reviewed-export", str(reviewed), "--skip-validation", "--out", str(gate)]),
                0,
            )

            code = run_cli(
                [
                    "trainer-preflight",
                    "--gate",
                    str(gate),
                    "--reviewed-export",
                    str(reviewed),
                    "--out",
                    str(preflight),
                ]
            )

            self.assertEqual(code, 1)
            result = json.loads(preflight.read_text(encoding="utf-8"))
            self.assertIn("gate_validation_passed", {check["id"] for check in result["checks"] if not check["passed"]})

    def test_trainer_preflight_blocks_unvalidated_review_calibration_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reviewed = make_reviewed_export(root)
            calibration = root / "review_calibration.json"
            preflight = root / "trainer_preflight.json"
            self.assertEqual(
                run_cli(["review-calibration", "--reviewed-export", str(reviewed), "--skip-validation", "--out", str(calibration)]),
                0,
            )

            code = run_cli(
                [
                    "trainer-preflight",
                    "--gate",
                    str(calibration),
                    "--reviewed-export",
                    str(reviewed),
                    "--out",
                    str(preflight),
                ]
            )

            self.assertEqual(code, 1)
            result = json.loads(preflight.read_text(encoding="utf-8"))
            self.assertIn("gate_validation_passed", {check["id"] for check in result["checks"] if not check["passed"]})

    def test_trainer_preflight_blocks_unvalidated_improvement_gate_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gate = root / "improvement_ledger_gate.json"
            evidence_bundle = root / "evidence_bundle.json"
            preflight = root / "trainer_preflight.json"
            write_improvement_ledger_gate(gate)
            write_passed_evidence_bundle(evidence_bundle)

            code = run_cli(
                [
                    "trainer-preflight",
                    "--gate",
                    str(gate),
                    "--evidence-bundle",
                    str(evidence_bundle),
                    "--out",
                    str(preflight),
                ]
            )

            self.assertEqual(code, 1)
            result = json.loads(preflight.read_text(encoding="utf-8"))
            self.assertEqual(result["gates"][0]["id"], "improvement_ledger_gate")
            self.assertIn("gate_validation_passed", {check["id"] for check in result["checks"] if not check["passed"]})

    def test_trainer_preflight_accepts_external_validation_for_improvement_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gate = root / "improvement_ledger_gate.json"
            evidence_bundle = root / "evidence_bundle.json"
            validation = root / "validation.json"
            preflight = root / "trainer_preflight.json"
            write_improvement_ledger_gate(gate)
            write_passed_evidence_bundle(evidence_bundle)
            self.assertEqual(run_cli(["validate", "--improvement-ledger-gate", str(gate), "--strict", "--out", str(validation)]), 0)

            code = run_cli(
                [
                    "trainer-preflight",
                    "--gate",
                    str(gate),
                    "--validation",
                    str(validation),
                    "--evidence-bundle",
                    str(evidence_bundle),
                    "--require-gate",
                    "improvement_ledger_gate",
                    "--trainer-command",
                    "python train.py --dataset runs/training_export",
                    "--preserve-paths",
                    "--out",
                    str(preflight),
                ]
            )

            self.assertEqual(code, 0)
            result = json.loads(preflight.read_text(encoding="utf-8"))
            self.assertTrue(result["passed"])
            self.assertEqual(result["gate_count"], 1)
            self.assertEqual(result["passed_gate_count"], 1)
            gate_validation = result["gates"][0]["validation"]
            self.assertTrue(gate_validation["available"])
            self.assertTrue(gate_validation["passed"])
            self.assertTrue(gate_validation["summary_passed"])
            self.assertEqual(gate_validation["target_type"], "improvement_ledger_gate")
            self.assertEqual(gate_validation["source"], str(validation))
            self.assertEqual(result["validation_summaries"][0]["path"], str(validation))
            self.assertEqual(result["validation_summaries"][0]["targets"][0]["type"], "improvement_ledger_gate")
            self.assertEqual(len(result["validation_summaries"][0]["sha256"]), 64)
            self.assertEqual(run_cli(["validate", "--trainer-preflight", str(preflight), "--strict"]), 0)

    def test_trainer_preflight_rejects_failed_external_validation_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gate = root / "improvement_ledger_gate.json"
            evidence_bundle = root / "evidence_bundle.json"
            validation = root / "validation.json"
            preflight = root / "trainer_preflight.json"
            write_improvement_ledger_gate(gate)
            write_passed_evidence_bundle(evidence_bundle)
            run_cli(["validate", "--improvement-ledger-gate", str(gate), "--strict", "--out", str(validation)])
            validation_payload = json.loads(validation.read_text(encoding="utf-8"))
            validation_payload["passed"] = False
            validation_payload["error_count"] = 1
            validation.write_text(json.dumps(validation_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            code = run_cli(
                [
                    "trainer-preflight",
                    "--gate",
                    str(gate),
                    "--validation",
                    str(validation),
                    "--evidence-bundle",
                    str(evidence_bundle),
                    "--out",
                    str(preflight),
                ]
            )

            self.assertEqual(code, 1)
            result = json.loads(preflight.read_text(encoding="utf-8"))
            gate_validation = result["gates"][0]["validation"]
            self.assertTrue(gate_validation["available"])
            self.assertFalse(gate_validation["passed"])
            self.assertFalse(gate_validation["summary_passed"])
            self.assertIn("gate_validation_passed", {check["id"] for check in result["checks"] if not check["passed"]})

    def test_validate_rejects_stale_external_validation_summary_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gate = root / "improvement_ledger_gate.json"
            evidence_bundle = root / "evidence_bundle.json"
            validation = root / "validation.json"
            preflight = root / "trainer_preflight.json"
            summary = root / "preflight_validation.json"
            write_improvement_ledger_gate(gate)
            write_passed_evidence_bundle(evidence_bundle)
            run_cli(["validate", "--improvement-ledger-gate", str(gate), "--strict", "--out", str(validation)])
            run_cli(
                [
                    "trainer-preflight",
                    "--gate",
                    str(gate),
                    "--validation",
                    str(validation),
                    "--evidence-bundle",
                    str(evidence_bundle),
                    "--preserve-paths",
                    "--out",
                    str(preflight),
                ]
            )
            validation_payload = json.loads(validation.read_text(encoding="utf-8"))
            validation_payload["warning_count"] = 1
            validation.write_text(json.dumps(validation_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            code = run_cli(["validate", "--trainer-preflight", str(preflight), "--out", str(summary)])

            self.assertEqual(code, 1)
            validation_result = json.loads(summary.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in validation_result["targets"] for error in target["errors"])
            self.assertIn("trainer_preflight.validation_summaries[0].sha256", errors)

    def test_trainer_preflight_accepts_validated_reviewed_gate_and_calibration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reviewed = make_reviewed_export(root)
            gate = root / "reviewed_gate.json"
            calibration = root / "review_calibration.json"
            preflight = root / "trainer_preflight.json"
            self.assertEqual(run_cli(["gate-reviewed", "--reviewed-export", str(reviewed), "--out", str(gate)]), 0)
            self.assertEqual(run_cli(["review-calibration", "--reviewed-export", str(reviewed), "--out", str(calibration)]), 0)

            code = run_cli(
                [
                    "trainer-preflight",
                    "--gate",
                    str(gate),
                    "--gate",
                    str(calibration),
                    "--reviewed-export",
                    str(reviewed),
                    "--require-gate",
                    "reviewed_gate",
                    "--require-gate",
                    "review_calibration",
                    "--trainer-command",
                    "python train.py --dataset runs/reviewed_export",
                    "--out",
                    str(preflight),
                ]
            )

            self.assertEqual(code, 0)
            result = json.loads(preflight.read_text(encoding="utf-8"))
            self.assertTrue(result["passed"])
            self.assertEqual({gate["id"] for gate in result["gates"]}, {"reviewed_gate", "review_calibration"})
            self.assertTrue(all(gate["validation"]["passed"] for gate in result["gates"]))
            self.assertIn("reviewed_export_reviewed_labels_jsonl", result["artifacts"])

    def test_trainer_preflight_blocks_symlinked_training_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = root / "runs"
            gate = root / "training_gate.json"
            preflight = root / "trainer_preflight.json"
            run_cli(["run", "--scenario", str(ROOT / "scenarios" / "prompt_injection_good.json"), "--out", str(runs / "good")])
            run_cli(["export-rl", "--runs", str(runs), "--out", str(runs / "training_export")])
            run_cli(["gate-export", "--training-export", str(runs / "training_export"), "--out", str(gate)])
            sft_path = runs / "training_export" / "sft.jsonl"
            external_path = root / "external_sft.jsonl"
            external_path.write_text(sft_path.read_text(encoding="utf-8"), encoding="utf-8")
            sft_path.unlink()
            try:
                sft_path.symlink_to(external_path)
            except (NotImplementedError, OSError) as exc:
                self.skipTest(f"symlink creation unavailable: {exc}")

            code = run_cli(
                [
                    "trainer-preflight",
                    "--gate",
                    str(gate),
                    "--training-export",
                    str(runs / "training_export"),
                    "--out",
                    str(preflight),
                ]
            )

            self.assertEqual(code, 1)
            result = json.loads(preflight.read_text(encoding="utf-8"))
            self.assertEqual(result["recommendation"], "block_launch")
            self.assertFalse(result["artifacts"]["training_export_sft_jsonl"]["regular_file"])
            self.assertTrue(result["artifacts"]["training_export_sft_jsonl"]["symlink"])
            failed_checks = {check["id"] for check in result["checks"] if not check["passed"]}
            self.assertIn("artifact_file_regular", failed_checks)
            self.assertEqual(run_cli(["validate", "--trainer-preflight", str(preflight), "--strict"]), 0)

    def test_trainer_preflight_blocks_malformed_training_schema_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = root / "runs"
            gate = root / "training_gate.json"
            preflight = root / "trainer_preflight.json"
            run_cli(["run", "--scenario", str(ROOT / "scenarios" / "prompt_injection_good.json"), "--out", str(runs / "good")])
            run_cli(["export-rl", "--runs", str(runs), "--out", str(runs / "training_export")])
            run_cli(["gate-export", "--training-export", str(runs / "training_export"), "--out", str(gate)])
            sft_path = runs / "training_export" / "sft.jsonl"
            sft_rows = read_jsonl(sft_path)
            sft_rows[0].pop("response", None)
            sft_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in sft_rows), encoding="utf-8")

            code = run_cli(
                [
                    "trainer-preflight",
                    "--gate",
                    str(gate),
                    "--training-export",
                    str(runs / "training_export"),
                    "--out",
                    str(preflight),
                ]
            )

            self.assertEqual(code, 1)
            result = json.loads(preflight.read_text(encoding="utf-8"))
            contract = result["schema_contracts"]["training_export_sft_jsonl"]
            self.assertFalse(contract["passed"])
            self.assertEqual(contract["schema_name"], "rl_sft")
            self.assertIn("missing required property 'response'", "\n".join(contract["errors"]))
            self.assertIn("schema_contract_passed", {check["id"] for check in result["checks"] if not check["passed"]})
            self.assertEqual(run_cli(["validate", "--trainer-preflight", str(preflight), "--strict"]), 0)

    def test_trainer_preflight_blocks_symlinked_training_split_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = root / "runs"
            gate = root / "training_gate.json"
            preflight = root / "trainer_preflight.json"
            run_cli(["run", "--scenario", str(ROOT / "scenarios" / "prompt_injection_good.json"), "--out", str(runs / "good")])
            run_cli(["export-rl", "--runs", str(runs), "--out", str(runs / "training_export")])
            run_cli(["gate-export", "--training-export", str(runs / "training_export"), "--out", str(gate)])
            split_path = runs / "training_export" / "splits" / "train" / "episodes.jsonl"
            external_path = root / "external_split_episodes.jsonl"
            external_path.write_text(split_path.read_text(encoding="utf-8"), encoding="utf-8")
            split_path.unlink()
            try:
                split_path.symlink_to(external_path)
            except (NotImplementedError, OSError) as exc:
                self.skipTest(f"symlink creation unavailable: {exc}")

            code = run_cli(
                [
                    "trainer-preflight",
                    "--gate",
                    str(gate),
                    "--training-export",
                    str(runs / "training_export"),
                    "--out",
                    str(preflight),
                ]
            )

            self.assertEqual(code, 1)
            result = json.loads(preflight.read_text(encoding="utf-8"))
            self.assertEqual(result["recommendation"], "block_launch")
            split_record = result["artifacts"]["training_export_splits_train_episodes_jsonl"]
            self.assertFalse(split_record["regular_file"])
            self.assertTrue(split_record["symlink"])
            failed_checks = {check["id"] for check in result["checks"] if not check["passed"]}
            self.assertIn("artifact_file_regular", failed_checks)
            self.assertEqual(run_cli(["validate", "--trainer-preflight", str(preflight), "--strict"]), 0)

    def test_validate_rejects_stale_trainer_preflight_gate_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp) / "runs"
            gate = Path(tmp) / "training_gate.json"
            preflight = Path(tmp) / "trainer_preflight.json"
            summary = Path(tmp) / "validation.json"
            run_cli(["run", "--scenario", str(ROOT / "scenarios" / "prompt_injection_good.json"), "--out", str(runs / "good")])
            run_cli(["export-rl", "--runs", str(runs), "--out", str(runs / "training_export")])
            run_cli(["gate-export", "--training-export", str(runs / "training_export"), "--out", str(gate)])
            run_cli(
                [
                    "trainer-preflight",
                    "--gate",
                    str(gate),
                    "--training-export",
                    str(runs / "training_export"),
                    "--preserve-paths",
                    "--out",
                    str(preflight),
                ]
            )
            gate_payload = json.loads(gate.read_text(encoding="utf-8"))
            gate_payload["passed"] = False
            gate.write_text(json.dumps(gate_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            code = run_cli(["validate", "--trainer-preflight", str(preflight), "--out", str(summary)])

            self.assertEqual(code, 1)
            validation = json.loads(summary.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in validation["targets"] for error in target["errors"])
            self.assertIn("trainer_preflight.gates[0].sha256", errors)

            launch_check = Path(tmp) / "trainer_launch_check.json"
            self.assertEqual(
                run_cli(["trainer-launch-check", "--preflight", str(preflight), "--out", str(launch_check)]),
                1,
            )
            launch = json.loads(launch_check.read_text(encoding="utf-8"))
            self.assertIn("preflight_validation_passed", {check["id"] for check in launch["checks"] if not check["passed"]})

    def test_validate_rejects_stale_trainer_preflight_artifact_symlink(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = root / "runs"
            gate = root / "training_gate.json"
            preflight = root / "trainer_preflight.json"
            summary = root / "validation.json"
            run_cli(["run", "--scenario", str(ROOT / "scenarios" / "prompt_injection_good.json"), "--out", str(runs / "good")])
            run_cli(["export-rl", "--runs", str(runs), "--out", str(runs / "training_export")])
            run_cli(["gate-export", "--training-export", str(runs / "training_export"), "--out", str(gate)])
            run_cli(
                [
                    "trainer-preflight",
                    "--gate",
                    str(gate),
                    "--training-export",
                    str(runs / "training_export"),
                    "--preserve-paths",
                    "--out",
                    str(preflight),
                ]
            )
            episodes_path = runs / "training_export" / "episodes.jsonl"
            external_path = root / "external_episodes.jsonl"
            external_path.write_text(episodes_path.read_text(encoding="utf-8"), encoding="utf-8")
            episodes_path.unlink()
            try:
                episodes_path.symlink_to(external_path)
            except (NotImplementedError, OSError) as exc:
                self.skipTest(f"symlink creation unavailable: {exc}")

            code = run_cli(["validate", "--trainer-preflight", str(preflight), "--out", str(summary)])

            self.assertEqual(code, 1)
            validation = json.loads(summary.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in validation["targets"] for error in target["errors"])
            self.assertIn("trainer_preflight.artifacts.training_export_episodes_jsonl.path must not resolve to a symlink", errors)

            launch_check = root / "trainer_launch_check.json"
            self.assertEqual(
                run_cli(["trainer-launch-check", "--preflight", str(preflight), "--out", str(launch_check)]),
                1,
            )
            launch = json.loads(launch_check.read_text(encoding="utf-8"))
            self.assertIn("preflight_validation_passed", {check["id"] for check in launch["checks"] if not check["passed"]})


if __name__ == "__main__":
    unittest.main()
