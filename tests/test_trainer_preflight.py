import json
import os
import shlex
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from flightrecorder.agentic_training_plan import build_agentic_training_plan, write_agentic_training_plan
from flightrecorder.cli import main
from flightrecorder.schema_registry import check_schema_contract


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
        row["reviewer_confidence"] = "high"
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


def write_agentic_plan_fixture(root: Path) -> Path:
    model = root / "agentic_model.json"
    dataset = root / "agentic_dataset.json"
    plan_path = root / "agentic_training_plan.json"
    model.write_text(
        json.dumps(
            {
                "schema_version": "hfr.model_candidate.test.v1",
                "model_id": "local/agentic-preflight-model",
                "candidate_id": "candidate",
                "license": {"status": "approved", "allow_training": True},
                "compatibility": {"passed": True},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    dataset.write_text(
        json.dumps(
            {
                "schema_version": "hfr.dataset_manifest.test.v1",
                "dataset_id": "agentic-preflight-dataset",
                "dataset_version": "v1",
                "license": {"status": "approved", "allow_training": True},
                "redaction": {"status": "redacted", "passed": True, "contains_unredacted_traces": False},
                "views": {
                    "sft": {"path": "sft.jsonl", "row_count": 2, "schema_version": "hfr.rl.sft.v1"},
                    "dpo": {"path": "dpo.jsonl", "row_count": 2, "schema_version": "hfr.rl.dpo.v1"},
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    plan = build_agentic_training_plan(
        out_path=plan_path,
        mode="sft_then_dpo",
        model_manifest_path=model,
        dataset_manifest_path=dataset,
        trainer_backend="axolotl",
        output_dir=root / "adapters",
        limit=2,
        created_at="2026-07-02T00:00:00+00:00",
    )
    write_agentic_training_plan(plan_path, plan)
    return plan_path


def write_improvement_ledger_gate(path: Path) -> None:
    source_dir = path.parent / "runs"
    source_dir.mkdir(parents=True, exist_ok=True)
    plan_1 = source_dir / "improvement_plan_1.json"
    plan_2 = source_dir / "improvement_plan_2.json"
    ledger = source_dir / "improvement_ledger.json"
    plan = {
        "schema_version": "hfr.improvement_plan.v1",
        "passed": True,
        "readiness": "ready",
        "work_item_count": 1,
        "decision": {
            "recommendation": "run_improvement_iteration",
            "critical_or_high_count": 1,
        },
        "work_items": [
            {
                "category": "repair",
                "priority": "high",
                "summary": "Repair prompt-injection evidence coverage.",
                "suggested_action": "Add targeted repair evidence before promotion.",
                "scenario_id": "prompt_injection_bad",
                "task_family": "prompt_injection",
                "rule_id": "forbidden_actions",
                "rule_name": "Forbidden actions",
                "score": 0,
                "task_completion_status": "failed",
                "evidence_refs": [],
            }
        ],
    }
    plan_1.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    plan_2.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    assert run_cli(["improvement-ledger", "--plan", str(plan_1), "--plan", str(plan_2), "--out", str(ledger)]) == 0
    assert (
        run_cli(
            [
                "gate-improvement-ledger",
                "--improvement-ledger",
                str(ledger),
                "--max-recurring-work-items",
                "1",
                "--out",
                str(path),
            ]
        )
        == 0
    )


class TrainerPreflightTests(unittest.TestCase):
    def _assert_trainer_wrapper_validation_rejects(
        self,
        root: Path,
        payload: dict,
        name: str,
        expected_scope: str,
        expected_field: str,
    ) -> None:
        forged_receipt = root / f"{name}.json"
        forged_summary = root / f"{name}_summary.json"
        forged_receipt.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        code = run_cli(
            [
                "validate",
                "--trainer-wrapper-dry-run",
                str(forged_receipt),
                "--strict",
                "--out",
                str(forged_summary),
            ]
        )
        self.assertEqual(code, 1)
        validation = json.loads(forged_summary.read_text(encoding="utf-8"))
        errors = "\n".join(error for target in validation["targets"] for error in target["errors"])
        self.assertIn(expected_scope, errors)
        self.assertIn(expected_field, errors)

    def test_trainer_preflight_archives_agentic_training_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gate = root / "evidence_bundle.json"
            agentic_plan = write_agentic_plan_fixture(root)
            preflight = root / "trainer_preflight.json"
            write_passed_evidence_bundle(gate)

            self.assertEqual(
                run_cli(
                    [
                        "trainer-preflight",
                        "--gate",
                        str(gate),
                        "--agentic-training-plan",
                        str(agentic_plan),
                        "--require-gate",
                        "evidence_bundle",
                        "--trainer-command",
                        f"python train.py --agentic-plan {agentic_plan}",
                        "--metadata",
                        "launcher=agentic-dry-run",
                        "--out",
                        str(preflight),
                        "--preserve-paths",
                    ]
                ),
                0,
            )
            result = json.loads(preflight.read_text(encoding="utf-8"))
            self.assertTrue(result["passed"])
            self.assertIn("agentic_training_plan", result["artifacts"])
            self.assertEqual(len(result["artifacts"]["agentic_training_plan"]["sha256"]), 64)
            self.assertTrue(result["schema_contracts"]["agentic_training_plan"]["passed"])
            self.assertEqual(result["schema_contracts"]["agentic_training_plan"]["schema_name"], "agentic_training_plan")
            self.assertFalse(any(check["id"] == "agentic_training_plan_ready" and not check["passed"] for check in result["checks"]))
            self.assertEqual(run_cli(["validate", "--trainer-preflight", str(preflight), "--strict"]), 0)

            launch_check = root / "trainer_launch_check.json"
            self.assertEqual(
                run_cli(
                    [
                        "trainer-launch-check",
                        "--preflight",
                        str(preflight),
                        "--require-gate",
                        "evidence_bundle",
                        "--require-metadata",
                        "launcher=agentic-dry-run",
                        "--out",
                        str(launch_check),
                        "--strict",
                        "--preserve-paths",
                    ]
                ),
                0,
            )

            archive = root / "trainer_archive"
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
                        "--preserve-paths",
                    ]
                ),
                0,
            )
            archive_manifest = json.loads((archive / "trainer_archive.json").read_text(encoding="utf-8"))
            self.assertTrue(archive_manifest["passed"])
            self.assertIn("agentic_training_plan", {item["artifact_name"] for item in archive_manifest["trainer_inputs"]})
            self.assertEqual(run_cli(["validate", "--trainer-archive", str(archive), "--strict"]), 0)

            trainer_code = root / "trainer_code"
            trainer_code.mkdir()
            (trainer_code / "train.py").write_text("print('agentic dry run only')\n", encoding="utf-8")
            archive_check = root / "trainer_archive_check.json"
            self.assertEqual(
                run_cli(
                    [
                        "trainer-archive-check",
                        "--archive",
                        str(archive),
                        "--external-code-root",
                        str(trainer_code),
                        "--out",
                        str(archive_check),
                        "--strict",
                        "--preserve-paths",
                    ]
                ),
                0,
            )

            consumer_plan = root / "trainer_consumer_plan.json"
            self.assertEqual(
                run_cli(
                    [
                        "trainer-consumer-plan",
                        "--archive-check",
                        str(archive_check),
                        "--out",
                        str(consumer_plan),
                        "--strict",
                        "--preserve-paths",
                    ]
                ),
                0,
            )
            plan = json.loads(consumer_plan.read_text(encoding="utf-8"))
            self.assertTrue(plan["passed"])
            self.assertIn("agentic_training_plan", {item["artifact_name"] for item in plan["execution"]["trainer_inputs"]})
            self.assertEqual(run_cli(["validate", "--trainer-consumer-plan", str(consumer_plan), "--strict"]), 0)

    def test_trainer_preflight_writes_output_relative_source_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = root / "src"
            output_dir = root / "out"
            source_dir.mkdir()
            output_dir.mkdir()
            gate = source_dir / "evidence_bundle.json"
            preflight = output_dir / "trainer_preflight.json"
            write_passed_evidence_bundle(gate)

            self.assertEqual(
                run_cli(
                    [
                        "trainer-preflight",
                        "--gate",
                        str(gate),
                        "--evidence-bundle",
                        str(gate),
                        "--trainer-command",
                        "python train.py --bundle ../src/evidence_bundle.json",
                        "--out",
                        str(preflight),
                    ]
                ),
                0,
            )

            result = json.loads(preflight.read_text(encoding="utf-8"))
            self.assertEqual(result["gates"][0]["path"], "../src/evidence_bundle.json")
            self.assertEqual(result["artifacts"]["evidence_bundle"]["path"], "../src/evidence_bundle.json")
            self.assertEqual(result["schema_contracts"]["evidence_bundle"]["path"], "../src/evidence_bundle.json")
            self.assertEqual(run_cli(["validate", "--trainer-preflight", str(preflight), "--strict"]), 0)

    def test_validate_rejects_trainer_preflight_cwd_relative_source_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = root / "src"
            output_dir = root / "out"
            source_dir.mkdir()
            output_dir.mkdir()
            gate = source_dir / "evidence_bundle.json"
            preflight = output_dir / "trainer_preflight.json"
            summary = root / "validation.json"
            write_passed_evidence_bundle(gate)
            run_cli(
                [
                    "trainer-preflight",
                    "--gate",
                    str(gate),
                    "--evidence-bundle",
                    str(gate),
                    "--trainer-command",
                    "python train.py --bundle ../src/evidence_bundle.json",
                    "--out",
                    str(preflight),
                ]
            )
            result = json.loads(preflight.read_text(encoding="utf-8"))
            result["gates"][0]["path"] = "evidence_bundle.json"
            result["artifacts"]["evidence_bundle"]["path"] = "evidence_bundle.json"
            result["schema_contracts"]["evidence_bundle"]["path"] = "evidence_bundle.json"
            preflight.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            previous_cwd = Path.cwd()
            try:
                os.chdir(source_dir)
                code = run_cli(["validate", "--trainer-preflight", str(preflight), "--strict", "--out", str(summary)])
            finally:
                os.chdir(previous_cwd)

            self.assertEqual(code, 1)
            validation = json.loads(summary.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in validation["targets"] for error in target["errors"])
            self.assertIn("trainer_preflight.gates[0].path does not resolve to an existing file", errors)
            self.assertIn("trainer_preflight.artifacts.evidence_bundle.path does not resolve to an existing file", errors)

    def test_trainer_archive_includes_output_relative_parent_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = root / "src"
            output_dir = root / "out"
            archive = root / "trainer_archive"
            source_dir.mkdir()
            output_dir.mkdir()
            gate = source_dir / "evidence_bundle.json"
            preflight = output_dir / "trainer_preflight.json"
            launch_check = output_dir / "trainer_launch_check.json"
            write_passed_evidence_bundle(gate)
            run_cli(
                [
                    "trainer-preflight",
                    "--gate",
                    str(gate),
                    "--evidence-bundle",
                    str(gate),
                    "--trainer-command",
                    "python train.py --bundle ../src/evidence_bundle.json",
                    "--out",
                    str(preflight),
                ]
            )
            self.assertEqual(run_cli(["trainer-launch-check", "--preflight", str(preflight), "--out", str(launch_check)]), 0)

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
            manifest = json.loads((archive / "trainer_archive.json").read_text(encoding="utf-8"))
            self.assertTrue(manifest["passed"])
            self.assertTrue(manifest["self_contained"])
            self.assertEqual(manifest["metrics"]["missing_count"], 0)
            self.assertIn("gate", {artifact["role"] for artifact in manifest["artifacts"]})

    def test_trainer_archive_rejects_cwd_relative_source_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = root / "src"
            output_dir = root / "out"
            archive = root / "trainer_archive"
            source_dir.mkdir()
            output_dir.mkdir()
            gate = source_dir / "evidence_bundle.json"
            preflight = output_dir / "trainer_preflight.json"
            launch_check = output_dir / "trainer_launch_check.json"
            write_passed_evidence_bundle(gate)
            run_cli(
                [
                    "trainer-preflight",
                    "--gate",
                    str(gate),
                    "--evidence-bundle",
                    str(gate),
                    "--trainer-command",
                    "python train.py --bundle ../src/evidence_bundle.json",
                    "--out",
                    str(preflight),
                ]
            )
            self.assertEqual(run_cli(["trainer-launch-check", "--preflight", str(preflight), "--out", str(launch_check)]), 0)
            result = json.loads(preflight.read_text(encoding="utf-8"))
            result["gates"][0]["path"] = "evidence_bundle.json"
            result["artifacts"]["evidence_bundle"]["path"] = "evidence_bundle.json"
            result["schema_contracts"]["evidence_bundle"]["path"] = "evidence_bundle.json"
            preflight.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            previous_cwd = Path.cwd()
            try:
                os.chdir(source_dir)
                code = run_cli(
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
                )
            finally:
                os.chdir(previous_cwd)

            self.assertEqual(code, 1)
            manifest = json.loads((archive / "trainer_archive.json").read_text(encoding="utf-8"))
            self.assertFalse(manifest["passed"])
            self.assertFalse(manifest["self_contained"])
            self.assertGreater(manifest["metrics"]["missing_count"], 0)
            self.assertIn("gate", {item["role"] for item in manifest["missing"]})

    def test_trainer_archive_rejects_stale_preflight_source_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = root / "src"
            output_dir = root / "out"
            archive = root / "trainer_archive"
            source_dir.mkdir()
            output_dir.mkdir()
            gate = source_dir / "evidence_bundle.json"
            preflight = output_dir / "trainer_preflight.json"
            launch_check = output_dir / "trainer_launch_check.json"
            write_passed_evidence_bundle(gate)
            self.assertEqual(
                run_cli(
                    [
                        "trainer-preflight",
                        "--gate",
                        str(gate),
                        "--evidence-bundle",
                        str(gate),
                        "--trainer-command",
                        "python train.py --bundle ../src/evidence_bundle.json",
                        "--out",
                        str(preflight),
                    ]
                ),
                0,
            )
            self.assertEqual(run_cli(["trainer-launch-check", "--preflight", str(preflight), "--out", str(launch_check)]), 0)
            stale_bundle = json.loads(gate.read_text(encoding="utf-8"))
            stale_bundle["decision"]["summary"] = "Stale mutation after trainer preflight approval."
            stale_bundle["notes"] = ["stale-after-preflight"]
            gate.write_text(json.dumps(stale_bundle, sort_keys=True) + "\n", encoding="utf-8")

            code = run_cli(
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
            )

            self.assertEqual(code, 1)
            manifest = json.loads((archive / "trainer_archive.json").read_text(encoding="utf-8"))
            self.assertFalse(manifest["passed"])
            self.assertFalse(manifest["self_contained"])
            missing_roles = {item["role"] for item in manifest["missing"]}
            self.assertIn("gate", missing_roles)
            self.assertIn("trainer_artifact", missing_roles)
            self.assertTrue(any("sha256 does not match preflight record" in item["reason"] for item in manifest["missing"]))
            archived_text = "\n".join(path.read_text(encoding="utf-8") for path in (archive / "artifacts").rglob("*.json"))
            self.assertNotIn("stale-after-preflight", archived_text)
            self.assertEqual(run_cli(["validate", "--trainer-archive", str(archive), "--strict"]), 0)

    def test_trainer_archive_rejects_stale_preflight_directory_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = root / "runs"
            gate = root / "training_gate.json"
            preflight = root / "trainer_preflight.json"
            launch_check = root / "trainer_launch_check.json"
            archive = root / "trainer_archive"
            run_cli(["run", "--scenario", str(ROOT / "scenarios" / "prompt_injection_good.json"), "--out", str(runs / "good")])
            run_cli(["export-rl", "--runs", str(runs), "--out", str(runs / "training_export")])
            self.assertEqual(run_cli(["gate-export", "--training-export", str(runs / "training_export"), "--out", str(gate)]), 0)
            dataset_version = json.loads((runs / "training_export" / "manifest.json").read_text(encoding="utf-8"))["dataset_version"]
            self.assertEqual(
                run_cli(
                    [
                        "trainer-preflight",
                        "--gate",
                        str(gate),
                        "--training-export",
                        str(runs / "training_export"),
                        "--require-dataset-version",
                        dataset_version,
                        "--trainer-command",
                        f"python train.py --dataset {runs / 'training_export'}",
                        "--out",
                        str(preflight),
                    ]
                ),
                0,
            )
            self.assertEqual(run_cli(["trainer-launch-check", "--preflight", str(preflight), "--out", str(launch_check)]), 0)
            late_file = runs / "training_export" / "late_after_preflight.json"
            late_file.write_text(json.dumps({"stale": "directory-mutation-after-preflight"}) + "\n", encoding="utf-8")

            code = run_cli(
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
            )

            self.assertEqual(code, 1)
            manifest = json.loads((archive / "trainer_archive.json").read_text(encoding="utf-8"))
            self.assertFalse(manifest["passed"])
            self.assertFalse(manifest["self_contained"])
            self.assertTrue(
                any(
                    item["role"] == "trainer_artifact"
                    and item.get("name") == "training_export"
                    and "sha256 does not match preflight record" in item["reason"]
                    for item in manifest["missing"]
                )
            )
            self.assertFalse(any(path.name == "late_after_preflight.json" for path in (archive / "artifacts").rglob("*.json")))
            self.assertEqual(run_cli(["validate", "--trainer-archive", str(archive), "--strict"]), 0)

    def test_trainer_preflight_accepts_passed_training_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp) / "runs"
            gate = Path(tmp) / "training_gate.json"
            preflight = Path(tmp) / "trainer_preflight.json"
            run_cli(["run-suite", "--scenarios", str(ROOT / "scenarios"), "--out", str(runs), "--export-rl"])
            dataset_version = json.loads((runs / "training_export" / "manifest.json").read_text(encoding="utf-8"))["dataset_version"]
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
                    "--require-dataset-version",
                    dataset_version,
                    "--trainer-command",
                    f"python train.py --dataset {runs / 'training_export'}",
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
            self.assertEqual(result["required_dataset_versions"], [dataset_version])
            self.assertEqual(result["dataset_selection"][0]["dataset_version"], dataset_version)
            self.assertTrue(result["dataset_selection"][0]["matches_required"])
            self.assertTrue(result["dataset_selection"][0]["redaction_passed"])
            self.assertTrue(result["dataset_selection"][0]["heldout_scenario_exclusive"])
            self.assertIn("action_sft", result["dataset_selection"][0]["trainer_modes"])
            self.assertIn("process_reward", result["dataset_selection"][0]["trainer_modes"])
            self.assertEqual(
                result["dataset_selection"][0]["trainer_views"]["mode_to_view"]["action_sft"],
                "action_sft",
            )
            self.assertEqual(
                result["dataset_selection"][0]["trainer_views"]["mode_to_view"]["process_reward"],
                "process_reward",
            )
            self.assertTrue(result["gates"][0]["validation"]["passed"])
            self.assertEqual(result["metadata"]["launcher"], "dry-run")
            self.assertEqual(result["trainer_command"]["argv"][:2], ["python", "train.py"])
            self.assertEqual(
                result["artifacts"]["training_export"]["tree_hash_algorithm"],
                "sha256(sorted-relative-path-size-file-sha256)",
            )
            self.assertEqual(len(result["artifacts"]["training_export"]["sha256"]), 64)
            self.assertGreater(result["artifacts"]["training_export"]["file_count"], 0)
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
            schema = check_schema_contract(result, name_or_id="trainer_preflight")
            self.assertTrue(schema["passed"], schema["errors"])
            for field_name in ("sha256", "size_bytes"):
                forged = json.loads(json.dumps(result))
                forged["artifacts"]["training_export_sft_jsonl"].pop(field_name)
                forged_schema = check_schema_contract(forged, name_or_id="trainer_preflight")
                self.assertFalse(forged_schema["passed"])
                self.assertTrue(any("oneOf" in error for error in forged_schema["errors"]))
            for field_name in ("sha256", "size_bytes"):
                forged = json.loads(json.dumps(result))
                forged["artifacts"]["training_export"].pop(field_name)
                forged_schema = check_schema_contract(forged, name_or_id="trainer_preflight")
                self.assertFalse(forged_schema["passed"])
                self.assertTrue(any("oneOf" in error for error in forged_schema["errors"]))
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
                        "--require-dataset-version",
                        dataset_version,
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
            self.assertEqual(launch["required_dataset_versions"], [dataset_version])
            self.assertEqual(launch["dataset_selection"][0]["dataset_version"], dataset_version)
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
                        "--preserve-paths",
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
            self.assertGreater(len(result["trainer_inputs"]), 0)
            self.assertGreater(len(result["path_rewrites"]), 0)
            self.assertEqual(result["approved_command"]["argv"][:2], ["python", "train.py"])
            self.assertTrue(result["portable_command"]["approved"])
            self.assertTrue(result["portable_command"]["rewritten"])
            self.assertIn("artifacts/trainer_artifacts", " ".join(result["portable_command"]["argv"]))
            self.assertNotIn(str(runs / "training_export"), " ".join(result["portable_command"]["argv"]))
            contract = result["consumer_contract"]
            self.assertEqual(contract["execution_cwd"], "archive_root")
            self.assertEqual(contract["command_kind"], "advisory_portable_command")
            self.assertTrue(contract["portable_command_available"])
            self.assertTrue(contract["portable_command_rewritten"])
            self.assertEqual(contract["trainer_input_count"], len(result["trainer_inputs"]))
            self.assertEqual(contract["path_rewrite_count"], len(result["path_rewrites"]))
            self.assertTrue(contract["external_code_required"])
            self.assertEqual(contract["external_command_path_count"], len(contract["external_command_paths"]))
            self.assertIn("train.py", {item["path"] for item in contract["external_command_paths"]})
            self.assertEqual(result["metrics"]["trainer_input_count"], len(result["trainer_inputs"]))
            self.assertEqual(result["metrics"]["path_rewrite_count"], len(result["path_rewrites"]))
            self.assertEqual(result["metrics"]["external_command_path_count"], contract["external_command_path_count"])
            self.assertEqual(run_cli(["validate", "--trainer-archive", str(archive), "--strict"]), 0)
            self.assertEqual(run_cli(["schemas", "--check", str(manifest_path)]), 0)

            trainer_code = Path(tmp) / "trainer_code"
            trainer_code.mkdir()
            (trainer_code / "train.py").write_text("print('dry run only')\n", encoding="utf-8")
            archive_check = Path(tmp) / "trainer_archive_check.json"
            self.assertEqual(
                run_cli(
                    [
                        "trainer-archive-check",
                        "--archive",
                        str(archive),
                        "--external-code-root",
                        str(trainer_code),
                        "--out",
                        str(archive_check),
                        "--strict",
                        "--preserve-paths",
                    ]
                ),
                0,
            )
            check = json.loads(archive_check.read_text(encoding="utf-8"))
            self.assertEqual(check["schema_version"], "hfr.trainer_archive_check.v1")
            self.assertTrue(check["passed"])
            self.assertEqual(check["recommendation"], "consumer_ready")
            self.assertEqual(check["metrics"]["missing_external_code_count"], 0)
            self.assertEqual(check["metrics"]["trainer_input_count"], len(result["trainer_inputs"]))
            self.assertEqual(check["metrics"]["trainer_input_available_count"], len(result["trainer_inputs"]))
            external = check["external_code_checks"]
            self.assertEqual(len(external), contract["external_command_path_count"])
            self.assertIn("train.py", {item["path"] for item in external})
            self.assertTrue(all(len(item["sha256"]) == 64 for item in external if item["passed"]))
            self.assertTrue(all("expected_size_bytes" in item for item in check["trainer_input_checks"] if item["passed"]))
            self.assertEqual(run_cli(["schemas", "--check", str(archive_check)]), 0)
            schema = check_schema_contract(check, name_or_id="trainer_archive_check")
            self.assertTrue(schema["passed"], schema["errors"])
            forged_external = json.loads(json.dumps(check))
            passed_external = next(item for item in forged_external["external_code_checks"] if item["passed"])
            passed_external.pop("sha256")
            forged_external_schema = check_schema_contract(forged_external, name_or_id="trainer_archive_check")
            self.assertFalse(forged_external_schema["passed"])
            self.assertIn("expected exactly one matching schema from oneOf, got 0", "\n".join(forged_external_schema["errors"]))
            forged_external = json.loads(json.dumps(check))
            passed_external = next(item for item in forged_external["external_code_checks"] if item["passed"])
            passed_external.pop("size_bytes")
            forged_external_schema = check_schema_contract(forged_external, name_or_id="trainer_archive_check")
            self.assertFalse(forged_external_schema["passed"])
            self.assertIn("expected exactly one matching schema from oneOf, got 0", "\n".join(forged_external_schema["errors"]))
            passed_input = next(item for item in check["trainer_input_checks"] if item["passed"] and item["kind"] == "file")
            self.assertEqual(len(passed_input["sha256"]), 64)
            self.assertIn("size_bytes", passed_input)
            for field_name in ("sha256", "size_bytes", "expected_sha256", "expected_size_bytes"):
                forged = json.loads(json.dumps(check))
                next(item for item in forged["trainer_input_checks"] if item["passed"] and item["kind"] == "file").pop(
                    field_name
                )
                forged_schema = check_schema_contract(forged, name_or_id="trainer_archive_check")
                self.assertFalse(forged_schema["passed"])
                self.assertTrue(any("oneOf" in error for error in forged_schema["errors"]))
            forged = json.loads(json.dumps(check))
            next(item for item in forged["trainer_input_checks"] if item["passed"] and item["kind"] == "file")[
                "expected_sha256"
            ] = "not-a-hash"
            forged_schema = check_schema_contract(forged, name_or_id="trainer_archive_check")
            self.assertFalse(forged_schema["passed"])
            self.assertTrue(any("oneOf" in error for error in forged_schema["errors"]))
            passed_directory = next(
                item for item in check["trainer_input_checks"] if item["passed"] and item["kind"] == "directory"
            )
            self.assertEqual(len(passed_directory["sha256"]), 64)
            self.assertIn("size_bytes", passed_directory)
            self.assertIn("file_count", passed_directory)
            self.assertIn("expected_file_count", passed_directory)
            for field_name in ("file_count", "expected_file_count"):
                forged = json.loads(json.dumps(check))
                next(
                    item
                    for item in forged["trainer_input_checks"]
                    if item["passed"] and item["kind"] == "directory"
                ).pop(field_name)
                forged_schema = check_schema_contract(forged, name_or_id="trainer_archive_check")
                self.assertFalse(forged_schema["passed"])
                self.assertTrue(any("oneOf" in error for error in forged_schema["errors"]))
            failed_input = json.loads(json.dumps(check))
            failed_input["trainer_input_checks"][0]["passed"] = False
            failed_input["trainer_input_checks"][0].pop("size_bytes")
            failed_input["trainer_input_checks"][0].pop("expected_size_bytes")
            failed_input_schema = check_schema_contract(failed_input, name_or_id="trainer_archive_check")
            self.assertTrue(failed_input_schema["passed"], failed_input_schema["errors"])
            self.assertEqual(run_cli(["validate", "--trainer-archive-check", str(archive_check), "--strict"]), 0)

            missing_external_path_check = Path(tmp) / "trainer_archive_check_missing_external_path.json"
            missing_external_path_summary = Path(tmp) / "trainer_archive_check_missing_external_path_summary.json"
            forged = json.loads(json.dumps(check))
            next(item for item in forged["external_code_checks"] if item["passed"])["resolved_path"] = str(
                Path(tmp) / "missing_train.py"
            )
            missing_external_path_check.write_text(json.dumps(forged, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            code = run_cli(
                [
                    "validate",
                    "--trainer-archive-check",
                    str(missing_external_path_check),
                    "--strict",
                    "--out",
                    str(missing_external_path_summary),
                ]
            )
            self.assertEqual(code, 1)
            validation = json.loads(missing_external_path_summary.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in validation["targets"] for error in target["errors"])
            self.assertIn("trainer_archive_check.external_code_checks", errors)
            self.assertIn("resolved_path must resolve to an existing file on disk", errors)

            missing_input_path_check = Path(tmp) / "trainer_archive_check_missing_input_path.json"
            missing_input_path_summary = Path(tmp) / "trainer_archive_check_missing_input_path_summary.json"
            forged = json.loads(json.dumps(check))
            next(item for item in forged["trainer_input_checks"] if item["passed"] and item["kind"] == "file")[
                "resolved_path"
            ] = str(Path(tmp) / "missing_input.json")
            missing_input_path_check.write_text(json.dumps(forged, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            code = run_cli(
                [
                    "validate",
                    "--trainer-archive-check",
                    str(missing_input_path_check),
                    "--strict",
                    "--out",
                    str(missing_input_path_summary),
                ]
            )
            self.assertEqual(code, 1)
            validation = json.loads(missing_input_path_summary.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in validation["targets"] for error in target["errors"])
            self.assertIn("trainer_archive_check.trainer_input_checks", errors)
            self.assertIn("resolved_path must resolve to an existing file on disk", errors)

            forged_check = Path(tmp) / "trainer_archive_check_forged_input_size.json"
            forged_summary = Path(tmp) / "trainer_archive_check_forged_input_size_summary.json"
            forged = json.loads(json.dumps(check))
            for item in forged["trainer_input_checks"]:
                if item["passed"]:
                    item["resolved_path"] = "<redacted:trainer-input>"
                    item["size_bytes"] += 1
                    break
            forged_check.write_text(json.dumps(forged, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            code = run_cli(
                [
                    "validate",
                    "--trainer-archive-check",
                    str(forged_check),
                    "--strict",
                    "--out",
                    str(forged_summary),
                ]
            )
            self.assertEqual(code, 1)
            validation = json.loads(forged_summary.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in validation["targets"] for error in target["errors"])
            self.assertIn("trainer_archive_check.trainer_input_checks", errors)
            self.assertIn("size_bytes", errors)

            consumer_plan = Path(tmp) / "trainer_consumer_plan.json"
            self.assertEqual(
                run_cli(
                    [
                        "trainer-consumer-plan",
                        "--archive-check",
                        str(archive_check),
                        "--out",
                        str(consumer_plan),
                        "--strict",
                        "--preserve-paths",
                    ]
                ),
                0,
            )
            plan = json.loads(consumer_plan.read_text(encoding="utf-8"))
            self.assertEqual(plan["schema_version"], "hfr.trainer_consumer_plan.v1")
            self.assertTrue(plan["passed"])
            self.assertEqual(plan["recommendation"], "ready_for_external_trainer")
            self.assertFalse(plan["handoff_contract"]["flight_recorder_executed_command"])
            self.assertTrue(plan["handoff_contract"]["runner_owns_execution"])
            self.assertEqual(plan["execution"]["execution_cwd"], "archive_root")
            self.assertEqual(plan["execution"]["command_argv"][:2], ["python", "train.py"])
            self.assertEqual(plan["execution"]["command_shell"], shlex.join(plan["execution"]["command_argv"]))
            self.assertTrue(any("command_argv is canonical" in note for note in plan["handoff_contract"]["notes"]))
            self.assertTrue(any("Local path integrity is rechecked" in note for note in plan["notes"]))
            self.assertIn("train.py", {item["path"] for item in plan["execution"]["external_code_files"]})
            self.assertEqual(plan["metrics"]["trainer_input_count"], len(result["trainer_inputs"]))
            self.assertEqual(plan["metrics"]["external_code_file_count"], contract["external_command_path_count"])
            self.assertEqual(run_cli(["schemas", "--check", str(consumer_plan)]), 0)
            schema = check_schema_contract(plan, name_or_id="trainer_consumer_plan")
            self.assertTrue(schema["passed"], schema["errors"])
            forged = json.loads(json.dumps(plan))
            forged["provider_console_url"] = "redacted-provider-console"
            forged["checks"][0]["provider_call"] = "forged"
            forged["validation"]["credential_hint"] = "redacted"
            forged["source_archive_check"]["local_source_path"] = "redacted-source-path"
            forged["execution"]["trainer_process_pid"] = 123
            forged["execution"]["external_code_files"][0]["execution_receipt"] = "not-created"
            forged["execution"]["trainer_inputs"][0]["credential_value"] = "redacted"
            forged["handoff_contract"]["cloud_job_url"] = "redacted-cloud-job-url"
            forged["metrics"]["cloud_cost_incurred_usd"] = 0
            forged_schema = check_schema_contract(forged, name_or_id="trainer_consumer_plan")
            self.assertFalse(forged_schema["passed"])
            schema_errors = "\n".join(forged_schema["errors"])
            for field_name in (
                "provider_console_url",
                "provider_call",
                "credential_hint",
                "local_source_path",
                "trainer_process_pid",
                "execution_receipt",
                "credential_value",
                "cloud_job_url",
                "cloud_cost_incurred_usd",
            ):
                self.assertIn(field_name, schema_errors)
            forged_unknown_plan = Path(tmp) / "trainer_consumer_plan_forged_side_effect_fields.json"
            forged_unknown_summary = Path(tmp) / "trainer_consumer_plan_forged_side_effect_fields_summary.json"
            forged_unknown_plan.write_text(json.dumps(forged, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            code = run_cli(
                [
                    "validate",
                    "--trainer-consumer-plan",
                    str(forged_unknown_plan),
                    "--strict",
                    "--out",
                    str(forged_unknown_summary),
                ]
            )
            self.assertEqual(code, 1)
            validation = json.loads(forged_unknown_summary.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in validation["targets"] for error in target["errors"])
            self.assertIn("trainer_consumer_plan contains unknown field(s): ['provider_console_url'].", errors)
            self.assertIn("trainer_consumer_plan.checks[0] contains unknown field(s): ['provider_call'].", errors)
            self.assertIn("trainer_consumer_plan.validation contains unknown field(s): ['credential_hint'].", errors)
            self.assertIn(
                "trainer_consumer_plan.source_archive_check contains unknown field(s): ['local_source_path'].",
                errors,
            )
            self.assertIn("trainer_consumer_plan.execution contains unknown field(s): ['trainer_process_pid'].", errors)
            self.assertIn(
                "trainer_consumer_plan.execution.external_code_files[0] contains unknown field(s): ['execution_receipt'].",
                errors,
            )
            self.assertIn(
                "trainer_consumer_plan.execution.trainer_inputs[0] contains unknown field(s): ['credential_value'].",
                errors,
            )
            self.assertIn(
                "trainer_consumer_plan.handoff_contract contains unknown field(s): ['cloud_job_url'].",
                errors,
            )
            self.assertIn("trainer_consumer_plan.metrics contains unknown field(s): ['cloud_cost_incurred_usd'].", errors)
            passed_external = next(item for item in plan["execution"]["external_code_files"] if item["passed"])
            self.assertEqual(len(passed_external["sha256"]), 64)
            self.assertIn("size_bytes", passed_external)
            for field_name in ("sha256", "size_bytes"):
                forged = json.loads(json.dumps(plan))
                next(item for item in forged["execution"]["external_code_files"] if item["passed"]).pop(field_name)
                forged_schema = check_schema_contract(forged, name_or_id="trainer_consumer_plan")
                self.assertFalse(forged_schema["passed"])
                self.assertTrue(any("oneOf" in error for error in forged_schema["errors"]))
            failed_external = json.loads(json.dumps(plan))
            failed_external["execution"]["external_code_files"][0]["passed"] = False
            failed_external["execution"]["external_code_files"][0].pop("sha256")
            failed_external["execution"]["external_code_files"][0].pop("size_bytes")
            failed_external_schema = check_schema_contract(failed_external, name_or_id="trainer_consumer_plan")
            self.assertTrue(failed_external_schema["passed"], failed_external_schema["errors"])
            passed_input = next(item for item in plan["execution"]["trainer_inputs"] if item["passed"])
            self.assertEqual(len(passed_input["sha256"]), 64)
            self.assertIn("size_bytes", passed_input)
            for field_name in ("sha256", "size_bytes", "expected_sha256", "expected_size_bytes"):
                forged = json.loads(json.dumps(plan))
                next(item for item in forged["execution"]["trainer_inputs"] if item["passed"]).pop(field_name)
                forged_schema = check_schema_contract(forged, name_or_id="trainer_consumer_plan")
                self.assertFalse(forged_schema["passed"])
                self.assertTrue(any("oneOf" in error for error in forged_schema["errors"]))
            forged = json.loads(json.dumps(plan))
            next(item for item in forged["execution"]["trainer_inputs"] if item["passed"] and item["kind"] == "file")[
                "expected_sha256"
            ] = "not-a-hash"
            forged_schema = check_schema_contract(forged, name_or_id="trainer_consumer_plan")
            self.assertFalse(forged_schema["passed"])
            self.assertTrue(any("oneOf" in error for error in forged_schema["errors"]))
            passed_directory = next(
                item for item in plan["execution"]["trainer_inputs"] if item["passed"] and item["kind"] == "directory"
            )
            self.assertEqual(len(passed_directory["sha256"]), 64)
            self.assertIn("file_count", passed_directory)
            self.assertIn("expected_file_count", passed_directory)
            forged = json.loads(json.dumps(plan))
            next(
                item for item in forged["execution"]["trainer_inputs"] if item["passed"] and item["kind"] == "directory"
            )["expected_sha256"] = "not-a-hash"
            forged_schema = check_schema_contract(forged, name_or_id="trainer_consumer_plan")
            self.assertFalse(forged_schema["passed"])
            self.assertTrue(any("oneOf" in error for error in forged_schema["errors"]))
            failed_input = json.loads(json.dumps(plan))
            failed_input["execution"]["trainer_inputs"][0]["passed"] = False
            failed_input["execution"]["trainer_inputs"][0].pop("sha256")
            failed_input["execution"]["trainer_inputs"][0].pop("size_bytes")
            failed_input["execution"]["trainer_inputs"][0]["expected_sha256"] = ""
            failed_input_schema = check_schema_contract(failed_input, name_or_id="trainer_consumer_plan")
            self.assertTrue(failed_input_schema["passed"], failed_input_schema["errors"])
            malformed_failed_input = json.loads(json.dumps(plan))
            malformed_failed_input["execution"]["trainer_inputs"][0]["passed"] = False
            malformed_failed_input["execution"]["trainer_inputs"][0]["expected_sha256"] = "not-a-hash"
            malformed_failed_input_schema = check_schema_contract(
                malformed_failed_input,
                name_or_id="trainer_consumer_plan",
            )
            self.assertFalse(malformed_failed_input_schema["passed"])
            self.assertTrue(any("expected_sha256" in error for error in malformed_failed_input_schema["errors"]))
            self.assertEqual(run_cli(["validate", "--trainer-consumer-plan", str(consumer_plan), "--strict"]), 0)

            missing_consumer_external_path_plan = Path(tmp) / "trainer_consumer_plan_missing_external_path.json"
            missing_consumer_external_path_summary = Path(tmp) / "trainer_consumer_plan_missing_external_path_summary.json"
            forged = json.loads(json.dumps(plan))
            next(item for item in forged["execution"]["external_code_files"] if item["passed"])["resolved_path"] = str(
                Path(tmp) / "missing_consumer_train.py"
            )
            missing_consumer_external_path_plan.write_text(json.dumps(forged, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            code = run_cli(
                [
                    "validate",
                    "--trainer-consumer-plan",
                    str(missing_consumer_external_path_plan),
                    "--strict",
                    "--out",
                    str(missing_consumer_external_path_summary),
                ]
            )
            self.assertEqual(code, 1)
            validation = json.loads(missing_consumer_external_path_summary.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in validation["targets"] for error in target["errors"])
            self.assertIn("trainer_consumer_plan.execution.external_code_files", errors)
            self.assertIn("resolved_path must resolve to an existing file on disk", errors)

            redacted_consumer_external_path_plan = Path(tmp) / "trainer_consumer_plan_redacted_external_path.json"
            forged = json.loads(json.dumps(plan))
            redacted_external = next(item for item in forged["execution"]["external_code_files"] if item["passed"])
            redacted_external["resolved_path"] = "<redacted:external-code>"
            redacted_external["size_bytes"] += 1
            redacted_consumer_external_path_plan.write_text(json.dumps(forged, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            self.assertEqual(
                run_cli(["validate", "--trainer-consumer-plan", str(redacted_consumer_external_path_plan), "--strict"]),
                0,
            )

            missing_consumer_input_path_plan = Path(tmp) / "trainer_consumer_plan_missing_input_path.json"
            missing_consumer_input_path_summary = Path(tmp) / "trainer_consumer_plan_missing_input_path_summary.json"
            forged = json.loads(json.dumps(plan))
            next(item for item in forged["execution"]["trainer_inputs"] if item["passed"] and item["kind"] == "file")[
                "resolved_path"
            ] = str(Path(tmp) / "missing_consumer_input.json")
            missing_consumer_input_path_plan.write_text(json.dumps(forged, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            code = run_cli(
                [
                    "validate",
                    "--trainer-consumer-plan",
                    str(missing_consumer_input_path_plan),
                    "--strict",
                    "--out",
                    str(missing_consumer_input_path_summary),
                ]
            )
            self.assertEqual(code, 1)
            validation = json.loads(missing_consumer_input_path_summary.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in validation["targets"] for error in target["errors"])
            self.assertIn("trainer_consumer_plan.execution.trainer_inputs", errors)
            self.assertIn("resolved_path must resolve to an existing file on disk", errors)

            redacted_consumer_input_path_plan = Path(tmp) / "trainer_consumer_plan_redacted_input_path.json"
            forged = json.loads(json.dumps(plan))
            redacted_input = next(item for item in forged["execution"]["trainer_inputs"] if item["passed"] and item["kind"] == "file")
            redacted_input["resolved_path"] = "<redacted:trainer-input>"
            redacted_input["size_bytes"] += 1
            redacted_input["expected_size_bytes"] += 1
            redacted_consumer_input_path_plan.write_text(json.dumps(forged, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            self.assertEqual(
                run_cli(["validate", "--trainer-consumer-plan", str(redacted_consumer_input_path_plan), "--strict"]),
                0,
            )

            missing_source_plan = Path(tmp) / "trainer_consumer_plan_missing_source.json"
            missing_source_summary = Path(tmp) / "trainer_consumer_plan_missing_source_summary.json"
            forged = json.loads(json.dumps(plan))
            forged["source_archive_check"]["path"] = str(Path(tmp) / "missing_archive_check.json")
            missing_source_plan.write_text(json.dumps(forged, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            code = run_cli(
                [
                    "validate",
                    "--trainer-consumer-plan",
                    str(missing_source_plan),
                    "--strict",
                    "--out",
                    str(missing_source_summary),
                ]
            )
            self.assertEqual(code, 1)
            validation = json.loads(missing_source_summary.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in validation["targets"] for error in target["errors"])
            self.assertIn("trainer_consumer_plan.source_archive_check.path must resolve to an existing file", errors)

            missing_fingerprint_plan = Path(tmp) / "trainer_consumer_plan_missing_source_fingerprint.json"
            missing_fingerprint_summary = Path(tmp) / "trainer_consumer_plan_missing_source_fingerprint_summary.json"
            forged = json.loads(json.dumps(plan))
            forged["source_archive_check"].pop("size_bytes")
            forged["source_archive_check"].pop("sha256")
            missing_fingerprint_plan.write_text(json.dumps(forged, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            code = run_cli(
                [
                    "validate",
                    "--trainer-consumer-plan",
                    str(missing_fingerprint_plan),
                    "--strict",
                    "--out",
                    str(missing_fingerprint_summary),
                ]
            )
            self.assertEqual(code, 1)
            validation = json.loads(missing_fingerprint_summary.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in validation["targets"] for error in target["errors"])
            self.assertIn("trainer_consumer_plan.source_archive_check.size_bytes must be present", errors)
            self.assertIn("trainer_consumer_plan.source_archive_check.sha256 must be present", errors)

            stale_source_plan = Path(tmp) / "trainer_consumer_plan_stale_source.json"
            stale_source_summary = Path(tmp) / "trainer_consumer_plan_stale_source_summary.json"
            forged = json.loads(json.dumps(plan))
            forged["source_archive_check"]["size_bytes"] += 1
            forged["source_archive_check"]["sha256"] = "0" * 64
            stale_source_plan.write_text(json.dumps(forged, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            code = run_cli(
                [
                    "validate",
                    "--trainer-consumer-plan",
                    str(stale_source_plan),
                    "--strict",
                    "--out",
                    str(stale_source_summary),
                ]
            )
            self.assertEqual(code, 1)
            validation = json.loads(stale_source_summary.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in validation["targets"] for error in target["errors"])
            self.assertIn("trainer_consumer_plan.source_archive_check.size_bytes does not match path", errors)
            self.assertIn("trainer_consumer_plan.source_archive_check.sha256 does not match path", errors)

            symlink_source_plan = Path(tmp) / "trainer_consumer_plan_symlink_source.json"
            symlink_source_summary = Path(tmp) / "trainer_consumer_plan_symlink_source_summary.json"
            source_link = Path(tmp) / "trainer_archive_check_link.json"
            try:
                source_link.symlink_to(archive_check)
            except (NotImplementedError, OSError) as exc:
                self.skipTest(f"symlink unavailable: {exc}")
            forged = json.loads(json.dumps(plan))
            forged["source_archive_check"]["path"] = source_link.name
            symlink_source_plan.write_text(json.dumps(forged, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            code = run_cli(
                [
                    "validate",
                    "--trainer-consumer-plan",
                    str(symlink_source_plan),
                    "--strict",
                    "--out",
                    str(symlink_source_summary),
                ]
            )
            self.assertEqual(code, 1)
            validation = json.loads(symlink_source_summary.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in validation["targets"] for error in target["errors"])
            self.assertIn(
                "trainer_consumer_plan.source_archive_check.path must resolve to a regular non-symlink file",
                errors,
            )

            broken_symlink_source_plan = Path(tmp) / "trainer_consumer_plan_broken_symlink_source.json"
            broken_symlink_source_summary = Path(tmp) / "trainer_consumer_plan_broken_symlink_source_summary.json"
            broken_source_link = Path(tmp) / "broken_archive_check_link.json"
            try:
                broken_source_link.symlink_to(Path(tmp) / "missing_archive_check_target.json")
            except (NotImplementedError, OSError) as exc:
                self.skipTest(f"symlink unavailable: {exc}")
            forged = json.loads(json.dumps(plan))
            forged["source_archive_check"]["path"] = broken_source_link.name
            broken_symlink_source_plan.write_text(json.dumps(forged, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            code = run_cli(
                [
                    "validate",
                    "--trainer-consumer-plan",
                    str(broken_symlink_source_plan),
                    "--strict",
                    "--out",
                    str(broken_symlink_source_summary),
                ]
            )
            self.assertEqual(code, 1)
            validation = json.loads(broken_symlink_source_summary.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in validation["targets"] for error in target["errors"])
            self.assertIn(
                "trainer_consumer_plan.source_archive_check.path must resolve to a regular non-symlink file",
                errors,
            )

            symlink_parent_source_plan = Path(tmp) / "trainer_consumer_plan_symlink_parent_source.json"
            symlink_parent_source_summary = Path(tmp) / "trainer_consumer_plan_symlink_parent_source_summary.json"
            linked_target = Path(tmp) / "linked_target"
            linked_target.mkdir()
            linked_archive_check = linked_target / archive_check.name
            linked_archive_check.write_text(archive_check.read_text(encoding="utf-8"), encoding="utf-8")
            linked_parent = Path(tmp) / "linked_artifacts"
            try:
                linked_parent.symlink_to(linked_target, target_is_directory=True)
            except (NotImplementedError, OSError) as exc:
                self.skipTest(f"symlink unavailable: {exc}")
            forged = json.loads(json.dumps(plan))
            forged["source_archive_check"]["path"] = str(Path(linked_parent.name) / archive_check.name)
            symlink_parent_source_plan.write_text(json.dumps(forged, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            code = run_cli(
                [
                    "validate",
                    "--trainer-consumer-plan",
                    str(symlink_parent_source_plan),
                    "--strict",
                    "--out",
                    str(symlink_parent_source_summary),
                ]
            )
            self.assertEqual(code, 1)
            validation = json.loads(symlink_parent_source_summary.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in validation["targets"] for error in target["errors"])
            self.assertIn(
                "trainer_consumer_plan.source_archive_check.path must resolve to a regular non-symlink file",
                errors,
            )

            forged_plan = Path(tmp) / "trainer_consumer_plan_forged_validation.json"
            forged_summary = Path(tmp) / "trainer_consumer_plan_forged_validation_summary.json"
            forged = json.loads(json.dumps(plan))
            forged["validation"]["passed"] = True
            forged["validation"]["error_count"] = 1
            forged["validation"]["errors"] = ["forged validation failure"]
            forged["metrics"]["archive_check_error_count"] = 1
            forged_plan.write_text(json.dumps(forged, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            code = run_cli(
                [
                    "validate",
                    "--trainer-consumer-plan",
                    str(forged_plan),
                    "--strict",
                    "--out",
                    str(forged_summary),
                ]
            )
            self.assertEqual(code, 1)
            validation = json.loads(forged_summary.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in validation["targets"] for error in target["errors"])
            self.assertIn("trainer_consumer_plan.validation.passed", errors)

            wrapper_receipt = Path(tmp) / "trainer_wrapper_dry_run.json"
            wrapper = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "examples" / "trainer-wrapper" / "consume_trainer_plan.py"),
                    "--plan",
                    str(consumer_plan),
                    "--out",
                    str(wrapper_receipt),
                    "--strict",
                ],
                check=False,
                cwd=ROOT,
                capture_output=True,
                text=True,
            )
            self.assertEqual(wrapper.returncode, 0, wrapper.stderr + wrapper.stdout)
            receipt = json.loads(wrapper_receipt.read_text(encoding="utf-8"))
            self.assertEqual(receipt["schema_version"], "hfr.example_trainer_wrapper_dry_run.v1")
            self.assertTrue(receipt["passed"])
            self.assertEqual(receipt["recommendation"], "dry_run_ready")
            self.assertEqual(receipt["would_run"]["argv"][:2], ["python", "train.py"])
            self.assertEqual(receipt["metrics"]["trainer_input_count"], len(result["trainer_inputs"]))
            self.assertEqual(receipt["metrics"]["external_code_file_count"], contract["external_command_path_count"])
            self.assertEqual(run_cli(["schemas", "--check", str(wrapper_receipt)]), 0)
            schema = check_schema_contract(receipt, name_or_id="trainer_wrapper_dry_run")
            self.assertTrue(schema["passed"], schema["errors"])
            wrapper_external = next(item for item in receipt["inputs"]["external_code_files"] if item["passed"])
            self.assertEqual(len(wrapper_external["sha256"]), 64)
            self.assertIn("size_bytes", wrapper_external)
            for field_name in ("sha256", "size_bytes"):
                forged = json.loads(json.dumps(receipt))
                next(item for item in forged["inputs"]["external_code_files"] if item["passed"]).pop(field_name)
                forged_schema = check_schema_contract(forged, name_or_id="trainer_wrapper_dry_run")
                self.assertFalse(forged_schema["passed"])
                self.assertTrue(any("oneOf" in error for error in forged_schema["errors"]))
                self._assert_trainer_wrapper_validation_rejects(
                    Path(tmp),
                    forged,
                    f"trainer_wrapper_dry_run_missing_external_{field_name}",
                    "trainer_wrapper_dry_run.inputs.external_code_files",
                    field_name,
                )
            failed_external = json.loads(json.dumps(receipt))
            failed_external["inputs"]["external_code_files"][0]["passed"] = False
            failed_external["inputs"]["external_code_files"][0].pop("size_bytes")
            failed_external_schema = check_schema_contract(failed_external, name_or_id="trainer_wrapper_dry_run")
            self.assertTrue(failed_external_schema["passed"], failed_external_schema["errors"])
            missing_external_path = json.loads(json.dumps(receipt))
            next(item for item in missing_external_path["inputs"]["external_code_files"] if item["passed"])["resolved_path"] = str(
                Path(tmp) / "missing_wrapper_external.py"
            )
            self._assert_trainer_wrapper_validation_rejects(
                Path(tmp),
                missing_external_path,
                "trainer_wrapper_dry_run_missing_external_path",
                "trainer_wrapper_dry_run.inputs.external_code_files",
                "resolved_path must resolve to an existing file on disk",
            )
            redacted_external_path = json.loads(json.dumps(receipt))
            redacted_external = next(item for item in redacted_external_path["inputs"]["external_code_files"] if item["passed"])
            redacted_external["resolved_path"] = "<redacted:external-code>"
            redacted_external["size_bytes"] += 1
            redacted_external_receipt = Path(tmp) / "trainer_wrapper_dry_run_redacted_external_path.json"
            redacted_external_receipt.write_text(json.dumps(redacted_external_path, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            self.assertEqual(
                run_cli(["validate", "--trainer-wrapper-dry-run", str(redacted_external_receipt), "--strict"]),
                0,
            )
            wrapper_input = next(item for item in receipt["inputs"]["trainer_inputs"] if item["passed"] and item["kind"] == "file")
            self.assertEqual(len(wrapper_input["sha256"]), 64)
            self.assertIn("size_bytes", wrapper_input)
            for field_name in ("sha256", "size_bytes", "expected_sha256", "expected_size_bytes"):
                forged = json.loads(json.dumps(receipt))
                next(item for item in forged["inputs"]["trainer_inputs"] if item["passed"] and item["kind"] == "file").pop(field_name)
                forged_schema = check_schema_contract(forged, name_or_id="trainer_wrapper_dry_run")
                self.assertFalse(forged_schema["passed"])
                self.assertTrue(any("oneOf" in error for error in forged_schema["errors"]))
                if field_name in {"sha256", "size_bytes"}:
                    self._assert_trainer_wrapper_validation_rejects(
                        Path(tmp),
                        forged,
                        f"trainer_wrapper_dry_run_missing_input_{field_name}",
                        "trainer_wrapper_dry_run.inputs.trainer_inputs",
                        field_name,
                    )
            forged = json.loads(json.dumps(receipt))
            next(item for item in forged["inputs"]["trainer_inputs"] if item["passed"] and item["kind"] == "file")[
                "expected_sha256"
            ] = "not-a-hash"
            forged_schema = check_schema_contract(forged, name_or_id="trainer_wrapper_dry_run")
            self.assertFalse(forged_schema["passed"])
            self.assertTrue(any("oneOf" in error for error in forged_schema["errors"]))
            missing_input_path = json.loads(json.dumps(receipt))
            next(item for item in missing_input_path["inputs"]["trainer_inputs"] if item["passed"] and item["kind"] == "file")[
                "resolved_path"
            ] = str(Path(tmp) / "missing_wrapper_input.json")
            self._assert_trainer_wrapper_validation_rejects(
                Path(tmp),
                missing_input_path,
                "trainer_wrapper_dry_run_missing_input_path",
                "trainer_wrapper_dry_run.inputs.trainer_inputs",
                "resolved_path must resolve to an existing file on disk",
            )
            redacted_input_path = json.loads(json.dumps(receipt))
            redacted_input = next(
                item for item in redacted_input_path["inputs"]["trainer_inputs"] if item["passed"] and item["kind"] == "file"
            )
            redacted_input["resolved_path"] = "<redacted:trainer-input>"
            redacted_input["size_bytes"] += 1
            redacted_input["expected_size_bytes"] += 1
            redacted_input_receipt = Path(tmp) / "trainer_wrapper_dry_run_redacted_input_path.json"
            redacted_input_receipt.write_text(json.dumps(redacted_input_path, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            self.assertEqual(
                run_cli(["validate", "--trainer-wrapper-dry-run", str(redacted_input_receipt), "--strict"]),
                0,
            )
            wrapper_directory = next(item for item in receipt["inputs"]["trainer_inputs"] if item["passed"] and item["kind"] == "directory")
            self.assertEqual(len(wrapper_directory["sha256"]), 64)
            self.assertIn("size_bytes", wrapper_directory)
            self.assertIn("file_count", wrapper_directory)
            self.assertIn("expected_file_count", wrapper_directory)
            for field_name in ("file_count", "expected_file_count"):
                forged = json.loads(json.dumps(receipt))
                next(item for item in forged["inputs"]["trainer_inputs"] if item["passed"] and item["kind"] == "directory").pop(field_name)
                forged_schema = check_schema_contract(forged, name_or_id="trainer_wrapper_dry_run")
                self.assertFalse(forged_schema["passed"])
                self.assertTrue(any("oneOf" in error for error in forged_schema["errors"]))
                if field_name == "file_count":
                    self._assert_trainer_wrapper_validation_rejects(
                        Path(tmp),
                        forged,
                        "trainer_wrapper_dry_run_missing_input_file_count",
                        "trainer_wrapper_dry_run.inputs.trainer_inputs",
                        "file_count",
                    )
            forged = json.loads(json.dumps(receipt))
            next(
                item for item in forged["inputs"]["trainer_inputs"] if item["passed"] and item["kind"] == "directory"
            )["expected_sha256"] = "not-a-hash"
            forged_schema = check_schema_contract(forged, name_or_id="trainer_wrapper_dry_run")
            self.assertFalse(forged_schema["passed"])
            self.assertTrue(any("oneOf" in error for error in forged_schema["errors"]))
            missing_directory_path = json.loads(json.dumps(receipt))
            next(item for item in missing_directory_path["inputs"]["trainer_inputs"] if item["passed"] and item["kind"] == "directory")[
                "resolved_path"
            ] = str(Path(tmp) / "missing_wrapper_input_dir")
            self._assert_trainer_wrapper_validation_rejects(
                Path(tmp),
                missing_directory_path,
                "trainer_wrapper_dry_run_missing_directory_path",
                "trainer_wrapper_dry_run.inputs.trainer_inputs",
                "resolved_path must resolve to an existing directory on disk",
            )
            failed_input = json.loads(json.dumps(receipt))
            failed_input["inputs"]["trainer_inputs"][0]["passed"] = False
            failed_input["inputs"]["trainer_inputs"][0].pop("size_bytes")
            failed_input["inputs"]["trainer_inputs"][0].pop("expected_size_bytes")
            failed_input["inputs"]["trainer_inputs"][0]["expected_sha256"] = ""
            failed_input_schema = check_schema_contract(failed_input, name_or_id="trainer_wrapper_dry_run")
            self.assertTrue(failed_input_schema["passed"], failed_input_schema["errors"])
            malformed_failed_input = json.loads(json.dumps(receipt))
            malformed_failed_input["inputs"]["trainer_inputs"][0]["passed"] = False
            malformed_failed_input["inputs"]["trainer_inputs"][0]["expected_sha256"] = "not-a-hash"
            malformed_failed_input_schema = check_schema_contract(
                malformed_failed_input,
                name_or_id="trainer_wrapper_dry_run",
            )
            self.assertFalse(malformed_failed_input_schema["passed"])
            self.assertTrue(any("expected_sha256" in error for error in malformed_failed_input_schema["errors"]))
            self.assertEqual(run_cli(["validate", "--trainer-wrapper-dry-run", str(wrapper_receipt), "--strict"]), 0)

            missing_check = Path(tmp) / "trainer_archive_check_missing.json"
            self.assertEqual(
                run_cli(
                    [
                        "trainer-archive-check",
                        "--archive",
                        str(archive),
                        "--external-code-root",
                        str(Path(tmp) / "missing_trainer_code"),
                        "--out",
                        str(missing_check),
                        "--strict",
                        "--preserve-paths",
                    ]
                ),
                1,
            )
            missing = json.loads(missing_check.read_text(encoding="utf-8"))
            self.assertFalse(missing["passed"])
            self.assertEqual(missing["recommendation"], "block_consumer_launch")
            self.assertGreater(missing["metrics"]["missing_external_code_count"], 0)
            self.assertTrue(any(not item["passed"] and "sha256" not in item for item in missing["external_code_checks"]))
            self.assertEqual(run_cli(["schemas", "--check", str(missing_check)]), 0)
            self.assertEqual(run_cli(["validate", "--trainer-archive-check", str(missing_check), "--strict"]), 0)

            blocked_plan = Path(tmp) / "trainer_consumer_plan_blocked.json"
            self.assertEqual(
                run_cli(
                    [
                        "trainer-consumer-plan",
                        "--archive-check",
                        str(missing_check),
                        "--out",
                        str(blocked_plan),
                        "--strict",
                        "--preserve-paths",
                    ]
                ),
                1,
            )
            blocked = json.loads(blocked_plan.read_text(encoding="utf-8"))
            self.assertFalse(blocked["passed"])
            self.assertEqual(blocked["recommendation"], "block_external_trainer")
            self.assertIn("archive_check_passed: passed=False", blocked["blocked_reasons"])
            self.assertEqual(run_cli(["validate", "--trainer-consumer-plan", str(blocked_plan), "--strict"]), 0)

            result["portable_command"]["argv"][-1] = "stale/training_export"
            result["portable_command"]["shell"] = "python train.py --dataset stale/training_export"
            manifest_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            self.assertEqual(run_cli(["validate", "--trainer-archive", str(archive), "--strict"]), 1)

            code, output = run_cli_output(["trainer-launch-check", "--preflight", str(preflight), "--print-command"])
            self.assertEqual(code, 0)
            self.assertEqual(output.strip(), f"python train.py --dataset {runs / 'training_export'}")

    def test_trainer_preflight_blocks_unselected_dataset_version(self):
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
                    "--require-dataset-version",
                    "hfrds-deadbeef",
                    "--trainer-command",
                    f"python train.py --dataset {runs / 'training_export'}",
                    "--out",
                    str(preflight),
                ]
            )

            self.assertEqual(code, 1)
            result = json.loads(preflight.read_text(encoding="utf-8"))
            self.assertFalse(result["passed"])
            self.assertFalse(result["dataset_selection"][0]["matches_required"])
            failed = {check["id"] for check in result["checks"] if check["passed"] is False}
            self.assertIn("dataset_version_matches_required", failed)

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
            self.assertEqual(len(result["gates"][0]["sha256"]), 64)
            self.assertIn("size_bytes", result["gates"][0])
            self.assertTrue(gate_validation["available"])
            self.assertTrue(gate_validation["passed"])
            self.assertTrue(gate_validation["summary_passed"])
            self.assertEqual(gate_validation["target_type"], "improvement_ledger_gate")
            self.assertEqual(gate_validation["source"], str(validation))
            self.assertEqual(result["validation_summaries"][0]["path"], str(validation))
            self.assertEqual(result["validation_summaries"][0]["targets"][0]["type"], "improvement_ledger_gate")
            self.assertEqual(len(result["validation_summaries"][0]["sha256"]), 64)
            self.assertIn("size_bytes", result["validation_summaries"][0])
            schema = check_schema_contract(result, name_or_id="trainer_preflight")
            self.assertTrue(schema["passed"], schema["errors"])
            for field_name in ("sha256", "size_bytes"):
                forged = json.loads(json.dumps(result))
                forged["validation_summaries"][0].pop(field_name)
                forged_schema = check_schema_contract(forged, name_or_id="trainer_preflight")
                self.assertFalse(forged_schema["passed"])
                self.assertTrue(any("oneOf" in error for error in forged_schema["errors"]))
            non_regular = json.loads(json.dumps(result))
            non_regular["validation_summaries"][0]["regular_file"] = False
            non_regular["validation_summaries"][0].pop("sha256")
            non_regular["validation_summaries"][0].pop("size_bytes")
            non_regular_schema = check_schema_contract(non_regular, name_or_id="trainer_preflight")
            self.assertTrue(non_regular_schema["passed"], non_regular_schema["errors"])
            for field_name in ("sha256", "size_bytes"):
                forged = json.loads(json.dumps(result))
                forged["gates"][0].pop(field_name)
                forged_schema = check_schema_contract(forged, name_or_id="trainer_preflight")
                self.assertFalse(forged_schema["passed"])
                self.assertTrue(any("oneOf" in error for error in forged_schema["errors"]))
            missing_gate = json.loads(json.dumps(result))
            missing_gate["gates"][0]["exists"] = False
            missing_gate["gates"][0].pop("sha256")
            missing_gate["gates"][0].pop("size_bytes")
            missing_gate_schema = check_schema_contract(missing_gate, name_or_id="trainer_preflight")
            self.assertTrue(missing_gate_schema["passed"], missing_gate_schema["errors"])
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
            dataset_selection = result["dataset_selection"][0]
            self.assertIn("sft", dataset_selection["trainer_modes"])
            self.assertIn("action_sft", dataset_selection["trainer_modes"])
            self.assertIn("dpo", dataset_selection["trainer_modes"])
            self.assertIn("reward_model", dataset_selection["trainer_modes"])
            self.assertEqual(dataset_selection["trainer_views"]["contract_version"], "hfr.rl.trainer_views.v1")
            self.assertEqual(dataset_selection["trainer_views"]["mode_to_view"]["action_sft"], "reviewed_sft")
            self.assertEqual(dataset_selection["trainer_views"]["mode_to_view"]["dpo"], "reviewed_dpo")

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
            contract = result["schema_contracts"]["training_export_sft_jsonl"]
            self.assertFalse(contract["regular_file"])
            self.assertTrue(contract["symlink"])
            self.assertNotIn("sha256", contract)
            failed_checks = {check["id"] for check in result["checks"] if not check["passed"]}
            self.assertIn("artifact_file_regular", failed_checks)
            schema = check_schema_contract(result, name_or_id="trainer_preflight")
            self.assertTrue(schema["passed"], schema["errors"])
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
            self.assertEqual(len(contract["sha256"]), 64)
            self.assertIn("size_bytes", contract)
            schema = check_schema_contract(result, name_or_id="trainer_preflight")
            self.assertTrue(schema["passed"], schema["errors"])
            for field_name in ("sha256", "size_bytes"):
                forged = json.loads(json.dumps(result))
                forged["schema_contracts"]["training_export_sft_jsonl"].pop(field_name)
                forged_schema = check_schema_contract(forged, name_or_id="trainer_preflight")
                self.assertFalse(forged_schema["passed"])
                self.assertTrue(any("oneOf" in error for error in forged_schema["errors"]))
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
            schema = check_schema_contract(result, name_or_id="trainer_preflight")
            self.assertTrue(schema["passed"], schema["errors"])
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
