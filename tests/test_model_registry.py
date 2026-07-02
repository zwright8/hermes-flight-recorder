from __future__ import annotations

import json
import socket
import subprocess
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from flightrecorder.cli import main
from flightrecorder.model_registry import (
    MODEL_CANDIDATE_SCHEMA_VERSION,
    MODEL_REGISTRY_LINK_COLLECTIONS,
    MODEL_SCOUT_MANIFEST_SCHEMA_VERSION,
    ModelRegistryError,
    build_dry_run_training_plan,
    build_model_compatibility_report,
    build_model_serving_probe_receipt,
    link_model_registry_artifact,
    model_candidate_errors,
    model_serving_probe_receipt_errors,
    move_model_alias,
    new_model_registry,
    register_model_candidate,
    select_model_for_training,
    training_plan_errors,
)
from flightrecorder.schema_registry import check_schema_contract
from flightrecorder.validation import validate_artifacts


def run_cli(args):
    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        return main(args)


def run_cli_output(args):
    stdout = StringIO()
    with redirect_stdout(stdout), redirect_stderr(StringIO()):
        code = main(args)
    return code, stdout.getvalue()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def approved_candidate(candidate_id: str = "local-mock-tiny-chat") -> dict:
    return {
        "schema_version": MODEL_CANDIDATE_SCHEMA_VERSION,
        "candidate_id": candidate_id,
        "model_id": f"local/{candidate_id}",
        "source": {
            "type": "local_fixture",
            "url": f"file://fixtures/models/{candidate_id}",
            "revision": "metadata-only",
        },
        "license": {
            "status": "approved",
            "license_id": "mit",
            "source_url": "https://opensource.org/license/mit/",
            "terms_url": "https://opensource.org/license/mit/",
            "review_status": "approved",
            "accepted_terms": True,
            "training_allowed": True,
            "reviewed_at": "2026-07-02T00:00:00Z",
            "reviewer": "model-layer-test",
            "notes": ["Synthetic metadata-only local fixture."],
        },
        "compatibility": {
            "context_length": 2048,
            "tokenizer": {"status": "metadata_only", "verified": False, "notes": "Tokenizer metadata recorded."},
            "chat_template": {"status": "metadata_only", "verified": False, "notes": "Chat template metadata recorded."},
            "serving": {"status": "metadata_only", "verified": False, "notes": "Serving metadata recorded."},
            "tool_calls": {
                "status": "metadata_only",
                "supported": True,
                "verified": False,
                "notes": "Tool-call support is metadata-only.",
            },
            "structured_outputs": {
                "status": "metadata_only",
                "supported": True,
                "verified": False,
                "notes": "Structured output support is metadata-only.",
            },
            "quantization": {"status": "metadata_only", "verified": False, "notes": "4-bit LoRA-compatible path expected."},
            "memory": {"status": "metadata_only", "verified": False, "notes": "Tiny fixture memory footprint."},
        },
        "notes": ["No weights are downloaded by model-layer tests."],
    }


def unknown_license_candidate() -> dict:
    candidate = approved_candidate("unknown-license-chat")
    candidate["license"] = {
        "status": "unknown",
        "license_id": "unknown",
        "source_url": "https://example.invalid/license",
        "terms_url": "https://example.invalid/terms",
        "review_status": "pending",
        "accepted_terms": False,
        "training_allowed": False,
        "notes": ["Unknown license may be scouted but cannot be selected for training."],
    }
    return candidate


class ModelRegistryTests(unittest.TestCase):
    def test_registry_alias_and_dry_run_plan_accept_approved_license(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate = approved_candidate()
            dataset_manifest = root / "dataset_manifest.json"
            write_json(dataset_manifest, {"schema_version": "hfr.test_dataset_manifest.v1", "dataset_id": "local_mock_dataset_v1"})
            report_path = root / "compatibility_report.json"
            report = build_model_compatibility_report(candidate, out_path=report_path)
            write_json(report_path, report)

            registry = new_model_registry(registry_path=root / "model_registry.json")
            registry = register_model_candidate(registry, candidate)
            registry = link_model_registry_artifact(
                registry,
                entry_id=candidate["candidate_id"],
                collection="datasets",
                artifact_id="local_mock_dataset_v1",
                kind="dataset_manifest",
                status="dry_run_stub",
                path=dataset_manifest,
            )
            registry = link_model_registry_artifact(
                registry,
                entry_id=candidate["candidate_id"],
                collection="compatibility_reports",
                artifact_id="local_mock_tiny_chat_compatibility_report",
                kind="model_compatibility_report",
                status="metadata_report",
                path=report_path,
            )
            registry = move_model_alias(registry, alias="candidate", target=candidate["candidate_id"], reason="unit test")
            entry = select_model_for_training(registry, "candidate")
            self.assertTrue(entry["training_eligible"])
            self.assertEqual(set(entry["links"]), set(MODEL_REGISTRY_LINK_COLLECTIONS))

            plan = build_dry_run_training_plan(
                registry,
                model_ref="candidate",
                dataset_id="local_mock_dataset_v1",
                dataset_manifest=dataset_manifest,
                trainer="local-dry-run",
                mode="sft",
                output_dir=root / "outputs",
                out_path=root / "training_plan.json",
                compatibility_report=report,
                compatibility_report_path=report_path,
            )

            self.assertEqual(training_plan_errors(plan), [])
            self.assertTrue(plan["dry_run"])
            self.assertTrue(plan["no_weight_download"])
            self.assertEqual(plan["dataset"]["registry_link_id"], "local_mock_dataset_v1")
            self.assertEqual(plan["compatibility_report"]["registry_link_id"], "local_mock_tiny_chat_compatibility_report")
            self.assertFalse(plan["gpu_execution"])
            self.assertFalse(plan["execution"]["flight_recorder_imported_heavy_ml"])
            self.assertTrue(check_schema_contract(candidate)["passed"])
            self.assertTrue(check_schema_contract(report)["passed"])
            self.assertTrue(check_schema_contract(plan)["passed"])

    def test_dry_run_plan_requires_registry_links(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate = approved_candidate()
            dataset_manifest = root / "dataset_manifest.json"
            report_path = root / "compatibility_report.json"
            write_json(dataset_manifest, {"schema_version": "hfr.test_dataset_manifest.v1", "dataset_id": "local_mock_dataset_v1"})
            report = build_model_compatibility_report(candidate, out_path=report_path)
            write_json(report_path, report)

            registry = register_model_candidate(new_model_registry(registry_path=root / "model_registry.json"), candidate)
            registry = move_model_alias(registry, alias="candidate", target=candidate["candidate_id"], reason="unit test")
            with self.assertRaisesRegex(ModelRegistryError, "missing linked dataset_manifest artifact"):
                build_dry_run_training_plan(
                    registry,
                    model_ref="candidate",
                    dataset_id="local_mock_dataset_v1",
                    dataset_manifest=dataset_manifest,
                    trainer="local-dry-run",
                    mode="sft",
                    output_dir=root / "outputs",
                    out_path=root / "training_plan.json",
                    compatibility_report=report,
                    compatibility_report_path=report_path,
                )

            registry = link_model_registry_artifact(
                registry,
                entry_id=candidate["candidate_id"],
                collection="datasets",
                artifact_id="local_mock_dataset_v1",
                kind="dataset_manifest",
                status="dry_run_stub",
                path=dataset_manifest,
            )
            with self.assertRaisesRegex(ModelRegistryError, "missing linked compatibility_report artifact"):
                build_dry_run_training_plan(
                    registry,
                    model_ref="candidate",
                    dataset_id="local_mock_dataset_v1",
                    dataset_manifest=dataset_manifest,
                    trainer="local-dry-run",
                    mode="sft",
                    output_dir=root / "outputs",
                    out_path=root / "training_plan.json",
                    compatibility_report=report,
                    compatibility_report_path=report_path,
                )

            registry = link_model_registry_artifact(
                registry,
                entry_id=candidate["candidate_id"],
                collection="compatibility_reports",
                artifact_id="local_mock_tiny_chat_compatibility_report",
                kind="model_compatibility_report",
                status="metadata_report",
                path=report_path,
            )
            plan = build_dry_run_training_plan(
                registry,
                model_ref="candidate",
                dataset_id="local_mock_dataset_v1",
                dataset_manifest=dataset_manifest,
                trainer="local-dry-run",
                mode="sft",
                output_dir=root / "outputs",
                out_path=root / "training_plan.json",
                compatibility_report=report,
                compatibility_report_path=report_path,
            )
            self.assertEqual(training_plan_errors(plan), [])
            self.assertEqual(plan["compatibility_report"]["registry_link_id"], "local_mock_tiny_chat_compatibility_report")

    def test_registry_links_artifacts_with_hashes_and_upserts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate = approved_candidate()
            dataset_manifest = root / "dataset_manifest.json"
            write_json(dataset_manifest, {"dataset_id": "local_mock_dataset_v1"})
            registry = register_model_candidate(new_model_registry(registry_path=root / "model_registry.json"), candidate)

            registry = link_model_registry_artifact(
                registry,
                entry_id=candidate["candidate_id"],
                collection="datasets",
                artifact_id="local_mock_dataset_v1",
                kind="dataset_manifest",
                status="dry_run_stub",
                path=dataset_manifest,
                metadata={"role": "training_input"},
            )
            links = registry["entries"][candidate["candidate_id"]]["links"]
            self.assertEqual(len(links["datasets"]), 1)
            self.assertEqual(links["datasets"][0]["id"], "local_mock_dataset_v1")
            self.assertEqual(len(links["datasets"][0]["sha256"]), 64)

            registry = link_model_registry_artifact(
                registry,
                entry_id=candidate["candidate_id"],
                collection="datasets",
                artifact_id="local_mock_dataset_v1",
                kind="dataset_manifest",
                status="verified",
                path=dataset_manifest,
            )
            links = registry["entries"][candidate["candidate_id"]]["links"]
            self.assertEqual(len(links["datasets"]), 1)
            self.assertEqual(links["datasets"][0]["status"], "verified")
            rows = [row for row in registry["entries"].values()]
            self.assertEqual(rows[0]["links"]["training_runs"], [])
            with self.assertRaises(ModelRegistryError):
                link_model_registry_artifact(
                    registry,
                    entry_id=candidate["candidate_id"],
                    collection="datasets",
                    artifact_id="local_mock_dataset_v1",
                    kind="dataset_manifest",
                    path=dataset_manifest,
                    sha256="0" * 64,
                )
            with self.assertRaises(ModelRegistryError):
                link_model_registry_artifact(
                    registry,
                    entry_id=candidate["candidate_id"],
                    collection="unknown",
                    artifact_id="x",
                    kind="x",
                )

    def test_serving_probe_receipt_avoids_launches_and_links(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate = approved_candidate()
            report_path = root / "compatibility_report.json"
            receipt_path = root / "serving_probe_receipt.json"
            report = build_model_compatibility_report(candidate, out_path=report_path)
            write_json(report_path, report)
            registry = register_model_candidate(new_model_registry(registry_path=root / "model_registry.json"), candidate)
            registry = move_model_alias(registry, alias="candidate", target=candidate["candidate_id"], reason="unit test")
            real_import = __import__
            heavy_modules = {"accelerate", "datasets", "peft", "torch", "transformers", "trl"}

            def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
                if name.split(".", 1)[0] in heavy_modules:
                    raise AssertionError(f"heavy ML import attempted: {name}")
                return real_import(name, globals, locals, fromlist, level)

            with patch("builtins.__import__", side_effect=guarded_import), patch.object(
                socket,
                "create_connection",
                side_effect=AssertionError("network connection attempted"),
            ), patch.object(subprocess, "Popen", side_effect=AssertionError("process launch attempted")):
                receipt = build_model_serving_probe_receipt(
                    registry,
                    model_ref="candidate",
                    out_path=receipt_path,
                    profile_id="local_mock_tiny_chat_metadata",
                    provider="metadata_only",
                    serving_engine="not_launched",
                    base_url="metadata://not-launched",
                    compatibility_report=report,
                    compatibility_report_path=report_path,
                )

            self.assertEqual(model_serving_probe_receipt_errors(receipt), [])
            self.assertTrue(check_schema_contract(receipt)["passed"])
            self.assertEqual(receipt["readiness"], "metadata_recorded")
            self.assertEqual(receipt["recommendation"], "ready_for_external_serving_probe")
            self.assertFalse(receipt["download_policy"]["downloaded_weights"])
            self.assertFalse(receipt["download_policy"]["launched_server"])
            self.assertFalse(receipt["execution"]["flight_recorder_opened_network_connection"])
            self.assertEqual(receipt["summary"]["probe_count"], 7)
            write_json(receipt_path, receipt)

            registry = link_model_registry_artifact(
                registry,
                entry_id=candidate["candidate_id"],
                collection="serving_probes",
                artifact_id="local_mock_tiny_chat_metadata_serving_probe",
                kind="model_serving_probe_receipt",
                status="metadata_receipt",
                path=receipt_path,
            )
            links = registry["entries"][candidate["candidate_id"]]["links"]
            self.assertEqual(len(links["serving_probes"]), 1)
            self.assertEqual(len(links["serving_probes"][0]["sha256"]), 64)

    def test_unknown_license_blocks_training_selection_and_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate = unknown_license_candidate()
            dataset_manifest = root / "dataset_manifest.json"
            write_json(dataset_manifest, {"dataset_id": "local_mock_dataset_v1"})
            registry = register_model_candidate(new_model_registry(registry_path=root / "model_registry.json"), candidate)
            registry = move_model_alias(registry, alias="candidate", target=candidate["candidate_id"], reason="unit test")

            self.assertFalse(registry["entries"][candidate["candidate_id"]]["training_eligible"])
            training_errors = "\n".join(model_candidate_errors(candidate, require_training_eligible=True))
            self.assertIn("status 'unknown' is not approved for training", training_errors)
            with self.assertRaises(ModelRegistryError):
                select_model_for_training(registry, "candidate")
            with self.assertRaises(ModelRegistryError):
                build_dry_run_training_plan(
                    registry,
                    model_ref="candidate",
                    dataset_id="local_mock_dataset_v1",
                    dataset_manifest=dataset_manifest,
                    trainer="local-dry-run",
                    mode="sft",
                    output_dir=root / "outputs",
                    out_path=root / "training_plan.json",
                )

    def test_champion_alias_requires_explicit_rollback(self):
        registry = new_model_registry()
        champion = approved_candidate("champion-fixture")
        rollback = approved_candidate("rollback-fixture")
        registry = register_model_candidate(registry, champion)
        registry = register_model_candidate(registry, rollback)

        with self.assertRaises(ModelRegistryError):
            move_model_alias(registry, alias="champion", target=champion["candidate_id"], reason="missing rollback")

        registry = move_model_alias(
            registry,
            alias="champion",
            target=champion["candidate_id"],
            rollback_target=rollback["candidate_id"],
            reason="unit test champion move",
        )
        self.assertEqual(registry["aliases"]["champion"], champion["candidate_id"])
        self.assertEqual(registry["aliases"]["rollback"], rollback["candidate_id"])

    def test_model_scout_manifest_validates_refs_and_blocks_false_eligibility(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate_path = root / "model_candidates" / "approved.json"
            report_path = root / "compatibility" / "approved_report.json"
            manifest_path = root / "model_scout_manifest.json"
            candidate = approved_candidate()
            report = build_model_compatibility_report(candidate, out_path=report_path)
            write_json(candidate_path, candidate)
            write_json(report_path, report)
            write_json(
                manifest_path,
                scout_manifest(
                    candidate_id=candidate["candidate_id"],
                    manifest_path="model_candidates/approved.json",
                    compatibility_report_path="compatibility/approved_report.json",
                    license_status="approved",
                    review_status="approved",
                    training_selection_eligible=True,
                ),
            )

            summary = validate_artifacts(model_scout_manifest_paths=[manifest_path], strict=True)
            self.assertTrue(summary["passed"], summary["targets"])
            self.assertEqual(summary["targets"][0]["details"]["training_eligible_count"], 1)

            blocked_candidate_path = root / "model_candidates" / "unknown.json"
            blocked_manifest_path = root / "blocked_model_scout_manifest.json"
            blocked_candidate = unknown_license_candidate()
            write_json(blocked_candidate_path, blocked_candidate)
            write_json(
                blocked_manifest_path,
                scout_manifest(
                    candidate_id=blocked_candidate["candidate_id"],
                    manifest_path="model_candidates/unknown.json",
                    compatibility_report_path=None,
                    license_status="unknown",
                    review_status="pending",
                    training_selection_eligible=True,
                ),
            )

            blocked_summary = validate_artifacts(model_scout_manifest_paths=[blocked_manifest_path])
            self.assertFalse(blocked_summary["passed"])
            errors = "\n".join(error for target in blocked_summary["targets"] for error in target["errors"])
            self.assertIn("training_selection_eligible cannot be true", errors)

    def test_cli_register_alias_report_plan_and_validate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate_path = root / "candidate.json"
            report_path = root / "compatibility_report.json"
            registry_path = root / "model_registry.json"
            entry_path = root / "registry_entry.json"
            dataset_manifest = root / "dataset_manifest.json"
            plan_path = root / "training_plan.json"
            receipt_path = root / "serving_probe_receipt.json"
            validation_path = root / "validation.json"
            candidate = approved_candidate()
            write_json(candidate_path, candidate)
            write_json(dataset_manifest, {"schema_version": "hfr.test_dataset_manifest.v1", "dataset_id": "local_mock_dataset_v1"})

            self.assertEqual(
                run_cli(["model-candidate", "validate", str(candidate_path), "--require-training-eligible"]),
                0,
            )
            self.assertEqual(
                run_cli(["model-candidate", "compatibility-report", "--candidate", str(candidate_path), "--out", str(report_path)]),
                0,
            )
            self.assertEqual(
                run_cli(
                    [
                        "model-registry",
                        "register",
                        "--registry",
                        str(registry_path),
                        "--candidate",
                        str(candidate_path),
                        "--entry-out",
                        str(entry_path),
                    ]
                ),
                0,
            )
            self.assertEqual(
                run_cli(
                    [
                        "model-registry",
                        "alias",
                        "--registry",
                        str(registry_path),
                        "--alias",
                        "candidate",
                        "--target",
                        candidate["candidate_id"],
                        "--reason",
                        "unit test candidate selection",
                    ]
                ),
                0,
            )
            list_code, list_output = run_cli_output(["model-registry", "list", "--registry", str(registry_path), "--json"])
            self.assertEqual(list_code, 0)
            self.assertEqual(json.loads(list_output)[0]["aliases"], ["candidate"])
            self.assertEqual(
                run_cli(
                    [
                        "model-registry",
                        "link",
                        "--registry",
                        str(registry_path),
                        "--entry",
                        candidate["candidate_id"],
                        "--collection",
                        "datasets",
                        "--artifact-id",
                        "local_mock_dataset_v1",
                        "--kind",
                        "dataset_manifest",
                        "--status",
                        "dry_run_stub",
                        "--path",
                        str(dataset_manifest),
                        "--entry-out",
                        str(entry_path),
                        "--metadata",
                        "role=training_input",
                    ]
                ),
                0,
            )
            self.assertEqual(
                run_cli(
                    [
                        "model-registry",
                        "link",
                        "--registry",
                        str(registry_path),
                        "--entry",
                        candidate["candidate_id"],
                        "--collection",
                        "compatibility_reports",
                        "--artifact-id",
                        "local_mock_tiny_chat_compatibility_report",
                        "--kind",
                        "model_compatibility_report",
                        "--status",
                        "metadata_report",
                        "--path",
                        str(report_path),
                        "--entry-out",
                        str(entry_path),
                    ]
                ),
                0,
            )
            self.assertEqual(
                run_cli(
                    [
                        "model-registry",
                        "serving-probe-receipt",
                        "--registry",
                        str(registry_path),
                        "--model-ref",
                        "candidate",
                        "--out",
                        str(receipt_path),
                        "--profile-id",
                        "local_mock_tiny_chat_metadata",
                        "--provider",
                        "metadata_only",
                        "--serving-engine",
                        "not_launched",
                        "--base-url",
                        "metadata://not-launched",
                        "--compatibility-report",
                        str(report_path),
                        "--link",
                        "--artifact-id",
                        "local_mock_tiny_chat_metadata_serving_probe",
                        "--entry-out",
                        str(entry_path),
                    ]
                ),
                0,
            )
            self.assertEqual(
                run_cli(
                    [
                        "training-plan",
                        "dry-run",
                        "--registry",
                        str(registry_path),
                        "--model-ref",
                        "candidate",
                        "--dataset-id",
                        "local_mock_dataset_v1",
                        "--dataset-manifest",
                        str(dataset_manifest),
                        "--trainer",
                        "local-dry-run",
                        "--mode",
                        "sft",
                        "--output-dir",
                        str(root / "outputs"),
                        "--out",
                        str(plan_path),
                        "--compatibility-report",
                        str(report_path),
                        "--hyperparameter",
                        "learning_rate=0.0001",
                        "--compute",
                        "target_device=cpu",
                    ]
                ),
                0,
            )
            self.assertEqual(
                run_cli(
                    [
                        "model-registry",
                        "link",
                        "--registry",
                        str(registry_path),
                        "--entry",
                        candidate["candidate_id"],
                        "--collection",
                        "training_runs",
                        "--artifact-id",
                        "local_mock_tiny_chat_sft_dry_run",
                        "--kind",
                        "training_plan",
                        "--status",
                        "dry_run_plan",
                        "--path",
                        str(plan_path),
                        "--entry-out",
                        str(entry_path),
                    ]
                ),
                0,
            )
            self.assertEqual(
                run_cli(
                    [
                        "validate",
                        "--model-candidate",
                        str(candidate_path),
                        "--model-compatibility-report",
                        str(report_path),
                        "--model-serving-probe-receipt",
                        str(receipt_path),
                        "--model-registry-entry",
                        str(entry_path),
                        "--model-registry",
                        str(registry_path),
                        "--training-plan",
                        str(plan_path),
                        "--out",
                        str(validation_path),
                        "--strict",
                    ]
                ),
                0,
            )
            validation = json.loads(validation_path.read_text(encoding="utf-8"))
            self.assertTrue(validation["passed"], validation["targets"])
            self.assertEqual(validation["target_count"], 6)
            registry = json.loads(registry_path.read_text(encoding="utf-8"))
            link_counts = registry["entries"][candidate["candidate_id"]]["links"]
            self.assertEqual(len(link_counts["datasets"]), 1)
            self.assertEqual(len(link_counts["compatibility_reports"]), 1)
            self.assertEqual(len(link_counts["serving_probes"]), 1)
            self.assertEqual(len(link_counts["training_runs"]), 1)

    def test_dry_run_plan_avoids_heavy_imports_network_and_process_launches(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate = approved_candidate()
            registry = register_model_candidate(new_model_registry(registry_path=root / "model_registry.json"), candidate)
            registry = move_model_alias(registry, alias="candidate", target=candidate["candidate_id"], reason="unit test")
            dataset_manifest = root / "dataset_manifest.json"
            write_json(dataset_manifest, {"dataset_id": "local_mock_dataset_v1"})
            registry = link_model_registry_artifact(
                registry,
                entry_id=candidate["candidate_id"],
                collection="datasets",
                artifact_id="local_mock_dataset_v1",
                kind="dataset_manifest",
                status="dry_run_stub",
                path=dataset_manifest,
            )
            real_import = __import__
            heavy_modules = {"accelerate", "datasets", "peft", "torch", "transformers", "trl"}

            def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
                if name.split(".", 1)[0] in heavy_modules:
                    raise AssertionError(f"heavy ML import attempted: {name}")
                return real_import(name, globals, locals, fromlist, level)

            with patch("builtins.__import__", side_effect=guarded_import), patch.object(
                socket,
                "create_connection",
                side_effect=AssertionError("network connection attempted"),
            ), patch.object(subprocess, "Popen", side_effect=AssertionError("process launch attempted")):
                plan = build_dry_run_training_plan(
                    registry,
                    model_ref="candidate",
                    dataset_id="local_mock_dataset_v1",
                    dataset_manifest=dataset_manifest,
                    trainer="local-dry-run",
                    mode="sft",
                    output_dir=root / "outputs",
                    out_path=root / "training_plan.json",
                )

            self.assertTrue(plan["dry_run"])
            self.assertFalse(plan["execution"]["flight_recorder_downloaded_weights"])
            self.assertFalse(plan["execution"]["flight_recorder_launched_gpu_job"])

    def test_checked_in_qwen_metadata_candidate_is_no_download_training_eligible(self):
        root = Path(__file__).resolve().parents[1]
        candidate_path = root / "experiments/registry/model_candidates/qwen3_4b_instruct_2507.json"
        report_path = root / "experiments/registry/compatibility/qwen3_4b_instruct_2507.compatibility_report.json"
        registry_path = root / "experiments/registry/model_registry.json"
        entry_path = root / "experiments/registry/model_registry_entries/qwen3_4b_instruct_2507.json"
        plan_path = root / "experiments/registry/training_plans/qwen3_4b_instruct_2507_sft_dry_run.json"
        receipt_path = root / "experiments/registry/serving_probes/qwen3_4b_instruct_2507_metadata_serving_probe.json"

        candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
        report = json.loads(report_path.read_text(encoding="utf-8"))
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        entry = json.loads(entry_path.read_text(encoding="utf-8"))
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))

        self.assertEqual(model_candidate_errors(candidate, require_training_eligible=True), [])
        self.assertEqual(candidate["license"]["license_id"], "apache-2.0")
        self.assertFalse(candidate["source"]["weight_download_allowed"])
        self.assertEqual(candidate["compatibility"]["context_length"], 262144)
        self.assertEqual(len(candidate["metadata_evidence"]["files"]), 4)
        self.assertTrue(report["passed"])
        self.assertEqual(report["download_policy"]["downloaded_weights"], False)
        self.assertEqual(report["download_policy"]["downloaded_tokenizer"], False)
        self.assertEqual(report["download_policy"]["gpu_execution"], False)
        self.assertTrue(entry["training_eligible"])
        self.assertIn("qwen3_4b_instruct_2507", registry["entries"])
        self.assertEqual(len(entry["links"]["datasets"]), 1)
        self.assertEqual(len(entry["links"]["compatibility_reports"]), 1)
        self.assertEqual(len(entry["links"]["serving_probes"]), 1)
        self.assertEqual(len(entry["links"]["training_runs"]), 1)
        self.assertTrue(plan["dry_run"])
        self.assertTrue(plan["no_weight_download"])
        self.assertFalse(plan["gpu_execution"])
        self.assertFalse(plan["execution"]["flight_recorder_downloaded_weights"])
        self.assertFalse(plan["execution"]["flight_recorder_downloaded_tokenizer"])
        self.assertEqual(plan["model"]["candidate_id"], "qwen3_4b_instruct_2507")
        self.assertEqual(plan["dataset"]["registry_link_id"], "local_mock_dataset_v1")
        self.assertEqual(plan["compatibility_report"]["registry_link_id"], "qwen3_4b_instruct_2507_compatibility_report")
        self.assertEqual(model_serving_probe_receipt_errors(receipt), [])
        self.assertEqual(receipt["candidate_id"], "qwen3_4b_instruct_2507")
        self.assertEqual(receipt["readiness"], "metadata_recorded")
        self.assertFalse(receipt["download_policy"]["launched_server"])
        self.assertFalse(receipt["execution"]["flight_recorder_launched_server"])


def scout_manifest(
    *,
    candidate_id: str,
    manifest_path: str,
    compatibility_report_path: str | None,
    license_status: str,
    review_status: str,
    training_selection_eligible: bool,
) -> dict:
    ref = {
        "candidate_id": candidate_id,
        "manifest_path": manifest_path,
        "priority": "high",
        "reason": "Unit test scout candidate.",
        "license_status": license_status,
        "review_status": review_status,
        "training_selection_eligible": training_selection_eligible,
    }
    if compatibility_report_path is not None:
        ref["compatibility_report_path"] = compatibility_report_path
    return {
        "schema_version": MODEL_SCOUT_MANIFEST_SCHEMA_VERSION,
        "updated_at": "2026-07-02T00:00:00Z",
        "selection_policy": {
            "allow_unknown_license_for_scouting": True,
            "block_unknown_license_from_training": True,
            "require_candidate_source": True,
            "require_compatibility_metadata": True,
            "require_terms_review_for_training": True,
        },
        "candidates": [ref],
        "notes": ["Synthetic scout manifest for model-layer validation tests."],
    }
