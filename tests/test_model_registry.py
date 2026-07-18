import builtins
import json
import socket
import subprocess
import tempfile
import unittest
import urllib.request
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock

from flightrecorder.cli import main
from flightrecorder.model_registry import (
    ModelRegistryError,
    add_model_registry_link,
    build_dry_run_training_plan,
    build_model_compatibility_report,
    model_candidate_errors,
    model_compatibility_report_errors,
    model_registry_entry_errors,
    model_scout_manifest_errors,
    new_model_registry,
    move_model_alias,
    register_model_candidate,
    select_model_for_training,
    training_plan_errors,
)
from flightrecorder.schema_registry import check_schema_contract


def run_cli(args):
    with redirect_stdout(StringIO()):
        return main(args)


def run_cli_output(args):
    output = StringIO()
    with redirect_stdout(output):
        code = main(args)
    return code, output.getvalue()


def approved_candidate(candidate_id: str = "local_mock_tiny_chat") -> dict:
    return {
        "schema_version": "hfr.model_candidate.v1",
        "candidate_id": candidate_id,
        "model_id": "local/mock-tiny-chat",
        "source": {
            "type": "local_mock",
            "url": "local://models/mock-tiny-chat",
            "revision": "fixture-v1",
        },
        "license": {
            "status": "approved",
            "license_id": "project-local-fixture",
            "source_url": "local://licenses/project-local-fixture",
            "review_status": "approved",
            "reviewed_at": "2026-07-02T00:00:00Z",
            "reviewer": "model-layer-test",
            "accepted_terms": True,
            "terms_url": "local://licenses/project-local-fixture/terms",
            "training_allowed": True,
            "notes": ["Local mock fixture; no external model weights."],
        },
        "compatibility": {
            "context_length": 2048,
            "tokenizer": {"status": "metadata_only", "notes": "No tokenizer download required for mock."},
            "chat_template": {"status": "metadata_only", "notes": "Template supplied by harness."},
            "serving": {"engines": ["mock"], "notes": "Mock serving only."},
            "tool_calls": {"supported": False, "notes": "Not probed for mock candidate."},
            "structured_outputs": {"supported": False, "notes": "Not probed for mock candidate."},
            "quantization": {"options": ["none"], "notes": "No quantization for mock candidate."},
            "memory": {"minimum_vram_gb": 0, "notes": "No GPU required."},
        },
        "notes": [],
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


class ModelRegistryTests(unittest.TestCase):
    def test_candidate_registry_entry_and_training_plan_accept_approved_license(self):
        candidate = approved_candidate()
        registry = register_model_candidate(new_model_registry(), candidate)
        registry = move_model_alias(registry, alias="candidate", target=candidate["candidate_id"])
        dataset_manifest = self._dataset_manifest()

        with tempfile.TemporaryDirectory() as tmp:
            dataset_path = Path(tmp) / "dataset_manifest.json"
            report_path = Path(tmp) / "compatibility_report.json"
            write_json(dataset_path, dataset_manifest)
            report = build_model_compatibility_report(candidate, out_path=report_path)
            write_json(report_path, report)
            plan = build_dry_run_training_plan(
                registry,
                model_ref="candidate",
                dataset_id="dataset-v1",
                dataset_manifest=dataset_path,
                trainer="local-test-trainer",
                mode="sft",
                output_dir=Path(tmp) / "adapters",
                out_path=Path(tmp) / "training_plan.json",
                compatibility_report=report,
                compatibility_report_path=report_path,
            )

        entry = registry["entries"][candidate["candidate_id"]]
        self.assertFalse(model_candidate_errors(candidate, require_training_eligible=True))
        self.assertFalse(model_registry_entry_errors(entry))
        self.assertFalse(model_compatibility_report_errors(report))
        self.assertTrue(check_schema_contract(candidate)["passed"])
        self.assertTrue(check_schema_contract(entry)["passed"])
        self.assertTrue(check_schema_contract(registry)["passed"])
        self.assertTrue(check_schema_contract(plan)["passed"])
        self.assertTrue(check_schema_contract(report)["passed"])
        self.assertTrue(entry["training_eligible"])
        self.assertEqual(report["summary"]["probe_count"], 8)
        self.assertFalse(report["download_policy"]["downloaded_weights"])
        self.assertEqual(plan["compatibility_report"]["candidate_id"], candidate["candidate_id"])
        self.assertEqual(plan["compatibility_report"]["model_id"], candidate["model_id"])
        self.assertEqual(plan["compatibility_report"]["probe_count"], 8)
        self.assertEqual(len(plan["compatibility_report"]["sha256"]), 64)
        self.assertEqual(plan["schema_version"], "hfr.training_plan.v1")
        self.assertTrue(plan["dry_run"])
        self.assertTrue(plan["no_weight_download"])
        self.assertFalse(plan["gpu_execution"])
        self.assertFalse(training_plan_errors(plan))

    def test_non_approved_license_matrix_blocks_selection_and_plan_generation(self):
        for status in ("unknown", "restricted", "rejected", "noncommercial", "needs-review"):
            with self.subTest(status=status):
                candidate = approved_candidate(f"candidate_{status.replace('-', '_')}")
                candidate["license"]["status"] = status
                registry = register_model_candidate(new_model_registry(), candidate)
                registry = move_model_alias(registry, alias="candidate", target=candidate["candidate_id"])

                errors = model_candidate_errors(candidate, require_training_eligible=True)
                self.assertTrue(any("not approved for training" in error for error in errors))
                with self.assertRaises(ModelRegistryError):
                    select_model_for_training(registry, "candidate")
                with tempfile.TemporaryDirectory() as tmp:
                    dataset_path = Path(tmp) / "dataset_manifest.json"
                    write_json(dataset_path, self._dataset_manifest())
                    with self.assertRaises(ModelRegistryError):
                        build_dry_run_training_plan(
                            registry,
                            model_ref="candidate",
                            dataset_id="dataset-v1",
                            dataset_manifest=dataset_path,
                            trainer="local-test-trainer",
                            mode="sft",
                            output_dir=Path(tmp) / "adapters",
                            out_path=Path(tmp) / "training_plan.json",
                        )

    def test_champion_alias_requires_explicit_registered_rollback(self):
        candidate = approved_candidate("new_candidate")
        previous = approved_candidate("previous_champion")
        registry = new_model_registry()
        registry = register_model_candidate(registry, previous)
        registry = register_model_candidate(registry, candidate)

        with self.assertRaises(ModelRegistryError):
            move_model_alias(registry, alias="champion", target="new_candidate")
        with self.assertRaises(ModelRegistryError):
            move_model_alias(registry, alias="champion", target="new_candidate", rollback_target="missing")

        registry = move_model_alias(
            registry,
            alias="champion",
            target="new_candidate",
            rollback_target="previous_champion",
            reason="test promotion",
        )
        self.assertEqual(registry["aliases"]["champion"], "new_candidate")
        self.assertEqual(registry["aliases"]["rollback"], "previous_champion")
        self.assertEqual([row["alias"] for row in registry["alias_history"]], ["rollback", "champion"])

    def test_cli_register_alias_plan_and_validate_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate_path = root / "candidate.json"
            registry_path = root / "registry.json"
            dataset_path = root / "dataset_manifest.json"
            compatibility_path = root / "compatibility_report.json"
            plan_path = root / "training_plan.json"
            write_json(candidate_path, approved_candidate())
            write_json(dataset_path, self._dataset_manifest())

            self.assertEqual(run_cli(["model-candidate", "validate", "--candidate", str(candidate_path), "--require-training-eligible"]), 0)
            self.assertEqual(
                run_cli(["model-registry", "register", "--registry", str(registry_path), "--candidate", str(candidate_path)]),
                0,
            )
            code, output = run_cli_output(["model-registry", "list", "--registry", str(registry_path), "--json"])
            self.assertEqual(code, 0)
            self.assertEqual(json.loads(output)[0]["entry_id"], "local_mock_tiny_chat")
            self.assertEqual(
                run_cli(
                    [
                        "model-candidate",
                        "compatibility-report",
                        "--candidate",
                        str(candidate_path),
                        "--out",
                        str(compatibility_path),
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
                        "local_mock_tiny_chat",
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
                        "--model",
                        "candidate",
                        "--dataset-id",
                        "dataset-v1",
                        "--dataset-manifest",
                        str(dataset_path),
                        "--compatibility-report",
                        str(compatibility_path),
                        "--trainer",
                        "local-test-trainer",
                        "--mode",
                        "sft",
                        "--output-dir",
                        str(root / "adapters"),
                        "--hyperparameter",
                        "learning_rate=0.0001",
                        "--compute",
                        "accelerator=none",
                        "--out",
                        str(plan_path),
                    ]
                ),
                0,
            )
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            self.assertEqual(plan["compatibility_report"]["candidate_id"], "local_mock_tiny_chat")
            self.assertEqual(plan["compatibility_report"]["path"], str(compatibility_path))
            link_commands = [
                ("dataset", "dataset-v1", dataset_path, "split=validation"),
                ("training-run", "train-run-001", plan_path, "mode=sft"),
                ("adapter", "adapter-001", root / "adapters", "format=peft"),
                ("eval", "eval-001", root / "eval_summary.json", "arm=flightrecorder"),
                ("promotion-decision", "promotion-001", root / "promotion_decision.json", "decision=blocked"),
            ]
            for link_type, artifact_id, path, metadata in link_commands:
                self.assertEqual(
                    run_cli(
                        [
                            "model-registry",
                            "link",
                            "--registry",
                            str(registry_path),
                            "--entry",
                            "candidate",
                            "--type",
                            link_type,
                            "--artifact-id",
                            artifact_id,
                            "--path",
                            str(path),
                            "--metadata",
                            metadata,
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
                        "local_mock_tiny_chat",
                        "--type",
                        "dataset",
                        "--artifact-id",
                        "dataset-v1",
                        "--path",
                        str(dataset_path),
                        "--note",
                        "updated existing dataset link without duplication",
                    ]
                ),
                0,
            )
            registry = json.loads(registry_path.read_text(encoding="utf-8"))
            entry = registry["entries"]["local_mock_tiny_chat"]
            self.assertEqual(len(entry["datasets"]), 1)
            self.assertEqual(entry["datasets"][0]["artifact_id"], "dataset-v1")
            self.assertEqual(entry["datasets"][0]["note"], "updated existing dataset link without duplication")
            self.assertEqual(entry["training_runs"][0]["metadata"]["mode"], "sft")
            self.assertEqual(entry["adapters"][0]["artifact_id"], "adapter-001")
            self.assertEqual(entry["evals"][0]["metadata"]["arm"], "flightrecorder")
            self.assertEqual(entry["promotion_decisions"][0]["metadata"]["decision"], "blocked")
            entry_path = root / "entry.json"
            write_json(entry_path, registry["entries"]["local_mock_tiny_chat"])
            self.assertEqual(
                run_cli(
                    [
                        "validate",
                        "--model-candidate",
                        str(candidate_path),
                        "--model-compatibility-report",
                        str(compatibility_path),
                        "--model-registry-entry",
                        str(entry_path),
                        "--model-registry",
                        str(registry_path),
                        "--training-plan",
                        str(plan_path),
                        "--strict",
                    ]
                ),
                0,
            )

    def test_training_plan_rejects_mismatched_or_failed_compatibility_report(self):
        candidate = approved_candidate()
        registry = register_model_candidate(new_model_registry(), candidate)
        registry = move_model_alias(registry, alias="candidate", target=candidate["candidate_id"])

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_path = root / "dataset_manifest.json"
            report_path = root / "compatibility_report.json"
            write_json(dataset_path, self._dataset_manifest())
            report = build_model_compatibility_report(candidate, out_path=report_path)
            mismatched = dict(report)
            mismatched["candidate_id"] = "other_candidate"
            write_json(report_path, mismatched)
            with self.assertRaises(ModelRegistryError):
                build_dry_run_training_plan(
                    registry,
                    model_ref="candidate",
                    dataset_id="dataset-v1",
                    dataset_manifest=dataset_path,
                    trainer="local-test-trainer",
                    mode="sft",
                    output_dir=root / "adapters",
                    out_path=root / "training_plan.json",
                    compatibility_report=mismatched,
                    compatibility_report_path=report_path,
                )

            blocked = dict(report)
            blocked["passed"] = False
            blocked["readiness"] = "blocked"
            blocked["recommendation"] = "block_training_selection"
            write_json(report_path, blocked)
            with self.assertRaises(ModelRegistryError):
                build_dry_run_training_plan(
                    registry,
                    model_ref="candidate",
                    dataset_id="dataset-v1",
                    dataset_manifest=dataset_path,
                    trainer="local-test-trainer",
                    mode="sft",
                    output_dir=root / "adapters",
                    out_path=root / "training_plan.json",
                    compatibility_report=blocked,
                    compatibility_report_path=report_path,
                )

    def test_model_scout_manifest_validates_candidate_and_compatibility_refs(self):
        candidate = approved_candidate()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate_path = root / "model_candidates" / "candidate.json"
            report_path = root / "compatibility_reports" / "candidate.json"
            manifest_path = root / "model_scout_manifest.json"
            report = build_model_compatibility_report(candidate, out_path=report_path)
            write_json(candidate_path, candidate)
            write_json(report_path, report)
            manifest = self._scout_manifest(
                candidate_id=candidate["candidate_id"],
                candidate_path=candidate_path,
                compatibility_report_path=report_path,
                training_selection_eligible=True,
            )
            write_json(manifest_path, manifest)

            self.assertFalse(model_scout_manifest_errors(manifest))
            self.assertTrue(check_schema_contract(manifest)["passed"])
            self.assertEqual(run_cli(["model-scout", "validate", "--manifest", str(manifest_path)]), 0)
            self.assertEqual(run_cli(["validate", "--model-scout-manifest", str(manifest_path), "--strict"]), 0)
            self.assertEqual(run_cli(["schemas", "--check", str(manifest_path), "--name", "model_scout_manifest"]), 0)

    def test_model_scout_manifest_blocks_unknown_license_training_selection_claim(self):
        candidate = approved_candidate()
        candidate["license"]["status"] = "unknown"
        candidate["license"]["review_status"] = "pending"
        candidate["license"]["accepted_terms"] = False
        candidate["license"]["training_allowed"] = False
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate_path = root / "candidate.json"
            manifest_path = root / "model_scout_manifest.json"
            write_json(candidate_path, candidate)
            manifest = self._scout_manifest(
                candidate_id=candidate["candidate_id"],
                candidate_path=candidate_path,
                compatibility_report_path=None,
                license_status="unknown",
                review_status="pending",
                training_selection_eligible=True,
            )
            write_json(manifest_path, manifest)

            self.assertFalse(model_scout_manifest_errors(manifest))
            code, output = run_cli_output(["validate", "--model-scout-manifest", str(manifest_path), "--strict"])
            self.assertEqual(code, 1)
            self.assertIn("training_selection_eligible cannot be true", output)

    def test_model_registry_link_helper_requires_registered_entry_and_artifact(self):
        registry = register_model_candidate(new_model_registry(), approved_candidate())
        with self.assertRaises(ModelRegistryError):
            add_model_registry_link(registry, entry_ref="missing", link_type="dataset", artifact_id="dataset-v1")
        with self.assertRaises(ModelRegistryError):
            add_model_registry_link(registry, entry_ref="local_mock_tiny_chat", link_type="dataset")
        with self.assertRaises(ModelRegistryError):
            add_model_registry_link(registry, entry_ref="local_mock_tiny_chat", link_type="unknown", artifact_id="dataset-v1")

    def test_dry_run_plan_avoids_heavy_imports_network_and_gpu_launches(self):
        candidate = approved_candidate()
        registry = register_model_candidate(new_model_registry(), candidate)
        registry = move_model_alias(registry, alias="candidate", target=candidate["candidate_id"])
        real_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name.split(".", 1)[0] in {"torch", "transformers", "huggingface_hub"}:
                raise AssertionError(f"heavy ML import attempted: {name}")
            return real_import(name, *args, **kwargs)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_path = root / "dataset_manifest.json"
            write_json(dataset_path, self._dataset_manifest())
            with mock.patch("builtins.__import__", side_effect=guarded_import), mock.patch.object(
                urllib.request, "urlopen", side_effect=AssertionError("network download attempted")
            ), mock.patch.object(
                urllib.request, "urlretrieve", side_effect=AssertionError("file download attempted")
            ), mock.patch.object(
                subprocess, "Popen", side_effect=AssertionError("process launch attempted")
            ), mock.patch.object(
                socket, "socket", side_effect=AssertionError("socket attempted")
            ):
                plan = build_dry_run_training_plan(
                    registry,
                    model_ref="candidate",
                    dataset_id="dataset-v1",
                    dataset_manifest=dataset_path,
                    trainer="local-test-trainer",
                    mode="sft",
                    output_dir=root / "adapters",
                    out_path=root / "training_plan.json",
                )

        self.assertFalse(plan["execution"]["flight_recorder_downloaded_weights"])
        self.assertFalse(plan["execution"]["flight_recorder_downloaded_tokenizer"])
        self.assertFalse(plan["execution"]["flight_recorder_imported_heavy_ml"])
        self.assertFalse(plan["execution"]["flight_recorder_launched_gpu_job"])

    def _dataset_manifest(self) -> dict:
        return {
            "schema_version": "hfr.test_dataset_manifest.v1",
            "dataset_id": "dataset-v1",
            "artifact_count": 0,
            "redaction_status": "redacted",
        }

    def _scout_manifest(
        self,
        *,
        candidate_id: str,
        candidate_path: Path,
        compatibility_report_path,
        training_selection_eligible: bool,
        license_status: str = "approved",
        review_status: str = "approved",
    ) -> dict:
        candidate_ref = {
            "candidate_id": candidate_id,
            "manifest_path": str(candidate_path),
            "priority": "test",
            "reason": "Validate model scout manifest references.",
            "license_status": license_status,
            "review_status": review_status,
            "training_selection_eligible": training_selection_eligible,
        }
        if compatibility_report_path is not None:
            candidate_ref["compatibility_report_path"] = str(compatibility_report_path)
        return {
            "schema_version": "hfr.model_scout_manifest.v1",
            "updated_at": "2026-07-02T00:00:00Z",
            "selection_policy": {
                "allow_unknown_license_for_scouting": True,
                "block_unknown_license_from_training": True,
                "require_candidate_source": True,
                "require_compatibility_metadata": True,
                "require_terms_review_for_training": True,
            },
            "candidates": [candidate_ref],
            "notes": ["Test scout manifest."],
        }


if __name__ == "__main__":
    unittest.main()
