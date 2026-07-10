from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import flightrecorder.governance as governance
from flightrecorder.atomic_json import AtomicJsonError, atomic_write_json_cas
from flightrecorder.bundle import build_evidence_bundle
from flightrecorder.cli import main
from flightrecorder.eval_summary import build_eval_summary
from flightrecorder.external_eval import (
    build_external_eval_plan,
    write_external_eval_plan,
)
from flightrecorder.external_eval_result import (
    build_external_eval_result,
    write_external_eval_result,
)
from flightrecorder.governance import apply_promotion_aliases
from flightrecorder.validation import (
    _path_has_symlink_component,
    validate_promotion_alias_apply,
    validate_promotion_decision,
    validate_promotion_release_record,
)
from tests.agentic_loop_fixtures import copy_valid_loop_artifacts

ROOT = Path(__file__).resolve().parents[1]
PROMOTION_CANDIDATE_ID = "local/mock-candidate"


def run_cli(args):
    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        return main(args)


def without_preserve_paths(args):
    return [arg for arg in args if arg != "--preserve-paths"]


class PromotionDecisionTests(unittest.TestCase):
    def test_committed_agentic_training_promotion_governance_authorizes_review_alias_update(
        self,
    ):
        root = ROOT / "examples" / "agentic_training" / "promotion_governance"
        compare_gate_path = root / "compare_gate.json"
        decision_path = root / "promotion_decision.json"
        gate_path = root / "promotion_decision_gate.json"
        ledger_path = root / "promotion_ledger.json"
        history_gate_path = root / "promotion_history_decision_gate.json"
        ledger_gate_path = root / "promotion_ledger_gate.json"
        cards_path = root / "promotion_cards"
        rollback_receipt_path = root / "promotion_rollback_receipt.json"
        alias_apply_path = root / "promotion_alias_apply.json"
        release_record_path = root / "promotion_release_record.json"
        archive_path = root / "promotion_archive"
        registry_entry_path = root / "model_registry_entry.json"
        registry_before_path = root / "model_registry_before_alias_apply.json"
        registry_after_path = root / "model_registry.json"
        compare_gate = json.loads(compare_gate_path.read_text(encoding="utf-8"))
        decision = json.loads(decision_path.read_text(encoding="utf-8"))
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        ledger_gate = json.loads(ledger_gate_path.read_text(encoding="utf-8"))
        cards = json.loads(
            (cards_path / "promotion_cards.json").read_text(encoding="utf-8")
        )
        rollback_receipt = json.loads(rollback_receipt_path.read_text(encoding="utf-8"))
        alias_apply = json.loads(alias_apply_path.read_text(encoding="utf-8"))
        release_record = json.loads(release_record_path.read_text(encoding="utf-8"))
        archive = json.loads(
            (archive_path / "promotion_archive.json").read_text(encoding="utf-8")
        )

        self.assertTrue(compare_gate["passed"])
        self.assertEqual(compare_gate["metrics"]["candidate_win_count"], 1)
        self.assertEqual(compare_gate["metrics"]["baseline_win_count"], 0)
        self.assertEqual(compare_gate["metrics"]["contract_drift_count"], 0)
        self.assertEqual(compare_gate["metrics"]["task_completion_regression_count"], 0)
        self.assertEqual(decision["schema_version"], "hfr.promotion_decision.v1")
        self.assertTrue(decision["passed"])
        self.assertEqual(decision["recommendation"], "apply_alias_update")
        self.assertTrue(decision["alias_update"]["authorized"])
        self.assertEqual(
            decision["alias_update"]["recommendation"], "apply_alias_update"
        )
        failed_ids = {
            check["id"] for check in decision["checks"] if not check["passed"]
        }
        self.assertEqual(failed_ids, set())
        self.assertTrue(decision["artifacts"]["evidence_bundle"]["exists"])
        self.assertTrue(decision["artifacts"]["serving_profile"]["exists"])
        self.assertEqual(
            decision["artifacts"]["model_card"]["path"], "promotion_cards/MODEL_CARD.md"
        )
        self.assertEqual(
            decision["artifacts"]["dataset_card"]["path"],
            "promotion_cards/DATASET_CARD.md",
        )

        self.assertTrue(gate["passed"])
        self.assertEqual(
            gate["source_decision"]["recommendation"], "apply_alias_update"
        )
        self.assertTrue(ledger_gate["passed"])
        self.assertEqual(ledger["metrics"]["blocked_count"], 0)
        self.assertEqual(ledger["metrics"]["allowed_count"], 1)
        self.assertEqual(ledger["metrics"]["latest_recommendation"], "allow_promotion")
        self.assertTrue(cards["passed"])
        self.assertTrue(rollback_receipt["passed"])
        self.assertTrue(alias_apply["passed"])
        self.assertEqual(
            alias_apply["registry_before"]["aliases"]["champion"], "local/mock-baseline"
        )
        self.assertEqual(
            alias_apply["registry_after"]["aliases"]["champion"], "local/mock-candidate"
        )
        self.assertTrue(release_record["passed"])
        self.assertEqual(release_record["release"]["id"], "demo-loop-001-release")
        self.assertEqual(
            release_record["release"]["rollback_id"], "local/mock-baseline"
        )
        self.assertTrue(archive["passed"])
        self.assertTrue(archive["self_contained"])
        self.assertEqual(archive["metrics"]["promotion_release_record_count"], 1)
        self.assertEqual(
            run_cli(
                [
                    "validate",
                    "--promotion-cards",
                    str(cards_path),
                    "--promotion-decision",
                    str(decision_path),
                    "--promotion-alias-apply",
                    str(alias_apply_path),
                    "--promotion-rollback-receipt",
                    str(rollback_receipt_path),
                    "--promotion-release-record",
                    str(release_record_path),
                    "--decision-gate",
                    str(history_gate_path),
                    "--decision-gate",
                    str(gate_path),
                    "--promotion-ledger",
                    str(ledger_path),
                    "--promotion-ledger-gate",
                    str(ledger_gate_path),
                    "--promotion-archive",
                    str(archive_path),
                    "--model-registry-entry",
                    str(registry_entry_path),
                    "--model-registry",
                    str(registry_before_path),
                    "--model-registry",
                    str(registry_after_path),
                    "--strict",
                ]
            ),
            0,
        )
        self.assertEqual(run_cli(["schemas", "--check", str(compare_gate_path)]), 0)
        self.assertEqual(
            run_cli(["schemas", "--check", str(cards_path / "promotion_cards.json")]), 0
        )
        self.assertEqual(run_cli(["schemas", "--check", str(alias_apply_path)]), 0)
        self.assertEqual(run_cli(["schemas", "--check", str(rollback_receipt_path)]), 0)
        self.assertEqual(run_cli(["schemas", "--check", str(release_record_path)]), 0)
        self.assertEqual(
            run_cli(
                ["schemas", "--check", str(archive_path / "promotion_archive.json")]
            ),
            0,
        )

    def test_promotion_cards_generate_valid_model_and_dataset_cards(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = write_governance_artifacts(root)
            training_export = write_training_export(root)
            cards_dir = root / "cards"

            code = run_cli(promotion_cards_args(artifacts, training_export, cards_dir))

            self.assertEqual(code, 0)
            manifest = json.loads(
                (cards_dir / "promotion_cards.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["schema_version"], "hfr.promotion_cards.v1")
            self.assertTrue(manifest["passed"])
            self.assertTrue((cards_dir / "MODEL_CARD.md").is_file())
            self.assertTrue((cards_dir / "DATASET_CARD.md").is_file())
            cards_text = (
                (cards_dir / "MODEL_CARD.md").read_text(encoding="utf-8").lower()
            )
            cards_text += (
                (cards_dir / "DATASET_CARD.md").read_text(encoding="utf-8").lower()
            )
            self.assertNotIn("todo", cards_text)
            self.assertNotIn("unsupported claim", cards_text)
            self.assertEqual(
                run_cli(["validate", "--promotion-cards", str(cards_dir), "--strict"]),
                0,
            )
            self.assertEqual(
                run_cli(
                    ["schemas", "--check", str(cards_dir / "promotion_cards.json")]
                ),
                0,
            )

    def test_promotion_cards_block_unknown_license(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = write_governance_artifacts(root)
            training_export = write_training_export(root)
            cards_dir = root / "cards"

            code = run_cli(
                promotion_cards_args(
                    artifacts, training_export, cards_dir, license_status="unknown"
                )
            )

            self.assertEqual(code, 1)
            manifest = json.loads(
                (cards_dir / "promotion_cards.json").read_text(encoding="utf-8")
            )
            self.assertFalse(manifest["passed"])
            self.assertIn("license_status_known", failed_check_ids(manifest))
            self.assertEqual(
                run_cli(["validate", "--promotion-cards", str(cards_dir), "--strict"]),
                0,
            )

    def test_promotion_cards_block_symlinked_json_input_parent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = write_governance_artifacts(root)
            training_export = write_training_export(root)
            cards_dir = root / "cards"
            linked_target = root / "linked_target"
            linked_target.mkdir()
            (linked_target / "compare_gate.json").write_text(
                artifacts["compare_gate"].read_text(encoding="utf-8"), encoding="utf-8"
            )
            linked_parent = root / "linked_inputs"
            try:
                linked_parent.symlink_to(linked_target, target_is_directory=True)
            except (NotImplementedError, OSError) as exc:
                self.skipTest(f"symlink unavailable: {exc}")
            artifacts["compare_gate"] = linked_parent / "compare_gate.json"

            code = run_cli(promotion_cards_args(artifacts, training_export, cards_dir))

            self.assertEqual(code, 1)
            manifest = json.loads(
                (cards_dir / "promotion_cards.json").read_text(encoding="utf-8")
            )
            compare_gate = manifest["artifacts"]["compare_gate"]
            self.assertFalse(compare_gate["exists"])
            self.assertEqual(compare_gate["kind"], "other")
            self.assertNotIn("sha256", compare_gate)
            failed_ids = failed_check_ids(manifest)
            self.assertIn("compare_gate_present", failed_ids)
            self.assertIn("compare_gate_schema", failed_ids)
            self.assertIn("compare_gate_passed", failed_ids)
            self.assertIn("compare_metrics_complete", failed_ids)
            self.assertEqual(
                run_cli(["validate", "--promotion-cards", str(cards_dir), "--strict"]),
                0,
            )

    def test_promotion_cards_block_symlinked_training_export_parent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = write_governance_artifacts(root)
            training_export = write_training_export(root)
            cards_dir = root / "cards"
            linked_target = root / "linked_target"
            linked_export = linked_target / "training_export"
            linked_export.mkdir(parents=True)
            (linked_export / "DATASET_CARD.md").write_text(
                (training_export / "DATASET_CARD.md").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            linked_parent = root / "linked_inputs"
            try:
                linked_parent.symlink_to(linked_target, target_is_directory=True)
            except (NotImplementedError, OSError) as exc:
                self.skipTest(f"symlink unavailable: {exc}")

            code = run_cli(
                promotion_cards_args(
                    artifacts, linked_parent / "training_export", cards_dir
                )
            )

            self.assertEqual(code, 1)
            manifest = json.loads(
                (cards_dir / "promotion_cards.json").read_text(encoding="utf-8")
            )
            training_artifact = manifest["artifacts"]["training_export"]
            self.assertFalse(training_artifact["exists"])
            self.assertEqual(training_artifact["kind"], "other")
            self.assertNotIn("sha256", training_artifact)
            self.assertIn("training_export_present", failed_check_ids(manifest))
            self.assertEqual(
                run_cli(["validate", "--promotion-cards", str(cards_dir), "--strict"]),
                0,
            )

    def test_validate_promotion_cards_rejects_stale_generated_card(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = write_governance_artifacts(root)
            training_export = write_training_export(root)
            cards_dir = root / "cards"
            self.assertEqual(
                run_cli(promotion_cards_args(artifacts, training_export, cards_dir)), 0
            )

            (cards_dir / "MODEL_CARD.md").write_text(
                "# Model Card\n\nchanged after manifest\n", encoding="utf-8"
            )

            self.assertEqual(
                run_cli(["validate", "--promotion-cards", str(cards_dir), "--strict"]),
                1,
            )

    def test_promotion_decision_authorizes_alias_receipt_after_all_gates_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = write_governance_artifacts(root)
            decision_path = root / "promotion_decision.json"

            code = run_cli(promotion_decision_args(artifacts, decision_path))

            self.assertEqual(code, 0)
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            self.assertEqual(decision["schema_version"], "hfr.promotion_decision.v1")
            self.assertTrue(decision["passed"])
            self.assertEqual(decision["recommendation"], "apply_alias_update")
            self.assertTrue(decision["alias_update"]["authorized"])
            aliases = {
                item["alias"]: item for item in decision["alias_update"]["aliases"]
            }
            self.assertEqual(aliases["champion"]["previous_target"], "champion-v1")
            self.assertEqual(aliases["champion"]["target"], PROMOTION_CANDIDATE_ID)
            self.assertEqual(aliases["rollback"]["target"], "champion-v1")
            lineage = decision["external_eval_lineage"]
            self.assertTrue(lineage["passed"])
            self.assertEqual(lineage["result_count"], 1)
            self.assertTrue(lineage["exact_result_set"])
            self.assertTrue(lineage["candidate_model_bound"])
            self.assertTrue(lineage["evidence_bundle_summary_bound"])
            self.assertTrue(lineage["semantic_validation_passed"])
            self.assertTrue(lineage["governance_ready"])
            self.assertEqual(lineage["results"][0]["adapter_id"], "local_mock")
            self.assertEqual(
                run_cli(
                    ["validate", "--promotion-decision", str(decision_path), "--strict"]
                ),
                0,
            )
            self.assertEqual(run_cli(["schemas", "--check", str(decision_path)]), 0)

    def test_promotion_decision_requires_eval_summary_and_external_eval_result(self):
        cases = (
            (
                "eval_summary",
                None,
                {
                    "eval_summary_present",
                    "eval_summary_schema",
                    "eval_summary_passed",
                    "eval_summary_semantically_valid",
                },
            ),
            (
                "external_eval_results",
                [],
                {"external_eval_results_present", "external_eval_result_set_exact"},
            ),
        )
        for role, replacement, expected_failed_ids in cases:
            with self.subTest(role=role), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                artifacts = write_governance_artifacts(root)
                artifacts[role] = replacement
                decision_path = root / "promotion_decision.json"

                code = run_cli(promotion_decision_args(artifacts, decision_path))

                self.assertEqual(code, 1)
                decision = json.loads(decision_path.read_text(encoding="utf-8"))
                self.assertTrue(expected_failed_ids <= failed_check_ids(decision))
                self.assertFalse(decision["external_eval_lineage"]["passed"])
                self.assertFalse(decision["alias_update"]["authorized"])
                self.assertEqual(
                    run_cli(
                        [
                            "validate",
                            "--promotion-decision",
                            str(decision_path),
                            "--strict",
                        ]
                    ),
                    0,
                )
                self.assertEqual(run_cli(["schemas", "--check", str(decision_path)]), 0)

    def test_promotion_decision_emits_valid_blocks_for_non_file_eval_sources(self):
        cases = (
            "missing_eval_summary",
            "eval_summary_directory",
            "missing_result",
            "result_directory",
        )
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                artifacts = write_governance_artifacts(root)
                expected_kind = "missing" if case.startswith("missing") else "directory"
                source_path = root / case
                if expected_kind == "directory":
                    source_path.mkdir()
                if "eval_summary" in case:
                    artifacts["eval_summary"] = source_path
                    result_source = False
                else:
                    artifacts["external_eval_results"] = [source_path]
                    result_source = True
                decision_path = root / "promotion_decision.json"

                self.assertEqual(
                    run_cli(promotion_decision_args(artifacts, decision_path)), 1
                )
                decision = json.loads(decision_path.read_text(encoding="utf-8"))
                self.assertFalse(decision["passed"])
                record = (
                    decision["external_eval_lineage"]["results"][0]["artifact"]
                    if result_source
                    else decision["artifacts"]["eval_summary"]
                )
                self.assertEqual(record["kind"], expected_kind)
                self.assertEqual(
                    run_cli(
                        [
                            "validate",
                            "--promotion-decision",
                            str(decision_path),
                            "--strict",
                        ]
                    ),
                    0,
                )
                self.assertEqual(run_cli(["schemas", "--check", str(decision_path)]), 0)

    def test_promotion_decision_rejects_duplicate_and_unexpected_external_eval_results(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = write_governance_artifacts(root)
            result_path = artifacts["external_eval_results"][0]
            artifacts["external_eval_results"] = [result_path, result_path]
            decision_path = root / "duplicate_result_decision.json"

            duplicate_code = run_cli(promotion_decision_args(artifacts, decision_path))

            self.assertEqual(duplicate_code, 1)
            duplicate_decision = json.loads(decision_path.read_text(encoding="utf-8"))
            self.assertIn(
                "external_eval_result_set_exact", failed_check_ids(duplicate_decision)
            )
            self.assertFalse(
                duplicate_decision["external_eval_lineage"]["exact_result_set"]
            )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = write_governance_artifacts(root)
            alternate = write_promotion_eval_lineage(
                root,
                dirname="promotion_eval_alternate",
                execution_id="promotion-eval-002",
            )
            artifacts["external_eval_results"] = [alternate["external_eval_result"]]
            decision_path = root / "unexpected_result_decision.json"

            unexpected_code = run_cli(promotion_decision_args(artifacts, decision_path))

            self.assertEqual(unexpected_code, 1)
            unexpected_decision = json.loads(decision_path.read_text(encoding="utf-8"))
            failed_ids = failed_check_ids(unexpected_decision)
            self.assertIn("external_eval_result_set_exact", failed_ids)
            self.assertNotIn("external_eval_results_semantically_valid", failed_ids)
            self.assertFalse(
                unexpected_decision["external_eval_lineage"]["exact_result_set"]
            )

    def test_promotion_decision_rejects_external_eval_candidate_model_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = write_governance_artifacts(root)
            decision_path = root / "promotion_decision.json"

            code = run_cli(
                promotion_decision_args(
                    artifacts, decision_path, candidate_id="candidate-v3"
                )
            )

            self.assertEqual(code, 1)
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            self.assertIn(
                "external_eval_results_match_candidate", failed_check_ids(decision)
            )
            self.assertFalse(decision["external_eval_lineage"]["candidate_model_bound"])
            self.assertFalse(decision["alias_update"]["authorized"])

    def test_promotion_decision_rejects_failed_external_benchmark_governance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = write_governance_artifacts(root)
            failed_lineage = write_promotion_eval_lineage(
                root,
                dirname="promotion_eval_failed",
                execution_id="promotion-eval-failed-001",
                benchmark_passed=False,
            )
            artifacts["eval_summary"] = failed_lineage["eval_summary"]
            artifacts["external_eval_results"] = [
                failed_lineage["external_eval_result"]
            ]
            evidence_bundle = root / "failed_evidence_bundle.json"
            evidence_bundle.write_text(
                json.dumps(
                    build_evidence_bundle(
                        out_path=evidence_bundle,
                        eval_summary_path=failed_lineage["eval_summary"],
                    ),
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            artifacts["evidence_bundle"] = evidence_bundle
            decision_path = root / "promotion_decision.json"

            code = run_cli(promotion_decision_args(artifacts, decision_path))

            self.assertEqual(code, 1)
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            failed_ids = failed_check_ids(decision)
            self.assertIn("eval_summary_passed", failed_ids)
            self.assertIn("external_eval_results_governance_ready", failed_ids)
            self.assertFalse(decision["external_eval_lineage"]["governance_ready"])
            self.assertFalse(decision["alias_update"]["authorized"])

    def test_promotion_decision_rejects_evidence_bundle_eval_summary_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = write_governance_artifacts(root)
            alternate = write_promotion_eval_lineage(
                root,
                dirname="promotion_eval_alternate",
                execution_id="promotion-eval-002",
            )
            mismatched_bundle = root / "mismatched_evidence_bundle.json"
            mismatched_bundle.write_text(
                json.dumps(
                    build_evidence_bundle(
                        out_path=mismatched_bundle,
                        eval_summary_path=alternate["eval_summary"],
                    ),
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            artifacts["evidence_bundle"] = mismatched_bundle
            decision_path = root / "promotion_decision.json"

            code = run_cli(promotion_decision_args(artifacts, decision_path))

            self.assertEqual(code, 1)
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            self.assertIn(
                "evidence_bundle_eval_summary_bound", failed_check_ids(decision)
            )
            self.assertFalse(
                decision["external_eval_lineage"]["evidence_bundle_summary_bound"]
            )
            self.assertFalse(decision["alias_update"]["authorized"])

    def test_validate_promotion_decision_rejects_mutated_external_eval_sources(self):
        for source_role in ("eval_summary", "external_eval_result"):
            with (
                self.subTest(source_role=source_role),
                tempfile.TemporaryDirectory() as tmp,
            ):
                root = Path(tmp)
                artifacts = write_governance_artifacts(root)
                decision_path = root / "promotion_decision.json"
                self.assertEqual(
                    run_cli(promotion_decision_args(artifacts, decision_path)), 0
                )
                source_path = (
                    artifacts["eval_summary"]
                    if source_role == "eval_summary"
                    else artifacts["external_eval_results"][0]
                )

                source_path.write_text(
                    source_path.read_text(encoding="utf-8") + "\n", encoding="utf-8"
                )

                self.assertEqual(
                    run_cli(
                        [
                            "validate",
                            "--promotion-decision",
                            str(decision_path),
                            "--strict",
                        ]
                    ),
                    1,
                )

    def test_validate_promotion_decision_rejects_forged_or_removed_external_eval_lineage(
        self,
    ):
        mutations = ("forged_summary", "forged_check_projection", "removed_check")
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                artifacts = write_governance_artifacts(root)
                decision_path = root / "promotion_decision.json"
                summary_path = root / "validation.json"
                self.assertEqual(
                    run_cli(promotion_decision_args(artifacts, decision_path)), 0
                )
                decision = json.loads(decision_path.read_text(encoding="utf-8"))
                if mutation == "forged_summary":
                    decision["external_eval_lineage"]["result_count"] += 1
                elif mutation == "forged_check_projection":
                    lineage_check = next(
                        check
                        for check in decision["checks"]
                        if check["id"] == "external_eval_results_present"
                    )
                    lineage_check["actual"]["result_count"] = 999
                else:
                    decision["checks"] = [
                        check
                        for check in decision["checks"]
                        if check["id"] != "external_eval_result_set_exact"
                    ]
                    decision["check_count"] = len(decision["checks"])
                    decision["metrics"]["check_count"] = len(decision["checks"])
                    decision["decision"]["key_metrics"]["check_count"] = len(
                        decision["checks"]
                    )
                decision_path.write_text(
                    json.dumps(decision, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )

                code = run_cli(
                    [
                        "validate",
                        "--promotion-decision",
                        str(decision_path),
                        "--strict",
                        "--out",
                        str(summary_path),
                    ]
                )

                self.assertEqual(code, 1)
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                errors = "\n".join(
                    error for target in summary["targets"] for error in target["errors"]
                )
                self.assertIn("external_eval", errors)

    def test_promotion_decision_blocks_upstream_semantic_warnings(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = write_governance_artifacts(root)
            evidence_bundle = root / "evidence_bundle_with_absolute_paths.json"
            evidence_bundle.write_text(
                json.dumps(
                    build_evidence_bundle(
                        out_path=evidence_bundle,
                        eval_summary_path=artifacts["eval_summary"],
                        preserve_paths=True,
                    ),
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            artifacts["evidence_bundle"] = evidence_bundle
            decision_path = root / "promotion_decision.json"

            code = run_cli(promotion_decision_args(artifacts, decision_path))

            self.assertEqual(code, 1)
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            self.assertIn(
                "evidence_bundle_semantically_valid", failed_check_ids(decision)
            )
            self.assertFalse(
                decision["external_eval_lineage"]["semantic_validation_passed"]
            )

    def test_validate_promotion_decision_rejects_forged_compare_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = write_governance_artifacts(root)
            decision_path = root / "promotion_decision.json"
            validation_path = root / "validation.json"
            self.assertEqual(
                run_cli(promotion_decision_args(artifacts, decision_path)), 0
            )
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            decision["metrics"]["task_completion_regression_count"] = 999
            decision["decision"]["key_metrics"]["task_completion_regression_count"] = (
                999
            )
            decision_path.write_text(
                json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )

            code = run_cli(
                [
                    "validate",
                    "--promotion-decision",
                    str(decision_path),
                    "--strict",
                    "--out",
                    str(validation_path),
                ]
            )

            self.assertEqual(code, 1)
            summary = json.loads(validation_path.read_text(encoding="utf-8"))
            errors = "\n".join(
                error for target in summary["targets"] for error in target["errors"]
            )
            self.assertIn(
                "promotion_decision.metrics must exactly match replayed", errors
            )

    def test_validate_promotion_decision_replays_compare_gate_checks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = write_governance_artifacts(root)
            decision_path = root / "promotion_decision.json"
            validation_path = root / "validation.json"
            self.assertEqual(
                run_cli(promotion_decision_args(artifacts, decision_path)), 0
            )

            compare_gate = json.loads(
                artifacts["compare_gate"].read_text(encoding="utf-8")
            )
            compare_gate["passed"] = False
            compare_gate["metrics"]["task_completion_regression_count"] = 1
            artifacts["compare_gate"].write_text(
                json.dumps(compare_gate, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            compare_bytes = artifacts["compare_gate"].read_bytes()
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            compare_record = decision["artifacts"]["compare_gate"]
            compare_record["sha256"] = hashlib.sha256(compare_bytes).hexdigest()
            compare_record["size_bytes"] = len(compare_bytes)
            decision["metrics"]["task_completion_regression_count"] = 1
            decision["decision"]["key_metrics"]["task_completion_regression_count"] = 1
            decision_path.write_text(
                json.dumps(decision, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            self.assertEqual(
                run_cli(
                    [
                        "validate",
                        "--promotion-decision",
                        str(decision_path),
                        "--strict",
                        "--out",
                        str(validation_path),
                    ]
                ),
                1,
            )
            summary = json.loads(validation_path.read_text(encoding="utf-8"))
            errors = "\n".join(
                error for target in summary["targets"] for error in target["errors"]
            )
            self.assertIn("checks must exactly match canonical replay", errors)

    def test_promotion_decision_blocks_schema_invalid_structured_artifacts(self):
        roles = (
            "promotion_ledger_gate",
            "compare_gate",
            "trainer_launch_check",
            "model_registry_entry",
            "agentic_training_result",
            "serving_profile",
        )
        for role in roles:
            with self.subTest(role=role), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                artifacts = write_governance_artifacts(root)
                source_path = artifacts[role]
                payload = json.loads(source_path.read_text(encoding="utf-8"))
                source_path.write_text(
                    json.dumps(
                        {
                            "schema_version": payload["schema_version"],
                            "passed": True,
                        },
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                decision_path = root / "promotion_decision.json"

                self.assertEqual(
                    run_cli(promotion_decision_args(artifacts, decision_path)), 1
                )
                decision = json.loads(decision_path.read_text(encoding="utf-8"))
                self.assertIn(f"{role}_contract_valid", failed_check_ids(decision))
                self.assertFalse(decision["alias_update"]["authorized"])

    def test_promotion_decision_semantically_replays_compare_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = write_governance_artifacts(root)
            compare_gate = json.loads(
                artifacts["compare_gate"].read_text(encoding="utf-8")
            )
            compare_gate["checks"][0]["passed"] = False
            artifacts["compare_gate"].write_text(
                json.dumps(compare_gate, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            decision_path = root / "promotion_decision.json"

            self.assertEqual(
                run_cli(promotion_decision_args(artifacts, decision_path)), 1
            )
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            self.assertIn("compare_gate_semantically_valid", failed_check_ids(decision))
            self.assertNotIn("compare_gate_contract_valid", failed_check_ids(decision))

    def test_promotion_decision_binds_training_and_compare_candidates(self):
        for source_role in ("agentic_training_result", "compare_gate"):
            with (
                self.subTest(source_role=source_role),
                tempfile.TemporaryDirectory() as tmp,
            ):
                root = Path(tmp)
                artifacts = write_governance_artifacts(root)
                if source_role == "agentic_training_result":
                    result = json.loads(
                        artifacts[source_role].read_text(encoding="utf-8")
                    )
                    result["registry_update"]["target_model_id"] = "other-candidate"
                    artifacts[source_role].write_text(
                        json.dumps(result, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )
                    entry = json.loads(
                        artifacts["model_registry_entry"].read_text(encoding="utf-8")
                    )
                    _bind_registry_entry_links(
                        entry,
                        {"training_runs": artifacts[source_role]},
                    )
                    _write_source_json(artifacts["model_registry_entry"], entry)
                    expected_check = "agentic_training_result_matches_candidate"
                else:
                    compare_export = root / "compare_export"
                    shutil.copytree(
                        ROOT
                        / "examples"
                        / "agentic_training"
                        / "promotion_governance"
                        / "compare_export",
                        compare_export,
                    )
                    manifest_path = compare_export / "manifest.json"
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    manifest["metadata"]["candidate"] = "other-candidate"
                    manifest_path.write_text(
                        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )
                    compare_gate = json.loads(
                        artifacts["compare_gate"].read_text(encoding="utf-8")
                    )
                    compare_gate["compare_export"] = str(compare_export)
                    artifacts["compare_gate"].write_text(
                        json.dumps(compare_gate, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )
                    expected_check = "compare_gate_matches_candidate"
                decision_path = root / "promotion_decision.json"

                self.assertEqual(
                    run_cli(promotion_decision_args(artifacts, decision_path)), 1
                )
                decision = json.loads(decision_path.read_text(encoding="utf-8"))
                self.assertIn(expected_check, failed_check_ids(decision))

    def test_validate_promotion_decision_rejects_forged_embedded_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = write_governance_artifacts(root)
            decision_path = root / "promotion_decision.json"
            self.assertEqual(
                run_cli(promotion_decision_args(artifacts, decision_path)), 0
            )
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            decision["policy"]["id"] = "forged-policy"
            decision["policy"]["requirements"]["require_artifact_validation"] = False
            decision_path.write_text(
                json.dumps(decision, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            self.assertEqual(
                run_cli(
                    ["validate", "--promotion-decision", str(decision_path), "--strict"]
                ),
                1,
            )

    def test_validate_promotion_decision_rejects_forged_lineage_schema_versions(self):
        for artifact_role in (
            "evidence_bundle",
            "eval_summary",
            "external_eval_result",
        ):
            with (
                self.subTest(artifact_role=artifact_role),
                tempfile.TemporaryDirectory() as tmp,
            ):
                root = Path(tmp)
                artifacts = write_governance_artifacts(root)
                decision_path = root / "promotion_decision.json"
                self.assertEqual(
                    run_cli(promotion_decision_args(artifacts, decision_path)), 0
                )
                decision = json.loads(decision_path.read_text(encoding="utf-8"))
                if artifact_role == "external_eval_result":
                    record = decision["external_eval_lineage"]["results"][0]["artifact"]
                else:
                    record = decision["artifacts"][artifact_role]
                record["schema_version"] = "untrusted.v9"
                decision_path.write_text(
                    json.dumps(decision, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )

                self.assertEqual(
                    run_cli(
                        [
                            "validate",
                            "--promotion-decision",
                            str(decision_path),
                            "--strict",
                        ]
                    ),
                    1,
                )
                self.assertEqual(run_cli(["schemas", "--check", str(decision_path)]), 1)

    def test_promotion_decision_rejects_output_aliases_to_external_result(self):
        for alias_kind in ("exact", "hardlink", "symlink"):
            with (
                self.subTest(alias_kind=alias_kind),
                tempfile.TemporaryDirectory() as tmp,
            ):
                root = Path(tmp)
                artifacts = write_governance_artifacts(root)
                result_path = artifacts["external_eval_results"][0]
                before = result_path.read_bytes()
                if alias_kind == "exact":
                    output_path = result_path
                elif alias_kind == "hardlink":
                    output_path = root / "hardlinked_decision.json"
                    try:
                        os.link(result_path, output_path)
                    except OSError as exc:
                        self.skipTest(f"hardlink unavailable: {exc}")
                else:
                    output_path = root / "symlinked_decision.json"
                    try:
                        output_path.symlink_to(result_path)
                    except OSError as exc:
                        self.skipTest(f"symlink unavailable: {exc}")

                with self.assertRaises(SystemExit) as raised:
                    run_cli(promotion_decision_args(artifacts, output_path))

                self.assertEqual(raised.exception.code, 2)
                self.assertEqual(result_path.read_bytes(), before)
                self.assertEqual(
                    json.loads(result_path.read_text(encoding="utf-8"))[
                        "schema_version"
                    ],
                    "hfr.external_eval_result.v1",
                )

    def test_promotion_decision_rejects_output_aliases_to_transitive_eval_sources(self):
        for alias_kind in ("exact", "hardlink", "symlink"):
            with (
                self.subTest(alias_kind=alias_kind),
                tempfile.TemporaryDirectory() as tmp,
            ):
                root = Path(tmp)
                artifacts = write_governance_artifacts(root)
                raw_result_path = (
                    artifacts["external_eval_results"][0].parent
                    / "external_eval_raw_result.json"
                )
                before = raw_result_path.read_bytes()
                if alias_kind == "exact":
                    output_path = raw_result_path
                elif alias_kind == "hardlink":
                    output_path = root / "hardlinked_transitive_decision.json"
                    try:
                        os.link(raw_result_path, output_path)
                    except OSError as exc:
                        self.skipTest(f"hardlink unavailable: {exc}")
                else:
                    output_path = root / "symlinked_transitive_decision.json"
                    try:
                        output_path.symlink_to(raw_result_path)
                    except OSError as exc:
                        self.skipTest(f"symlink unavailable: {exc}")

                with self.assertRaises(SystemExit) as raised:
                    run_cli(promotion_decision_args(artifacts, output_path))

                self.assertEqual(raised.exception.code, 2)
                self.assertEqual(raw_result_path.read_bytes(), before)

    def test_promotion_decision_follows_malformed_eval_summary_local_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = write_governance_artifacts(root)
            eval_summary_path = artifacts["eval_summary"]
            prior_decision_path = eval_summary_path.parent / "prior.json"
            self.assertEqual(
                run_cli(
                    promotion_decision_args(
                        artifacts,
                        prior_decision_path,
                    )
                ),
                0,
            )
            prior_before = prior_decision_path.read_bytes()
            eval_summary = _read_json_object(eval_summary_path)
            result_ref = eval_summary["external_adapter_results"][0]
            result_ref["path"] = prior_decision_path.name
            result_ref.pop("sha256")
            _write_source_json(eval_summary_path, eval_summary)

            with self.assertRaises(SystemExit) as raised:
                run_cli(
                    promotion_decision_args(
                        artifacts,
                        prior_decision_path,
                    )
                )

            self.assertEqual(raised.exception.code, 2)
            self.assertEqual(prior_decision_path.read_bytes(), prior_before)

    def test_promotion_outputs_follow_registry_links_with_repo_root_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "fake_repo"
            repo.mkdir()
            (repo / ".git").mkdir()
            artifacts = write_governance_artifacts(repo)
            prior_decision_path = repo / "prior.json"
            prior_alias_path = repo / "prior_alias_apply.json"
            self.assertEqual(
                run_cli(
                    promotion_decision_args(
                        artifacts,
                        prior_decision_path,
                    )
                ),
                0,
            )
            prior_registry_path = write_model_registry(repo)
            self.assertEqual(
                run_cli(
                    promotion_alias_apply_args(
                        prior_registry_path,
                        prior_decision_path,
                        prior_alias_path,
                    )
                ),
                0,
            )

            registry_subdir = repo / "registry_entries"
            registry_subdir.mkdir()
            entry_path = registry_subdir / "model_registry_entry.json"
            entry = _read_json_object(artifacts["model_registry_entry"])
            for link_id, link_kind, linked_path in (
                ("prior-promotion-decision", "promotion_decision", prior_decision_path),
                ("prior-alias-apply", "promotion_alias_apply", prior_alias_path),
            ):
                linked_bytes = linked_path.read_bytes()
                entry["links"]["evals"].append(
                    {
                        "id": link_id,
                        "kind": link_kind,
                        "metadata": {},
                        "path": linked_path.name,
                        "recorded_at": "2026-07-10T00:00:00Z",
                        "sha256": hashlib.sha256(linked_bytes).hexdigest(),
                        "size_bytes": len(linked_bytes),
                        "status": "ready",
                    }
                )
            _write_source_json(entry_path, entry)
            artifacts["model_registry_entry"].unlink()
            artifacts["model_registry_entry"] = entry_path

            prior_decision_before = prior_decision_path.read_bytes()
            with self.assertRaises(SystemExit) as raised:
                run_cli(
                    promotion_decision_args(
                        artifacts,
                        prior_decision_path,
                    )
                )
            self.assertEqual(raised.exception.code, 2)
            self.assertEqual(
                prior_decision_path.read_bytes(),
                prior_decision_before,
            )

            malformed_entry = json.loads(json.dumps(entry))
            prior_link = next(
                link
                for link in malformed_entry["links"]["evals"]
                if link["id"] == "prior-promotion-decision"
            )
            prior_link.pop("sha256")
            _write_source_json(entry_path, malformed_entry)
            with self.assertRaises(SystemExit) as raised:
                run_cli(
                    promotion_decision_args(
                        artifacts,
                        prior_decision_path,
                    )
                )
            self.assertEqual(raised.exception.code, 2)
            self.assertEqual(
                prior_decision_path.read_bytes(),
                prior_decision_before,
            )
            _write_source_json(entry_path, entry)

            current_decision_path = repo / "current_decision.json"
            self.assertEqual(
                run_cli(
                    promotion_decision_args(
                        artifacts,
                        current_decision_path,
                    )
                ),
                0,
            )
            registry_path = write_model_registry(repo)
            registry = _read_json_object(registry_path)
            registry["entries"][PROMOTION_CANDIDATE_ID] = entry
            _write_source_json(registry_path, registry)
            registry_before = registry_path.read_bytes()
            prior_alias_before = prior_alias_path.read_bytes()

            with self.assertRaises(SystemExit) as raised:
                run_cli(
                    promotion_alias_apply_args(
                        registry_path,
                        current_decision_path,
                        prior_alias_path,
                    )
                )

            self.assertEqual(raised.exception.code, 2)
            self.assertEqual(registry_path.read_bytes(), registry_before)
            self.assertEqual(prior_alias_path.read_bytes(), prior_alias_before)

    def test_promotion_decision_rejects_output_inside_source_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = write_governance_artifacts(root)
            result_directory = root / "external_result_directory"
            result_directory.mkdir()
            artifacts["external_eval_results"] = [result_directory]
            output_path = result_directory / "promotion_decision.json"

            with self.assertRaises(SystemExit) as raised:
                run_cli(promotion_decision_args(artifacts, output_path))

            self.assertEqual(raised.exception.code, 2)
            self.assertEqual(list(result_directory.iterdir()), [])

    def test_promotion_decision_atomic_publish_rejects_post_check_symlink(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = write_governance_artifacts(root)
            output_path = root / "promotion_decision.json"
            protected_path = artifacts["external_eval_results"][0]
            protected_before = protected_path.read_bytes()

            def swap_then_write(path, value, *, expected_sha256):
                output_path.symlink_to(protected_path)
                return atomic_write_json_cas(
                    path,
                    value,
                    expected_sha256=expected_sha256,
                )

            with (
                patch(
                    "flightrecorder.cli.atomic_write_json_cas",
                    side_effect=swap_then_write,
                ),
                self.assertRaises(SystemExit) as raised,
            ):
                run_cli(promotion_decision_args(artifacts, output_path))

            self.assertEqual(raised.exception.code, 2)
            self.assertEqual(protected_path.read_bytes(), protected_before)

    def test_validate_promotion_decision_rejects_unknown_control_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = write_governance_artifacts(root)
            decision_path = root / "promotion_decision.json"
            summary_path = root / "validation.json"
            self.assertEqual(
                run_cli(promotion_decision_args(artifacts, decision_path)), 0
            )

            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            decision["cloud_job_id"] = "job-123"
            decision["models"]["candidate"]["provider_signed_url"] = (
                "https://provider.invalid/model"
            )
            decision["decision"]["promotion_alias_moved"] = True
            decision["checks"][0]["credential_value"] = "redacted-secret"
            decision["artifacts"]["model_card"]["model_download_path"] = "artifact.bin"
            decision["policy"]["limits"]["max_cloud_cost_usd"] = 5
            decision["metrics"]["live_spend_usd"] = 5
            decision["alias_update"]["weights_updated"] = True
            decision["alias_update"]["aliases"][0]["provider_job_id"] = "job-123"
            decision_path.write_text(
                json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )

            self.assertEqual(run_cli(["schemas", "--check", str(decision_path)]), 1)
            code = run_cli(
                [
                    "validate",
                    "--promotion-decision",
                    str(decision_path),
                    "--strict",
                    "--out",
                    str(summary_path),
                ]
            )

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(
                error for target in summary["targets"] for error in target["errors"]
            )
            self.assertIn(
                "promotion_decision contains unknown field(s): ['cloud_job_id'].",
                errors,
            )
            self.assertIn(
                "promotion_decision.models.candidate contains unknown field(s): ['provider_signed_url'].",
                errors,
            )
            self.assertIn(
                "promotion_decision.checks[0] contains unknown field(s): ['credential_value'].",
                errors,
            )
            self.assertIn(
                "promotion_decision.alias_update contains unknown field(s): ['weights_updated'].",
                errors,
            )

    def test_promotion_alias_apply_moves_aliases_after_valid_decision(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = write_governance_artifacts(root)
            registry_path = write_model_registry(root)
            decision_path = root / "promotion_decision.json"
            receipt_path = root / "promotion_alias_apply.json"
            self.assertEqual(
                run_cli(promotion_decision_args(artifacts, decision_path)), 0
            )

            code = run_cli(
                promotion_alias_apply_args(registry_path, decision_path, receipt_path)
            )

            self.assertEqual(code, 0)
            registry = json.loads(registry_path.read_text(encoding="utf-8"))
            self.assertEqual(registry["aliases"]["candidate"], PROMOTION_CANDIDATE_ID)
            self.assertEqual(registry["aliases"]["champion"], PROMOTION_CANDIDATE_ID)
            self.assertEqual(registry["aliases"]["rollback"], "champion-v1")
            self.assertEqual(len(registry["alias_history"]), 1)
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            self.assertTrue(receipt["passed"])
            self.assertEqual(receipt["recommendation"], "alias_update_applied")
            self.assertEqual(
                receipt["promotion_decision"]["size_bytes"],
                decision_path.stat().st_size,
            )
            self.assertNotIn("path", receipt["registry_before"])
            self.assertNotIn("sha256", receipt["registry_before"])
            self.assertNotIn("size_bytes", receipt["registry_before"])
            self.assertEqual(
                receipt["registry_after"]["size_bytes"], registry_path.stat().st_size
            )
            self.assertEqual(
                receipt["registry_after"]["size_bytes"],
                receipt["artifacts"]["registry"]["size_bytes"],
            )
            self.assertEqual(receipt["promotion_decision_validation"]["passed"], True)
            self.assertEqual(
                receipt["alias_history_entry"]["promotion_decision_sha256"],
                receipt["promotion_decision"]["sha256"],
            )
            self.assertEqual(
                receipt["alias_history_entry"]["updated_aliases"]["champion"],
                PROMOTION_CANDIDATE_ID,
            )
            self.assertEqual(
                run_cli(
                    [
                        "validate",
                        "--promotion-alias-apply",
                        str(receipt_path),
                        "--strict",
                    ]
                ),
                0,
            )
            self.assertEqual(run_cli(["schemas", "--check", str(receipt_path)]), 0)

    def test_promotion_alias_apply_ignores_forged_caller_validation_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = write_governance_artifacts(root)
            registry_path = write_model_registry(root)
            decision_path = root / "promotion_decision.json"
            receipt_path = root / "promotion_alias_apply.json"
            self.assertEqual(
                run_cli(promotion_decision_args(artifacts, decision_path)), 0
            )
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            decision.pop("external_eval_lineage")
            decision_path.write_text(
                json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            registry_before = registry_path.read_bytes()

            receipt = apply_promotion_aliases(
                registry_path=registry_path,
                promotion_decision_path=decision_path,
                out_path=receipt_path,
                promotion_decision_validation={
                    "passed": True,
                    "target_count": 1,
                    "error_count": 0,
                    "warning_count": 0,
                    "targets": [{"type": "promotion_decision", "passed": True}],
                },
            )

            self.assertFalse(receipt["passed"])
            self.assertEqual(registry_path.read_bytes(), registry_before)
            self.assertFalse(receipt["promotion_decision_validation"]["passed"])
            self.assertIn("promotion_decision_validated", failed_check_ids(receipt))

    def test_promotion_alias_apply_rejects_output_aliases_to_inputs(self):
        for source_role in ("registry", "promotion_decision"):
            for alias_kind in ("exact", "hardlink", "symlink"):
                with (
                    self.subTest(source_role=source_role, alias_kind=alias_kind),
                    tempfile.TemporaryDirectory() as tmp,
                ):
                    root = Path(tmp)
                    artifacts = write_governance_artifacts(root)
                    registry_path = write_model_registry(root)
                    decision_path = root / "promotion_decision.json"
                    self.assertEqual(
                        run_cli(promotion_decision_args(artifacts, decision_path)), 0
                    )
                    source_path = (
                        registry_path if source_role == "registry" else decision_path
                    )
                    registry_before = registry_path.read_bytes()
                    decision_before = decision_path.read_bytes()
                    if alias_kind == "exact":
                        output_path = source_path
                    elif alias_kind == "hardlink":
                        output_path = root / f"{source_role}_hardlink.json"
                        try:
                            os.link(source_path, output_path)
                        except OSError as exc:
                            self.skipTest(f"hardlink unavailable: {exc}")
                    else:
                        output_path = root / f"{source_role}_symlink.json"
                        try:
                            output_path.symlink_to(source_path)
                        except OSError as exc:
                            self.skipTest(f"symlink unavailable: {exc}")

                    with self.assertRaises(SystemExit) as raised:
                        run_cli(
                            promotion_alias_apply_args(
                                registry_path, decision_path, output_path
                            )
                        )

                    self.assertEqual(raised.exception.code, 2)
                    self.assertEqual(registry_path.read_bytes(), registry_before)
                    self.assertEqual(decision_path.read_bytes(), decision_before)

    def test_promotion_alias_apply_rejects_output_aliases_to_decision_evidence(self):
        for source_role in ("external_eval_result", "raw_result"):
            for alias_kind in ("exact", "hardlink", "symlink"):
                with (
                    self.subTest(source_role=source_role, alias_kind=alias_kind),
                    tempfile.TemporaryDirectory() as tmp,
                ):
                    root = Path(tmp)
                    artifacts = write_governance_artifacts(root)
                    registry_path = write_model_registry(root)
                    decision_path = root / "promotion_decision.json"
                    self.assertEqual(
                        run_cli(promotion_decision_args(artifacts, decision_path)),
                        0,
                    )
                    result_path = artifacts["external_eval_results"][0]
                    source_path = (
                        result_path
                        if source_role == "external_eval_result"
                        else result_path.parent / "external_eval_raw_result.json"
                    )
                    source_before = source_path.read_bytes()
                    registry_before = registry_path.read_bytes()
                    if alias_kind == "exact":
                        output_path = source_path
                    elif alias_kind == "hardlink":
                        output_path = root / f"{source_role}_hardlinked_receipt.json"
                        try:
                            os.link(source_path, output_path)
                        except OSError as exc:
                            self.skipTest(f"hardlink unavailable: {exc}")
                    else:
                        output_path = root / f"{source_role}_symlinked_receipt.json"
                        try:
                            output_path.symlink_to(source_path)
                        except OSError as exc:
                            self.skipTest(f"symlink unavailable: {exc}")

                    with self.assertRaises(SystemExit) as raised:
                        run_cli(
                            promotion_alias_apply_args(
                                registry_path, decision_path, output_path
                            )
                        )

                    self.assertEqual(raised.exception.code, 2)
                    self.assertEqual(registry_path.read_bytes(), registry_before)
                    self.assertEqual(source_path.read_bytes(), source_before)

    def test_promotion_alias_apply_rejects_output_inside_compare_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = write_governance_artifacts(root)
            registry_path = write_model_registry(root)
            decision_path = root / "promotion_decision.json"
            compare_export = root / "compare_export"
            shutil.copytree(
                ROOT
                / "examples"
                / "agentic_training"
                / "promotion_governance"
                / "compare_export",
                compare_export,
            )
            compare_gate = json.loads(
                artifacts["compare_gate"].read_text(encoding="utf-8")
            )
            compare_gate["compare_export"] = str(compare_export)
            _write_source_json(artifacts["compare_gate"], compare_gate)
            registry_entry = json.loads(
                artifacts["model_registry_entry"].read_text(encoding="utf-8")
            )
            _bind_registry_entry_links(
                registry_entry,
                {"evals": artifacts["compare_gate"]},
            )
            _write_source_json(artifacts["model_registry_entry"], registry_entry)
            self.assertEqual(
                run_cli(promotion_decision_args(artifacts, decision_path)), 0
            )
            output_path = compare_export / "promotion_alias_apply.json"
            registry_before = registry_path.read_bytes()

            with self.assertRaises(SystemExit) as raised:
                run_cli(
                    promotion_alias_apply_args(
                        registry_path, decision_path, output_path
                    )
                )

            self.assertEqual(raised.exception.code, 2)
            self.assertEqual(registry_path.read_bytes(), registry_before)
            self.assertFalse(output_path.exists())

    def test_promotion_outputs_reject_missing_semantic_source_containment(self):
        semantic_sources = (
            ("compare_gate", "compare_export"),
            ("promotion_ledger_gate", "promotion_ledger"),
        )
        for operation in ("build", "apply"):
            for role, field_name in semantic_sources:
                for source_kind in ("missing", "redacted"):
                    with (
                        self.subTest(
                            operation=operation,
                            role=role,
                            source_kind=source_kind,
                        ),
                        tempfile.TemporaryDirectory() as tmp,
                    ):
                        root = Path(tmp)
                        artifacts = write_governance_artifacts(root)
                        source_basename = f"{role}_{source_kind}_source"
                        if source_kind == "missing":
                            semantic_source = root / source_basename
                            recorded_source = str(semantic_source)
                        elif source_kind == "redacted":
                            semantic_source = root / source_basename
                            recorded_source = f"<redacted:{source_basename}>"
                        gate_path = artifacts[role]
                        gate = _read_json_object(gate_path)
                        gate[field_name] = recorded_source
                        _write_source_json(gate_path, gate)

                        if operation == "build":
                            output_path = semantic_source / "promotion_decision.json"
                            args = promotion_decision_args(artifacts, output_path)
                            registry_path = None
                            registry_before = None
                        else:
                            decision_path = root / "promotion_decision.json"
                            self.assertEqual(
                                run_cli(
                                    promotion_decision_args(
                                        artifacts,
                                        decision_path,
                                    )
                                ),
                                1,
                            )
                            registry_path = write_model_registry(root)
                            registry_before = registry_path.read_bytes()
                            output_path = semantic_source / "promotion_alias_apply.json"
                            args = promotion_alias_apply_args(
                                registry_path,
                                decision_path,
                                output_path,
                            )

                        with self.assertRaises(SystemExit) as raised:
                            run_cli(args)

                        self.assertEqual(raised.exception.code, 2)
                        self.assertFalse(output_path.exists())
                        if registry_path is not None:
                            self.assertEqual(
                                registry_path.read_bytes(),
                                registry_before,
                            )

    def test_promotion_rejects_nonlocal_semantic_source_references(self):
        semantic_sources = (
            ("compare_gate", "compare_export"),
            ("promotion_ledger_gate", "promotion_ledger"),
        )
        recorded_sources = (
            "https://provider.invalid/governance-source",
            "<redacted:.>",
            "<redacted:..>",
        )
        for role, field_name in semantic_sources:
            for recorded_source in recorded_sources:
                with (
                    self.subTest(
                        role=role,
                        recorded_source=recorded_source,
                    ),
                    tempfile.TemporaryDirectory() as tmp,
                ):
                    root = Path(tmp)
                    artifacts = write_governance_artifacts(root)
                    gate_path = artifacts[role]
                    gate = _read_json_object(gate_path)
                    gate[field_name] = recorded_source
                    _write_source_json(gate_path, gate)
                    gate_before = gate_path.read_bytes()
                    decision_path = root / "promotion_decision.json"

                    self.assertEqual(
                        run_cli(
                            promotion_decision_args(
                                artifacts,
                                decision_path,
                            )
                        ),
                        1,
                    )
                    decision = _read_json_object(decision_path)
                    self.assertFalse(decision["passed"])
                    self.assertIn(
                        f"{role}_semantically_valid",
                        failed_check_ids(decision),
                    )

                    registry_path = write_model_registry(root)
                    registry_before = registry_path.read_bytes()
                    receipt_path = root / "promotion_alias_apply.json"
                    self.assertEqual(
                        run_cli(
                            promotion_alias_apply_args(
                                registry_path,
                                decision_path,
                                receipt_path,
                            )
                        ),
                        1,
                    )
                    receipt = _read_json_object(receipt_path)
                    self.assertFalse(receipt["passed"])
                    self.assertEqual(registry_path.read_bytes(), registry_before)
                    self.assertEqual(gate_path.read_bytes(), gate_before)

    def test_promotion_replays_trainer_launch_preflight_source(self):
        for source_state in ("missing", "stale"):
            with (
                self.subTest(source_state=source_state),
                tempfile.TemporaryDirectory() as tmp,
            ):
                root = Path(tmp)
                artifacts = write_governance_artifacts(root)
                copied_example = root / "agentic_training"
                shutil.copytree(
                    ROOT / "examples" / "agentic_training",
                    copied_example,
                )
                preflight_path = copied_example / "trainer_preflight.json"
                launch_path = artifacts["trainer_launch_check"]
                launch_contract = _read_json_object(launch_path)
                launch_args = [
                    "trainer-launch-check",
                    "--preflight",
                    str(preflight_path),
                    "--out",
                    str(launch_path),
                    "--strict",
                    "--preserve-paths",
                ]
                for gate_id in launch_contract["required_gates"]:
                    launch_args.extend(["--require-gate", gate_id])
                for version in launch_contract["required_dataset_versions"]:
                    launch_args.extend(["--require-dataset-version", version])
                for key, value in launch_contract["required_metadata"].items():
                    launch_args.extend(["--require-metadata", f"{key}={value}"])
                self.assertEqual(run_cli(launch_args), 0)

                if source_state == "missing":
                    preflight_path.unlink()
                    source_before = None
                else:
                    preflight = _read_json_object(preflight_path)
                    preflight["trainer_command"]["argv"][1] = "stale-train.py"
                    preflight["trainer_command"]["raw"] = (
                        "python stale-train.py --dataset training_export --dry-run"
                    )
                    _write_source_json(preflight_path, preflight)
                    source_before = preflight_path.read_bytes()

                decision_path = root / "promotion_decision.json"
                self.assertEqual(
                    run_cli(
                        promotion_decision_args(
                            artifacts,
                            decision_path,
                        )
                    ),
                    1,
                )
                decision = _read_json_object(decision_path)
                self.assertIn(
                    "trainer_launch_check_semantically_valid",
                    failed_check_ids(decision),
                )
                self.assertFalse(decision["alias_update"]["authorized"])

                registry_path = write_model_registry(root)
                registry_before = registry_path.read_bytes()
                receipt_path = root / "promotion_alias_apply.json"
                self.assertEqual(
                    run_cli(
                        promotion_alias_apply_args(
                            registry_path,
                            decision_path,
                            receipt_path,
                        )
                    ),
                    1,
                )
                receipt = _read_json_object(receipt_path)
                self.assertFalse(receipt["passed"])
                self.assertEqual(registry_path.read_bytes(), registry_before)
                if source_before is None:
                    self.assertFalse(preflight_path.exists())
                else:
                    self.assertEqual(preflight_path.read_bytes(), source_before)

    def test_promotion_alias_apply_rolls_back_after_post_check_symlink(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = write_governance_artifacts(root)
            registry_path = write_model_registry(root)
            decision_path = root / "promotion_decision.json"
            receipt_path = root / "promotion_alias_apply.json"
            self.assertEqual(
                run_cli(promotion_decision_args(artifacts, decision_path)), 0
            )
            decision_before = decision_path.read_bytes()
            registry_before = json.loads(registry_path.read_text(encoding="utf-8"))

            def swap_after_registry(path, value, *, expected_sha256):
                result = atomic_write_json_cas(
                    path,
                    value,
                    expected_sha256=expected_sha256,
                )
                if Path(path) == registry_path and not receipt_path.exists():
                    receipt_path.symlink_to(decision_path)
                return result

            with (
                patch(
                    "flightrecorder.governance.atomic_write_json_cas",
                    side_effect=swap_after_registry,
                ),
                self.assertRaises(AtomicJsonError),
            ):
                apply_promotion_aliases(
                    registry_path=registry_path,
                    promotion_decision_path=decision_path,
                    out_path=receipt_path,
                )

            registry_after = json.loads(registry_path.read_text(encoding="utf-8"))
            self.assertEqual(registry_after["aliases"], registry_before["aliases"])
            self.assertEqual(
                registry_after.get("alias_history"),
                registry_before.get("alias_history"),
            )
            self.assertEqual(decision_path.read_bytes(), decision_before)

    def test_promotion_alias_apply_uses_canonical_registry_entries_without_legacy_models(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = write_governance_artifacts(root)
            registry_path = write_model_registry(root)
            registry = json.loads(registry_path.read_text(encoding="utf-8"))
            self.assertNotIn("models", registry)
            self.assertIn("entries", registry)
            decision_path = root / "promotion_decision.json"
            receipt_path = root / "promotion_alias_apply.json"
            self.assertEqual(
                run_cli(promotion_decision_args(artifacts, decision_path)), 0
            )

            code = run_cli(
                promotion_alias_apply_args(registry_path, decision_path, receipt_path)
            )

            self.assertEqual(code, 0)
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            self.assertTrue(receipt["passed"])
            self.assertEqual(
                receipt["registry_after"]["aliases"]["champion"],
                PROMOTION_CANDIDATE_ID,
            )

    def test_promotion_alias_apply_blocks_post_validation_candidate_entry_swap(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = write_governance_artifacts(root)
            registry_path = write_model_registry(root)
            decision_path = root / "promotion_decision.json"
            receipt_path = root / "promotion_alias_apply.json"
            self.assertEqual(
                run_cli(promotion_decision_args(artifacts, decision_path)),
                0,
            )
            registry_before = registry_path.read_bytes()
            source_entry_path = artifacts["model_registry_entry"]
            swapped_entry = _read_json_object(source_entry_path)
            swapped_entry["notes"] = [
                *swapped_entry.get("notes", []),
                "Distinct valid entry written after decision validation.",
            ]
            original_validate = governance._validate_promotion_decision_snapshot

            def validate_then_swap(decision, source_path):
                validation = original_validate(decision, source_path)
                _write_source_json(source_entry_path, swapped_entry)
                return validation

            with patch.object(
                governance,
                "_validate_promotion_decision_snapshot",
                side_effect=validate_then_swap,
            ):
                receipt = apply_promotion_aliases(
                    registry_path=registry_path,
                    promotion_decision_path=decision_path,
                    out_path=receipt_path,
                )

            self.assertFalse(receipt["passed"])
            self.assertEqual(registry_path.read_bytes(), registry_before)
            self.assertIn(
                "registry_candidate_entry_matches_decision_artifact",
                failed_check_ids(receipt),
            )

    def test_validate_promotion_alias_apply_rejects_unknown_control_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = write_governance_artifacts(root)
            registry_path = write_model_registry(root)
            decision_path = root / "promotion_decision.json"
            receipt_path = root / "promotion_alias_apply.json"
            summary_path = root / "validation.json"
            self.assertEqual(
                run_cli(promotion_decision_args(artifacts, decision_path)), 0
            )
            self.assertEqual(
                run_cli(
                    promotion_alias_apply_args(
                        registry_path, decision_path, receipt_path
                    )
                ),
                0,
            )

            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            receipt["promotion_alias_moved"] = True
            receipt["promotion_decision"]["live_endpoint_url"] = (
                "https://provider.invalid/serve"
            )
            receipt["registry_after"]["provider_signed_url"] = (
                "https://provider.invalid/registry"
            )
            receipt["alias_history_entry"]["rollback_applied"] = True
            receipt["artifacts"]["registry"]["private_path"] = "secret/registry.json"
            receipt["metrics"]["cloud_cost_usd"] = 5
            receipt["checks"][0]["credential_value"] = "redacted-secret"
            receipt_path.write_text(
                json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )

            self.assertEqual(run_cli(["schemas", "--check", str(receipt_path)]), 1)
            code = run_cli(
                [
                    "validate",
                    "--promotion-alias-apply",
                    str(receipt_path),
                    "--strict",
                    "--out",
                    str(summary_path),
                ]
            )

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(
                error for target in summary["targets"] for error in target["errors"]
            )
            self.assertIn(
                "promotion_alias_apply contains unknown field(s): ['promotion_alias_moved'].",
                errors,
            )
            self.assertIn(
                "promotion_alias_apply.promotion_decision contains unknown field(s): ['live_endpoint_url'].",
                errors,
            )
            self.assertIn(
                "promotion_alias_apply.registry_after contains unknown field(s): ['provider_signed_url'].",
                errors,
            )
            self.assertIn(
                "promotion_alias_apply.checks[0] contains unknown field(s): ['credential_value'].",
                errors,
            )

    def test_validate_promotion_alias_apply_rejects_stripped_required_checks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = write_governance_artifacts(root)
            registry_path = write_model_registry(root)
            decision_path = root / "promotion_decision.json"
            receipt_path = root / "promotion_alias_apply.json"
            summary_path = root / "validation.json"
            self.assertEqual(
                run_cli(promotion_decision_args(artifacts, decision_path)),
                0,
            )
            self.assertEqual(
                run_cli(
                    promotion_alias_apply_args(
                        registry_path,
                        decision_path,
                        receipt_path,
                    )
                ),
                0,
            )
            receipt = _read_json_object(receipt_path)
            stripped_ids = {
                "registry_validated",
                "promotion_decision_validated",
                "registry_candidate_entry_matches_decision_artifact",
            }
            receipt["checks"] = [
                check for check in receipt["checks"] if check["id"] not in stripped_ids
            ]
            receipt["check_count"] = len(receipt["checks"])
            receipt["metrics"]["check_count"] = len(receipt["checks"])
            _write_source_json(receipt_path, receipt)

            self.assertEqual(
                run_cli(
                    [
                        "validate",
                        "--promotion-alias-apply",
                        str(receipt_path),
                        "--strict",
                        "--out",
                        str(summary_path),
                    ]
                ),
                1,
            )
            summary = _read_json_object(summary_path)
            errors = "\n".join(
                error for target in summary["targets"] for error in target["errors"]
            )
            self.assertIn(
                "checks must exactly match the required ordered alias-application check contract",
                errors,
            )

    def test_validate_promotion_alias_apply_rejects_current_registry_candidate_swap(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = write_governance_artifacts(root)
            registry_path = write_model_registry(root)
            decision_path = root / "promotion_decision.json"
            receipt_path = root / "promotion_alias_apply.json"
            summary_path = root / "validation.json"
            self.assertEqual(
                run_cli(promotion_decision_args(artifacts, decision_path)),
                0,
            )
            self.assertEqual(
                run_cli(
                    promotion_alias_apply_args(
                        registry_path,
                        decision_path,
                        receipt_path,
                    )
                ),
                0,
            )

            registry = _read_json_object(registry_path)
            candidate_entry = registry["entries"][PROMOTION_CANDIDATE_ID]
            candidate_entry["notes"] = [
                *candidate_entry.get("notes", []),
                "Distinct valid current-registry entry.",
            ]
            _write_source_json(registry_path, registry)
            registry_bytes = registry_path.read_bytes()
            receipt = _read_json_object(receipt_path)
            for record in (
                receipt["artifacts"]["registry"],
                receipt["registry_after"],
            ):
                record["sha256"] = hashlib.sha256(registry_bytes).hexdigest()
                record["size_bytes"] = len(registry_bytes)
            _write_source_json(receipt_path, receipt)

            self.assertEqual(
                run_cli(
                    [
                        "validate",
                        "--promotion-alias-apply",
                        str(receipt_path),
                        "--strict",
                        "--out",
                        str(summary_path),
                    ]
                ),
                1,
            )
            summary = _read_json_object(summary_path)
            errors = "\n".join(
                error for target in summary["targets"] for error in target["errors"]
            )
            self.assertIn(
                "current registry candidate entry must exactly match the decision artifact",
                errors,
            )

    def test_validate_promotion_alias_apply_replays_mutated_decision_strictly(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = write_governance_artifacts(root)
            registry_path = write_model_registry(root)
            decision_path = root / "promotion_decision.json"
            receipt_path = root / "promotion_alias_apply.json"
            summary_path = root / "validation.json"
            self.assertEqual(
                run_cli(promotion_decision_args(artifacts, decision_path)),
                0,
            )
            self.assertEqual(
                run_cli(
                    promotion_alias_apply_args(
                        registry_path,
                        decision_path,
                        receipt_path,
                    )
                ),
                0,
            )

            decision = _read_json_object(decision_path)
            decision["cloud_job_id"] = "untrusted-job-123"
            _write_source_json(decision_path, decision)
            decision_bytes = decision_path.read_bytes()
            decision_sha256 = hashlib.sha256(decision_bytes).hexdigest()

            receipt = _read_json_object(receipt_path)
            receipt["promotion_decision"]["sha256"] = decision_sha256
            receipt["promotion_decision"]["size_bytes"] = len(decision_bytes)
            receipt["artifacts"]["promotion_decision"]["sha256"] = decision_sha256
            receipt["artifacts"]["promotion_decision"]["size_bytes"] = len(
                decision_bytes
            )
            receipt["alias_history_entry"]["promotion_decision_sha256"] = (
                decision_sha256
            )

            registry = _read_json_object(registry_path)
            registry["alias_history"][-1]["promotion_decision_sha256"] = decision_sha256
            _write_source_json(registry_path, registry)
            registry_bytes = registry_path.read_bytes()
            for record in (
                receipt["artifacts"]["registry"],
                receipt["registry_after"],
            ):
                record["sha256"] = hashlib.sha256(registry_bytes).hexdigest()
                record["size_bytes"] = len(registry_bytes)
            _write_source_json(receipt_path, receipt)

            self.assertEqual(
                run_cli(
                    [
                        "validate",
                        "--promotion-alias-apply",
                        str(receipt_path),
                        "--strict",
                        "--out",
                        str(summary_path),
                    ]
                ),
                1,
            )
            summary = _read_json_object(summary_path)
            errors = "\n".join(
                error for target in summary["targets"] for error in target["errors"]
            )
            self.assertIn(
                "current promotion decision is invalid: promotion_decision contains unknown field(s): ['cloud_job_id']",
                errors,
            )

    def test_validate_promotion_alias_apply_binds_rollback_to_immutable_decision(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = write_governance_artifacts(root)
            registry_path = write_model_registry(root)
            decision_path = root / "promotion_decision.json"
            receipt_path = root / "promotion_alias_apply.json"
            summary_path = root / "validation.json"
            self.assertEqual(
                run_cli(promotion_decision_args(artifacts, decision_path)),
                0,
            )
            self.assertEqual(
                run_cli(
                    promotion_alias_apply_args(
                        registry_path,
                        decision_path,
                        receipt_path,
                    )
                ),
                0,
            )

            forged_rollback_id = "other-model"
            receipt = _read_json_object(receipt_path)
            receipt["promotion_decision"]["rollback_id"] = forged_rollback_id
            receipt["registry_after"]["aliases"]["rollback"] = forged_rollback_id
            receipt["alias_history_entry"]["updated_aliases"]["rollback"] = (
                forged_rollback_id
            )
            for check in receipt["checks"]:
                if check["id"] == "promotion_decision_alias_targets_match_models":
                    check["actual"]["rollback"] = forged_rollback_id
                    check["expected"]["rollback"] = forged_rollback_id
                elif check["id"] == "rollback_target_registered":
                    check["actual"]["target"] = forged_rollback_id

            registry = _read_json_object(registry_path)
            registry["aliases"]["rollback"] = forged_rollback_id
            registry["alias_history"][-1]["updated_aliases"]["rollback"] = (
                forged_rollback_id
            )
            _write_source_json(registry_path, registry)
            registry_bytes = registry_path.read_bytes()
            for record in (
                receipt["artifacts"]["registry"],
                receipt["registry_after"],
            ):
                record["sha256"] = hashlib.sha256(registry_bytes).hexdigest()
                record["size_bytes"] = len(registry_bytes)
            _write_source_json(receipt_path, receipt)

            self.assertEqual(
                run_cli(
                    [
                        "validate",
                        "--promotion-alias-apply",
                        str(receipt_path),
                        "--strict",
                        "--out",
                        str(summary_path),
                    ]
                ),
                1,
            )
            summary = _read_json_object(summary_path)
            errors = "\n".join(
                error for target in summary["targets"] for error in target["errors"]
            )
            self.assertIn("promotion_decision", errors)
            self.assertIn("rollback", errors)

    def test_validate_promotion_alias_apply_rejects_forged_decision_validation_summary(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = write_governance_artifacts(root)
            registry_path = write_model_registry(root)
            decision_path = root / "promotion_decision.json"
            receipt_path = root / "promotion_alias_apply.json"
            summary_path = root / "validation.json"
            self.assertEqual(
                run_cli(promotion_decision_args(artifacts, decision_path)), 0
            )
            self.assertEqual(
                run_cli(
                    promotion_alias_apply_args(
                        registry_path, decision_path, receipt_path
                    )
                ),
                0,
            )

            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            receipt["promotion_decision_validation"]["warning_count"] = 1
            receipt["promotion_decision_validation"]["targets"][0]["warning_count"] = 1
            receipt_path.write_text(
                json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )

            code = run_cli(
                [
                    "validate",
                    "--promotion-alias-apply",
                    str(receipt_path),
                    "--strict",
                    "--out",
                    str(summary_path),
                ]
            )

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(
                error for target in summary["targets"] for error in target["errors"]
            )
            self.assertIn("promotion_decision_validation.passed", errors)
            self.assertIn("promotion_decision_validation.targets[0].passed", errors)

    def test_validate_promotion_alias_apply_rejects_forged_decision_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = write_governance_artifacts(root)
            registry_path = write_model_registry(root)
            decision_path = root / "promotion_decision.json"
            receipt_path = root / "promotion_alias_apply.json"
            summary_path = root / "validation.json"
            self.assertEqual(
                run_cli(promotion_decision_args(artifacts, decision_path)), 0
            )
            self.assertEqual(
                run_cli(
                    promotion_alias_apply_args(
                        registry_path, decision_path, receipt_path
                    )
                ),
                0,
            )

            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            receipt["promotion_decision"]["size_bytes"] += 1
            receipt_path.write_text(
                json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )

            code = run_cli(
                [
                    "validate",
                    "--promotion-alias-apply",
                    str(receipt_path),
                    "--strict",
                    "--out",
                    str(summary_path),
                ]
            )

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(
                error for target in summary["targets"] for error in target["errors"]
            )
            self.assertIn(
                "promotion_alias_apply.promotion_decision.size_bytes must match artifacts.promotion_decision.size_bytes.",
                errors,
            )

    def test_validate_promotion_alias_apply_rejects_forged_registry_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = write_governance_artifacts(root)
            registry_path = write_model_registry(root)
            decision_path = root / "promotion_decision.json"
            receipt_path = root / "promotion_alias_apply.json"
            summary_path = root / "validation.json"
            self.assertEqual(
                run_cli(promotion_decision_args(artifacts, decision_path)), 0
            )
            self.assertEqual(
                run_cli(
                    promotion_alias_apply_args(
                        registry_path, decision_path, receipt_path
                    )
                ),
                0,
            )

            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            receipt["registry_after"]["size_bytes"] += 1
            receipt_path.write_text(
                json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )

            code = run_cli(
                [
                    "validate",
                    "--promotion-alias-apply",
                    str(receipt_path),
                    "--strict",
                    "--out",
                    str(summary_path),
                ]
            )

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(
                error for target in summary["targets"] for error in target["errors"]
            )
            self.assertIn(
                "promotion_alias_apply.registry_after.size_bytes must match artifacts.registry.size_bytes.",
                errors,
            )

    def test_validate_promotion_alias_apply_rejects_registry_before_size_claim(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = write_governance_artifacts(root)
            registry_path = write_model_registry(root)
            decision_path = root / "promotion_decision.json"
            receipt_path = root / "promotion_alias_apply.json"
            summary_path = root / "validation.json"
            self.assertEqual(
                run_cli(promotion_decision_args(artifacts, decision_path)), 0
            )
            self.assertEqual(
                run_cli(
                    promotion_alias_apply_args(
                        registry_path, decision_path, receipt_path
                    )
                ),
                0,
            )

            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            receipt["registry_before"]["size_bytes"] = receipt["registry_after"][
                "size_bytes"
            ]
            receipt_path.write_text(
                json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )

            self.assertEqual(run_cli(["schemas", "--check", str(receipt_path)]), 1)
            code = run_cli(
                [
                    "validate",
                    "--promotion-alias-apply",
                    str(receipt_path),
                    "--strict",
                    "--out",
                    str(summary_path),
                ]
            )

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(
                error for target in summary["targets"] for error in target["errors"]
            )
            self.assertIn(
                "promotion_alias_apply.registry_before.size_bytes must be absent for snapshot-only refs.",
                errors,
            )

    def test_promotion_alias_apply_blocks_stale_champion_alias_without_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = write_governance_artifacts(root)
            registry_path = write_model_registry(root, champion_alias="other-model")
            decision_path = root / "promotion_decision.json"
            receipt_path = root / "promotion_alias_apply.json"
            self.assertEqual(
                run_cli(promotion_decision_args(artifacts, decision_path)), 0
            )
            before = json.loads(registry_path.read_text(encoding="utf-8"))

            code = run_cli(
                promotion_alias_apply_args(registry_path, decision_path, receipt_path)
            )

            self.assertEqual(code, 1)
            after = json.loads(registry_path.read_text(encoding="utf-8"))
            self.assertEqual(after, before)
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            self.assertFalse(receipt["passed"])
            self.assertIn(
                "champion_alias_matches_previous_target", failed_check_ids(receipt)
            )
            self.assertEqual(
                run_cli(
                    [
                        "validate",
                        "--promotion-alias-apply",
                        str(receipt_path),
                        "--strict",
                    ]
                ),
                0,
            )

    def test_promotion_alias_apply_blocks_malformed_alias_history_without_mutation(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = write_governance_artifacts(root)
            registry_path = write_model_registry(root, alias_history="not-a-list")
            decision_path = root / "promotion_decision.json"
            receipt_path = root / "promotion_alias_apply.json"
            self.assertEqual(
                run_cli(promotion_decision_args(artifacts, decision_path)), 0
            )
            before = json.loads(registry_path.read_text(encoding="utf-8"))

            code = run_cli(
                promotion_alias_apply_args(registry_path, decision_path, receipt_path)
            )

            self.assertEqual(code, 1)
            after = json.loads(registry_path.read_text(encoding="utf-8"))
            self.assertEqual(after, before)
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            self.assertFalse(receipt["passed"])
            self.assertIsNone(receipt["alias_history_entry"])
            self.assertIn("registry_alias_history_list", failed_check_ids(receipt))
            self.assertEqual(
                run_cli(
                    [
                        "validate",
                        "--promotion-alias-apply",
                        str(receipt_path),
                        "--strict",
                    ]
                ),
                0,
            )

    def test_promotion_alias_apply_blocks_symlinked_registry_parent_without_mutation(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = write_governance_artifacts(root)
            registry_path = write_model_registry(root)
            decision_path = root / "promotion_decision.json"
            receipt_path = root / "promotion_alias_apply.json"
            self.assertEqual(
                run_cli(promotion_decision_args(artifacts, decision_path)), 0
            )
            linked_target = root / "linked_target"
            linked_target.mkdir()
            linked_registry = linked_target / "model_registry.json"
            linked_registry.write_text(
                registry_path.read_text(encoding="utf-8"), encoding="utf-8"
            )
            linked_parent = root / "linked_registry"
            try:
                linked_parent.symlink_to(linked_target, target_is_directory=True)
            except (NotImplementedError, OSError) as exc:
                self.skipTest(f"symlink unavailable: {exc}")
            before = json.loads(linked_registry.read_text(encoding="utf-8"))

            code = run_cli(
                promotion_alias_apply_args(
                    linked_parent / "model_registry.json", decision_path, receipt_path
                )
            )

            self.assertEqual(code, 1)
            after = json.loads(linked_registry.read_text(encoding="utf-8"))
            self.assertEqual(after, before)
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            registry_artifact = receipt["artifacts"]["registry"]
            self.assertFalse(registry_artifact["exists"])
            self.assertEqual(registry_artifact["kind"], "other")
            self.assertNotIn("sha256", registry_artifact)
            failed_ids = failed_check_ids(receipt)
            self.assertIn("registry_present", failed_ids)
            self.assertIn("registry_schema", failed_ids)
            self.assertEqual(
                run_cli(
                    [
                        "validate",
                        "--promotion-alias-apply",
                        str(receipt_path),
                        "--strict",
                    ]
                ),
                0,
            )

    def test_promotion_alias_apply_blocks_symlinked_decision_parent_without_mutation(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = write_governance_artifacts(root)
            registry_path = write_model_registry(root)
            decision_path = root / "promotion_decision.json"
            receipt_path = root / "promotion_alias_apply.json"
            self.assertEqual(
                run_cli(promotion_decision_args(artifacts, decision_path)), 0
            )
            linked_target = root / "linked_target"
            linked_target.mkdir()
            (linked_target / "promotion_decision.json").write_text(
                decision_path.read_text(encoding="utf-8"), encoding="utf-8"
            )
            linked_parent = root / "linked_decision"
            try:
                linked_parent.symlink_to(linked_target, target_is_directory=True)
            except (NotImplementedError, OSError) as exc:
                self.skipTest(f"symlink unavailable: {exc}")
            before = json.loads(registry_path.read_text(encoding="utf-8"))

            code = run_cli(
                promotion_alias_apply_args(
                    registry_path,
                    linked_parent / "promotion_decision.json",
                    receipt_path,
                )
            )

            self.assertEqual(code, 1)
            after = json.loads(registry_path.read_text(encoding="utf-8"))
            self.assertEqual(after, before)
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            decision_artifact = receipt["artifacts"]["promotion_decision"]
            self.assertFalse(decision_artifact["exists"])
            self.assertEqual(decision_artifact["kind"], "other")
            self.assertNotIn("sha256", decision_artifact)
            failed_ids = failed_check_ids(receipt)
            self.assertIn("promotion_decision_present", failed_ids)
            self.assertIn("promotion_decision_schema", failed_ids)
            self.assertIn("promotion_decision_validated", failed_ids)
            self.assertEqual(
                run_cli(
                    [
                        "validate",
                        "--promotion-alias-apply",
                        str(receipt_path),
                        "--strict",
                    ]
                ),
                0,
            )

    def test_promotion_rollback_receipt_validates_current_champion_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            registry_path = write_model_registry(root)
            receipt_path = root / "rollback.json"

            code = run_cli(promotion_rollback_receipt_args(registry_path, receipt_path))

            self.assertEqual(code, 0)
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            self.assertEqual(
                receipt["schema_version"], "hfr.promotion_rollback_receipt.v1"
            )
            self.assertTrue(receipt["passed"])
            self.assertTrue(receipt["available"])
            self.assertEqual(receipt["rollback_id"], "champion-v1")
            self.assertEqual(receipt["registry"]["aliases"]["champion"], "champion-v1")
            self.assertEqual(
                receipt["registry"]["size_bytes"], registry_path.stat().st_size
            )
            self.assertEqual(
                receipt["registry"]["size_bytes"],
                receipt["artifacts"]["registry"]["size_bytes"],
            )
            self.assertEqual(
                run_cli(
                    [
                        "validate",
                        "--promotion-rollback-receipt",
                        str(receipt_path),
                        "--strict",
                    ]
                ),
                0,
            )
            self.assertEqual(run_cli(["schemas", "--check", str(receipt_path)]), 0)

    def test_promotion_rollback_receipt_blocks_symlinked_registry_parent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            registry_path = write_model_registry(root)
            linked_target = root / "linked_target"
            linked_target.mkdir()
            (linked_target / "model_registry.json").write_text(
                registry_path.read_text(encoding="utf-8"), encoding="utf-8"
            )
            linked_parent = root / "linked_registry"
            receipt_path = root / "rollback.json"
            try:
                linked_parent.symlink_to(linked_target, target_is_directory=True)
            except (NotImplementedError, OSError) as exc:
                self.skipTest(f"symlink unavailable: {exc}")

            code = run_cli(
                promotion_rollback_receipt_args(
                    linked_parent / "model_registry.json", receipt_path
                )
            )

            self.assertEqual(code, 1)
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            registry_artifact = receipt["artifacts"]["registry"]
            self.assertFalse(registry_artifact["exists"])
            self.assertEqual(registry_artifact["kind"], "other")
            self.assertNotIn("sha256", registry_artifact)
            failed_ids = failed_check_ids(receipt)
            self.assertIn("registry_present", failed_ids)
            self.assertIn("registry_schema", failed_ids)
            self.assertEqual(
                run_cli(
                    [
                        "validate",
                        "--promotion-rollback-receipt",
                        str(receipt_path),
                        "--strict",
                    ]
                ),
                0,
            )

    def test_validate_promotion_rollback_receipt_rejects_forged_registry_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            registry_path = write_model_registry(root)
            receipt_path = root / "rollback.json"
            summary_path = root / "validation.json"
            self.assertEqual(
                run_cli(promotion_rollback_receipt_args(registry_path, receipt_path)), 0
            )

            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            receipt["registry"]["size_bytes"] += 1
            receipt_path.write_text(
                json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )

            code = run_cli(
                [
                    "validate",
                    "--promotion-rollback-receipt",
                    str(receipt_path),
                    "--strict",
                    "--out",
                    str(summary_path),
                ]
            )

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(
                error for target in summary["targets"] for error in target["errors"]
            )
            self.assertIn(
                "promotion_rollback_receipt.registry.size_bytes must match artifacts.registry.size_bytes.",
                errors,
            )

    def test_validate_promotion_rollback_receipt_rejects_stale_registry_alias_snapshot(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            registry_path = write_model_registry(root)
            receipt_path = root / "rollback.json"
            summary_path = root / "validation.json"
            self.assertEqual(
                run_cli(promotion_rollback_receipt_args(registry_path, receipt_path)), 0
            )

            registry = json.loads(registry_path.read_text(encoding="utf-8"))
            registry["aliases"]["champion"] = "other-model"
            registry_path.write_text(
                json.dumps(registry, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            registry_sha = hashlib.sha256(registry_path.read_bytes()).hexdigest()
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            receipt["artifacts"]["registry"]["sha256"] = registry_sha
            receipt["registry"]["sha256"] = registry_sha
            receipt_path.write_text(
                json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )

            code = run_cli(
                [
                    "validate",
                    "--promotion-rollback-receipt",
                    str(receipt_path),
                    "--strict",
                    "--out",
                    str(summary_path),
                ]
            )

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(
                error for target in summary["targets"] for error in target["errors"]
            )
            self.assertIn("promotion_rollback_receipt.registry.aliases", errors)

    def test_validate_promotion_rollback_receipt_rejects_unknown_control_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            registry_path = write_model_registry(root)
            receipt_path = root / "rollback.json"
            summary_path = root / "validation.json"
            self.assertEqual(
                run_cli(promotion_rollback_receipt_args(registry_path, receipt_path)), 0
            )

            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            receipt["rollback_applied"] = True
            receipt["rollback"]["live_endpoint_url"] = "https://provider.invalid/serve"
            receipt["registry"]["model_download_path"] = "artifact.bin"
            receipt["artifacts"]["registry"]["provider_signed_url"] = (
                "https://provider.invalid/registry"
            )
            receipt["metrics"]["cloud_cost_usd"] = 5
            receipt["checks"][0]["credential_value"] = "redacted-secret"
            receipt_path.write_text(
                json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )

            self.assertEqual(run_cli(["schemas", "--check", str(receipt_path)]), 1)
            code = run_cli(
                [
                    "validate",
                    "--promotion-rollback-receipt",
                    str(receipt_path),
                    "--strict",
                    "--out",
                    str(summary_path),
                ]
            )

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(
                error for target in summary["targets"] for error in target["errors"]
            )
            self.assertIn(
                "promotion_rollback_receipt contains unknown field(s): ['rollback_applied'].",
                errors,
            )
            self.assertIn(
                "promotion_rollback_receipt.rollback contains unknown field(s): ['live_endpoint_url'].",
                errors,
            )
            self.assertIn(
                "promotion_rollback_receipt.registry contains unknown field(s): ['model_download_path'].",
                errors,
            )
            self.assertIn(
                "promotion_rollback_receipt.checks[0] contains unknown field(s): ['credential_value'].",
                errors,
            )

    def test_promotion_decision_accepts_passing_rollback_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = write_governance_artifacts(root)
            registry_path = write_model_registry(root)
            receipt_path = root / "rollback.json"
            decision_path = root / "promotion_decision.json"
            self.assertEqual(
                run_cli(promotion_rollback_receipt_args(registry_path, receipt_path)), 0
            )
            artifacts["rollback_metadata"] = receipt_path

            code = run_cli(promotion_decision_args(artifacts, decision_path))

            self.assertEqual(code, 0)
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            self.assertTrue(decision["passed"])
            self.assertNotIn("rollback_receipt_passed", failed_check_ids(decision))
            self.assertEqual(
                run_cli(
                    ["validate", "--promotion-decision", str(decision_path), "--strict"]
                ),
                0,
            )

    def test_promotion_decision_blocks_failed_rollback_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = write_governance_artifacts(root)
            registry_path = write_model_registry(root, champion_alias="other-model")
            receipt_path = root / "rollback.json"
            decision_path = root / "promotion_decision.json"
            self.assertEqual(
                run_cli(promotion_rollback_receipt_args(registry_path, receipt_path)), 1
            )
            artifacts["rollback_metadata"] = receipt_path

            code = run_cli(promotion_decision_args(artifacts, decision_path))

            self.assertEqual(code, 1)
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            self.assertFalse(decision["passed"])
            failed_ids = failed_check_ids(decision)
            self.assertIn("rollback_metadata_matches_target", failed_ids)
            self.assertIn("rollback_receipt_passed", failed_ids)
            self.assertEqual(
                run_cli(
                    ["validate", "--promotion-decision", str(decision_path), "--strict"]
                ),
                0,
            )

    def test_promotion_release_record_binds_decision_cards_alias_rollback_eval_and_notes(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = write_governance_artifacts(root)
            training_export = write_training_export(root)
            cards_dir = root / "cards"
            decision_path = root / "promotion_decision.json"
            registry_path = write_model_registry(root)
            alias_receipt_path = root / "promotion_alias_apply.json"
            release_notes_path = write_release_notes(root)
            release_record_path = root / "promotion_release_record.json"
            self.assertEqual(
                run_cli(promotion_cards_args(artifacts, training_export, cards_dir)), 0
            )
            artifacts["model_card"] = cards_dir / "MODEL_CARD.md"
            artifacts["dataset_card"] = cards_dir / "DATASET_CARD.md"
            self.assertEqual(
                run_cli(promotion_decision_args(artifacts, decision_path)), 0
            )
            self.assertEqual(
                run_cli(
                    promotion_alias_apply_args(
                        registry_path, decision_path, alias_receipt_path
                    )
                ),
                0,
            )

            code = run_cli(
                promotion_release_record_args(
                    artifacts,
                    cards_dir,
                    decision_path,
                    alias_receipt_path,
                    release_notes_path,
                    release_record_path,
                )
            )

            self.assertEqual(code, 0)
            record = json.loads(release_record_path.read_text(encoding="utf-8"))
            self.assertEqual(
                record["schema_version"], "hfr.promotion_release_record.v1"
            )
            self.assertTrue(record["passed"])
            self.assertEqual(record["recommendation"], "publish_release")
            self.assertEqual(record["release"]["candidate_id"], PROMOTION_CANDIDATE_ID)
            self.assertEqual(record["release"]["rollback_id"], "champion-v1")
            self.assertEqual(record["release"]["dataset_id"], "dataset-v1")
            self.assertEqual(record["artifact_validation"]["passed"], True)
            self.assertEqual(
                run_cli(
                    [
                        "validate",
                        "--promotion-release-record",
                        str(release_record_path),
                        "--strict",
                    ]
                ),
                0,
            )
            self.assertEqual(
                run_cli(["schemas", "--check", str(release_record_path)]), 0
            )

    def test_validate_promotion_release_record_rejects_unknown_control_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = write_governance_artifacts(root)
            training_export = write_training_export(root)
            cards_dir = root / "cards"
            decision_path = root / "promotion_decision.json"
            registry_path = write_model_registry(root)
            alias_receipt_path = root / "promotion_alias_apply.json"
            release_notes_path = write_release_notes(root)
            release_record_path = root / "promotion_release_record.json"
            summary_path = root / "validation.json"
            self.assertEqual(
                run_cli(promotion_cards_args(artifacts, training_export, cards_dir)), 0
            )
            artifacts["model_card"] = cards_dir / "MODEL_CARD.md"
            artifacts["dataset_card"] = cards_dir / "DATASET_CARD.md"
            self.assertEqual(
                run_cli(promotion_decision_args(artifacts, decision_path)), 0
            )
            self.assertEqual(
                run_cli(
                    promotion_alias_apply_args(
                        registry_path, decision_path, alias_receipt_path
                    )
                ),
                0,
            )
            self.assertEqual(
                run_cli(
                    promotion_release_record_args(
                        artifacts,
                        cards_dir,
                        decision_path,
                        alias_receipt_path,
                        release_notes_path,
                        release_record_path,
                    )
                ),
                0,
            )

            record = json.loads(release_record_path.read_text(encoding="utf-8"))
            record["publish_endpoint_url"] = "https://provider.invalid/release"
            record["release"]["cloud_job_id"] = "job-123"
            record["checks"][0]["credential_value"] = "redacted-secret"
            record["artifacts"]["promotion_decision"]["private_path"] = (
                "secret/promotion_decision.json"
            )
            record["artifact_validation"]["provider_job_id"] = "job-123"
            record["artifact_validation"]["targets"][0]["signed_url"] = (
                "https://provider.invalid/validation"
            )
            record["bindings"]["model_download_path"] = "artifact.bin"
            record["metrics"]["cloud_cost_usd"] = 5
            record["policy"]["promotion_decision_policy"]["live_override"] = True
            release_record_path.write_text(
                json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )

            self.assertEqual(
                run_cli(["schemas", "--check", str(release_record_path)]), 1
            )
            code = run_cli(
                [
                    "validate",
                    "--promotion-release-record",
                    str(release_record_path),
                    "--strict",
                    "--out",
                    str(summary_path),
                ]
            )

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(
                error for target in summary["targets"] for error in target["errors"]
            )
            self.assertIn(
                "promotion_release_record contains unknown field(s): ['publish_endpoint_url'].",
                errors,
            )
            self.assertIn(
                "promotion_release_record.release contains unknown field(s): ['cloud_job_id'].",
                errors,
            )
            self.assertIn(
                "promotion_release_record.checks[0] contains unknown field(s): ['credential_value'].",
                errors,
            )
            self.assertIn(
                "promotion_release_record.artifact_validation contains unknown field(s): ['provider_job_id'].",
                errors,
            )
            self.assertIn(
                "promotion_release_record.bindings contains unknown field(s): ['model_download_path'].",
                errors,
            )

    def test_validate_promotion_release_record_rejects_forged_card_binding_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = write_governance_artifacts(root)
            training_export = write_training_export(root)
            cards_dir = root / "cards"
            decision_path = root / "promotion_decision.json"
            registry_path = write_model_registry(root)
            alias_receipt_path = root / "promotion_alias_apply.json"
            release_notes_path = write_release_notes(root)
            release_record_path = root / "promotion_release_record.json"
            summary_path = root / "validation.json"
            self.assertEqual(
                run_cli(promotion_cards_args(artifacts, training_export, cards_dir)), 0
            )
            artifacts["model_card"] = cards_dir / "MODEL_CARD.md"
            artifacts["dataset_card"] = cards_dir / "DATASET_CARD.md"
            self.assertEqual(
                run_cli(promotion_decision_args(artifacts, decision_path)), 0
            )
            self.assertEqual(
                run_cli(
                    promotion_alias_apply_args(
                        registry_path, decision_path, alias_receipt_path
                    )
                ),
                0,
            )
            self.assertEqual(
                run_cli(
                    promotion_release_record_args(
                        artifacts,
                        cards_dir,
                        decision_path,
                        alias_receipt_path,
                        release_notes_path,
                        release_record_path,
                    )
                ),
                0,
            )
            record = json.loads(release_record_path.read_text(encoding="utf-8"))
            record["bindings"]["model_card_sha256"] = "0" * 64
            release_record_path.write_text(
                json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )

            code = run_cli(
                [
                    "validate",
                    "--promotion-release-record",
                    str(release_record_path),
                    "--strict",
                    "--out",
                    str(summary_path),
                ]
            )

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(
                error for target in summary["targets"] for error in target["errors"]
            )
            self.assertIn("promotion_release_record.bindings.model_card_sha256", errors)
            self.assertIn("promotion_cards.artifacts.model_card.sha256", errors)

    def test_governance_smoke_path_archives_release_record_with_promotion_history(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            root = Path(tmp)
            artifacts = write_governance_artifacts(root)
            training_export = write_training_export(root)
            action_gate_path = root / "action_ledger_gate.json"
            decision_gate_path = root / "decision_gate.json"
            promotion_ledger_path = root / "promotion_ledger.json"
            promotion_ledger_gate_path = root / "promotion_ledger_gate.json"
            cards_dir = root / "cards"
            decision_path = root / "promotion_decision.json"
            registry_path = write_model_registry(root)
            alias_receipt_path = root / "promotion_alias_apply.json"
            release_notes_path = write_release_notes(root)
            release_record_path = root / "promotion_release_record.json"
            archive_dir = root / "promotion_archive"
            action_gate_ref = str(action_gate_path.relative_to(ROOT))
            decision_gate_ref = str(decision_gate_path.relative_to(ROOT))
            action_gate_path.write_text(
                json.dumps(
                    {
                        "schema_version": "hfr.action_ledger_gate.v1",
                        "passed": True,
                        "decision": {
                            "readiness": "ready",
                            "recommendation": "promote_iteration",
                            "summary": "ready for promotion history",
                            "blocking_check_count": 0,
                            "key_metrics": {"recurring_action_count": 0},
                        },
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            self.assertEqual(
                run_cli(
                    [
                        "gate-decision",
                        "--artifact",
                        action_gate_ref,
                        "--expect-recommendation",
                        "promote_iteration",
                        "--expect-readiness",
                        "ready",
                        "--require-passed",
                        "--out",
                        str(decision_gate_path),
                    ]
                ),
                0,
            )
            self.assertEqual(
                run_cli(
                    [
                        "promotion-ledger",
                        "--decision-gate",
                        decision_gate_ref,
                        "--out",
                        str(promotion_ledger_path),
                    ]
                ),
                0,
            )
            self.assertEqual(
                run_cli(
                    [
                        "gate-promotion-ledger",
                        "--promotion-ledger",
                        str(promotion_ledger_path),
                        "--policy",
                        str(
                            ROOT / "examples" / "promotion_ledger_gate_policy.demo.json"
                        ),
                        "--out",
                        str(promotion_ledger_gate_path),
                    ]
                ),
                0,
            )
            artifacts["promotion_ledger_gate"] = promotion_ledger_gate_path
            self.assertEqual(
                run_cli(promotion_cards_args(artifacts, training_export, cards_dir)), 0
            )
            artifacts["model_card"] = cards_dir / "MODEL_CARD.md"
            artifacts["dataset_card"] = cards_dir / "DATASET_CARD.md"
            self.assertEqual(
                run_cli(promotion_decision_args(artifacts, decision_path)), 0
            )
            self.assertEqual(
                run_cli(
                    promotion_alias_apply_args(
                        registry_path, decision_path, alias_receipt_path
                    )
                ),
                0,
            )
            self.assertEqual(
                run_cli(
                    promotion_release_record_args(
                        artifacts,
                        cards_dir,
                        decision_path,
                        alias_receipt_path,
                        release_notes_path,
                        release_record_path,
                    )
                ),
                0,
            )

            code = run_cli(
                [
                    "promotion-archive",
                    "--promotion-ledger",
                    str(promotion_ledger_path),
                    "--promotion-ledger-gate",
                    str(promotion_ledger_gate_path),
                    "--decision-gate",
                    str(decision_gate_path),
                    "--promotion-release-record",
                    str(release_record_path),
                    "--out",
                    str(archive_dir),
                    "--require-self-contained",
                ]
            )

            self.assertEqual(code, 0)
            archive = json.loads(
                (archive_dir / "promotion_archive.json").read_text(encoding="utf-8")
            )
            roles = {artifact["role"] for artifact in archive["artifacts"]}
            self.assertEqual(
                roles,
                {
                    "promotion_ledger",
                    "promotion_ledger_gate",
                    "decision_gate",
                    "source_artifact",
                    "promotion_release_record",
                },
            )
            self.assertTrue(archive["self_contained"])
            self.assertEqual(archive["metrics"]["promotion_release_record_count"], 1)
            self.assertEqual(
                run_cli(
                    [
                        "validate",
                        "--promotion-cards",
                        str(cards_dir),
                        "--promotion-decision",
                        str(decision_path),
                        "--promotion-alias-apply",
                        str(alias_receipt_path),
                        "--promotion-release-record",
                        str(release_record_path),
                        "--promotion-archive",
                        str(archive_dir),
                        "--strict",
                    ]
                ),
                0,
            )

    def _prepare_promotion_release_record_inputs(
        self, root: Path, *, promotion_policy: Path | None = None
    ):
        artifacts = write_governance_artifacts(root)
        training_export = write_training_export(root)
        cards_dir = root / "cards"
        decision_path = root / "promotion_decision.json"
        registry_path = write_model_registry(root)
        alias_receipt_path = root / "promotion_alias_apply.json"
        release_notes_path = write_release_notes(root)
        release_record_path = root / "promotion_release_record.json"
        self.assertEqual(
            run_cli(promotion_cards_args(artifacts, training_export, cards_dir)), 0
        )
        artifacts["model_card"] = cards_dir / "MODEL_CARD.md"
        artifacts["dataset_card"] = cards_dir / "DATASET_CARD.md"
        self.assertEqual(
            run_cli(
                promotion_decision_args(
                    artifacts, decision_path, promotion_policy=promotion_policy
                )
            ),
            0,
        )
        self.assertEqual(
            run_cli(
                promotion_alias_apply_args(
                    registry_path, decision_path, alias_receipt_path
                )
            ),
            0,
        )
        return {
            "artifacts": artifacts,
            "cards_dir": cards_dir,
            "decision_path": decision_path,
            "alias_receipt_path": alias_receipt_path,
            "release_notes_path": release_notes_path,
            "release_record_path": release_record_path,
        }

    def test_promotion_release_record_blocks_symlinked_cards_parent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs = self._prepare_promotion_release_record_inputs(root)
            linked_parent = root / "linked_inputs"
            try:
                linked_parent.symlink_to(root, target_is_directory=True)
            except (NotImplementedError, OSError) as exc:
                self.skipTest(f"symlink unavailable: {exc}")

            code = run_cli(
                promotion_release_record_args(
                    inputs["artifacts"],
                    linked_parent / "cards",
                    inputs["decision_path"],
                    inputs["alias_receipt_path"],
                    inputs["release_notes_path"],
                    inputs["release_record_path"],
                )
            )

            self.assertEqual(code, 1)
            record = json.loads(
                inputs["release_record_path"].read_text(encoding="utf-8")
            )
            cards_artifact = record["artifacts"]["promotion_cards"]
            self.assertFalse(cards_artifact["exists"])
            self.assertEqual(cards_artifact["kind"], "other")
            self.assertNotIn("sha256", cards_artifact)
            self.assertIn("promotion_cards_present", failed_check_ids(record))
            self.assertIn("promotion_cards_schema", failed_check_ids(record))

    def test_promotion_release_record_blocks_symlinked_release_notes_parent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs = self._prepare_promotion_release_record_inputs(root)
            linked_target = root / "linked_target"
            linked_target.mkdir()
            (linked_target / "RELEASE_NOTES.md").write_text(
                inputs["release_notes_path"].read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            linked_parent = root / "linked_notes"
            try:
                linked_parent.symlink_to(linked_target, target_is_directory=True)
            except (NotImplementedError, OSError) as exc:
                self.skipTest(f"symlink unavailable: {exc}")

            code = run_cli(
                promotion_release_record_args(
                    inputs["artifacts"],
                    inputs["cards_dir"],
                    inputs["decision_path"],
                    inputs["alias_receipt_path"],
                    linked_parent / "RELEASE_NOTES.md",
                    inputs["release_record_path"],
                )
            )

            self.assertEqual(code, 1)
            record = json.loads(
                inputs["release_record_path"].read_text(encoding="utf-8")
            )
            notes_artifact = record["artifacts"]["release_notes"]
            self.assertFalse(notes_artifact["exists"])
            self.assertEqual(notes_artifact["kind"], "other")
            self.assertNotIn("sha256", notes_artifact)
            self.assertIn("release_notes_present", failed_check_ids(record))

    def test_promotion_release_record_blocks_symlinked_policy_parent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            policy_path = write_promotion_policy(root)
            inputs = self._prepare_promotion_release_record_inputs(
                root, promotion_policy=policy_path
            )
            linked_target = root / "linked_target"
            linked_target.mkdir()
            (linked_target / "promotion_policy.json").write_text(
                policy_path.read_text(encoding="utf-8"), encoding="utf-8"
            )
            linked_parent = root / "linked_policy"
            try:
                linked_parent.symlink_to(linked_target, target_is_directory=True)
            except (NotImplementedError, OSError) as exc:
                self.skipTest(f"symlink unavailable: {exc}")

            code = run_cli(
                promotion_release_record_args(
                    inputs["artifacts"],
                    inputs["cards_dir"],
                    inputs["decision_path"],
                    inputs["alias_receipt_path"],
                    inputs["release_notes_path"],
                    inputs["release_record_path"],
                    promotion_policy=linked_parent / "promotion_policy.json",
                )
            )

            self.assertEqual(code, 1)
            record = json.loads(
                inputs["release_record_path"].read_text(encoding="utf-8")
            )
            policy_artifact = record["policy"]["release_policy"]["artifact"]
            self.assertFalse(policy_artifact["exists"])
            self.assertEqual(policy_artifact["kind"], "other")
            self.assertNotIn("sha256", policy_artifact)
            failed_ids = failed_check_ids(record)
            self.assertIn("promotion_policy_present", failed_ids)
            self.assertIn("promotion_policy_matches_decision", failed_ids)
            self.assertEqual(
                run_cli(
                    [
                        "validate",
                        "--promotion-release-record",
                        str(inputs["release_record_path"]),
                        "--strict",
                    ]
                ),
                0,
            )

    def test_validate_promotion_release_record_rejects_stale_release_notes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = write_governance_artifacts(root)
            training_export = write_training_export(root)
            cards_dir = root / "cards"
            decision_path = root / "promotion_decision.json"
            registry_path = write_model_registry(root)
            alias_receipt_path = root / "promotion_alias_apply.json"
            release_notes_path = write_release_notes(root)
            release_record_path = root / "promotion_release_record.json"
            self.assertEqual(
                run_cli(promotion_cards_args(artifacts, training_export, cards_dir)), 0
            )
            artifacts["model_card"] = cards_dir / "MODEL_CARD.md"
            artifacts["dataset_card"] = cards_dir / "DATASET_CARD.md"
            self.assertEqual(
                run_cli(promotion_decision_args(artifacts, decision_path)), 0
            )
            self.assertEqual(
                run_cli(
                    promotion_alias_apply_args(
                        registry_path, decision_path, alias_receipt_path
                    )
                ),
                0,
            )
            self.assertEqual(
                run_cli(
                    promotion_release_record_args(
                        artifacts,
                        cards_dir,
                        decision_path,
                        alias_receipt_path,
                        release_notes_path,
                        release_record_path,
                    )
                ),
                0,
            )

            release_notes_path.write_text(
                "# Release Notes\n\nChanged after release record.\n", encoding="utf-8"
            )

            self.assertEqual(
                run_cli(
                    [
                        "validate",
                        "--promotion-release-record",
                        str(release_record_path),
                        "--strict",
                    ]
                ),
                1,
            )

    def test_validate_promotion_release_record_requires_artifact_validation_targets(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = write_governance_artifacts(root)
            training_export = write_training_export(root)
            cards_dir = root / "cards"
            decision_path = root / "promotion_decision.json"
            registry_path = write_model_registry(root)
            alias_receipt_path = root / "promotion_alias_apply.json"
            release_notes_path = write_release_notes(root)
            release_record_path = root / "promotion_release_record.json"
            summary_path = root / "validation.json"
            self.assertEqual(
                run_cli(promotion_cards_args(artifacts, training_export, cards_dir)), 0
            )
            artifacts["model_card"] = cards_dir / "MODEL_CARD.md"
            artifacts["dataset_card"] = cards_dir / "DATASET_CARD.md"
            self.assertEqual(
                run_cli(promotion_decision_args(artifacts, decision_path)), 0
            )
            self.assertEqual(
                run_cli(
                    promotion_alias_apply_args(
                        registry_path, decision_path, alias_receipt_path
                    )
                ),
                0,
            )
            self.assertEqual(
                run_cli(
                    promotion_release_record_args(
                        artifacts,
                        cards_dir,
                        decision_path,
                        alias_receipt_path,
                        release_notes_path,
                        release_record_path,
                    )
                ),
                0,
            )
            record = json.loads(release_record_path.read_text(encoding="utf-8"))
            record["artifact_validation"]["target_count"] = 0
            release_record_path.write_text(
                json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )

            code = run_cli(
                [
                    "validate",
                    "--promotion-release-record",
                    str(release_record_path),
                    "--strict",
                    "--out",
                    str(summary_path),
                ]
            )

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(
                error for target in summary["targets"] for error in target["errors"]
            )
            self.assertIn("artifact_validation.target_count", errors)

    def test_validate_promotion_release_record_requires_specific_validation_target_types(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = write_governance_artifacts(root)
            training_export = write_training_export(root)
            cards_dir = root / "cards"
            decision_path = root / "promotion_decision.json"
            registry_path = write_model_registry(root)
            alias_receipt_path = root / "promotion_alias_apply.json"
            release_notes_path = write_release_notes(root)
            release_record_path = root / "promotion_release_record.json"
            summary_path = root / "validation.json"
            self.assertEqual(
                run_cli(promotion_cards_args(artifacts, training_export, cards_dir)), 0
            )
            artifacts["model_card"] = cards_dir / "MODEL_CARD.md"
            artifacts["dataset_card"] = cards_dir / "DATASET_CARD.md"
            self.assertEqual(
                run_cli(promotion_decision_args(artifacts, decision_path)), 0
            )
            self.assertEqual(
                run_cli(
                    promotion_alias_apply_args(
                        registry_path, decision_path, alias_receipt_path
                    )
                ),
                0,
            )
            self.assertEqual(
                run_cli(
                    promotion_release_record_args(
                        artifacts,
                        cards_dir,
                        decision_path,
                        alias_receipt_path,
                        release_notes_path,
                        release_record_path,
                    )
                ),
                0,
            )
            record = json.loads(release_record_path.read_text(encoding="utf-8"))
            targets = record["artifact_validation"]["targets"]
            for item in targets:
                if item["type"] == "promotion_alias_apply":
                    item["type"] = "promotion_policy"
            release_record_path.write_text(
                json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )

            code = run_cli(
                [
                    "validate",
                    "--promotion-release-record",
                    str(release_record_path),
                    "--strict",
                    "--out",
                    str(summary_path),
                ]
            )

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(
                error for target in summary["targets"] for error in target["errors"]
            )
            self.assertIn("artifact_validation.targets missing required type", errors)
            self.assertIn("promotion_alias_apply", errors)

    def test_validate_promotion_release_record_rejects_forged_validation_warning_counts(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = write_governance_artifacts(root)
            training_export = write_training_export(root)
            cards_dir = root / "cards"
            decision_path = root / "promotion_decision.json"
            registry_path = write_model_registry(root)
            alias_receipt_path = root / "promotion_alias_apply.json"
            release_notes_path = write_release_notes(root)
            release_record_path = root / "promotion_release_record.json"
            summary_path = root / "validation.json"
            self.assertEqual(
                run_cli(promotion_cards_args(artifacts, training_export, cards_dir)), 0
            )
            artifacts["model_card"] = cards_dir / "MODEL_CARD.md"
            artifacts["dataset_card"] = cards_dir / "DATASET_CARD.md"
            self.assertEqual(
                run_cli(promotion_decision_args(artifacts, decision_path)), 0
            )
            self.assertEqual(
                run_cli(
                    promotion_alias_apply_args(
                        registry_path, decision_path, alias_receipt_path
                    )
                ),
                0,
            )
            self.assertEqual(
                run_cli(
                    promotion_release_record_args(
                        artifacts,
                        cards_dir,
                        decision_path,
                        alias_receipt_path,
                        release_notes_path,
                        release_record_path,
                    )
                ),
                0,
            )
            record = json.loads(release_record_path.read_text(encoding="utf-8"))
            record["artifact_validation"]["warning_count"] = 1
            record["artifact_validation"]["targets"][0]["warning_count"] = 1
            release_record_path.write_text(
                json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )

            code = run_cli(
                [
                    "validate",
                    "--promotion-release-record",
                    str(release_record_path),
                    "--strict",
                    "--out",
                    str(summary_path),
                ]
            )

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(
                error for target in summary["targets"] for error in target["errors"]
            )
            self.assertIn("artifact_validation.passed", errors)
            self.assertIn("artifact_validation.targets[0].passed", errors)

    def test_promotion_release_record_blocks_policy_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = write_governance_artifacts(root)
            training_export = write_training_export(root)
            cards_dir = root / "cards"
            decision_path = root / "promotion_decision.json"
            registry_path = write_model_registry(root)
            alias_receipt_path = root / "promotion_alias_apply.json"
            release_notes_path = write_release_notes(root)
            release_record_path = root / "promotion_release_record.json"
            policy_path = write_promotion_policy(root)
            mismatched_policy_path = write_promotion_policy(
                root,
                filename="promotion_policy_mismatch.json",
                policy_id="mismatch-policy",
            )
            self.assertEqual(
                run_cli(promotion_cards_args(artifacts, training_export, cards_dir)), 0
            )
            artifacts["model_card"] = cards_dir / "MODEL_CARD.md"
            artifacts["dataset_card"] = cards_dir / "DATASET_CARD.md"
            self.assertEqual(
                run_cli(
                    promotion_decision_args(
                        artifacts, decision_path, promotion_policy=policy_path
                    )
                ),
                0,
            )
            self.assertEqual(
                run_cli(
                    promotion_alias_apply_args(
                        registry_path, decision_path, alias_receipt_path
                    )
                ),
                0,
            )

            code = run_cli(
                promotion_release_record_args(
                    artifacts,
                    cards_dir,
                    decision_path,
                    alias_receipt_path,
                    release_notes_path,
                    release_record_path,
                    promotion_policy=mismatched_policy_path,
                )
            )

            self.assertEqual(code, 1)
            record = json.loads(release_record_path.read_text(encoding="utf-8"))
            self.assertIn("promotion_policy_matches_decision", failed_check_ids(record))
            self.assertEqual(
                run_cli(
                    [
                        "validate",
                        "--promotion-release-record",
                        str(release_record_path),
                        "--strict",
                    ]
                ),
                0,
            )

    def test_promotion_decision_blocks_missing_dataset_card_and_rollback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = write_governance_artifacts(root)
            artifacts["dataset_card"] = None
            decision_path = root / "promotion_decision.json"

            args = promotion_decision_args(artifacts, decision_path)
            rollback_index = args.index("--rollback-id")
            del args[rollback_index : rollback_index + 2]
            code = run_cli(args)

            self.assertEqual(code, 1)
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            failed_ids = failed_check_ids(decision)
            self.assertIn("dataset_card_present", failed_ids)
            self.assertIn("rollback_id_present", failed_ids)
            self.assertEqual(decision["recommendation"], "block_promotion")
            self.assertFalse(decision["alias_update"]["authorized"])
            self.assertEqual(
                run_cli(
                    ["validate", "--promotion-decision", str(decision_path), "--strict"]
                ),
                0,
            )

    def test_promotion_decision_rejects_directory_empty_and_wrong_heading_cards(self):
        for role in ("model_card", "dataset_card"):
            for card_kind in ("directory", "empty", "wrong_heading"):
                with (
                    self.subTest(role=role, card_kind=card_kind),
                    tempfile.TemporaryDirectory() as tmp,
                ):
                    root = Path(tmp)
                    artifacts = write_governance_artifacts(root)
                    card_path = artifacts[role]
                    if card_kind == "directory":
                        card_path.unlink()
                        card_path.mkdir()
                    elif card_kind == "empty":
                        card_path.write_text("", encoding="utf-8")
                    else:
                        card_path.write_text(
                            "# Unsupported Card\n\nNo required heading.\n",
                            encoding="utf-8",
                        )
                    decision_path = root / "promotion_decision.json"

                    self.assertEqual(
                        run_cli(
                            promotion_decision_args(
                                artifacts,
                                decision_path,
                            )
                        ),
                        1,
                    )
                    decision = _read_json_object(decision_path)
                    failed_ids = failed_check_ids(decision)
                    self.assertIn(f"{role}_claims_supported", failed_ids)
                    if card_kind == "directory":
                        self.assertIn(f"{role}_present", failed_ids)
                    self.assertFalse(decision["alias_update"]["authorized"])
                    self.assertEqual(
                        run_cli(
                            [
                                "validate",
                                "--promotion-decision",
                                str(decision_path),
                                "--strict",
                            ]
                        ),
                        0,
                    )

    def test_promotion_decision_requires_explicit_rollback_availability(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = write_governance_artifacts(root)
            artifacts["rollback_metadata"].write_text(
                json.dumps({"rollback_id": "champion-v1"}),
                encoding="utf-8",
            )
            decision_path = root / "promotion_decision.json"

            self.assertEqual(
                run_cli(promotion_decision_args(artifacts, decision_path)),
                1,
            )
            decision = _read_json_object(decision_path)
            self.assertIn(
                "rollback_metadata_matches_target",
                failed_check_ids(decision),
            )
            self.assertFalse(decision["alias_update"]["authorized"])
            self.assertEqual(
                run_cli(
                    [
                        "validate",
                        "--promotion-decision",
                        str(decision_path),
                        "--strict",
                    ]
                ),
                0,
            )

    def test_promotion_decision_blocks_missing_registry_training_and_serving_profile(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = write_governance_artifacts(root)
            artifacts["model_registry_entry"] = None
            artifacts["agentic_training_result"] = None
            artifacts["serving_profile"] = None
            decision_path = root / "promotion_decision.json"

            code = run_cli(promotion_decision_args(artifacts, decision_path))

            self.assertEqual(code, 1)
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            failed_ids = failed_check_ids(decision)
            self.assertIn("model_registry_entry_present", failed_ids)
            self.assertIn("agentic_training_result_present", failed_ids)
            self.assertIn("serving_profile_present", failed_ids)
            self.assertIn("agentic_training_result_passed", failed_ids)
            self.assertIn("model_registry_entry_matches_candidate", failed_ids)
            self.assertIn("serving_profile_ready", failed_ids)
            self.assertFalse(decision["alias_update"]["authorized"])
            self.assertEqual(
                run_cli(
                    ["validate", "--promotion-decision", str(decision_path), "--strict"]
                ),
                0,
            )

    def test_promotion_decision_blocks_unknown_license(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = write_governance_artifacts(root, license_status="unknown")
            decision_path = root / "promotion_decision.json"

            code = run_cli(promotion_decision_args(artifacts, decision_path))

            self.assertEqual(code, 1)
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            self.assertIn("license_status_known", failed_check_ids(decision))
            self.assertEqual(
                run_cli(
                    ["validate", "--promotion-decision", str(decision_path), "--strict"]
                ),
                0,
            )

    def test_promotion_decision_blocks_missing_accepted_terms(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = write_governance_artifacts(root, accepted_terms=None)
            decision_path = root / "promotion_decision.json"

            code = run_cli(promotion_decision_args(artifacts, decision_path))

            self.assertEqual(code, 1)
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            self.assertIn("license_terms_accepted", failed_check_ids(decision))
            self.assertFalse(decision["alias_update"]["authorized"])
            self.assertEqual(
                run_cli(
                    ["validate", "--promotion-decision", str(decision_path), "--strict"]
                ),
                0,
            )

    def test_promotion_decision_blocks_incomplete_compare_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = write_governance_artifacts(root)
            artifacts["compare_gate"].write_text(
                json.dumps(
                    {
                        "schema_version": "hfr.compare_gate.v1",
                        "passed": True,
                        "metrics": {},
                    }
                ),
                encoding="utf-8",
            )
            decision_path = root / "promotion_decision.json"

            code = run_cli(promotion_decision_args(artifacts, decision_path))

            self.assertEqual(code, 1)
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            self.assertIn("compare_metrics_complete", failed_check_ids(decision))
            self.assertFalse(decision["alias_update"]["authorized"])
            self.assertEqual(
                run_cli(
                    ["validate", "--promotion-decision", str(decision_path), "--strict"]
                ),
                0,
            )

    def test_promotion_decision_blocks_incomplete_policy_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = write_governance_artifacts(root)
            decision_path = root / "promotion_decision.json"
            policy_path = write_promotion_policy(
                root,
                required_artifacts=[
                    role
                    for role in promotion_decision_required_artifacts()
                    if role != "serving_report"
                ],
            )

            code = run_cli(
                promotion_decision_args(
                    artifacts, decision_path, promotion_policy=policy_path
                )
            )

            self.assertEqual(code, 1)
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            self.assertEqual(decision["policy"]["source"], "file")
            self.assertEqual(decision["policy"]["artifact"]["role"], "promotion_policy")
            self.assertIn(
                "promotion_policy_required_artifacts_complete",
                failed_check_ids(decision),
            )
            self.assertFalse(decision["alias_update"]["authorized"])
            self.assertEqual(
                run_cli(
                    ["validate", "--promotion-decision", str(decision_path), "--strict"]
                ),
                0,
            )

    def test_promotion_decision_blocks_symlinked_policy_parent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = write_governance_artifacts(root)
            decision_path = root / "promotion_decision.json"
            policy_path = write_promotion_policy(root)
            linked_target = root / "linked_target"
            linked_target.mkdir()
            (linked_target / "promotion_policy.json").write_text(
                policy_path.read_text(encoding="utf-8"), encoding="utf-8"
            )
            linked_parent = root / "linked_policy"
            try:
                linked_parent.symlink_to(linked_target, target_is_directory=True)
            except (NotImplementedError, OSError) as exc:
                self.skipTest(f"symlink unavailable: {exc}")

            code = run_cli(
                promotion_decision_args(
                    artifacts,
                    decision_path,
                    promotion_policy=linked_parent / "promotion_policy.json",
                )
            )

            self.assertEqual(code, 1)
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            policy_artifact = decision["policy"]["artifact"]
            self.assertFalse(policy_artifact["exists"])
            self.assertEqual(policy_artifact["kind"], "other")
            self.assertNotIn("sha256", policy_artifact)
            failed_ids = failed_check_ids(decision)
            self.assertIn("promotion_policy_present", failed_ids)
            self.assertIn("promotion_policy_fields_present", failed_ids)
            self.assertFalse(decision["alias_update"]["authorized"])
            self.assertEqual(
                run_cli(
                    ["validate", "--promotion-decision", str(decision_path), "--strict"]
                ),
                0,
            )

    def test_promotion_decision_blocks_symlinked_json_artifact_parent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = write_governance_artifacts(root)
            decision_path = root / "promotion_decision.json"
            linked_target = root / "linked_target"
            linked_target.mkdir()
            (linked_target / "compare_gate.json").write_text(
                artifacts["compare_gate"].read_text(encoding="utf-8"), encoding="utf-8"
            )
            linked_parent = root / "linked_artifacts"
            try:
                linked_parent.symlink_to(linked_target, target_is_directory=True)
            except (NotImplementedError, OSError) as exc:
                self.skipTest(f"symlink unavailable: {exc}")
            artifacts["compare_gate"] = linked_parent / "compare_gate.json"

            code = run_cli(promotion_decision_args(artifacts, decision_path))

            self.assertEqual(code, 1)
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            compare_gate = decision["artifacts"]["compare_gate"]
            self.assertFalse(compare_gate["exists"])
            self.assertEqual(compare_gate["kind"], "other")
            self.assertNotIn("sha256", compare_gate)
            failed_ids = failed_check_ids(decision)
            self.assertIn("compare_gate_present", failed_ids)
            self.assertIn("compare_gate_schema", failed_ids)
            self.assertIn("compare_gate_passed", failed_ids)
            self.assertIn("compare_metrics_complete", failed_ids)
            self.assertFalse(decision["alias_update"]["authorized"])
            self.assertEqual(
                run_cli(
                    ["validate", "--promotion-decision", str(decision_path), "--strict"]
                ),
                0,
            )

    def test_promotion_decision_blocks_symlinked_card_parent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = write_governance_artifacts(root)
            decision_path = root / "promotion_decision.json"
            linked_target = root / "linked_target"
            linked_target.mkdir()
            (linked_target / "MODEL_CARD.md").write_text(
                artifacts["model_card"].read_text(encoding="utf-8"), encoding="utf-8"
            )
            linked_parent = root / "linked_cards"
            try:
                linked_parent.symlink_to(linked_target, target_is_directory=True)
            except (NotImplementedError, OSError) as exc:
                self.skipTest(f"symlink unavailable: {exc}")
            artifacts["model_card"] = linked_parent / "MODEL_CARD.md"

            code = run_cli(promotion_decision_args(artifacts, decision_path))

            self.assertEqual(code, 1)
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            model_card = decision["artifacts"]["model_card"]
            self.assertFalse(model_card["exists"])
            self.assertEqual(model_card["kind"], "other")
            self.assertNotIn("sha256", model_card)
            failed_ids = failed_check_ids(decision)
            self.assertIn("model_card_present", failed_ids)
            self.assertFalse(decision["alias_update"]["authorized"])
            self.assertEqual(
                run_cli(
                    ["validate", "--promotion-decision", str(decision_path), "--strict"]
                ),
                0,
            )

    def test_validate_promotion_policy_rejects_relaxed_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            policy_path = write_promotion_policy(
                root,
                required_artifacts=[
                    role
                    for role in promotion_decision_required_artifacts()
                    if role != "serving_report"
                ],
                forbid_new_critical_rules=["forbidden_actions", "secret_exposure"],
                forbid_regressed_rules=["forbidden_actions", "secret_exposure"],
                limits={"max_task_completion_regressions": 1},
                require_accepted_terms=False,
            )
            summary_path = root / "policy_validation.json"

            code = run_cli(
                [
                    "validate",
                    "--promotion-policy",
                    str(policy_path),
                    "--strict",
                    "--out",
                    str(summary_path),
                ]
            )

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(
                error for target in summary["targets"] for error in target["errors"]
            )
            self.assertIn("missing required role", errors)
            self.assertIn("cannot exceed default maximum", errors)
            self.assertIn("zero-tolerance rules", errors)
            self.assertIn("require_accepted_terms must be true", errors)

    def test_promotion_decision_blocks_eval_regressions_and_secret_exposure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = write_governance_artifacts(
                root,
                compare_metrics={
                    "baseline_win_count": 1,
                    "contract_drift_count": 1,
                    "new_critical_failure_counts": {"secret_exposure": 1},
                    "regressed_rule_counts": {"forbidden_actions": 1},
                    "task_completion_regression_count": 1,
                    "unverified_contract_count": 0,
                },
            )
            decision_path = root / "promotion_decision.json"

            code = run_cli(promotion_decision_args(artifacts, decision_path))

            self.assertEqual(code, 1)
            failed_ids = failed_check_ids(
                json.loads(decision_path.read_text(encoding="utf-8"))
            )
            self.assertIn("task_completion_regressions_absent", failed_ids)
            self.assertIn("baseline_wins_absent", failed_ids)
            self.assertIn("contract_drifts_absent", failed_ids)
            self.assertIn("new_critical_failures_absent", failed_ids)
            self.assertIn("new_critical_secret_exposure_absent", failed_ids)
            self.assertIn("regression_forbidden_actions_absent", failed_ids)
            self.assertEqual(
                run_cli(
                    ["validate", "--promotion-decision", str(decision_path), "--strict"]
                ),
                0,
            )

    def test_validate_promotion_decision_rejects_stale_artifact_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = write_governance_artifacts(root)
            decision_path = root / "promotion_decision.json"
            self.assertEqual(
                run_cli(promotion_decision_args(artifacts, decision_path)), 0
            )

            artifacts["model_card"].write_text(
                "# Model Card\n\nchanged after decision\n", encoding="utf-8"
            )

            self.assertEqual(
                run_cli(
                    ["validate", "--promotion-decision", str(decision_path), "--strict"]
                ),
                1,
            )

    def test_validate_promotion_decision_rejects_symlink_artifact_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = write_governance_artifacts(root)
            decision_path = root / "promotion_decision.json"
            summary_path = root / "validation.json"
            model_card_link = root / "MODEL_CARD_LINK.md"
            try:
                model_card_link.symlink_to(artifacts["model_card"])
            except (NotImplementedError, OSError) as exc:
                self.skipTest(f"symlink unavailable: {exc}")
            self.assertEqual(
                run_cli(promotion_decision_args(artifacts, decision_path)), 0
            )
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            decision["artifacts"]["model_card"]["path"] = "MODEL_CARD_LINK.md"
            decision_path.write_text(
                json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )

            code = run_cli(
                [
                    "validate",
                    "--promotion-decision",
                    str(decision_path),
                    "--strict",
                    "--out",
                    str(summary_path),
                ]
            )

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(
                error for target in summary["targets"] for error in target["errors"]
            )
            self.assertIn(
                "promotion_decision.artifacts.model_card.path must not resolve to a symlink",
                errors,
            )

    def test_validate_promotion_decision_rejects_symlink_artifact_parent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = write_governance_artifacts(root)
            decision_path = root / "promotion_decision.json"
            summary_path = root / "validation.json"
            linked_target = root / "linked_target"
            linked_target.mkdir()
            (linked_target / "MODEL_CARD.md").write_text(
                artifacts["model_card"].read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            linked_parent = root / "linked_artifacts"
            try:
                linked_parent.symlink_to(linked_target, target_is_directory=True)
            except (NotImplementedError, OSError) as exc:
                self.skipTest(f"symlink unavailable: {exc}")
            self.assertEqual(
                run_cli(promotion_decision_args(artifacts, decision_path)), 0
            )
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            decision["artifacts"]["model_card"]["path"] = (
                "linked_artifacts/MODEL_CARD.md"
            )
            decision_path.write_text(
                json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )

            code = run_cli(
                [
                    "validate",
                    "--promotion-decision",
                    str(decision_path),
                    "--strict",
                    "--out",
                    str(summary_path),
                ]
            )

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(
                error for target in summary["targets"] for error in target["errors"]
            )
            self.assertIn(
                "promotion_decision.artifacts.model_card.path must not resolve through a symlink",
                errors,
            )

    def test_validate_promotion_decision_rejects_absolute_symlink_artifact_parent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = write_governance_artifacts(root)
            decision_path = root / "promotion_decision.json"
            summary_path = root / "validation.json"
            linked_target = root / "linked_target"
            linked_target.mkdir()
            (linked_target / "MODEL_CARD.md").write_text(
                artifacts["model_card"].read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            linked_parent = root / "linked_artifacts"
            try:
                linked_parent.symlink_to(linked_target, target_is_directory=True)
            except (NotImplementedError, OSError) as exc:
                self.skipTest(f"symlink unavailable: {exc}")
            self.assertEqual(
                run_cli(promotion_decision_args(artifacts, decision_path)), 0
            )
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            decision["artifacts"]["model_card"]["path"] = str(
                linked_parent / "MODEL_CARD.md"
            )
            decision_path.write_text(
                json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )

            code = run_cli(
                [
                    "validate",
                    "--promotion-decision",
                    str(decision_path),
                    "--strict",
                    "--out",
                    str(summary_path),
                ]
            )

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(
                error for target in summary["targets"] for error in target["errors"]
            )
            self.assertIn(
                "promotion_decision.artifacts.model_card.path must not resolve through a symlink",
                errors,
            )

    def test_promotion_symlink_scan_continues_after_root_alias(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            root_component = (
                Path(root.anchor) / root.parts[1]
                if root.is_absolute() and len(root.parts) > 1
                else None
            )
            if root_component is None or not root_component.is_symlink():
                self.skipTest("root-level symlink unavailable")
            linked_target = root / "linked_target"
            linked_target.mkdir()
            linked_parent = root / "linked_artifacts"
            try:
                linked_parent.symlink_to(linked_target, target_is_directory=True)
            except (NotImplementedError, OSError) as exc:
                self.skipTest(f"symlink unavailable: {exc}")

            self.assertTrue(
                _path_has_symlink_component(
                    linked_parent / "MODEL_CARD.md", include_leaf=False
                )
            )

    def test_validate_promotion_cards_rejects_symlink_directory_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = write_governance_artifacts(root)
            training_export = write_training_export(root)
            cards_dir = root / "cards"
            summary_path = root / "validation.json"
            training_export_link = root / "training_export_link"
            try:
                training_export_link.symlink_to(
                    training_export, target_is_directory=True
                )
            except (NotImplementedError, OSError) as exc:
                self.skipTest(f"symlink unavailable: {exc}")
            self.assertEqual(
                run_cli(promotion_cards_args(artifacts, training_export, cards_dir)), 0
            )
            manifest_path = cards_dir / "promotion_cards.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["artifacts"]["training_export"]["path"] = "../training_export_link"
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )

            code = run_cli(
                [
                    "validate",
                    "--promotion-cards",
                    str(cards_dir),
                    "--strict",
                    "--out",
                    str(summary_path),
                ]
            )

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(
                error for target in summary["targets"] for error in target["errors"]
            )
            self.assertIn(
                "promotion_cards.artifacts.training_export.path must not resolve to a symlink",
                errors,
            )

    def test_validate_promotion_cards_rejects_absolute_symlink_artifact_parent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = write_governance_artifacts(root)
            training_export = write_training_export(root)
            cards_dir = root / "cards"
            summary_path = root / "validation.json"
            linked_target = root / "linked_target"
            linked_export = linked_target / "training_export"
            linked_export.mkdir(parents=True)
            (linked_export / "DATASET_CARD.md").write_text(
                (training_export / "DATASET_CARD.md").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            linked_parent = root / "linked_artifacts"
            try:
                linked_parent.symlink_to(linked_target, target_is_directory=True)
            except (NotImplementedError, OSError) as exc:
                self.skipTest(f"symlink unavailable: {exc}")
            self.assertEqual(
                run_cli(promotion_cards_args(artifacts, training_export, cards_dir)), 0
            )
            manifest_path = cards_dir / "promotion_cards.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["artifacts"]["training_export"]["path"] = str(
                linked_parent / "training_export"
            )
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )

            code = run_cli(
                [
                    "validate",
                    "--promotion-cards",
                    str(cards_dir),
                    "--strict",
                    "--out",
                    str(summary_path),
                ]
            )

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(
                error for target in summary["targets"] for error in target["errors"]
            )
            self.assertIn(
                "promotion_cards.artifacts.training_export.path must not resolve through a symlink",
                errors,
            )

    def test_validate_promotion_cards_rejects_symlink_manifest_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = write_governance_artifacts(root)
            training_export = write_training_export(root)
            cards_dir = root / "cards"
            cards_link = root / "cards_link"
            summary_path = root / "validation.json"
            self.assertEqual(
                run_cli(promotion_cards_args(artifacts, training_export, cards_dir)), 0
            )
            try:
                cards_link.symlink_to(cards_dir, target_is_directory=True)
            except (NotImplementedError, OSError) as exc:
                self.skipTest(f"symlink unavailable: {exc}")

            code = run_cli(
                [
                    "validate",
                    "--promotion-cards",
                    str(cards_link),
                    "--strict",
                    "--out",
                    str(summary_path),
                ]
            )

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(
                error for target in summary["targets"] for error in target["errors"]
            )
            self.assertIn(
                "promotion_cards.json must resolve to a regular non-symlink file",
                errors,
            )

    def test_promotion_cards_writes_output_relative_artifact_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "src"
            output = root / "out"
            source.mkdir()
            output.mkdir()
            artifacts = write_governance_artifacts(source)
            training_export = write_training_export(source)
            cards_dir = output / "cards"

            code = run_cli(
                without_preserve_paths(
                    promotion_cards_args(artifacts, training_export, cards_dir)
                )
            )

            self.assertEqual(code, 0)
            manifest = json.loads(
                (cards_dir / "promotion_cards.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                manifest["artifacts"]["model_card"]["path"], "MODEL_CARD.md"
            )
            self.assertEqual(
                manifest["artifacts"]["dataset_card"]["path"], "DATASET_CARD.md"
            )
            self.assertEqual(
                manifest["artifacts"]["evidence_bundle"]["path"],
                "../../src/evidence/evidence_bundle.json",
            )
            self.assertEqual(
                manifest["artifacts"]["training_export"]["path"],
                "../../src/training_export",
            )
            self.assertEqual(
                run_cli(["validate", "--promotion-cards", str(cards_dir), "--strict"]),
                0,
            )

    def test_promotion_decision_writes_output_relative_artifact_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "src"
            output = root / "out"
            source.mkdir()
            output.mkdir()
            artifacts = write_governance_artifacts(source)
            decision_path = output / "promotion_decision.json"

            code = run_cli(
                without_preserve_paths(
                    promotion_decision_args(artifacts, decision_path)
                )
            )

            self.assertEqual(code, 0)
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            self.assertEqual(
                decision["artifacts"]["evidence_bundle"]["path"],
                "../src/evidence/evidence_bundle.json",
            )
            self.assertEqual(
                decision["artifacts"]["model_card"]["path"], "../src/MODEL_CARD.md"
            )
            self.assertEqual(
                run_cli(
                    ["validate", "--promotion-decision", str(decision_path), "--strict"]
                ),
                0,
            )

    def test_validate_promotion_decision_rejects_cwd_relative_artifact_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "src"
            output = root / "out"
            source.mkdir()
            output.mkdir()
            artifacts = write_governance_artifacts(source)
            decision_path = output / "promotion_decision.json"
            summary_path = output / "validation.json"
            self.assertEqual(
                run_cli(
                    without_preserve_paths(
                        promotion_decision_args(artifacts, decision_path)
                    )
                ),
                0,
            )
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            decision["artifacts"]["evidence_bundle"]["path"] = "evidence_bundle.json"
            decision_path.write_text(
                json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )

            previous_cwd = Path.cwd()
            try:
                os.chdir(source)
                code = run_cli(
                    [
                        "validate",
                        "--promotion-decision",
                        str(decision_path),
                        "--strict",
                        "--out",
                        str(summary_path),
                    ]
                )
            finally:
                os.chdir(previous_cwd)

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(
                error for target in summary["targets"] for error in target["errors"]
            )
            self.assertIn(
                "promotion_decision.artifacts.evidence_bundle.path does not exist at validation time",
                errors,
            )

    def test_promotion_release_record_writes_output_relative_artifact_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "src"
            output = root / "out"
            source.mkdir()
            output.mkdir()
            artifacts = write_governance_artifacts(source)
            training_export = write_training_export(source)
            cards_dir = output / "cards"
            registry_path = write_model_registry(source)
            decision_path = output / "promotion_decision.json"
            alias_receipt_path = output / "promotion_alias_apply.json"
            release_notes_path = write_release_notes(source)
            release_record_path = output / "promotion_release_record.json"
            self.assertEqual(
                run_cli(
                    without_preserve_paths(
                        promotion_cards_args(artifacts, training_export, cards_dir)
                    )
                ),
                0,
            )
            artifacts["model_card"] = cards_dir / "MODEL_CARD.md"
            artifacts["dataset_card"] = cards_dir / "DATASET_CARD.md"
            self.assertEqual(
                run_cli(
                    without_preserve_paths(
                        promotion_decision_args(artifacts, decision_path)
                    )
                ),
                0,
            )
            self.assertEqual(
                run_cli(
                    without_preserve_paths(
                        promotion_alias_apply_args(
                            registry_path, decision_path, alias_receipt_path
                        )
                    )
                ),
                0,
            )

            code = run_cli(
                without_preserve_paths(
                    promotion_release_record_args(
                        artifacts,
                        cards_dir,
                        decision_path,
                        alias_receipt_path,
                        release_notes_path,
                        release_record_path,
                    )
                )
            )

            self.assertEqual(code, 0)
            record = json.loads(release_record_path.read_text(encoding="utf-8"))
            self.assertEqual(
                record["artifacts"]["promotion_decision"]["path"],
                "promotion_decision.json",
            )
            self.assertEqual(record["artifacts"]["promotion_cards"]["path"], "cards")
            self.assertEqual(
                record["artifacts"]["rollback_metadata"]["path"], "../src/rollback.json"
            )
            self.assertEqual(
                record["artifacts"]["release_notes"]["path"], "../src/RELEASE_NOTES.md"
            )
            self.assertEqual(
                run_cli(
                    [
                        "validate",
                        "--promotion-release-record",
                        str(release_record_path),
                        "--strict",
                    ]
                ),
                0,
            )

    def test_policy_bound_release_validates_across_split_output_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "src"
            decision_dir = root / "governance" / "decision" / "nested"
            release_dir = root / "publish" / "release"
            policy_dir = root / "policies"
            source.mkdir()
            decision_dir.mkdir(parents=True)
            release_dir.mkdir(parents=True)
            policy_dir.mkdir()
            policy_path = write_promotion_policy(policy_dir)
            artifacts = write_governance_artifacts(source)
            training_export = write_training_export(source)
            cards_dir = decision_dir / "cards"
            registry_path = write_model_registry(source)
            decision_path = decision_dir / "promotion_decision.json"
            alias_receipt_path = decision_dir / "promotion_alias_apply.json"
            release_notes_path = write_release_notes(source)
            release_record_path = release_dir / "promotion_release_record.json"
            self.assertEqual(
                run_cli(
                    without_preserve_paths(
                        promotion_cards_args(
                            artifacts,
                            training_export,
                            cards_dir,
                        )
                    )
                ),
                0,
            )
            artifacts["model_card"] = cards_dir / "MODEL_CARD.md"
            artifacts["dataset_card"] = cards_dir / "DATASET_CARD.md"
            self.assertEqual(
                run_cli(
                    without_preserve_paths(
                        promotion_decision_args(
                            artifacts,
                            decision_path,
                            promotion_policy=policy_path,
                        )
                    )
                ),
                0,
            )
            self.assertEqual(
                run_cli(
                    without_preserve_paths(
                        promotion_alias_apply_args(
                            registry_path,
                            decision_path,
                            alias_receipt_path,
                        )
                    )
                ),
                0,
            )
            self.assertEqual(
                run_cli(
                    without_preserve_paths(
                        promotion_release_record_args(
                            artifacts,
                            cards_dir,
                            decision_path,
                            alias_receipt_path,
                            release_notes_path,
                            release_record_path,
                            promotion_policy=policy_path,
                        )
                    )
                ),
                0,
            )

            decision = _read_json_object(decision_path)
            record = _read_json_object(release_record_path)
            decision_policy_path = decision["policy"]["artifact"]["path"]
            release_policy_path = record["policy"]["release_policy"]["artifact"][
                "path"
            ]
            self.assertNotEqual(decision_policy_path, release_policy_path)
            self.assertFalse(Path(decision_policy_path).is_absolute())
            self.assertFalse(Path(release_policy_path).is_absolute())
            self.assertEqual(
                validate_promotion_release_record(release_record_path).errors,
                [],
            )

    def test_validate_promotion_decision_rejects_missing_required_pass_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = write_governance_artifacts(root)
            decision_path = root / "promotion_decision.json"
            summary_path = root / "validation.json"
            self.assertEqual(
                run_cli(promotion_decision_args(artifacts, decision_path)), 0
            )
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            decision["checks"] = [
                check
                for check in decision["checks"]
                if check["id"] != "license_terms_accepted"
            ]
            decision["check_count"] = len(decision["checks"])
            decision["metrics"]["check_count"] = len(decision["checks"])
            decision["decision"]["key_metrics"]["check_count"] = len(decision["checks"])
            decision_path.write_text(
                json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )

            code = run_cli(
                [
                    "validate",
                    "--promotion-decision",
                    str(decision_path),
                    "--strict",
                    "--out",
                    str(summary_path),
                ]
            )

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(
                error for target in summary["targets"] for error in target["errors"]
            )
            self.assertIn("missing required passing check", errors)
            self.assertIn("license_terms_accepted", errors)

    def test_validate_relocated_promotion_decision_replays_relocated_semantic_sources(
        self,
    ):
        for source_kind in ("trainer_preflight", "compare_export", "promotion_ledger"):
            with (
                self.subTest(source_kind=source_kind),
                tempfile.TemporaryDirectory() as tmp,
            ):
                artifacts = copy_valid_loop_artifacts(Path(tmp))
                decision_path = artifacts["promotion_decision"][0]
                fixture_root = decision_path.parent.parent
                self.assertEqual(validate_promotion_decision(decision_path).errors, [])

                if source_kind == "trainer_preflight":
                    source_path = fixture_root / "trainer_preflight.json"
                    source = _read_json_object(source_path)
                    source["trainer_command"]["argv"][1] = "relocated-blocked.py"
                    source["trainer_command"]["raw"] = "python relocated-blocked.py"
                elif source_kind == "compare_export":
                    source_path = (
                        fixture_root
                        / "promotion_governance"
                        / "compare_export"
                        / "manifest.json"
                    )
                    source = _read_json_object(source_path)
                    source["metadata"]["candidate"] = "relocated-tampered-candidate"
                else:
                    source_path = (
                        fixture_root / "promotion_governance" / "promotion_ledger.json"
                    )
                    source = {}
                _write_source_json(source_path, source)

                errors = validate_promotion_decision(decision_path).errors

                self.assertIn(
                    "promotion_decision.checks must exactly match canonical replay from current source artifacts.",
                    errors,
                )

    def test_validate_promotion_alias_apply_requires_authoritative_artifacts_for_applied_receipt(
        self,
    ):
        for role in ("registry", "promotion_decision"):
            with self.subTest(role=role), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                artifacts = write_governance_artifacts(root)
                registry_path = write_model_registry(root)
                decision_path = root / "promotion_decision.json"
                receipt_path = root / "promotion_alias_apply.json"
                self.assertEqual(
                    run_cli(promotion_decision_args(artifacts, decision_path)), 0
                )
                self.assertEqual(
                    run_cli(
                        promotion_alias_apply_args(
                            registry_path,
                            decision_path,
                            receipt_path,
                        )
                    ),
                    0,
                )

                receipt = _read_json_object(receipt_path)
                artifact = receipt["artifacts"][role]
                artifact["path"] = f"missing-{role}.json"
                artifact["exists"] = False
                artifact["kind"] = "missing"
                for field_name in ("sha256", "size_bytes", "schema_version"):
                    artifact.pop(field_name, None)
                if role == "registry":
                    receipt["registry_after"].update(
                        {
                            "path": artifact["path"],
                            "sha256": None,
                            "size_bytes": 0,
                        }
                    )
                else:
                    receipt["promotion_decision"].update(
                        {
                            "path": artifact["path"],
                            "sha256": None,
                            "size_bytes": 0,
                        }
                    )
                _write_source_json(receipt_path, receipt)

                errors = validate_promotion_alias_apply(receipt_path).errors

                self.assertIn(
                    f"promotion_alias_apply.artifacts.{role} must be an existing file for applied receipts.",
                    errors,
                )
                self.assertIn(
                    f"promotion_alias_apply.artifacts.{role}.sha256 must be a SHA-256 hex string for applied receipts.",
                    errors,
                )

    def test_validate_promotion_alias_apply_rejects_current_canonical_blocked_decision(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = write_governance_artifacts(root)
            registry_path = write_model_registry(root)
            decision_path = root / "promotion_decision.json"
            receipt_path = root / "promotion_alias_apply.json"
            self.assertEqual(
                run_cli(promotion_decision_args(artifacts, decision_path)), 0
            )
            self.assertEqual(
                run_cli(
                    promotion_alias_apply_args(
                        registry_path,
                        decision_path,
                        receipt_path,
                    )
                ),
                0,
            )

            license_review = _read_json_object(artifacts["license_review"])
            license_review["accepted_terms"] = False
            _write_source_json(artifacts["license_review"], license_review)
            self.assertEqual(
                run_cli(promotion_decision_args(artifacts, decision_path)), 1
            )
            blocked_decision = _read_json_object(decision_path)
            self.assertFalse(blocked_decision["passed"])
            self.assertEqual(validate_promotion_decision(decision_path).errors, [])

            decision_sha256, decision_size = _file_fingerprint(decision_path)
            receipt = _read_json_object(receipt_path)
            for record in (
                receipt["artifacts"]["promotion_decision"],
                receipt["promotion_decision"],
            ):
                record["sha256"] = decision_sha256
                record["size_bytes"] = decision_size
            receipt["alias_history_entry"]["promotion_decision_sha256"] = (
                decision_sha256
            )

            registry = _read_json_object(registry_path)
            registry["alias_history"][-1]["promotion_decision_sha256"] = decision_sha256
            _write_source_json(registry_path, registry)
            registry_sha256, registry_size = _file_fingerprint(registry_path)
            for record in (
                receipt["artifacts"]["registry"],
                receipt["registry_after"],
            ):
                record["sha256"] = registry_sha256
                record["size_bytes"] = registry_size
            _write_source_json(receipt_path, receipt)

            errors = validate_promotion_alias_apply(receipt_path).errors

            self.assertIn(
                "promotion_alias_apply current promotion decision must be passing, ready, and authorize alias application.",
                errors,
            )

    def test_promotion_alias_apply_rejects_registry_mutation_after_cas_without_receipt(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = write_governance_artifacts(root)
            registry_path = write_model_registry(root)
            decision_path = root / "promotion_decision.json"
            receipt_path = root / "promotion_alias_apply.json"
            self.assertEqual(
                run_cli(promotion_decision_args(artifacts, decision_path)), 0
            )
            original_snapshot_reader = governance._read_json_artifact_snapshot
            state = {"registry_cas_completed": False, "mutated": False}

            def tracked_cas(path, value, *, expected_sha256):
                result = atomic_write_json_cas(
                    path,
                    value,
                    expected_sha256=expected_sha256,
                )
                if Path(path) == registry_path:
                    state["registry_cas_completed"] = True
                return result

            def mutate_before_post_cas_read(path, *, reject_symlink_components=False):
                if (
                    Path(path) == registry_path
                    and state["registry_cas_completed"]
                    and not state["mutated"]
                ):
                    concurrent = _read_json_object(registry_path)
                    concurrent["notes"] = [
                        *concurrent.get("notes", []),
                        "concurrent mutation after alias CAS",
                    ]
                    _write_source_json(registry_path, concurrent)
                    state["mutated"] = True
                return original_snapshot_reader(
                    path,
                    reject_symlink_components=reject_symlink_components,
                )

            with (
                patch(
                    "flightrecorder.governance.atomic_write_json_cas",
                    side_effect=tracked_cas,
                ),
                patch(
                    "flightrecorder.governance._read_json_artifact_snapshot",
                    side_effect=mutate_before_post_cas_read,
                ),
                self.assertRaisesRegex(
                    governance.PromotionDecisionError,
                    "changed after alias update",
                ),
            ):
                apply_promotion_aliases(
                    registry_path=registry_path,
                    promotion_decision_path=decision_path,
                    out_path=receipt_path,
                )

            self.assertTrue(state["mutated"])
            self.assertFalse(receipt_path.exists())

    def test_validate_promotion_release_record_replays_rehashed_invalid_decision(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs = self._prepare_promotion_release_record_inputs(root)
            self.assertEqual(
                run_cli(
                    promotion_release_record_args(
                        inputs["artifacts"],
                        inputs["cards_dir"],
                        inputs["decision_path"],
                        inputs["alias_receipt_path"],
                        inputs["release_notes_path"],
                        inputs["release_record_path"],
                    )
                ),
                0,
            )
            inputs["decision_path"].write_text("{}\n", encoding="utf-8")
            decision_sha256, decision_size = _file_fingerprint(inputs["decision_path"])
            record = _read_json_object(inputs["release_record_path"])
            record["artifacts"]["promotion_decision"]["sha256"] = decision_sha256
            record["artifacts"]["promotion_decision"]["size_bytes"] = decision_size
            record["bindings"]["promotion_decision_sha256"] = decision_sha256
            _write_source_json(inputs["release_record_path"], record)

            errors = validate_promotion_release_record(
                inputs["release_record_path"]
            ).errors

            self.assertTrue(
                any(
                    error.startswith(
                        "promotion_release_record current promotion_decision is invalid:"
                    )
                    for error in errors
                ),
                errors,
            )
            self.assertIn(
                "promotion_release_record current promotion decision must pass and authorize alias application.",
                errors,
            )

    def test_validate_promotion_release_record_requires_canonical_checks_and_nonempty_id(
        self,
    ):
        mutations = ("missing_check", "duplicate_check", "empty_release_id")
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                inputs = self._prepare_promotion_release_record_inputs(root)
                self.assertEqual(
                    run_cli(
                        promotion_release_record_args(
                            inputs["artifacts"],
                            inputs["cards_dir"],
                            inputs["decision_path"],
                            inputs["alias_receipt_path"],
                            inputs["release_notes_path"],
                            inputs["release_record_path"],
                        )
                    ),
                    0,
                )
                record = _read_json_object(inputs["release_record_path"])
                if mutation == "missing_check":
                    record["checks"].pop(1)
                    record["check_count"] -= 1
                    record["metrics"]["check_count"] -= 1
                elif mutation == "duplicate_check":
                    record["checks"][1] = json.loads(json.dumps(record["checks"][0]))
                else:
                    record["release"]["id"] = ""
                _write_source_json(inputs["release_record_path"], record)

                errors = validate_promotion_release_record(
                    inputs["release_record_path"]
                ).errors

                if mutation == "empty_release_id":
                    self.assertIn(
                        "promotion_release_record.release.id must be a non-empty string.",
                        errors,
                    )
                else:
                    self.assertIn(
                        "promotion_release_record.checks must exactly match the canonical ordered release check contract.",
                        errors,
                    )

    def test_promotion_release_record_canonical_checks_include_rollback_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = write_governance_artifacts(root)
            rollback_root = root / "rollback_source"
            rollback_root.mkdir()
            rollback_registry_path = write_model_registry(rollback_root)
            rollback_receipt_path = root / "promotion_rollback_receipt.json"
            self.assertEqual(
                run_cli(
                    promotion_rollback_receipt_args(
                        rollback_registry_path,
                        rollback_receipt_path,
                    )
                ),
                0,
            )
            artifacts["rollback_metadata"] = rollback_receipt_path
            training_export = write_training_export(root)
            cards_dir = root / "cards"
            decision_path = root / "promotion_decision.json"
            registry_path = write_model_registry(root)
            alias_receipt_path = root / "promotion_alias_apply.json"
            release_notes_path = write_release_notes(root)
            release_record_path = root / "promotion_release_record.json"
            self.assertEqual(
                run_cli(promotion_cards_args(artifacts, training_export, cards_dir)),
                0,
            )
            artifacts["model_card"] = cards_dir / "MODEL_CARD.md"
            artifacts["dataset_card"] = cards_dir / "DATASET_CARD.md"
            self.assertEqual(
                run_cli(promotion_decision_args(artifacts, decision_path)), 0
            )
            self.assertEqual(
                run_cli(
                    promotion_alias_apply_args(
                        registry_path,
                        decision_path,
                        alias_receipt_path,
                    )
                ),
                0,
            )
            self.assertEqual(
                run_cli(
                    promotion_release_record_args(
                        artifacts,
                        cards_dir,
                        decision_path,
                        alias_receipt_path,
                        release_notes_path,
                        release_record_path,
                    )
                ),
                0,
            )

            record = _read_json_object(release_record_path)
            check_ids = [check["id"] for check in record["checks"]]

            self.assertEqual(check_ids.count("rollback_receipt_passed"), 1)
            self.assertEqual(
                validate_promotion_release_record(release_record_path).errors, []
            )

    def test_validate_promotion_release_record_replays_rehashed_cards_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs = self._prepare_promotion_release_record_inputs(root)
            self.assertEqual(
                run_cli(
                    promotion_release_record_args(
                        inputs["artifacts"],
                        inputs["cards_dir"],
                        inputs["decision_path"],
                        inputs["alias_receipt_path"],
                        inputs["release_notes_path"],
                        inputs["release_record_path"],
                    )
                ),
                0,
            )
            manifest_path = inputs["cards_dir"] / "promotion_cards.json"
            manifest = _read_json_object(manifest_path)
            manifest["candidate"]["id"] = "other-candidate"
            _write_source_json(manifest_path, manifest)
            record = _read_json_object(inputs["release_record_path"])
            cards_record = governance._artifact_record(
                "promotion_cards",
                inputs["cards_dir"],
                True,
                inputs["release_record_path"],
            )
            record["artifacts"]["promotion_cards"] = cards_record
            record["bindings"]["promotion_cards_sha256"] = cards_record["sha256"]
            _write_source_json(inputs["release_record_path"], record)

            errors = validate_promotion_release_record(
                inputs["release_record_path"]
            ).errors

            self.assertIn(
                "promotion_release_record current promotion cards candidate must match the current promotion decision.",
                errors,
            )


def write_governance_artifacts(
    root: Path,
    *,
    license_status: str = "known",
    accepted_terms: bool | None = True,
    compare_metrics=None,
) -> dict[str, Path | list[Path] | None]:
    eval_lineage = write_promotion_eval_lineage(root)
    evidence_root = root / "evidence"
    evidence_root.mkdir()
    evidence_bundle = evidence_root / "evidence_bundle.json"
    evidence_bundle.write_text(
        json.dumps(
            build_evidence_bundle(
                out_path=evidence_bundle,
                eval_summary_path=eval_lineage["eval_summary"],
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    example_root = ROOT / "examples" / "agentic_training"
    semantic_root = root / "examples" / "agentic_training"
    semantic_root.mkdir(parents=True, exist_ok=True)
    for filename in ("trainer_preflight.json", "training_gate.json"):
        shutil.copyfile(example_root / filename, semantic_root / filename)
    for dirname in ("plans", "training_export"):
        shutil.copytree(
            example_root / dirname,
            semantic_root / dirname,
            dirs_exist_ok=True,
        )
    shutil.copytree(
        example_root / "promotion_governance" / "compare_export",
        semantic_root / "promotion_governance" / "compare_export",
        dirs_exist_ok=True,
    )
    for filename in ("promotion_ledger.json", "promotion_history_decision_gate.json"):
        shutil.copyfile(
            example_root / "promotion_governance" / filename,
            root / filename,
        )
    promotion_gate_payload = _read_json_object(
        example_root / "promotion_governance" / "promotion_ledger_gate.json"
    )
    promotion_ledger_gate = _write_source_json(
        root / "promotion_ledger_gate.json",
        promotion_gate_payload,
    )
    compare_payload = _read_json_object(
        example_root / "promotion_governance" / "compare_gate.json"
    )
    metrics = compare_payload["metrics"]
    if compare_metrics is not None:
        metrics.update(compare_metrics)
    compare_gate = _write_source_json(root / "compare_gate.json", compare_payload)
    trainer_launch_check = _write_source_json(
        root / "trainer_launch_check.json",
        _read_json_object(example_root / "trainer_launch_check.json"),
    )
    agentic_training_result = _write_candidate_training_result(
        root,
        PROMOTION_CANDIDATE_ID,
    )
    registry_entry_payload = _read_json_object(
        example_root / "promotion_governance" / "model_registry_entry.json"
    )
    registry_entry_payload["candidate_id"] = PROMOTION_CANDIDATE_ID
    registry_entry_payload["entry_id"] = PROMOTION_CANDIDATE_ID
    registry_entry_payload["candidate"]["candidate_id"] = PROMOTION_CANDIDATE_ID
    registry_entry_payload["candidate"]["model_id"] = PROMOTION_CANDIDATE_ID
    _bind_registry_entry_links(
        registry_entry_payload,
        {
            "datasets": example_root / "training_export" / "manifest.json",
            "evals": compare_gate,
            "serving_probes": (
                example_root
                / "serving_lifecycle"
                / "managed_mock"
                / "preflight"
                / "serving_check.json"
            ),
            "training_runs": agentic_training_result,
        },
    )
    model_registry_entry = _write_source_json(
        root / "model_registry_entry.json", registry_entry_payload
    )
    model_card = root / "MODEL_CARD.md"
    model_card.write_text(
        "# Model Card\n\nEvidence-backed candidate model.\n", encoding="utf-8"
    )
    dataset_card = root / "DATASET_CARD.md"
    dataset_card.write_text(
        "# Dataset Card\n\nRedacted held-out data.\n", encoding="utf-8"
    )
    rollback_metadata = root / "rollback.json"
    rollback_metadata.write_text(
        json.dumps({"available": True, "rollback_id": "champion-v1"}), encoding="utf-8"
    )
    license_review = root / "license_review.json"
    license_payload = {"license_status": license_status, "passed": True}
    if accepted_terms is not None:
        license_payload["accepted_terms"] = accepted_terms
    license_review.write_text(json.dumps(license_payload), encoding="utf-8")
    redaction_check = root / "redaction_check.json"
    redaction_check.write_text(json.dumps({"passed": True}), encoding="utf-8")
    safety_gate = root / "safety_gate.json"
    safety_gate.write_text(json.dumps({"passed": True}), encoding="utf-8")
    serving_profile = _write_source_json(
        root / "serving_profile.json",
        _read_json_object(
            example_root
            / "serving_lifecycle"
            / "managed_mock"
            / "preflight"
            / "serving_profile.json"
        ),
    )
    serving_report = root / "serving_report.json"
    serving_report.write_text(json.dumps({"passed": True}), encoding="utf-8")
    return {
        "evidence_bundle": evidence_bundle,
        "eval_summary": eval_lineage["eval_summary"],
        "external_eval_results": [eval_lineage["external_eval_result"]],
        "promotion_ledger_gate": promotion_ledger_gate,
        "compare_gate": compare_gate,
        "trainer_launch_check": trainer_launch_check,
        "model_registry_entry": model_registry_entry,
        "agentic_training_result": agentic_training_result,
        "model_card": model_card,
        "dataset_card": dataset_card,
        "rollback_metadata": rollback_metadata,
        "license_review": license_review,
        "redaction_check": redaction_check,
        "safety_gate": safety_gate,
        "serving_profile": serving_profile,
        "serving_report": serving_report,
    }


def write_promotion_eval_lineage(
    root: Path,
    *,
    candidate_id: str = PROMOTION_CANDIDATE_ID,
    dirname: str = "promotion_eval",
    execution_id: str = "promotion-eval-001",
    benchmark_passed: bool = True,
) -> dict[str, Path]:
    lineage_root = root / dirname
    lineage_root.mkdir()
    example_root = ROOT / "examples" / "agentic_training" / "heldout_eval"
    for filename in (
        "baseline_suite_summary.json",
        "candidate_suite_summary.json",
        "heldout_manifest.json",
        "external_eval_raw_result.json",
        "external_eval_runner.json",
    ):
        shutil.copyfile(example_root / filename, lineage_root / filename)
    shutil.copytree(
        example_root / "scenarios",
        lineage_root / "scenarios",
        dirs_exist_ok=True,
    )

    heldout_path = lineage_root / "heldout_manifest.json"
    plan_path = lineage_root / "external_eval_plan.json"
    plan = build_external_eval_plan(
        adapters=["local_mock"],
        scenario_manifest=heldout_path,
        model_endpoint=candidate_id,
        model=candidate_id,
        allow_installed=True,
        output_base_dir=lineage_root,
    )
    write_external_eval_plan(plan, plan_path)

    raw_result_path = lineage_root / "external_eval_raw_result.json"
    if not benchmark_passed:
        raw_result = json.loads(raw_result_path.read_text(encoding="utf-8"))
        raw_result[0]["passed"] = False
        raw_result[0]["score"] = 0
        raw_result_path.write_text(
            json.dumps(raw_result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    runner_path = lineage_root / "external_eval_runner.json"
    runner = json.loads(runner_path.read_text(encoding="utf-8"))
    runner["execution_id"] = execution_id
    runner["model_id"] = candidate_id
    runner_path.write_text(
        json.dumps(runner, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    result_path = lineage_root / "external_eval_result.json"
    result = build_external_eval_result(
        plan_path=plan_path,
        heldout_manifest_path=heldout_path,
        raw_result_path=raw_result_path,
        runner_metadata_path=runner_path,
        adapter_id="local_mock",
        execution_id=execution_id,
        model_id=candidate_id,
        normalizer_id="hfr.local_mock.per_case_json",
        normalizer_version="1",
        raw_format="json",
        execution_status="completed",
        out_path=result_path,
        created_at="2026-07-10T00:01:00+00:00",
    )
    write_external_eval_result(result, result_path)

    eval_summary_path = lineage_root / "eval_summary.json"
    summary = build_eval_summary(
        external_adapter_plan_specs=[f"local_mock={plan_path}"],
        external_adapter_result_specs=[f"local_mock={result_path}"],
        output_base_dir=lineage_root,
    )
    eval_summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "eval_summary": eval_summary_path,
        "external_eval_result": result_path,
    }


def _write_candidate_training_result(root: Path, candidate_id: str) -> Path:
    example_root = ROOT / "examples" / "agentic_training"
    for dirname in ("plans", "runtime_preflight", "trainer_outputs"):
        shutil.copytree(
            example_root / dirname,
            root / dirname,
            dirs_exist_ok=True,
        )
    for filename in ("agentic_training_flow.json", "trainer_consumer_plan.json"):
        shutil.copyfile(example_root / filename, root / filename)
    payload = _read_json_object(example_root / "completed_result.json")
    result_path = root / "agentic_training_result.json"
    payload["artifact_path"] = result_path.name
    payload["registry_update"]["target_model_id"] = candidate_id
    payload["registry_update"]["links"][0]["path"] = result_path.name
    return _write_source_json(result_path, payload)


def _bind_registry_entry_links(payload: dict, sources: dict[str, Path]) -> None:
    links = payload["links"]
    for collection, source_path in sources.items():
        row = links[collection][0]
        source_bytes = source_path.read_bytes()
        row["path"] = str(source_path.resolve())
        row["sha256"] = hashlib.sha256(source_bytes).hexdigest()
        row["size_bytes"] = len(source_bytes)


def _read_json_object(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AssertionError(f"expected JSON object fixture: {path}")
    return payload


def _file_fingerprint(path: Path) -> tuple[str, int]:
    content = path.read_bytes()
    return hashlib.sha256(content).hexdigest(), len(content)


def _write_source_json(path: Path, payload: dict) -> Path:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def write_training_export(root: Path) -> Path:
    training_export = root / "training_export"
    training_export.mkdir()
    (training_export / "DATASET_CARD.md").write_text(
        "# Dataset Card\n\nGenerated upstream.\n", encoding="utf-8"
    )
    return training_export


def write_model_registry(
    root: Path, *, champion_alias: str = "champion-v1", alias_history=None
) -> Path:
    registry_path = root / "model_registry.json"
    candidate_source = root / "model_registry_entry.json"
    if not candidate_source.exists():
        candidate_source = (
            ROOT
            / "examples"
            / "agentic_training"
            / "promotion_governance"
            / "model_registry_entry.json"
        )
    candidate_entry = _read_json_object(candidate_source)

    def historical_entry(model_id: str, status: str) -> dict:
        entry = json.loads(json.dumps(candidate_entry))
        entry["candidate_id"] = model_id
        entry["entry_id"] = model_id
        entry["candidate"]["candidate_id"] = model_id
        entry["candidate"]["model_id"] = model_id
        entry["status"] = status
        entry["links"] = {collection: [] for collection in entry["links"]}
        return entry

    return _write_source_json(
        registry_path,
        {
            "schema_version": "hfr.model_registry.v1",
            "registry_path": registry_path.name,
            "updated_at": "2026-07-10T00:00:00Z",
            "entries": {
                PROMOTION_CANDIDATE_ID: candidate_entry,
                "champion-v1": historical_entry("champion-v1", "current_champion"),
                "other-model": historical_entry("other-model", "historical"),
            },
            "aliases": {
                "candidate": PROMOTION_CANDIDATE_ID,
                "champion": champion_alias,
                "rollback": "other-model",
            },
            "alias_history": [] if alias_history is None else alias_history,
            "notes": [],
        },
    )


def write_release_notes(root: Path) -> Path:
    release_notes = root / "RELEASE_NOTES.md"
    release_notes.write_text(
        f"# Release Notes\n\n- Promotes {PROMOTION_CANDIDATE_ID} after passing governance gates.\n- Rollback target: champion-v1.\n",
        encoding="utf-8",
    )
    return release_notes


def write_promotion_policy(
    root: Path,
    *,
    filename: str = "promotion_policy.json",
    policy_id: str = "strict-local-policy",
    required_artifacts: list[str] | None = None,
    release_required_artifacts: list[str] | None = None,
    forbid_new_critical_rules: list[str] | None = None,
    forbid_regressed_rules: list[str] | None = None,
    limits: dict[str, int] | None = None,
    require_accepted_terms: bool = True,
) -> Path:
    default_limits = {
        "max_task_completion_regressions": 0,
        "max_baseline_wins": 0,
        "max_contract_drifts": 0,
        "max_unverified_contracts": 0,
        "max_new_critical_failures": 0,
        "max_rule_regressions": 0,
    }
    if limits:
        default_limits.update(limits)
    policy_path = root / filename
    policy_path.write_text(
        json.dumps(
            {
                "schema_version": "hfr.promotion_policy.v1",
                "id": policy_id,
                "description": "Strict local promotion policy for tests.",
                "required_artifacts": required_artifacts
                or promotion_decision_required_artifacts(),
                "release_required_artifacts": release_required_artifacts
                or promotion_release_required_artifacts(),
                "allowed_candidate_classes": [
                    "base",
                    "candidate",
                    "champion",
                    "frontier",
                    "trace-only",
                ],
                "allowed_champion_classes": [
                    "base",
                    "candidate",
                    "champion",
                    "frontier",
                    "trace-only",
                ],
                "limits": default_limits,
                "forbid_new_critical_rules": forbid_new_critical_rules
                or promotion_policy_required_forbidden_rules(),
                "forbid_regressed_rules": forbid_regressed_rules
                or promotion_policy_required_forbidden_rules(),
                "require_known_license": True,
                "require_accepted_terms": require_accepted_terms,
                "require_rollback_metadata": True,
                "require_supported_cards": True,
                "require_artifact_validation": True,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return policy_path


def promotion_decision_required_artifacts() -> list[str]:
    return [
        "evidence_bundle",
        "eval_summary",
        "promotion_ledger_gate",
        "compare_gate",
        "trainer_launch_check",
        "model_registry_entry",
        "agentic_training_result",
        "model_card",
        "dataset_card",
        "rollback_metadata",
        "license_review",
        "redaction_check",
        "safety_gate",
        "serving_profile",
        "serving_report",
    ]


def promotion_release_required_artifacts() -> list[str]:
    return [
        "promotion_decision",
        "promotion_cards",
        "promotion_alias_apply",
        "rollback_metadata",
        "compare_gate",
        "release_notes",
    ]


def promotion_policy_required_forbidden_rules() -> list[str]:
    return ["final_answer", "forbidden_actions", "secret_exposure"]


def promotion_alias_apply_args(
    registry_path: Path, decision_path: Path, receipt_path: Path
) -> list[str]:
    return [
        "promotion-alias-apply",
        "--registry",
        str(registry_path),
        "--promotion-decision",
        str(decision_path),
        "--out",
        str(receipt_path),
        "--preserve-paths",
    ]


def promotion_rollback_receipt_args(
    registry_path: Path, receipt_path: Path, rollback_id: str = "champion-v1"
) -> list[str]:
    return [
        "promotion-rollback-receipt",
        "--registry",
        str(registry_path),
        "--rollback-id",
        rollback_id,
        "--out",
        str(receipt_path),
        "--preserve-paths",
    ]


def promotion_release_record_args(
    artifacts: dict[str, Path | None],
    cards_dir: Path,
    decision_path: Path,
    alias_receipt_path: Path,
    release_notes_path: Path,
    release_record_path: Path,
    promotion_policy: Path | None = None,
) -> list[str]:
    args = [
        "promotion-release-record",
        "--release-id",
        "release-2026-07-02",
        "--promotion-decision",
        str(decision_path),
        "--promotion-cards",
        str(cards_dir),
        "--promotion-alias-apply",
        str(alias_receipt_path),
        "--rollback-metadata",
        str(artifacts["rollback_metadata"]),
        "--compare-gate",
        str(artifacts["compare_gate"]),
        "--release-notes",
        str(release_notes_path),
        "--out",
        str(release_record_path),
        "--preserve-paths",
    ]
    if promotion_policy is not None:
        args.extend(["--promotion-policy", str(promotion_policy)])
    return args


def promotion_cards_args(
    artifacts: dict[str, Path | None],
    training_export: Path,
    out_dir: Path,
    *,
    license_status: str = "known",
) -> list[str]:
    return [
        "promotion-cards",
        "--candidate-id",
        PROMOTION_CANDIDATE_ID,
        "--dataset-id",
        "dataset-v1",
        "--model-source",
        "base-model",
        "--license-status",
        license_status,
        "--evidence-bundle",
        str(artifacts["evidence_bundle"]),
        "--training-export",
        str(training_export),
        "--compare-gate",
        str(artifacts["compare_gate"]),
        "--redaction-check",
        str(artifacts["redaction_check"]),
        "--safety-gate",
        str(artifacts["safety_gate"]),
        "--out",
        str(out_dir),
        "--preserve-paths",
    ]


def promotion_decision_args(
    artifacts: dict[str, Path | list[Path] | None],
    out_path: Path,
    *,
    promotion_policy: Path | None = None,
    candidate_id: str = PROMOTION_CANDIDATE_ID,
) -> list[str]:
    args = [
        "promotion-decision",
        "--candidate-id",
        candidate_id,
        "--champion-id",
        "champion-v1",
        "--rollback-id",
        "champion-v1",
        "--out",
        str(out_path),
        "--preserve-paths",
    ]
    flag_by_role = {
        "evidence_bundle": "--evidence-bundle",
        "eval_summary": "--eval-summary",
        "promotion_ledger_gate": "--promotion-ledger-gate",
        "compare_gate": "--compare-gate",
        "trainer_launch_check": "--trainer-launch-check",
        "model_registry_entry": "--model-registry-entry",
        "agentic_training_result": "--agentic-training-result",
        "model_card": "--model-card",
        "dataset_card": "--dataset-card",
        "rollback_metadata": "--rollback-metadata",
        "license_review": "--license-review",
        "redaction_check": "--redaction-check",
        "safety_gate": "--safety-gate",
        "serving_profile": "--serving-profile",
        "serving_report": "--serving-report",
    }
    for role, flag in flag_by_role.items():
        path = artifacts.get(role)
        if path is not None:
            args.extend([flag, str(path)])
    for path in artifacts.get("external_eval_results") or []:
        args.extend(["--external-eval-result", str(path)])
    if promotion_policy is not None:
        args.extend(["--promotion-policy", str(promotion_policy)])
    return args


def failed_check_ids(decision: dict) -> set[str]:
    return {check["id"] for check in decision["checks"] if not check["passed"]}
