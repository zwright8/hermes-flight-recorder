from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

from flightrecorder.cli import main
from flightrecorder.validation import _path_has_symlink_component

ROOT = Path(__file__).resolve().parents[1]


def run_cli(args):
    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        return main(args)


def without_preserve_paths(args):
    return [arg for arg in args if arg != "--preserve-paths"]


class PromotionDecisionTests(unittest.TestCase):
    def test_promotion_cards_generate_valid_model_and_dataset_cards(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = write_governance_artifacts(root)
            training_export = write_training_export(root)
            cards_dir = root / "cards"

            code = run_cli(promotion_cards_args(artifacts, training_export, cards_dir))

            self.assertEqual(code, 0)
            manifest = json.loads((cards_dir / "promotion_cards.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["schema_version"], "hfr.promotion_cards.v1")
            self.assertTrue(manifest["passed"])
            self.assertTrue((cards_dir / "MODEL_CARD.md").is_file())
            self.assertTrue((cards_dir / "DATASET_CARD.md").is_file())
            cards_text = (cards_dir / "MODEL_CARD.md").read_text(encoding="utf-8").lower()
            cards_text += (cards_dir / "DATASET_CARD.md").read_text(encoding="utf-8").lower()
            self.assertNotIn("todo", cards_text)
            self.assertNotIn("unsupported claim", cards_text)
            self.assertEqual(run_cli(["validate", "--promotion-cards", str(cards_dir), "--strict"]), 0)
            self.assertEqual(run_cli(["schemas", "--check", str(cards_dir / "promotion_cards.json")]), 0)

    def test_promotion_cards_block_unknown_license(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = write_governance_artifacts(root)
            training_export = write_training_export(root)
            cards_dir = root / "cards"

            code = run_cli(promotion_cards_args(artifacts, training_export, cards_dir, license_status="unknown"))

            self.assertEqual(code, 1)
            manifest = json.loads((cards_dir / "promotion_cards.json").read_text(encoding="utf-8"))
            self.assertFalse(manifest["passed"])
            self.assertIn("license_status_known", failed_check_ids(manifest))
            self.assertEqual(run_cli(["validate", "--promotion-cards", str(cards_dir), "--strict"]), 0)

    def test_validate_promotion_cards_rejects_stale_generated_card(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = write_governance_artifacts(root)
            training_export = write_training_export(root)
            cards_dir = root / "cards"
            self.assertEqual(run_cli(promotion_cards_args(artifacts, training_export, cards_dir)), 0)

            (cards_dir / "MODEL_CARD.md").write_text("# Model Card\n\nchanged after manifest\n", encoding="utf-8")

            self.assertEqual(run_cli(["validate", "--promotion-cards", str(cards_dir), "--strict"]), 1)

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
            aliases = {item["alias"]: item for item in decision["alias_update"]["aliases"]}
            self.assertEqual(aliases["champion"]["previous_target"], "champion-v1")
            self.assertEqual(aliases["champion"]["target"], "candidate-v2")
            self.assertEqual(aliases["rollback"]["target"], "champion-v1")
            self.assertEqual(run_cli(["validate", "--promotion-decision", str(decision_path), "--strict"]), 0)
            self.assertEqual(run_cli(["schemas", "--check", str(decision_path)]), 0)

    def test_promotion_alias_apply_moves_aliases_after_valid_decision(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = write_governance_artifacts(root)
            registry_path = write_model_registry(root)
            decision_path = root / "promotion_decision.json"
            receipt_path = root / "promotion_alias_apply.json"
            self.assertEqual(run_cli(promotion_decision_args(artifacts, decision_path)), 0)

            code = run_cli(promotion_alias_apply_args(registry_path, decision_path, receipt_path))

            self.assertEqual(code, 0)
            registry = json.loads(registry_path.read_text(encoding="utf-8"))
            self.assertEqual(registry["aliases"]["candidate"], "candidate-v2")
            self.assertEqual(registry["aliases"]["champion"], "candidate-v2")
            self.assertEqual(registry["aliases"]["rollback"], "champion-v1")
            self.assertEqual(len(registry["alias_history"]), 1)
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            self.assertTrue(receipt["passed"])
            self.assertEqual(receipt["recommendation"], "alias_update_applied")
            self.assertEqual(receipt["promotion_decision"]["size_bytes"], decision_path.stat().st_size)
            self.assertNotIn("path", receipt["registry_before"])
            self.assertNotIn("sha256", receipt["registry_before"])
            self.assertNotIn("size_bytes", receipt["registry_before"])
            self.assertEqual(receipt["registry_after"]["size_bytes"], registry_path.stat().st_size)
            self.assertEqual(receipt["registry_after"]["size_bytes"], receipt["artifacts"]["registry"]["size_bytes"])
            self.assertEqual(receipt["promotion_decision_validation"]["passed"], True)
            self.assertEqual(receipt["alias_history_entry"]["promotion_decision_sha256"], receipt["promotion_decision"]["sha256"])
            self.assertEqual(receipt["alias_history_entry"]["updated_aliases"]["champion"], "candidate-v2")
            self.assertEqual(run_cli(["validate", "--promotion-alias-apply", str(receipt_path), "--strict"]), 0)
            self.assertEqual(run_cli(["schemas", "--check", str(receipt_path)]), 0)

    def test_validate_promotion_alias_apply_rejects_forged_decision_validation_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = write_governance_artifacts(root)
            registry_path = write_model_registry(root)
            decision_path = root / "promotion_decision.json"
            receipt_path = root / "promotion_alias_apply.json"
            summary_path = root / "validation.json"
            self.assertEqual(run_cli(promotion_decision_args(artifacts, decision_path)), 0)
            self.assertEqual(run_cli(promotion_alias_apply_args(registry_path, decision_path, receipt_path)), 0)

            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            receipt["promotion_decision_validation"]["warning_count"] = 1
            receipt["promotion_decision_validation"]["targets"][0]["warning_count"] = 1
            receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            code = run_cli(["validate", "--promotion-alias-apply", str(receipt_path), "--strict", "--out", str(summary_path)])

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in summary["targets"] for error in target["errors"])
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
            self.assertEqual(run_cli(promotion_decision_args(artifacts, decision_path)), 0)
            self.assertEqual(run_cli(promotion_alias_apply_args(registry_path, decision_path, receipt_path)), 0)

            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            receipt["promotion_decision"]["size_bytes"] += 1
            receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            code = run_cli(["validate", "--promotion-alias-apply", str(receipt_path), "--strict", "--out", str(summary_path)])

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in summary["targets"] for error in target["errors"])
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
            self.assertEqual(run_cli(promotion_decision_args(artifacts, decision_path)), 0)
            self.assertEqual(run_cli(promotion_alias_apply_args(registry_path, decision_path, receipt_path)), 0)

            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            receipt["registry_after"]["size_bytes"] += 1
            receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            code = run_cli(["validate", "--promotion-alias-apply", str(receipt_path), "--strict", "--out", str(summary_path)])

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in summary["targets"] for error in target["errors"])
            self.assertIn("promotion_alias_apply.registry_after.size_bytes must match artifacts.registry.size_bytes.", errors)

    def test_validate_promotion_alias_apply_rejects_registry_before_size_claim(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = write_governance_artifacts(root)
            registry_path = write_model_registry(root)
            decision_path = root / "promotion_decision.json"
            receipt_path = root / "promotion_alias_apply.json"
            summary_path = root / "validation.json"
            self.assertEqual(run_cli(promotion_decision_args(artifacts, decision_path)), 0)
            self.assertEqual(run_cli(promotion_alias_apply_args(registry_path, decision_path, receipt_path)), 0)

            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            receipt["registry_before"]["size_bytes"] = receipt["registry_after"]["size_bytes"]
            receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            self.assertEqual(run_cli(["schemas", "--check", str(receipt_path)]), 1)
            code = run_cli(["validate", "--promotion-alias-apply", str(receipt_path), "--strict", "--out", str(summary_path)])

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in summary["targets"] for error in target["errors"])
            self.assertIn("promotion_alias_apply.registry_before.size_bytes must be absent for snapshot-only refs.", errors)

    def test_promotion_alias_apply_blocks_stale_champion_alias_without_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = write_governance_artifacts(root)
            registry_path = write_model_registry(root, champion_alias="other-model")
            decision_path = root / "promotion_decision.json"
            receipt_path = root / "promotion_alias_apply.json"
            self.assertEqual(run_cli(promotion_decision_args(artifacts, decision_path)), 0)
            before = json.loads(registry_path.read_text(encoding="utf-8"))

            code = run_cli(promotion_alias_apply_args(registry_path, decision_path, receipt_path))

            self.assertEqual(code, 1)
            after = json.loads(registry_path.read_text(encoding="utf-8"))
            self.assertEqual(after, before)
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            self.assertFalse(receipt["passed"])
            self.assertIn("champion_alias_matches_previous_target", failed_check_ids(receipt))
            self.assertEqual(run_cli(["validate", "--promotion-alias-apply", str(receipt_path), "--strict"]), 0)

    def test_promotion_alias_apply_blocks_malformed_alias_history_without_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = write_governance_artifacts(root)
            registry_path = write_model_registry(root, alias_history="not-a-list")
            decision_path = root / "promotion_decision.json"
            receipt_path = root / "promotion_alias_apply.json"
            self.assertEqual(run_cli(promotion_decision_args(artifacts, decision_path)), 0)
            before = json.loads(registry_path.read_text(encoding="utf-8"))

            code = run_cli(promotion_alias_apply_args(registry_path, decision_path, receipt_path))

            self.assertEqual(code, 1)
            after = json.loads(registry_path.read_text(encoding="utf-8"))
            self.assertEqual(after, before)
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            self.assertFalse(receipt["passed"])
            self.assertIsNone(receipt["alias_history_entry"])
            self.assertIn("registry_alias_history_list", failed_check_ids(receipt))
            self.assertEqual(run_cli(["validate", "--promotion-alias-apply", str(receipt_path), "--strict"]), 0)

    def test_promotion_rollback_receipt_validates_current_champion_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            registry_path = write_model_registry(root)
            receipt_path = root / "rollback.json"

            code = run_cli(promotion_rollback_receipt_args(registry_path, receipt_path))

            self.assertEqual(code, 0)
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            self.assertEqual(receipt["schema_version"], "hfr.promotion_rollback_receipt.v1")
            self.assertTrue(receipt["passed"])
            self.assertTrue(receipt["available"])
            self.assertEqual(receipt["rollback_id"], "champion-v1")
            self.assertEqual(receipt["registry"]["aliases"]["champion"], "champion-v1")
            self.assertEqual(receipt["registry"]["size_bytes"], registry_path.stat().st_size)
            self.assertEqual(receipt["registry"]["size_bytes"], receipt["artifacts"]["registry"]["size_bytes"])
            self.assertEqual(run_cli(["validate", "--promotion-rollback-receipt", str(receipt_path), "--strict"]), 0)
            self.assertEqual(run_cli(["schemas", "--check", str(receipt_path)]), 0)

    def test_validate_promotion_rollback_receipt_rejects_forged_registry_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            registry_path = write_model_registry(root)
            receipt_path = root / "rollback.json"
            summary_path = root / "validation.json"
            self.assertEqual(run_cli(promotion_rollback_receipt_args(registry_path, receipt_path)), 0)

            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            receipt["registry"]["size_bytes"] += 1
            receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            code = run_cli(["validate", "--promotion-rollback-receipt", str(receipt_path), "--strict", "--out", str(summary_path)])

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in summary["targets"] for error in target["errors"])
            self.assertIn("promotion_rollback_receipt.registry.size_bytes must match artifacts.registry.size_bytes.", errors)

    def test_validate_promotion_rollback_receipt_rejects_stale_registry_alias_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            registry_path = write_model_registry(root)
            receipt_path = root / "rollback.json"
            summary_path = root / "validation.json"
            self.assertEqual(run_cli(promotion_rollback_receipt_args(registry_path, receipt_path)), 0)

            registry = json.loads(registry_path.read_text(encoding="utf-8"))
            registry["aliases"]["champion"] = "other-model"
            registry_path.write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            registry_sha = hashlib.sha256(registry_path.read_bytes()).hexdigest()
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            receipt["artifacts"]["registry"]["sha256"] = registry_sha
            receipt["registry"]["sha256"] = registry_sha
            receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            code = run_cli(["validate", "--promotion-rollback-receipt", str(receipt_path), "--strict", "--out", str(summary_path)])

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in summary["targets"] for error in target["errors"])
            self.assertIn("promotion_rollback_receipt.registry.aliases", errors)

    def test_promotion_decision_accepts_passing_rollback_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = write_governance_artifacts(root)
            registry_path = write_model_registry(root)
            receipt_path = root / "rollback.json"
            decision_path = root / "promotion_decision.json"
            self.assertEqual(run_cli(promotion_rollback_receipt_args(registry_path, receipt_path)), 0)
            artifacts["rollback_metadata"] = receipt_path

            code = run_cli(promotion_decision_args(artifacts, decision_path))

            self.assertEqual(code, 0)
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            self.assertTrue(decision["passed"])
            self.assertNotIn("rollback_receipt_passed", failed_check_ids(decision))
            self.assertEqual(run_cli(["validate", "--promotion-decision", str(decision_path), "--strict"]), 0)

    def test_promotion_decision_blocks_failed_rollback_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = write_governance_artifacts(root)
            registry_path = write_model_registry(root, champion_alias="other-model")
            receipt_path = root / "rollback.json"
            decision_path = root / "promotion_decision.json"
            self.assertEqual(run_cli(promotion_rollback_receipt_args(registry_path, receipt_path)), 1)
            artifacts["rollback_metadata"] = receipt_path

            code = run_cli(promotion_decision_args(artifacts, decision_path))

            self.assertEqual(code, 1)
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            self.assertFalse(decision["passed"])
            failed_ids = failed_check_ids(decision)
            self.assertIn("rollback_metadata_matches_target", failed_ids)
            self.assertIn("rollback_receipt_passed", failed_ids)
            self.assertEqual(run_cli(["validate", "--promotion-decision", str(decision_path), "--strict"]), 0)

    def test_promotion_release_record_binds_decision_cards_alias_rollback_eval_and_notes(self):
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
            self.assertEqual(run_cli(promotion_cards_args(artifacts, training_export, cards_dir)), 0)
            artifacts["model_card"] = cards_dir / "MODEL_CARD.md"
            artifacts["dataset_card"] = cards_dir / "DATASET_CARD.md"
            self.assertEqual(run_cli(promotion_decision_args(artifacts, decision_path)), 0)
            self.assertEqual(run_cli(promotion_alias_apply_args(registry_path, decision_path, alias_receipt_path)), 0)

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
            self.assertEqual(record["schema_version"], "hfr.promotion_release_record.v1")
            self.assertTrue(record["passed"])
            self.assertEqual(record["recommendation"], "publish_release")
            self.assertEqual(record["release"]["candidate_id"], "candidate-v2")
            self.assertEqual(record["release"]["rollback_id"], "champion-v1")
            self.assertEqual(record["release"]["dataset_id"], "dataset-v1")
            self.assertEqual(record["artifact_validation"]["passed"], True)
            self.assertEqual(run_cli(["validate", "--promotion-release-record", str(release_record_path), "--strict"]), 0)
            self.assertEqual(run_cli(["schemas", "--check", str(release_record_path)]), 0)

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
            self.assertEqual(run_cli(promotion_cards_args(artifacts, training_export, cards_dir)), 0)
            artifacts["model_card"] = cards_dir / "MODEL_CARD.md"
            artifacts["dataset_card"] = cards_dir / "DATASET_CARD.md"
            self.assertEqual(run_cli(promotion_decision_args(artifacts, decision_path)), 0)
            self.assertEqual(run_cli(promotion_alias_apply_args(registry_path, decision_path, alias_receipt_path)), 0)
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
            release_record_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            code = run_cli(["validate", "--promotion-release-record", str(release_record_path), "--strict", "--out", str(summary_path)])

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in summary["targets"] for error in target["errors"])
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
                run_cli(["promotion-ledger", "--decision-gate", decision_gate_ref, "--out", str(promotion_ledger_path)]),
                0,
            )
            self.assertEqual(
                run_cli(
                    [
                        "gate-promotion-ledger",
                        "--promotion-ledger",
                        str(promotion_ledger_path),
                        "--policy",
                        str(ROOT / "examples" / "promotion_ledger_gate_policy.demo.json"),
                        "--out",
                        str(promotion_ledger_gate_path),
                    ]
                ),
                0,
            )
            artifacts["promotion_ledger_gate"] = promotion_ledger_gate_path
            self.assertEqual(run_cli(promotion_cards_args(artifacts, training_export, cards_dir)), 0)
            artifacts["model_card"] = cards_dir / "MODEL_CARD.md"
            artifacts["dataset_card"] = cards_dir / "DATASET_CARD.md"
            self.assertEqual(run_cli(promotion_decision_args(artifacts, decision_path)), 0)
            self.assertEqual(run_cli(promotion_alias_apply_args(registry_path, decision_path, alias_receipt_path)), 0)
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
            archive = json.loads((archive_dir / "promotion_archive.json").read_text(encoding="utf-8"))
            roles = {artifact["role"] for artifact in archive["artifacts"]}
            self.assertEqual(
                roles,
                {"promotion_ledger", "promotion_ledger_gate", "decision_gate", "source_artifact", "promotion_release_record"},
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
            self.assertEqual(run_cli(promotion_cards_args(artifacts, training_export, cards_dir)), 0)
            artifacts["model_card"] = cards_dir / "MODEL_CARD.md"
            artifacts["dataset_card"] = cards_dir / "DATASET_CARD.md"
            self.assertEqual(run_cli(promotion_decision_args(artifacts, decision_path)), 0)
            self.assertEqual(run_cli(promotion_alias_apply_args(registry_path, decision_path, alias_receipt_path)), 0)
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

            release_notes_path.write_text("# Release Notes\n\nChanged after release record.\n", encoding="utf-8")

            self.assertEqual(run_cli(["validate", "--promotion-release-record", str(release_record_path), "--strict"]), 1)

    def test_validate_promotion_release_record_requires_artifact_validation_targets(self):
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
            self.assertEqual(run_cli(promotion_cards_args(artifacts, training_export, cards_dir)), 0)
            artifacts["model_card"] = cards_dir / "MODEL_CARD.md"
            artifacts["dataset_card"] = cards_dir / "DATASET_CARD.md"
            self.assertEqual(run_cli(promotion_decision_args(artifacts, decision_path)), 0)
            self.assertEqual(run_cli(promotion_alias_apply_args(registry_path, decision_path, alias_receipt_path)), 0)
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
            release_record_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            code = run_cli(["validate", "--promotion-release-record", str(release_record_path), "--strict", "--out", str(summary_path)])

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in summary["targets"] for error in target["errors"])
            self.assertIn("artifact_validation.target_count", errors)

    def test_validate_promotion_release_record_requires_specific_validation_target_types(self):
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
            self.assertEqual(run_cli(promotion_cards_args(artifacts, training_export, cards_dir)), 0)
            artifacts["model_card"] = cards_dir / "MODEL_CARD.md"
            artifacts["dataset_card"] = cards_dir / "DATASET_CARD.md"
            self.assertEqual(run_cli(promotion_decision_args(artifacts, decision_path)), 0)
            self.assertEqual(run_cli(promotion_alias_apply_args(registry_path, decision_path, alias_receipt_path)), 0)
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
            release_record_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            code = run_cli(["validate", "--promotion-release-record", str(release_record_path), "--strict", "--out", str(summary_path)])

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in summary["targets"] for error in target["errors"])
            self.assertIn("artifact_validation.targets missing required type", errors)
            self.assertIn("promotion_alias_apply", errors)

    def test_validate_promotion_release_record_rejects_forged_validation_warning_counts(self):
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
            self.assertEqual(run_cli(promotion_cards_args(artifacts, training_export, cards_dir)), 0)
            artifacts["model_card"] = cards_dir / "MODEL_CARD.md"
            artifacts["dataset_card"] = cards_dir / "DATASET_CARD.md"
            self.assertEqual(run_cli(promotion_decision_args(artifacts, decision_path)), 0)
            self.assertEqual(run_cli(promotion_alias_apply_args(registry_path, decision_path, alias_receipt_path)), 0)
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
            release_record_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            code = run_cli(["validate", "--promotion-release-record", str(release_record_path), "--strict", "--out", str(summary_path)])

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in summary["targets"] for error in target["errors"])
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
            self.assertEqual(run_cli(promotion_cards_args(artifacts, training_export, cards_dir)), 0)
            artifacts["model_card"] = cards_dir / "MODEL_CARD.md"
            artifacts["dataset_card"] = cards_dir / "DATASET_CARD.md"
            self.assertEqual(run_cli(promotion_decision_args(artifacts, decision_path, promotion_policy=policy_path)), 0)
            self.assertEqual(run_cli(promotion_alias_apply_args(registry_path, decision_path, alias_receipt_path)), 0)

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
            self.assertEqual(run_cli(["validate", "--promotion-release-record", str(release_record_path), "--strict"]), 0)

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
            self.assertEqual(run_cli(["validate", "--promotion-decision", str(decision_path), "--strict"]), 0)

    def test_promotion_decision_blocks_missing_registry_training_and_serving_profile(self):
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
            self.assertEqual(run_cli(["validate", "--promotion-decision", str(decision_path), "--strict"]), 0)

    def test_promotion_decision_blocks_unknown_license(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = write_governance_artifacts(root, license_status="unknown")
            decision_path = root / "promotion_decision.json"

            code = run_cli(promotion_decision_args(artifacts, decision_path))

            self.assertEqual(code, 1)
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            self.assertIn("license_status_known", failed_check_ids(decision))
            self.assertEqual(run_cli(["validate", "--promotion-decision", str(decision_path), "--strict"]), 0)

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
            self.assertEqual(run_cli(["validate", "--promotion-decision", str(decision_path), "--strict"]), 0)

    def test_promotion_decision_blocks_incomplete_compare_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = write_governance_artifacts(root)
            artifacts["compare_gate"].write_text(
                json.dumps({"schema_version": "hfr.compare_gate.v1", "passed": True, "metrics": {}}),
                encoding="utf-8",
            )
            decision_path = root / "promotion_decision.json"

            code = run_cli(promotion_decision_args(artifacts, decision_path))

            self.assertEqual(code, 1)
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            self.assertIn("compare_metrics_complete", failed_check_ids(decision))
            self.assertFalse(decision["alias_update"]["authorized"])
            self.assertEqual(run_cli(["validate", "--promotion-decision", str(decision_path), "--strict"]), 0)

    def test_promotion_decision_blocks_incomplete_policy_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = write_governance_artifacts(root)
            decision_path = root / "promotion_decision.json"
            policy_path = write_promotion_policy(
                root,
                required_artifacts=[
                    role for role in promotion_decision_required_artifacts() if role != "serving_report"
                ],
            )

            code = run_cli(promotion_decision_args(artifacts, decision_path, promotion_policy=policy_path))

            self.assertEqual(code, 1)
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            self.assertEqual(decision["policy"]["source"], "file")
            self.assertEqual(decision["policy"]["artifact"]["role"], "promotion_policy")
            self.assertIn("promotion_policy_required_artifacts_complete", failed_check_ids(decision))
            self.assertFalse(decision["alias_update"]["authorized"])
            self.assertEqual(run_cli(["validate", "--promotion-decision", str(decision_path), "--strict"]), 0)

    def test_validate_promotion_policy_rejects_relaxed_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            policy_path = write_promotion_policy(
                root,
                required_artifacts=[
                    role for role in promotion_decision_required_artifacts() if role != "serving_report"
                ],
                forbid_new_critical_rules=["forbidden_actions", "secret_exposure"],
                forbid_regressed_rules=["forbidden_actions", "secret_exposure"],
                limits={"max_task_completion_regressions": 1},
                require_accepted_terms=False,
            )
            summary_path = root / "policy_validation.json"

            code = run_cli(["validate", "--promotion-policy", str(policy_path), "--strict", "--out", str(summary_path)])

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in summary["targets"] for error in target["errors"])
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
            failed_ids = failed_check_ids(json.loads(decision_path.read_text(encoding="utf-8")))
            self.assertIn("task_completion_regressions_absent", failed_ids)
            self.assertIn("baseline_wins_absent", failed_ids)
            self.assertIn("contract_drifts_absent", failed_ids)
            self.assertIn("new_critical_failures_absent", failed_ids)
            self.assertIn("new_critical_secret_exposure_absent", failed_ids)
            self.assertIn("regression_forbidden_actions_absent", failed_ids)
            self.assertEqual(run_cli(["validate", "--promotion-decision", str(decision_path), "--strict"]), 0)

    def test_validate_promotion_decision_rejects_stale_artifact_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = write_governance_artifacts(root)
            decision_path = root / "promotion_decision.json"
            self.assertEqual(run_cli(promotion_decision_args(artifacts, decision_path)), 0)

            artifacts["model_card"].write_text("# Model Card\n\nchanged after decision\n", encoding="utf-8")

            self.assertEqual(run_cli(["validate", "--promotion-decision", str(decision_path), "--strict"]), 1)

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
            self.assertEqual(run_cli(promotion_decision_args(artifacts, decision_path)), 0)
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            decision["artifacts"]["model_card"]["path"] = "MODEL_CARD_LINK.md"
            decision_path.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            code = run_cli(["validate", "--promotion-decision", str(decision_path), "--strict", "--out", str(summary_path)])

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in summary["targets"] for error in target["errors"])
            self.assertIn("promotion_decision.artifacts.model_card.path must not resolve to a symlink", errors)

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
            self.assertEqual(run_cli(promotion_decision_args(artifacts, decision_path)), 0)
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            decision["artifacts"]["model_card"]["path"] = "linked_artifacts/MODEL_CARD.md"
            decision_path.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            code = run_cli(["validate", "--promotion-decision", str(decision_path), "--strict", "--out", str(summary_path)])

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in summary["targets"] for error in target["errors"])
            self.assertIn("promotion_decision.artifacts.model_card.path must not resolve through a symlink", errors)

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
            self.assertEqual(run_cli(promotion_decision_args(artifacts, decision_path)), 0)
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            decision["artifacts"]["model_card"]["path"] = str(linked_parent / "MODEL_CARD.md")
            decision_path.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            code = run_cli(["validate", "--promotion-decision", str(decision_path), "--strict", "--out", str(summary_path)])

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in summary["targets"] for error in target["errors"])
            self.assertIn("promotion_decision.artifacts.model_card.path must not resolve through a symlink", errors)

    def test_promotion_symlink_scan_continues_after_root_alias(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            root_component = Path(root.anchor) / root.parts[1] if root.is_absolute() and len(root.parts) > 1 else None
            if root_component is None or not root_component.is_symlink():
                self.skipTest("root-level symlink unavailable")
            linked_target = root / "linked_target"
            linked_target.mkdir()
            linked_parent = root / "linked_artifacts"
            try:
                linked_parent.symlink_to(linked_target, target_is_directory=True)
            except (NotImplementedError, OSError) as exc:
                self.skipTest(f"symlink unavailable: {exc}")

            self.assertTrue(_path_has_symlink_component(linked_parent / "MODEL_CARD.md", include_leaf=False))

    def test_validate_promotion_cards_rejects_symlink_directory_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = write_governance_artifacts(root)
            training_export = write_training_export(root)
            cards_dir = root / "cards"
            summary_path = root / "validation.json"
            training_export_link = root / "training_export_link"
            try:
                training_export_link.symlink_to(training_export, target_is_directory=True)
            except (NotImplementedError, OSError) as exc:
                self.skipTest(f"symlink unavailable: {exc}")
            self.assertEqual(run_cli(promotion_cards_args(artifacts, training_export, cards_dir)), 0)
            manifest_path = cards_dir / "promotion_cards.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["artifacts"]["training_export"]["path"] = "../training_export_link"
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            code = run_cli(["validate", "--promotion-cards", str(cards_dir), "--strict", "--out", str(summary_path)])

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in summary["targets"] for error in target["errors"])
            self.assertIn("promotion_cards.artifacts.training_export.path must not resolve to a symlink", errors)

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
            self.assertEqual(run_cli(promotion_cards_args(artifacts, training_export, cards_dir)), 0)
            manifest_path = cards_dir / "promotion_cards.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["artifacts"]["training_export"]["path"] = str(linked_parent / "training_export")
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            code = run_cli(["validate", "--promotion-cards", str(cards_dir), "--strict", "--out", str(summary_path)])

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in summary["targets"] for error in target["errors"])
            self.assertIn("promotion_cards.artifacts.training_export.path must not resolve through a symlink", errors)

    def test_validate_promotion_cards_rejects_symlink_manifest_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = write_governance_artifacts(root)
            training_export = write_training_export(root)
            cards_dir = root / "cards"
            cards_link = root / "cards_link"
            summary_path = root / "validation.json"
            self.assertEqual(run_cli(promotion_cards_args(artifacts, training_export, cards_dir)), 0)
            try:
                cards_link.symlink_to(cards_dir, target_is_directory=True)
            except (NotImplementedError, OSError) as exc:
                self.skipTest(f"symlink unavailable: {exc}")

            code = run_cli(["validate", "--promotion-cards", str(cards_link), "--strict", "--out", str(summary_path)])

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in summary["targets"] for error in target["errors"])
            self.assertIn("promotion_cards.artifacts.model_card.path must not resolve through a symlink", errors)

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

            code = run_cli(without_preserve_paths(promotion_cards_args(artifacts, training_export, cards_dir)))

            self.assertEqual(code, 0)
            manifest = json.loads((cards_dir / "promotion_cards.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["artifacts"]["model_card"]["path"], "MODEL_CARD.md")
            self.assertEqual(manifest["artifacts"]["dataset_card"]["path"], "DATASET_CARD.md")
            self.assertEqual(manifest["artifacts"]["evidence_bundle"]["path"], "../../src/evidence_bundle.json")
            self.assertEqual(manifest["artifacts"]["training_export"]["path"], "../../src/training_export")
            self.assertEqual(run_cli(["validate", "--promotion-cards", str(cards_dir), "--strict"]), 0)

    def test_promotion_decision_writes_output_relative_artifact_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "src"
            output = root / "out"
            source.mkdir()
            output.mkdir()
            artifacts = write_governance_artifacts(source)
            decision_path = output / "promotion_decision.json"

            code = run_cli(without_preserve_paths(promotion_decision_args(artifacts, decision_path)))

            self.assertEqual(code, 0)
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            self.assertEqual(decision["artifacts"]["evidence_bundle"]["path"], "../src/evidence_bundle.json")
            self.assertEqual(decision["artifacts"]["model_card"]["path"], "../src/MODEL_CARD.md")
            self.assertEqual(run_cli(["validate", "--promotion-decision", str(decision_path), "--strict"]), 0)

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
            self.assertEqual(run_cli(without_preserve_paths(promotion_decision_args(artifacts, decision_path))), 0)
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            decision["artifacts"]["evidence_bundle"]["path"] = "evidence_bundle.json"
            decision_path.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            previous_cwd = Path.cwd()
            try:
                os.chdir(source)
                code = run_cli(["validate", "--promotion-decision", str(decision_path), "--strict", "--out", str(summary_path)])
            finally:
                os.chdir(previous_cwd)

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in summary["targets"] for error in target["errors"])
            self.assertIn("promotion_decision.artifacts.evidence_bundle.path does not exist at validation time", errors)

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
            self.assertEqual(run_cli(without_preserve_paths(promotion_cards_args(artifacts, training_export, cards_dir))), 0)
            artifacts["model_card"] = cards_dir / "MODEL_CARD.md"
            artifacts["dataset_card"] = cards_dir / "DATASET_CARD.md"
            self.assertEqual(run_cli(without_preserve_paths(promotion_decision_args(artifacts, decision_path))), 0)
            self.assertEqual(run_cli(without_preserve_paths(promotion_alias_apply_args(registry_path, decision_path, alias_receipt_path))), 0)

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
            self.assertEqual(record["artifacts"]["promotion_decision"]["path"], "promotion_decision.json")
            self.assertEqual(record["artifacts"]["promotion_cards"]["path"], "cards")
            self.assertEqual(record["artifacts"]["rollback_metadata"]["path"], "../src/rollback.json")
            self.assertEqual(record["artifacts"]["release_notes"]["path"], "../src/RELEASE_NOTES.md")
            self.assertEqual(run_cli(["validate", "--promotion-release-record", str(release_record_path), "--strict"]), 0)

    def test_validate_promotion_decision_rejects_missing_required_pass_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = write_governance_artifacts(root)
            decision_path = root / "promotion_decision.json"
            summary_path = root / "validation.json"
            self.assertEqual(run_cli(promotion_decision_args(artifacts, decision_path)), 0)
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            decision["checks"] = [check for check in decision["checks"] if check["id"] != "license_terms_accepted"]
            decision["check_count"] = len(decision["checks"])
            decision["metrics"]["check_count"] = len(decision["checks"])
            decision["decision"]["key_metrics"]["check_count"] = len(decision["checks"])
            decision_path.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            code = run_cli(["validate", "--promotion-decision", str(decision_path), "--strict", "--out", str(summary_path)])

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in summary["targets"] for error in target["errors"])
            self.assertIn("missing required passing check", errors)
            self.assertIn("license_terms_accepted", errors)


def write_governance_artifacts(
    root: Path,
    *,
    license_status: str = "known",
    accepted_terms: bool | None = True,
    compare_metrics=None,
) -> dict[str, Path | None]:
    evidence_bundle = root / "evidence_bundle.json"
    evidence_bundle.write_text(json.dumps({"schema_version": "hfr.evidence_bundle.v1", "passed": True}), encoding="utf-8")
    promotion_ledger_gate = root / "promotion_ledger_gate.json"
    promotion_ledger_gate.write_text(
        json.dumps({"schema_version": "hfr.promotion_ledger_gate.v1", "passed": True}),
        encoding="utf-8",
    )
    compare_gate = root / "compare_gate.json"
    metrics = {
        "baseline_win_count": 0,
        "contract_drift_count": 0,
        "new_critical_failure_counts": {},
        "regressed_rule_counts": {},
        "task_completion_regression_count": 0,
        "unverified_contract_count": 0,
    }
    if compare_metrics is not None:
        metrics.update(compare_metrics)
    compare_gate.write_text(
        json.dumps({"schema_version": "hfr.compare_gate.v1", "passed": True, "metrics": metrics}),
        encoding="utf-8",
    )
    trainer_launch_check = root / "trainer_launch_check.json"
    trainer_launch_check.write_text(
        json.dumps({"schema_version": "hfr.trainer_launch_check.v1", "passed": True}),
        encoding="utf-8",
    )
    model_registry_entry = root / "model_registry_entry.json"
    model_registry_entry.write_text(
        json.dumps(
            {
                "schema_version": "hfr.model_registry_entry.v1",
                "candidate_id": "candidate-v2",
                "entry_id": "candidate-v2",
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    agentic_training_result = root / "agentic_training_result.json"
    agentic_training_result.write_text(
        json.dumps({"schema_version": "hfr.agentic_training_result.v1", "passed": True}),
        encoding="utf-8",
    )
    model_card = root / "MODEL_CARD.md"
    model_card.write_text("# Model Card\n\nEvidence-backed candidate model.\n", encoding="utf-8")
    dataset_card = root / "DATASET_CARD.md"
    dataset_card.write_text("# Dataset Card\n\nRedacted held-out data.\n", encoding="utf-8")
    rollback_metadata = root / "rollback.json"
    rollback_metadata.write_text(json.dumps({"available": True, "rollback_id": "champion-v1"}), encoding="utf-8")
    license_review = root / "license_review.json"
    license_payload = {"license_status": license_status, "passed": True}
    if accepted_terms is not None:
        license_payload["accepted_terms"] = accepted_terms
    license_review.write_text(json.dumps(license_payload), encoding="utf-8")
    redaction_check = root / "redaction_check.json"
    redaction_check.write_text(json.dumps({"passed": True}), encoding="utf-8")
    safety_gate = root / "safety_gate.json"
    safety_gate.write_text(json.dumps({"passed": True}), encoding="utf-8")
    serving_profile = root / "serving_profile.json"
    serving_profile.write_text(
        json.dumps(
            {
                "schema_version": "hfr.serving_profile.v1",
                "eval_preflight": {"ready": True, "readiness": "ready", "failed_checks": []},
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    serving_report = root / "serving_report.json"
    serving_report.write_text(json.dumps({"passed": True}), encoding="utf-8")
    return {
        "evidence_bundle": evidence_bundle,
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


def write_training_export(root: Path) -> Path:
    training_export = root / "training_export"
    training_export.mkdir()
    (training_export / "DATASET_CARD.md").write_text("# Dataset Card\n\nGenerated upstream.\n", encoding="utf-8")
    return training_export


def write_model_registry(root: Path, *, champion_alias: str = "champion-v1", alias_history=None) -> Path:
    registry_path = root / "model_registry.json"
    registry_path.write_text(
        json.dumps(
            {
                "schema_version": "hfr.model_registry.v1",
                "models": {
                    "candidate-v2": {"id": "candidate-v2", "role": "candidate"},
                    "champion-v1": {"id": "champion-v1", "role": "champion"},
                    "other-model": {"id": "other-model", "role": "historical"},
                },
                "aliases": {
                    "candidate": "candidate-v2",
                    "champion": champion_alias,
                    "rollback": "other-model",
                },
                "alias_history": [] if alias_history is None else alias_history,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return registry_path


def write_release_notes(root: Path) -> Path:
    release_notes = root / "RELEASE_NOTES.md"
    release_notes.write_text(
        "# Release Notes\n\n- Promotes candidate-v2 after passing governance gates.\n- Rollback target: champion-v1.\n",
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
                "required_artifacts": required_artifacts or promotion_decision_required_artifacts(),
                "release_required_artifacts": release_required_artifacts or promotion_release_required_artifacts(),
                "allowed_candidate_classes": ["base", "candidate", "champion", "frontier", "trace-only"],
                "allowed_champion_classes": ["base", "candidate", "champion", "frontier", "trace-only"],
                "limits": default_limits,
                "forbid_new_critical_rules": forbid_new_critical_rules or promotion_policy_required_forbidden_rules(),
                "forbid_regressed_rules": forbid_regressed_rules or promotion_policy_required_forbidden_rules(),
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


def promotion_alias_apply_args(registry_path: Path, decision_path: Path, receipt_path: Path) -> list[str]:
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


def promotion_rollback_receipt_args(registry_path: Path, receipt_path: Path, rollback_id: str = "champion-v1") -> list[str]:
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
        "candidate-v2",
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
    artifacts: dict[str, Path | None],
    out_path: Path,
    *,
    promotion_policy: Path | None = None,
) -> list[str]:
    args = [
        "promotion-decision",
        "--candidate-id",
        "candidate-v2",
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
    if promotion_policy is not None:
        args.extend(["--promotion-policy", str(promotion_policy)])
    return args


def failed_check_ids(decision: dict) -> set[str]:
    return {check["id"] for check in decision["checks"] if not check["passed"]}
