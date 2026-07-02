from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

from flightrecorder.cli import main


def run_cli(args):
    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        return main(args)


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
            self.assertEqual(receipt["promotion_decision_validation"]["passed"], True)
            self.assertEqual(receipt["alias_history_entry"]["promotion_decision_sha256"], receipt["promotion_decision"]["sha256"])
            self.assertEqual(receipt["alias_history_entry"]["updated_aliases"]["champion"], "candidate-v2")
            self.assertEqual(run_cli(["validate", "--promotion-alias-apply", str(receipt_path), "--strict"]), 0)

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


def write_governance_artifacts(root: Path, *, license_status: str = "known", compare_metrics=None) -> dict[str, Path | None]:
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
    if compare_metrics:
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
    model_card = root / "MODEL_CARD.md"
    model_card.write_text("# Model Card\n\nEvidence-backed candidate model.\n", encoding="utf-8")
    dataset_card = root / "DATASET_CARD.md"
    dataset_card.write_text("# Dataset Card\n\nRedacted held-out data.\n", encoding="utf-8")
    rollback_metadata = root / "rollback.json"
    rollback_metadata.write_text(json.dumps({"available": True, "rollback_id": "champion-v1"}), encoding="utf-8")
    license_review = root / "license_review.json"
    license_review.write_text(
        json.dumps({"accepted_terms": True, "license_status": license_status, "passed": True}),
        encoding="utf-8",
    )
    redaction_check = root / "redaction_check.json"
    redaction_check.write_text(json.dumps({"passed": True}), encoding="utf-8")
    safety_gate = root / "safety_gate.json"
    safety_gate.write_text(json.dumps({"passed": True}), encoding="utf-8")
    serving_report = root / "serving_report.json"
    serving_report.write_text(json.dumps({"passed": True}), encoding="utf-8")
    return {
        "evidence_bundle": evidence_bundle,
        "promotion_ledger_gate": promotion_ledger_gate,
        "compare_gate": compare_gate,
        "trainer_launch_check": trainer_launch_check,
        "model_card": model_card,
        "dataset_card": dataset_card,
        "rollback_metadata": rollback_metadata,
        "license_review": license_review,
        "redaction_check": redaction_check,
        "safety_gate": safety_gate,
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


def promotion_release_record_args(
    artifacts: dict[str, Path | None],
    cards_dir: Path,
    decision_path: Path,
    alias_receipt_path: Path,
    release_notes_path: Path,
    release_record_path: Path,
) -> list[str]:
    return [
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


def promotion_decision_args(artifacts: dict[str, Path | None], out_path: Path) -> list[str]:
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
        "model_card": "--model-card",
        "dataset_card": "--dataset-card",
        "rollback_metadata": "--rollback-metadata",
        "license_review": "--license-review",
        "redaction_check": "--redaction-check",
        "safety_gate": "--safety-gate",
        "serving_report": "--serving-report",
    }
    for role, flag in flag_by_role.items():
        path = artifacts.get(role)
        if path is not None:
            args.extend([flag, str(path)])
    return args


def failed_check_ids(decision: dict) -> set[str]:
    return {check["id"] for check in decision["checks"] if not check["passed"]}
