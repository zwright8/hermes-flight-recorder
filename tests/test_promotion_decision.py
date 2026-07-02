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
