import hashlib
import json
import shutil
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

from flightrecorder.agentic_loop_ledger import AgenticLoopLedgerError, build_agentic_loop_ledger
from flightrecorder.agentic_training_loop_plan import build_agentic_training_loop_plan, write_agentic_training_loop_plan
from flightrecorder.agentic_loop_governance import build_agentic_loop_governance_receipt
from flightrecorder.cli import main
from flightrecorder.eval_summary import build_eval_summary
from flightrecorder.external_eval_result import (
    build_external_eval_result,
    write_external_eval_result,
)
from flightrecorder.schema_registry import list_schema_records
from flightrecorder.source_contract import inspect_artifact_source
from flightrecorder.validation import validate_promotion_archive, validate_promotion_cards
from tests.agentic_loop_fixtures import copy_valid_loop_artifacts


def run_cli(args):
    with redirect_stdout(StringIO()):
        return main(args)


ROOT = Path(__file__).resolve().parents[1]


def _write_valid_training_export(root: Path) -> Path:
    runs = root / "runs"
    assert run_cli(
        [
            "run",
            "--scenario",
            str(ROOT / "scenarios" / "email_reply_completion_good.json"),
            "--out",
            str(runs / "email_reply_completion_good"),
        ]
    ) == 0
    training_export = root / "training_export"
    assert run_cli(["export-rl", "--runs", str(runs), "--out", str(training_export)]) == 0
    return training_export


class AgenticLoopLedgerTests(unittest.TestCase):
    def test_ledger_rejects_schema_invalid_loop_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan_path = self.write_loop_plan(root / "plan.json", "loop-invalid", {})
            payload = json.loads(plan_path.read_text(encoding="utf-8"))
            payload.pop("notes")
            plan_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            with self.assertRaises(AgenticLoopLedgerError):
                build_agentic_loop_ledger([plan_path])

    def test_promotion_directory_sources_require_valid_contents(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shutil.copytree(
                ROOT / "examples" / "agentic_training",
                root / "agentic_training",
            )
            cards = root / "agentic_training" / "promotion_governance" / "promotion_cards"
            archive = root / "agentic_training" / "promotion_governance" / "promotion_archive"

            self.assertTrue(inspect_artifact_source(cards, "promotion_cards")["ready"])
            self.assertTrue(inspect_artifact_source(archive, "promotion_archive")["ready"])

            cards_manifest_path = cards / "promotion_cards.json"
            archive_manifest_path = archive / "promotion_archive.json"
            original_cards_manifest = cards_manifest_path.read_text(encoding="utf-8")
            original_archive_manifest = archive_manifest_path.read_text(encoding="utf-8")
            blocked_cards = json.loads(original_cards_manifest)
            blocked_cards["checks"][0]["passed"] = False
            blocked_cards["passed"] = False
            blocked_cards["readiness"] = "blocked"
            blocked_cards["recommendation"] = "regenerate_or_block_promotion"
            blocked_cards["failed_check_count"] = 1
            blocked_cards["metrics"]["failed_check_count"] = 1
            cards_manifest_path.write_text(json.dumps(blocked_cards, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            blocked_archive = json.loads(original_archive_manifest)
            blocked_archive["passed"] = False
            blocked_archive["self_contained"] = False
            blocked_archive["missing"] = [{"index": 0, "reason": "source unavailable", "role": "source_artifact"}]
            blocked_archive["metrics"]["missing_count"] = 1
            blocked_archive["metrics"]["missing_role_counts"] = [{"count": 1, "id": "source_artifact"}]
            archive_manifest_path.write_text(json.dumps(blocked_archive, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            self.assertEqual(validate_promotion_cards(cards).errors, [])
            self.assertEqual(validate_promotion_archive(archive).errors, [])
            self.assertFalse(inspect_artifact_source(cards, "promotion_cards")["semantic_valid"])
            self.assertFalse(inspect_artifact_source(archive, "promotion_archive")["semantic_valid"])

            cards_manifest_path.write_text(original_cards_manifest, encoding="utf-8")
            archive_manifest_path.write_text(original_archive_manifest, encoding="utf-8")
            (cards / "MODEL_CARD.md").unlink()
            archived_ledger = archive / "artifacts" / "promotion_ledger.json"
            archived_ledger.write_text(archived_ledger.read_text(encoding="utf-8") + "\n", encoding="utf-8")

            cards_source = inspect_artifact_source(cards, "promotion_cards")
            archive_source = inspect_artifact_source(archive, "promotion_archive")
            self.assertTrue(cards_source["schema_valid"])
            self.assertTrue(archive_source["schema_valid"])
            self.assertFalse(cards_source["semantic_valid"])
            self.assertFalse(archive_source["semantic_valid"])
            self.assertFalse(cards_source["ready"])
            self.assertFalse(archive_source["ready"])

    def test_promotion_directory_sources_reject_empty_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cards = root / "promotion_cards"
            archive = root / "promotion_archive"
            cards.mkdir()
            archive.mkdir()

            self.assertFalse(inspect_artifact_source(cards, "promotion_cards")["ready"])
            self.assertFalse(inspect_artifact_source(archive, "promotion_archive")["ready"])

    def test_training_export_source_rejects_empty_episode_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            training_export = _write_valid_training_export(Path(tmp))
            self.assertTrue(inspect_artifact_source(training_export, "training_export")["ready"])

            (training_export / "episodes.jsonl").write_text("", encoding="utf-8")

            source = inspect_artifact_source(training_export, "training_export")
            self.assertTrue(source["schema_valid"])
            self.assertFalse(source["semantic_valid"])
            self.assertFalse(source["ready"])

    def test_training_export_source_rejects_tampered_episode_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            training_export = _write_valid_training_export(Path(tmp))
            episodes_path = training_export / "episodes.jsonl"
            self.assertTrue(inspect_artifact_source(training_export, "training_export")["ready"])

            episodes_path.write_text(episodes_path.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")

            source = inspect_artifact_source(training_export, "training_export")
            self.assertTrue(source["schema_valid"])
            self.assertFalse(source["semantic_valid"])
            self.assertFalse(source["ready"])

    def test_committed_example_loop_ledger_replays_loop_plan(self):
        ledger_path = ROOT / "examples" / "agentic_training" / "loop_ledger.json"
        loop_plan_path = ROOT / "examples" / "agentic_training" / "loop_plan.json"
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))

        iteration = ledger["iterations"][0]
        self.assertEqual(iteration["path"], "loop_plan.json")
        self.assertEqual(iteration["size_bytes"], loop_plan_path.stat().st_size)
        self.assertEqual(iteration["sha256"], hashlib.sha256(loop_plan_path.read_bytes()).hexdigest())
        self.assertTrue(ledger["passed"])
        self.assertEqual(ledger["decision"]["recommended_governance_action"], "approve")
        self.assertTrue(ledger["readiness_digest"]["ready_for_governance_review"])
        self.assertTrue(ledger["readiness_digest"]["cloud_training_receipts_fail_closed"])
        self.assertEqual(ledger["readiness_digest"]["cloud_training_cost_incurred_usd"], 0)
        groups = {row["group"]: row["count"] for row in iteration["artifact_group_counts"]}
        self.assertEqual(groups["review"], 6)
        self.assertEqual(groups["datasets"], 3)
        self.assertEqual(groups["improvement"], 3)
        self.assertEqual(groups["next_iteration"], 1)
        self.assertEqual(groups["rollouts"], 3)
        self.assertEqual(groups["training"], 6)
        self.assertEqual(groups["eval"], 5)
        self.assertEqual(groups["evidence"], 1)
        self.assertEqual(groups["serving"], 1)
        self.assertEqual(groups["governance"], 4)
        role_counts = {row["role"]: row["count"] for row in iteration["artifact_role_counts"]}
        self.assertEqual(role_counts["promotion_decision"], 1)
        self.assertEqual(role_counts["promotion_ledger"], 1)
        self.assertEqual(role_counts["promotion_cards"], 1)
        self.assertEqual(role_counts["promotion_alias_apply"], 1)
        self.assertEqual(role_counts["promotion_rollback_receipt"], 1)
        self.assertEqual(role_counts["promotion_release_record"], 1)
        self.assertEqual(role_counts["promotion_archive"], 1)
        self.assertEqual(ledger["readiness_digest"]["missing_phase_input_count"], 0)
        self.assertEqual(ledger["readiness_digest"]["missing_artifact_group_count"], 0)
        self.assertTrue(ledger["readiness_digest"]["cloud_training_lineage_bound"])
        self.assertEqual(ledger["readiness_digest"]["cloud_training_missing_link_count"], 0)
        self.assertEqual(ledger["readiness_digest"]["external_eval_adapter_count"], 1)
        self.assertEqual(ledger["readiness_digest"]["external_eval_ready_adapter_count"], 1)
        self.assertEqual(ledger["readiness_digest"]["external_eval_receipt_count"], 1)
        self.assertTrue(ledger["readiness_digest"]["external_eval_receipts_fail_closed"])
        self.assertTrue(ledger["readiness_digest"]["external_eval_receipts_passed"])
        self.assertTrue(ledger["readiness_digest"]["rollback_receipt_present"])
        actions = {row["action"]: row for row in ledger["decision"]["governance_actions"]}
        self.assertTrue(actions["rollback"]["available"])
        self.assertEqual(actions["rollback"]["blocked_reasons"], [])
        self.assertFalse(ledger["execution_boundary"]["cloud_jobs_started"])
        self.assertFalse(ledger["execution_boundary"]["paid_model_grader_calls_started"])
        self.assertFalse(ledger["execution_boundary"]["weights_updated_by_flight_recorder"])
        self.assertEqual(run_cli(["schemas", "--check", str(ledger_path)]), 0)
        self.assertEqual(run_cli(["validate", "--agentic-loop-ledger", str(ledger_path), "--strict"]), 0)

    def test_committed_example_governance_receipt_replays_ledger(self):
        receipt_path = ROOT / "examples" / "agentic_training" / "loop_governance_receipt.json"
        ledger_path = ROOT / "examples" / "agentic_training" / "loop_ledger.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))

        source_ledger = receipt["source_ledger"]
        self.assertEqual(source_ledger["path"], "loop_ledger.json")
        self.assertEqual(source_ledger["size_bytes"], ledger_path.stat().st_size)
        self.assertEqual(source_ledger["sha256"], hashlib.sha256(ledger_path.read_bytes()).hexdigest())
        self.assertTrue(receipt["passed"])
        self.assertEqual(receipt["requested_action"]["action"], "approve")
        self.assertTrue(receipt["requested_action"]["available"])
        self.assertEqual(receipt["decision"]["recommendation"], "record_approval_for_promotion_review")
        self.assertFalse(receipt["execution_boundary"]["cloud_jobs_started"])
        self.assertFalse(receipt["execution_boundary"]["promotion_alias_moved"])
        self.assertFalse(receipt["execution_boundary"]["rollback_applied"])
        self.assertFalse(receipt["execution_boundary"]["weights_updated_by_flight_recorder"])
        self.assertEqual(run_cli(["schemas", "--check", str(receipt_path)]), 0)
        self.assertEqual(run_cli(["validate", "--agentic-loop-governance-receipt", str(receipt_path), "--strict"]), 0)

    def test_agentic_loop_ledger_tracks_blocked_and_ready_iterations(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            blocked_plan = self.write_loop_plan(root / "plans" / "blocked.json", "loop-001", {})
            ready_artifacts = self.write_ready_artifacts(root / "ready")
            ready_plan = self.write_loop_plan(root / "ready" / "plan.json", "loop-002", ready_artifacts)
            ledger = root / "ledger.json"

            code = run_cli(["agentic-loop", "ledger", "--plan", str(blocked_plan), "--plan", str(ready_plan), "--out", str(ledger)])

            self.assertEqual(code, 0)
            payload = json.loads(ledger.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema_version"], "hfr.agentic_loop_ledger.v1")
            self.assertTrue(payload["passed"])
            self.assertEqual(payload["iteration_count"], 2)
            self.assertEqual(payload["metrics"]["ready_iteration_count"], 1)
            self.assertEqual(payload["metrics"]["blocked_iteration_count"], 1)
            self.assertEqual(payload["metrics"]["latest_iteration_id"], "loop-002")
            self.assertEqual(payload["decision"]["recommendation"], "ready_for_governance_review")
            self.assertEqual(payload["decision"]["recommended_governance_action"], "approve")
            actions = {row["action"]: row for row in payload["decision"]["governance_actions"]}
            self.assertEqual(set(actions), {"approve", "reject", "rollback", "request_another_iteration"})
            self.assertTrue(actions["approve"]["available"])
            self.assertTrue(actions["reject"]["available"])
            self.assertFalse(actions["rollback"]["available"])
            self.assertTrue(actions["request_another_iteration"]["available"])
            digest = payload["readiness_digest"]
            self.assertEqual(digest["latest_iteration_id"], "loop-002")
            self.assertTrue(digest["ready_for_governance_review"])
            self.assertEqual(digest["recommended_governance_action"], "approve")
            self.assertTrue(digest["promotion_ledger_present"])
            self.assertEqual(digest["missing_phase_input_count"], 0)
            self.assertEqual(digest["missing_artifact_group_count"], 0)
            self.assertFalse(digest["side_effects_started"])
            self.assertFalse(payload["execution_boundary"]["cloud_jobs_started"])
            ready_record = payload["iterations"][1]
            groups = {row["group"]: row["count"] for row in ready_record["artifact_group_counts"]}
            self.assertGreater(groups["rollouts"], 0)
            self.assertGreater(groups["review"], 0)
            self.assertGreater(groups["cloud_training"], 0)
            self.assertGreater(groups["training"], 0)
            self.assertGreater(groups["eval"], 0)
            role_names = {row["role"] for row in ready_record["artifact_role_counts"]}
            self.assertIn("cloud_training_launch_receipt", role_names)
            self.assertIn("model_grader_disagreement_queue", role_names)
            self.assertIn("model_grader_override_receipt", role_names)
            self.assertTrue(ready_record["cloud_training"]["status_receipt_present"])
            self.assertTrue(ready_record["cloud_training_receipt_state"]["fail_closed"])
            self.assertEqual(ready_record["cloud_training_receipt_state"]["launch_mode"], "dry_run")
            self.assertEqual(ready_record["cloud_training_receipt_state"]["status_provider_status"], "not_started")
            self.assertTrue(ready_record["cloud_training_lineage"]["passed"])
            self.assertEqual(ready_record["cloud_training_lineage"]["provider"]["pipeline_provider_id"], "modal")
            self.assertFalse(ready_record["cloud_training"]["provider_api_calls_started"])
            self.assertIn("external_eval_receipt", ready_record["evals"]["roles_present"])
            self.assertTrue(ready_record["external_eval_receipt_state"]["receipts_passed"])
            self.assertTrue(ready_record["external_eval_receipt_state"]["fail_closed"])
            self.assertEqual(ready_record["external_eval_receipt_state"]["launch_mode"], "dry_run")
            self.assertTrue(ready_record["governance"]["promotion_decision_present"])
            self.assertFalse(ready_record["governance"]["weights_updated_by_flight_recorder"])
            self.assertTrue(digest["cloud_training_lineage_bound"])
            self.assertTrue(digest["cloud_training_receipts_fail_closed"])
            self.assertFalse(digest["cloud_training_live_launch_requested"])
            self.assertEqual(digest["cloud_training_cost_incurred_usd"], 0)
            self.assertEqual(digest["cloud_training_launch_mode"], "dry_run")
            self.assertEqual(digest["cloud_training_status_provider_status"], "not_started")
            self.assertEqual(digest["cloud_training_provider_id"], "modal")
            self.assertEqual(digest["cloud_training_missing_link_count"], 0)
            self.assertEqual(digest["cloud_training_mismatched_link_count"], 0)
            self.assertEqual(digest["cloud_training_ambiguous_link_count"], 0)
            self.assertEqual(digest["cloud_training_duplicate_role_count"], 0)
            self.assertTrue(digest["external_eval_receipts_passed"])
            self.assertTrue(digest["external_eval_receipts_fail_closed"])
            self.assertFalse(digest["external_eval_live_benchmark_requested"])
            self.assertFalse(digest["external_eval_live_benchmarks_started"])
            self.assertFalse(digest["external_eval_provider_api_calls_started"])
            self.assertFalse(digest["external_eval_model_downloads_started"])
            self.assertFalse(digest["external_eval_credential_values_recorded"])
            self.assertEqual(digest["external_eval_cost_incurred_usd"], 0)
            self.assertEqual(digest["external_eval_launch_mode"], "dry_run")
            self.assertEqual(digest["external_eval_receipt_count"], 1)
            self.assertEqual(digest["external_eval_adapter_count"], 1)
            self.assertEqual(digest["external_eval_ready_adapter_count"], 1)
            self.assertEqual(run_cli(["validate", "--agentic-loop-ledger", str(ledger), "--strict"]), 0)
            self.assertEqual(run_cli(["schemas", "--check", str(ledger)]), 0)

    def test_governance_actions_recommend_another_iteration_when_blocked(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            root = Path(tmp)
            blocked_plan = self.write_loop_plan(root / "plans" / "blocked.json", "loop-001", {})
            ledger = root / "ledger.json"

            self.assertEqual(run_cli(["agentic-loop", "ledger", "--plan", str(blocked_plan), "--out", str(ledger)]), 0)

            payload = json.loads(ledger.read_text(encoding="utf-8"))
            self.assertEqual(payload["decision"]["readiness"], "blocked")
            self.assertEqual(payload["decision"]["recommended_governance_action"], "request_another_iteration")
            actions = {row["action"]: row for row in payload["decision"]["governance_actions"]}
            self.assertFalse(actions["approve"]["available"])
            self.assertIn("latest_iteration_not_ready_for_governance_review", actions["approve"]["blocked_reasons"])
            self.assertIn("missing_promotion_decision", actions["approve"]["blocked_reasons"])
            self.assertIn("missing_promotion_ledger", actions["approve"]["blocked_reasons"])
            self.assertTrue(actions["reject"]["available"])
            self.assertFalse(actions["rollback"]["available"])
            self.assertTrue(actions["request_another_iteration"]["available"])
            self.assertEqual(payload["readiness_digest"]["recommended_governance_action"], "request_another_iteration")
            self.assertEqual(run_cli(["validate", "--agentic-loop-ledger", str(ledger), "--strict"]), 0)
            self.assertEqual(run_cli(["schemas", "--check", str(ledger)]), 0)

    def test_approval_requires_promotion_ledger_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ready_artifacts = self.write_ready_artifacts(root / "ready")
            ready_artifacts.pop("promotion_ledger")
            plan = self.write_loop_plan(root / "ready" / "plan.json", "loop-001", ready_artifacts)
            ledger = root / "ledger.json"
            summary = root / "summary.json"

            self.assertEqual(run_cli(["agentic-loop", "ledger", "--plan", str(plan), "--out", str(ledger)]), 0)

            payload = json.loads(ledger.read_text(encoding="utf-8"))
            actions = {row["action"]: row for row in payload["decision"]["governance_actions"]}
            self.assertFalse(actions["approve"]["available"])
            self.assertIn("missing_promotion_ledger", actions["approve"]["blocked_reasons"])
            self.assertFalse(payload["readiness_digest"]["promotion_ledger_present"])
            self.assertTrue(payload["readiness_digest"]["promotion_decision_present"])
            self.assertEqual(payload["decision"]["recommended_governance_action"], "request_another_iteration")
            self.assertEqual(run_cli(["validate", "--agentic-loop-ledger", str(ledger), "--strict"]), 0)
            self.assertEqual(run_cli(["schemas", "--check", str(ledger)]), 0)

            payload["readiness_digest"]["promotion_ledger_present"] = True
            ledger.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            code = run_cli(["validate", "--agentic-loop-ledger", str(ledger), "--out", str(summary)])

            self.assertEqual(code, 1)
            errors = "\n".join(error for target in json.loads(summary.read_text(encoding="utf-8"))["targets"] for error in target["errors"])
            self.assertIn("readiness_digest.promotion_ledger_present must match the latest iteration", errors)

    def test_governance_counts_exclude_invalid_or_missing_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ready_artifacts = self.write_ready_artifacts(root / "ready")
            ready_artifacts["promotion_decision"] = [
                self.write_json(root / "ready" / "invalid_promotion_decision.json", "hfr.promotion_decision.v1")
            ]
            ready_artifacts["promotion_ledger"] = [root / "ready" / "missing_promotion_ledger.json"]
            ready_artifacts["promotion_rollback_receipt"] = [
                self.write_json(
                    root / "ready" / "invalid_promotion_rollback_receipt.json",
                    "hfr.promotion_rollback_receipt.v1",
                )
            ]
            plan = self.write_loop_plan(root / "ready" / "plan.json", "loop-001", ready_artifacts)
            ledger = build_agentic_loop_ledger([plan])

            iteration = ledger["iterations"][0]
            role_counts = {row["role"]: row["count"] for row in iteration["artifact_role_counts"]}
            self.assertNotIn("promotion_decision", role_counts)
            self.assertNotIn("promotion_ledger", role_counts)
            self.assertNotIn("promotion_rollback_receipt", role_counts)
            self.assertFalse(iteration["governance"]["promotion_decision_present"])
            self.assertFalse(iteration["governance"]["promotion_ledger_present"])
            self.assertFalse(iteration["governance"]["rollback_receipt_present"])
            actions = {row["action"]: row for row in ledger["decision"]["governance_actions"]}
            self.assertIn("missing_promotion_decision", actions["approve"]["blocked_reasons"])
            self.assertIn("missing_promotion_ledger", actions["approve"]["blocked_reasons"])
            self.assertIn("missing_rollback_receipt", actions["rollback"]["blocked_reasons"])
            self.assertEqual(ledger["decision"]["recommended_governance_action"], "request_another_iteration")

    def test_governance_actions_mark_rollback_available_when_receipt_is_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ready_artifacts = copy_valid_loop_artifacts(root / "ready")
            plan = self.write_loop_plan(root / "ready" / "plan.json", "loop-001", ready_artifacts)
            ledger = root / "ledger.json"

            self.assertEqual(run_cli(["agentic-loop", "ledger", "--plan", str(plan), "--out", str(ledger)]), 0)

            payload = json.loads(ledger.read_text(encoding="utf-8"))
            actions = {row["action"]: row for row in payload["decision"]["governance_actions"]}
            self.assertTrue(actions["approve"]["available"])
            self.assertTrue(actions["rollback"]["available"])
            self.assertEqual(actions["rollback"]["blocked_reasons"], [])
            self.assertEqual(payload["decision"]["recommended_governance_action"], "approve")
            self.assertTrue(payload["readiness_digest"]["rollback_receipt_present"])
            self.assertEqual(run_cli(["validate", "--agentic-loop-ledger", str(ledger), "--strict"]), 0)

    def test_validate_rejects_stale_agentic_loop_ledger_source_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = self.write_loop_plan(root / "plans" / "loop.json", "loop-001", {})
            ledger = root / "ledger.json"
            summary = root / "summary.json"
            self.assertEqual(run_cli(["agentic-loop", "ledger", "--plan", str(plan), "--out", str(ledger)]), 0)
            self.assertEqual(run_cli(["validate", "--agentic-loop-ledger", str(ledger), "--strict"]), 0)
            payload = json.loads(plan.read_text(encoding="utf-8"))
            payload["objective"] = "tampered after ledger"
            plan.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            code = run_cli(["validate", "--agentic-loop-ledger", str(ledger), "--out", str(summary)])

            self.assertEqual(code, 1)
            errors = "\n".join(error for target in json.loads(summary.read_text(encoding="utf-8"))["targets"] for error in target["errors"])
            self.assertIn("sha256 does not match the current file", errors)

    def test_validate_rejects_parent_symlink_agentic_loop_ledger_source_plan(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            root = Path(tmp)
            source = root / "source"
            plan = self.write_loop_plan(source / "plans" / "loop.json", "loop-001", {})
            ledger = root / "ledger.json"
            summary = root / "summary.json"
            self.assertEqual(run_cli(["agentic-loop", "ledger", "--plan", str(plan), "--out", str(ledger)]), 0)
            linked_source = root / "linked_source"
            linked_source.symlink_to(source, target_is_directory=True)
            payload = json.loads(ledger.read_text(encoding="utf-8"))
            payload["iterations"][0]["path"] = str(Path("linked_source") / "plans" / "loop.json")
            ledger.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            code = run_cli(["validate", "--agentic-loop-ledger", str(ledger), "--out", str(summary)])

            self.assertEqual(code, 1)
            errors = "\n".join(error for target in json.loads(summary.read_text(encoding="utf-8"))["targets"] for error in target["errors"])
            self.assertIn(
                "agentic_loop_ledger.iterations[0].path must resolve to a regular non-symlink source loop plan.",
                errors,
            )

    def test_agentic_loop_ledger_blocks_unreplayable_source_plan_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = self.write_loop_plan(root / "plans" / "loop.json", "loop-001", {})
            report_dir = root / "reports"
            report_dir.mkdir()
            ledger = report_dir / "agentic_loop_ledger.json"
            stderr = StringIO()

            with redirect_stderr(stderr), self.assertRaises(SystemExit) as exc:
                run_cli(["agentic-loop", "ledger", "--plan", str(plan), "--out", str(ledger), "--preserve-paths"])

            self.assertEqual(exc.exception.code, 2)
            self.assertFalse(ledger.exists())
            error_text = stderr.getvalue()
            self.assertIn("Loop plan source must be replayable from the ledger output directory", error_text)
            self.assertNotIn(str(root), error_text)

    def test_agentic_loop_ledger_blocks_parent_symlink_source_plan(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            root = Path(tmp)
            source = root / "source"
            plan = self.write_loop_plan(source / "plans" / "loop.json", "loop-001", {})
            linked_source = root / "linked_source"
            linked_source.symlink_to(source, target_is_directory=True)
            ledger = root / "ledger.json"
            stderr = StringIO()

            with redirect_stderr(stderr), self.assertRaises(SystemExit) as exc:
                run_cli(["agentic-loop", "ledger", "--plan", str(linked_source / plan.relative_to(source)), "--out", str(ledger)])

            self.assertEqual(exc.exception.code, 2)
            self.assertFalse(ledger.exists())
            self.assertIn("Loop plan must resolve to a regular non-symlink file", stderr.getvalue())

    def test_validate_replays_source_plan_eval_summary_readiness(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = self.write_ready_artifacts(root / "ready")
            eval_summary_path = artifacts["eval_summary"][0]
            eval_summary = json.loads(eval_summary_path.read_text(encoding="utf-8"))
            del eval_summary["arms"]
            eval_summary_path.write_text(json.dumps(eval_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            plan = self.write_loop_plan(root / "ready" / "plan.json", "loop-001", artifacts)
            ledger = root / "ledger.json"
            summary = root / "summary.json"
            loop_plan = json.loads(plan.read_text(encoding="utf-8"))

            heldout_check = next(check for check in loop_plan["checks"] if check["id"] == "heldout_eval_is_fail_closed")
            self.assertFalse(heldout_check["passed"])
            self.assertFalse(heldout_check["actual"]["eval_summary_valid"])
            self.assertEqual(run_cli(["agentic-loop", "ledger", "--plan", str(plan), "--out", str(ledger)]), 0)
            code = run_cli(["validate", "--agentic-loop-ledger", str(ledger), "--out", str(summary)])

            self.assertEqual(code, 0)
            payload = json.loads(summary.read_text(encoding="utf-8"))
            self.assertTrue(payload["passed"], payload)

    def test_validate_replays_source_plan_promotion_decision_readiness(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = self.write_ready_artifacts(root / "ready")
            decision_path = artifacts["promotion_decision"][0]
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            del decision["checks"]
            decision_path.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            plan = self.write_loop_plan(root / "ready" / "plan.json", "loop-001", artifacts)
            ledger = root / "ledger.json"
            summary = root / "summary.json"

            self.assertEqual(run_cli(["agentic-loop", "ledger", "--plan", str(plan), "--out", str(ledger)]), 0)
            code = run_cli(["validate", "--agentic-loop-ledger", str(ledger), "--out", str(summary)])

            self.assertEqual(code, 0)
            validation = json.loads(summary.read_text(encoding="utf-8"))
            self.assertTrue(validation["passed"], validation)
            ledger_payload = json.loads(ledger.read_text(encoding="utf-8"))
            self.assertEqual(ledger_payload["decision"]["readiness"], "blocked")

    def test_validate_rejects_tampered_readiness_digest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ready_artifacts = self.write_ready_artifacts(root / "ready")
            plan = self.write_loop_plan(root / "ready" / "plan.json", "loop-001", ready_artifacts)
            ledger = root / "ledger.json"
            summary = root / "summary.json"
            self.assertEqual(run_cli(["agentic-loop", "ledger", "--plan", str(plan), "--out", str(ledger)]), 0)
            payload = json.loads(ledger.read_text(encoding="utf-8"))
            payload["readiness_digest"]["missing_phase_input_count"] = 9
            ledger.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            code = run_cli(["validate", "--agentic-loop-ledger", str(ledger), "--out", str(summary)])

            self.assertEqual(code, 1)
            errors = "\n".join(error for target in json.loads(summary.read_text(encoding="utf-8"))["targets"] for error in target["errors"])
            self.assertIn("readiness_digest.missing_phase_input_count must match missing_phase_inputs", errors)

    def test_validate_rejects_tampered_cloud_training_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ready_artifacts = self.write_ready_artifacts(root / "ready")
            plan = self.write_loop_plan(root / "ready" / "plan.json", "loop-001", ready_artifacts)
            ledger = root / "ledger.json"
            summary = root / "summary.json"
            self.assertEqual(run_cli(["agentic-loop", "ledger", "--plan", str(plan), "--out", str(ledger)]), 0)
            payload = json.loads(ledger.read_text(encoding="utf-8"))
            payload["iterations"][0]["cloud_training"]["artifact_count"] = 0
            payload["iterations"][0]["cloud_training"]["cloud_jobs_started"] = True
            payload["iterations"][0]["cloud_training_receipt_state"]["fail_closed"] = False
            payload["iterations"][0]["cloud_training_lineage"]["matched_link_count"] = 0
            payload["iterations"][0]["external_eval_receipt_state"]["fail_closed"] = False
            ledger.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            code = run_cli(["validate", "--agentic-loop-ledger", str(ledger), "--out", str(summary)])

            self.assertEqual(code, 1)
            errors = "\n".join(error for target in json.loads(summary.read_text(encoding="utf-8"))["targets"] for error in target["errors"])
            self.assertIn("cloud_training.artifact_count must match ledger cloud training role counts", errors)
            self.assertIn("cloud_training.cloud_jobs_started must match ledger cloud training role counts", errors)
            self.assertIn("cloud_training_receipt_state.fail_closed must match source loop plan cloud training receipt artifacts", errors)
            self.assertIn("cloud_training_lineage must match the source loop plan cloud_training_lineage", errors)
            self.assertIn(
                "external_eval_receipt_state.fail_closed must match source loop plan external eval receipt artifacts",
                errors,
            )

    def test_external_eval_side_effects_flow_into_ledger_digest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ready_artifacts = self.write_ready_artifacts(root / "ready")
            receipt = ready_artifacts["external_eval_receipt"][0]
            receipt_payload = json.loads(receipt.read_text(encoding="utf-8"))
            receipt_payload["passed"] = False
            receipt_payload["readiness"] = "blocked"
            receipt_payload["recommendation"] = "keep_external_eval_claims_disabled"
            receipt_payload["launch"]["mode"] = "live"
            receipt_payload["launch"]["live_benchmarks_started"] = True
            receipt_payload["launch"]["provider_api_called"] = True
            receipt_payload["launch"]["model_downloads_started"] = True
            receipt_payload["launch"]["cost_incurred_usd"] = 5
            receipt.write_text(json.dumps(receipt_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            plan = self.write_loop_plan(root / "ready" / "plan.json", "loop-001", ready_artifacts)
            ledger = root / "ledger.json"
            summary = root / "summary.json"
            self.assertEqual(run_cli(["agentic-loop", "ledger", "--plan", str(plan), "--out", str(ledger)]), 0)
            payload = json.loads(ledger.read_text(encoding="utf-8"))

            self.assertFalse(payload["iterations"][0]["external_eval_receipt_state"]["fail_closed"])
            self.assertFalse(payload["iterations"][0]["external_eval_receipt_state"]["receipts_passed"])
            self.assertTrue(payload["iterations"][0]["external_eval_receipt_state"]["live_benchmark_requested"])
            self.assertTrue(payload["iterations"][0]["external_eval_receipt_state"]["provider_api_calls_started"])
            self.assertEqual(payload["readiness_digest"]["external_eval_cost_incurred_usd"], 5)
            self.assertFalse(payload["readiness_digest"]["external_eval_receipts_fail_closed"])
            self.assertTrue(payload["readiness_digest"]["external_eval_live_benchmark_requested"])

            payload["readiness_digest"]["external_eval_receipts_fail_closed"] = True
            payload["readiness_digest"]["external_eval_live_benchmark_requested"] = False
            payload["readiness_digest"]["external_eval_cost_incurred_usd"] = 0
            ledger.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            code = run_cli(["validate", "--agentic-loop-ledger", str(ledger), "--out", str(summary)])

            self.assertEqual(code, 1)
            errors = "\n".join(error for target in json.loads(summary.read_text(encoding="utf-8"))["targets"] for error in target["errors"])
            self.assertIn(
                "readiness_digest.external_eval_receipts_fail_closed must match the latest iteration",
                errors,
            )
            self.assertIn(
                "readiness_digest.external_eval_live_benchmark_requested must match the latest iteration",
                errors,
            )

    def test_forged_external_eval_receipt_keeps_ledger_state_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ready_artifacts = self.write_ready_artifacts(root / "ready")
            receipt_path = ready_artifacts["external_eval_receipt"][0]
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            receipt["source_plan"]["sha256"] = "0" * 64
            receipt["passed"] = True
            receipt["readiness"] = "dry_run_recorded"
            receipt["recommendation"] = "archive_external_eval_dry_run"
            receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            plan = self.write_loop_plan(root / "ready" / "plan.json", "loop-001", ready_artifacts)
            ledger = root / "ledger.json"
            summary = root / "summary.json"

            self.assertEqual(run_cli(["agentic-loop", "ledger", "--plan", str(plan), "--out", str(ledger)]), 0)
            payload = json.loads(ledger.read_text(encoding="utf-8"))
            self.assertFalse(payload["iterations"][0]["external_eval_receipt_state"]["receipts_passed"])
            self.assertEqual(payload["iterations"][0]["external_eval_receipt_state"]["receipt_passed_count"], 0)
            self.assertFalse(payload["readiness_digest"]["external_eval_receipts_passed"])
            code = run_cli(["validate", "--agentic-loop-ledger", str(ledger), "--out", str(summary), "--strict"])
            receipt_code = run_cli(["validate", "--external-eval-receipt", str(receipt_path), "--strict"])

            self.assertEqual(code, 0)
            self.assertEqual(receipt_code, 1)

    def test_duplicate_receipt_side_effects_flow_into_ledger_digest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ready_artifacts = self.write_ready_artifacts(root / "ready")
            duplicate_receipt = root / "ready" / "cloud_training_launch_receipt_duplicate.json"
            duplicate_payload = json.loads(ready_artifacts["cloud_training_launch_receipt"][0].read_text(encoding="utf-8"))
            duplicate_payload["execution_boundary"]["provider_api_called"] = True
            duplicate_payload["execution_boundary"]["cloud_job_started"] = True
            duplicate_payload["execution_boundary"]["live_requested"] = True
            duplicate_payload["execution_boundary"]["cloud_cost_incurred_usd"] = 4
            duplicate_receipt.write_text(json.dumps(duplicate_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            ready_artifacts["cloud_training_launch_receipt"].append(duplicate_receipt)
            plan = self.write_loop_plan(root / "ready" / "plan.json", "loop-001", ready_artifacts)
            ledger = root / "ledger.json"
            summary = root / "summary.json"
            self.assertEqual(run_cli(["agentic-loop", "ledger", "--plan", str(plan), "--out", str(ledger)]), 0)
            payload = json.loads(ledger.read_text(encoding="utf-8"))
            self.assertFalse(payload["iterations"][0]["cloud_training_receipt_state"]["fail_closed"])
            self.assertTrue(payload["iterations"][0]["cloud_training_receipt_state"]["provider_api_calls_started"])
            self.assertTrue(payload["iterations"][0]["cloud_training_receipt_state"]["live_launch_requested"])
            self.assertEqual(payload["readiness_digest"]["cloud_training_cost_incurred_usd"], 4)
            original_payload = json.loads(json.dumps(payload))
            payload["readiness_digest"]["cloud_training_receipts_fail_closed"] = True
            payload["readiness_digest"]["cloud_training_cost_incurred_usd"] = 0
            payload["readiness_digest"]["cloud_training_live_launch_requested"] = False
            ledger.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            code = run_cli(["validate", "--agentic-loop-ledger", str(ledger), "--out", str(summary)])

            self.assertEqual(code, 1)
            errors = "\n".join(error for target in json.loads(summary.read_text(encoding="utf-8"))["targets"] for error in target["errors"])
            self.assertIn(
                "readiness_digest.cloud_training_receipts_fail_closed must match the latest iteration",
                errors,
            )
            self.assertIn("readiness_digest.cloud_training_cost_incurred_usd must match the latest iteration", errors)

            original_payload["iterations"][0]["cloud_training_receipt_state"]["fail_closed"] = True
            original_payload["iterations"][0]["cloud_training_receipt_state"]["provider_api_calls_started"] = False
            original_payload["iterations"][0]["cloud_training_receipt_state"]["cloud_jobs_started"] = False
            original_payload["iterations"][0]["cloud_training_receipt_state"]["cost_incurred_usd"] = 0
            original_payload["iterations"][0]["cloud_training_receipt_state"]["live_launch_requested"] = False
            original_payload["readiness_digest"]["cloud_training_receipts_fail_closed"] = True
            original_payload["readiness_digest"]["cloud_training_cost_incurred_usd"] = 0
            original_payload["readiness_digest"]["cloud_training_live_launch_requested"] = False
            ledger.write_text(json.dumps(original_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            code = run_cli(["validate", "--agentic-loop-ledger", str(ledger), "--out", str(summary)])

            self.assertEqual(code, 1)
            errors = "\n".join(error for target in json.loads(summary.read_text(encoding="utf-8"))["targets"] for error in target["errors"])
            self.assertIn("cloud_training_receipt_state.fail_closed must match source loop plan cloud training receipt artifacts", errors)
            self.assertIn("cloud_training_receipt_state.provider_api_calls_started must match source loop plan cloud training receipt artifacts", errors)
            self.assertIn("cloud_training_receipt_state.cost_incurred_usd must match source loop plan cloud training receipt artifacts", errors)
            self.assertIn("cloud_training_receipt_state.live_launch_requested must match source loop plan cloud training receipt artifacts", errors)

    def test_forged_cloud_launch_receipt_keeps_ledger_receipt_state_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ready_artifacts = self.write_ready_artifacts(root / "ready")
            launch_receipt = ready_artifacts["cloud_training_launch_receipt"][0]
            launch_payload = json.loads(launch_receipt.read_text(encoding="utf-8"))
            launch_payload["source_artifacts"]["launch_plan"]["sha256"] = "0" * 64
            launch_payload["passed"] = True
            launch_receipt.write_text(json.dumps(launch_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            plan = self.write_loop_plan(root / "ready" / "plan.json", "loop-001", ready_artifacts)
            ledger = root / "ledger.json"
            summary = root / "summary.json"

            self.assertEqual(run_cli(["agentic-loop", "ledger", "--plan", str(plan), "--out", str(ledger)]), 0)
            payload = json.loads(ledger.read_text(encoding="utf-8"))
            self.assertFalse(payload["iterations"][0]["cloud_training_receipt_state"]["launch_receipt_passed"])
            self.assertFalse(payload["iterations"][0]["cloud_training_receipt_state"]["receipts_passed"])
            self.assertEqual(run_cli(["validate", "--agentic-loop-ledger", str(ledger), "--out", str(summary), "--strict"]), 0)
            self.assertEqual(run_cli(["validate", "--cloud-training-launch-receipt", str(launch_receipt), "--strict"]), 1)

    def test_live_launch_request_alone_blocks_ledger_governance_readiness(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ready_artifacts = self.write_ready_artifacts(root / "ready")
            launch_receipt = ready_artifacts["cloud_training_launch_receipt"][0]
            launch_payload = json.loads(launch_receipt.read_text(encoding="utf-8"))
            launch_payload["launch"]["mode"] = "live"
            launch_receipt.write_text(json.dumps(launch_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            plan = self.write_loop_plan(root / "ready" / "plan.json", "loop-001", ready_artifacts)
            ledger = root / "ledger.json"
            summary = root / "summary.json"
            self.assertEqual(run_cli(["agentic-loop", "ledger", "--plan", str(plan), "--out", str(ledger)]), 0)
            payload = json.loads(ledger.read_text(encoding="utf-8"))

            self.assertFalse(payload["iterations"][0]["cloud_training_receipt_state"]["fail_closed"])
            self.assertTrue(payload["iterations"][0]["cloud_training_receipt_state"]["live_launch_requested"])
            self.assertFalse(payload["readiness_digest"]["ready_for_governance_review"])
            self.assertFalse(payload["readiness_digest"]["cloud_training_receipts_fail_closed"])
            self.assertTrue(payload["readiness_digest"]["cloud_training_live_launch_requested"])

            forged_ready_payload = json.loads(json.dumps(payload))
            forged_ready_payload["iterations"][0]["readiness"] = "ready_for_governance_review"
            forged_ready_payload["iterations"][0]["recommendation"] = "approve_iteration_execution"
            forged_ready_payload["metrics"]["ready_iteration_count"] = 1
            forged_ready_payload["metrics"]["blocked_iteration_count"] = 0
            forged_ready_payload["metrics"]["latest_readiness"] = "ready_for_governance_review"
            forged_ready_payload["metrics"]["latest_recommendation"] = "approve_iteration_execution"
            forged_ready_payload["readiness_digest"]["readiness"] = "ready_for_governance_review"
            forged_ready_payload["readiness_digest"]["recommendation"] = "approve_iteration_execution"
            forged_ready_payload["readiness_digest"]["ready_for_governance_review"] = True
            ledger.write_text(json.dumps(forged_ready_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            code = run_cli(["validate", "--agentic-loop-ledger", str(ledger), "--out", str(summary)])

            self.assertEqual(code, 1)
            errors = "\n".join(error for target in json.loads(summary.read_text(encoding="utf-8"))["targets"] for error in target["errors"])
            self.assertIn(
                "agentic_loop_ledger.readiness_digest.ready_for_governance_review must match latest iteration readiness.",
                errors,
            )

            payload["iterations"][0]["cloud_training_receipt_state"]["fail_closed"] = True
            payload["iterations"][0]["cloud_training_receipt_state"]["live_launch_requested"] = False
            payload["readiness_digest"]["cloud_training_receipts_fail_closed"] = True
            payload["readiness_digest"]["cloud_training_live_launch_requested"] = False
            ledger.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            code = run_cli(["validate", "--agentic-loop-ledger", str(ledger), "--out", str(summary)])

            self.assertEqual(code, 1)
            errors = "\n".join(error for target in json.loads(summary.read_text(encoding="utf-8"))["targets"] for error in target["errors"])
            self.assertIn("cloud_training_receipt_state.fail_closed must match source loop plan cloud training receipt artifacts", errors)
            self.assertIn("cloud_training_receipt_state.live_launch_requested must match source loop plan cloud training receipt artifacts", errors)

    def test_validate_rejects_tampered_governance_actions(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            root = Path(tmp)
            blocked_plan = self.write_loop_plan(root / "plans" / "blocked.json", "loop-001", {})
            ledger = root / "ledger.json"
            summary = root / "summary.json"
            self.assertEqual(run_cli(["agentic-loop", "ledger", "--plan", str(blocked_plan), "--out", str(ledger)]), 0)
            payload = json.loads(ledger.read_text(encoding="utf-8"))
            payload["decision"]["recommended_governance_action"] = "approve"
            payload["decision"]["summary"] = "forged ready summary"
            payload["decision"]["governance_actions"][0]["available"] = True
            payload["decision"]["governance_actions"][0]["blocked_reasons"] = []
            payload["decision"]["governance_actions"][0]["blocked_reason_count"] = 0
            payload["readiness_digest"]["summary"] = "forged digest summary"
            ledger.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            code = run_cli(["validate", "--agentic-loop-ledger", str(ledger), "--out", str(summary)])

            self.assertEqual(code, 1)
            errors = "\n".join(error for target in json.loads(summary.read_text(encoding="utf-8"))["targets"] for error in target["errors"])
            self.assertIn("agentic_loop_ledger.decision.recommended_governance_action must match latest iteration state.", errors)
            self.assertIn("agentic_loop_ledger.decision.governance_actions must match latest iteration state.", errors)
            self.assertIn("agentic_loop_ledger.decision.summary must match latest iteration state.", errors)
            self.assertIn("agentic_loop_ledger.readiness_digest.summary must match latest iteration readiness.", errors)

    def test_validate_and_schema_reject_unknown_agentic_loop_ledger_fields(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            root = Path(tmp)
            blocked_plan = self.write_loop_plan(root / "plans" / "blocked.json", "loop-001", {})
            ledger = root / "ledger.json"
            summary = root / "summary.json"
            self.assertEqual(run_cli(["agentic-loop", "ledger", "--plan", str(blocked_plan), "--out", str(ledger)]), 0)
            payload = json.loads(ledger.read_text(encoding="utf-8"))
            payload["cloud_job_id"] = "job_live"
            payload["metrics"]["provider_invoice_id"] = "invoice_live"
            payload["metrics"]["readiness_counts"][0]["private_metric"] = "hidden"
            payload["decision"]["approval_token"] = "token_redacted"
            payload["decision"]["governance_actions"][0]["signed_url"] = "https://example.invalid/action"
            payload["readiness_digest"]["promotion_alias_moved"] = True
            payload["execution_boundary"]["cloud_job_id"] = "job_live"
            ledger.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            self.assertEqual(run_cli(["schemas", "--check", str(ledger)]), 1)
            code = run_cli(["validate", "--agentic-loop-ledger", str(ledger), "--out", str(summary)])

            self.assertEqual(code, 1)
            errors = "\n".join(error for target in json.loads(summary.read_text(encoding="utf-8"))["targets"] for error in target["errors"])
            self.assertIn("agentic_loop_ledger contains unknown field(s): ['cloud_job_id'].", errors)
            self.assertIn("agentic_loop_ledger.metrics contains unknown field(s): ['provider_invoice_id'].", errors)
            self.assertIn(
                "agentic_loop_ledger.metrics.readiness_counts[0] contains unknown field(s): ['private_metric'].",
                errors,
            )
            self.assertIn("agentic_loop_ledger.decision contains unknown field(s): ['approval_token'].", errors)
            self.assertIn(
                "agentic_loop_ledger.decision.governance_actions[0] contains unknown field(s): ['signed_url'].",
                errors,
            )
            self.assertIn("agentic_loop_ledger.readiness_digest contains unknown field(s): ['promotion_alias_moved'].", errors)
            self.assertIn("agentic_loop_ledger.execution_boundary contains unknown field(s): ['cloud_job_id'].", errors)

    def test_validate_and_schema_reject_unknown_agentic_loop_iteration_fields(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            root = Path(tmp)
            blocked_plan = self.write_loop_plan(root / "plans" / "blocked.json", "loop-001", {})
            ledger = root / "ledger.json"
            summary = root / "summary.json"
            self.assertEqual(run_cli(["agentic-loop", "ledger", "--plan", str(blocked_plan), "--out", str(ledger)]), 0)
            payload = json.loads(ledger.read_text(encoding="utf-8"))
            iteration = payload["iterations"][0]
            iteration["provider_job_id"] = "job_live"
            iteration["artifact_group_counts"][0]["private_metric"] = "hidden"
            iteration["cost_estimate"]["provider_invoice_id"] = "invoice_live"
            iteration["cloud_training"]["provider_console_url"] = "https://example.invalid/cloud"
            iteration["cloud_training_lineage"]["provider_trace_url"] = "https://example.invalid/trace"
            iteration["serving"]["live_endpoint_url"] = "https://example.invalid/serve"
            iteration["evals"]["benchmark_job_id"] = "bench_live"
            iteration["external_eval_receipt_state"]["benchmark_job_id"] = "bench_live"
            iteration["training_outputs"]["model_download_path"] = "/redacted/model"
            iteration["governance"]["promotion_alias_moved"] = True
            iteration["next_actions"]["automation_thread_id"] = "thread_redacted"
            ledger.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            self.assertEqual(run_cli(["schemas", "--check", str(ledger)]), 1)
            code = run_cli(["validate", "--agentic-loop-ledger", str(ledger), "--out", str(summary)])

            self.assertEqual(code, 1)
            errors = "\n".join(error for target in json.loads(summary.read_text(encoding="utf-8"))["targets"] for error in target["errors"])
            self.assertIn("agentic_loop_ledger.iterations[0] contains unknown field(s): ['provider_job_id'].", errors)
            self.assertIn(
                "agentic_loop_ledger.iterations[0].artifact_group_counts[0] contains unknown field(s): ['private_metric'].",
                errors,
            )
            self.assertIn(
                "agentic_loop_ledger.iterations[0].cost_estimate contains unknown field(s): ['provider_invoice_id'].",
                errors,
            )
            self.assertIn(
                "agentic_loop_ledger.iterations[0].cloud_training contains unknown field(s): ['provider_console_url'].",
                errors,
            )
            self.assertIn(
                "agentic_loop_ledger.iterations[0].cloud_training_lineage contains unknown field(s): ['provider_trace_url'].",
                errors,
            )
            self.assertIn("agentic_loop_ledger.iterations[0].serving contains unknown field(s): ['live_endpoint_url'].", errors)
            self.assertIn("agentic_loop_ledger.iterations[0].evals contains unknown field(s): ['benchmark_job_id'].", errors)
            self.assertIn(
                "agentic_loop_ledger.iterations[0].external_eval_receipt_state contains unknown field(s): ['benchmark_job_id'].",
                errors,
            )
            self.assertIn(
                "agentic_loop_ledger.iterations[0].training_outputs contains unknown field(s): ['model_download_path'].",
                errors,
            )
            self.assertIn("agentic_loop_ledger.iterations[0].governance contains unknown field(s): ['promotion_alias_moved'].", errors)
            self.assertIn("agentic_loop_ledger.iterations[0].next_actions contains unknown field(s): ['automation_thread_id'].", errors)

    def test_agentic_loop_governance_receipt_records_reject_without_side_effects(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            root = Path(tmp)
            blocked_plan = self.write_loop_plan(root / "plans" / "blocked.json", "loop-001", {})
            ledger = root / "ledger.json"
            receipt = root / "governance_reject.json"
            self.assertEqual(run_cli(["agentic-loop", "ledger", "--plan", str(blocked_plan), "--out", str(ledger)]), 0)

            code = run_cli(
                [
                    "agentic-loop",
                    "governance",
                    "--ledger",
                    str(ledger),
                    "--action",
                    "reject",
                    "--requested-by",
                    "governance-ci",
                    "--reason",
                    "blocked loop stays rejected",
                    "--created-at",
                    "2026-07-03T00:00:00+00:00",
                    "--out",
                    str(receipt),
                ]
            )

            self.assertEqual(code, 0)
            payload = json.loads(receipt.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema_version"], "hfr.agentic_loop_governance_receipt.v1")
            self.assertTrue(payload["passed"])
            self.assertEqual(payload["recommendation"], "record_rejection")
            self.assertEqual(payload["requested_action"]["action"], "reject")
            self.assertTrue(payload["requested_action"]["available"])
            self.assertEqual(payload["source_ledger"]["path"], "ledger.json")
            self.assertTrue(payload["execution_boundary"]["receipt_only"])
            self.assertFalse(payload["execution_boundary"]["promotion_alias_moved"])
            self.assertFalse(payload["execution_boundary"]["rollback_applied"])
            self.assertFalse(payload["execution_boundary"]["weights_updated_by_flight_recorder"])
            self.assertEqual(run_cli(["validate", "--agentic-loop-governance-receipt", str(receipt), "--strict"]), 0)
            self.assertEqual(run_cli(["schemas", "--check", str(receipt)]), 0)

    def test_failed_external_benchmark_cannot_reach_governance_approval(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            root = Path(tmp)
            artifacts = copy_valid_loop_artifacts(root)
            heldout_root = artifacts["heldout_manifest"][0].parent
            result_path = heldout_root / "failed_external_eval_result.json"
            result = build_external_eval_result(
                plan_path=artifacts["external_eval_plan"][0],
                heldout_manifest_path=artifacts["heldout_manifest"][0],
                raw_result_path=heldout_root / "candidate_suite_summary.json",
                runner_metadata_path=heldout_root / "external_eval_runner.json",
                adapter_id="local_mock",
                execution_id="local-mock-eval-001",
                model_id="local/mock-candidate",
                normalizer_id="hfr.local_mock.run_suite",
                normalizer_version="1",
                raw_format="hfr.run_suite.v1",
                execution_status="completed",
                out_path=result_path,
                created_at="2026-07-03T00:00:00+00:00",
            )
            write_external_eval_result(result, result_path)
            summary_path = heldout_root / "failed_eval_summary.json"
            summary = build_eval_summary(
                suite_summary_specs=[
                    f"candidate={heldout_root / 'candidate_suite_summary.json'}"
                ],
                external_adapter_plan_specs=[
                    f"external={artifacts['external_eval_plan'][0]}"
                ],
                external_adapter_result_specs=[f"external={result_path}"],
                output_base_dir=heldout_root,
            )
            summary_path.write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            artifacts["external_eval_result"] = [result_path]
            artifacts["eval_summary"] = [summary_path]
            fixture_root = heldout_root.parent
            loop_plan = self.write_loop_plan(
                fixture_root / "failed_loop_plan.json",
                "failed-external-benchmark",
                artifacts,
            )
            ledger = fixture_root / "failed_loop_ledger.json"
            receipt = fixture_root / "failed_loop_governance.json"

            self.assertEqual(
                run_cli(
                    [
                        "agentic-loop",
                        "ledger",
                        "--plan",
                        str(loop_plan),
                        "--out",
                        str(ledger),
                    ]
                ),
                0,
            )
            governance_code = run_cli(
                [
                    "agentic-loop",
                    "governance",
                    "--ledger",
                    str(ledger),
                    "--action",
                    "approve",
                    "--created-at",
                    "2026-07-03T00:00:00+00:00",
                    "--out",
                    str(receipt),
                ]
            )

            loop = json.loads(loop_plan.read_text(encoding="utf-8"))
            ledger_payload = json.loads(ledger.read_text(encoding="utf-8"))
            receipt_payload = json.loads(receipt.read_text(encoding="utf-8"))
            self.assertEqual(result["benchmark_outcome"]["status"], "failed")
            self.assertEqual(loop["execution_completion"], "completed")
            self.assertEqual(loop["governance_readiness"], "blocked")
            self.assertEqual(
                ledger_payload["readiness_digest"]["execution_completion"],
                "completed",
            )
            self.assertEqual(
                ledger_payload["readiness_digest"]["governance_readiness"],
                "blocked",
            )
            self.assertEqual(governance_code, 1)
            self.assertFalse(receipt_payload["requested_action"]["available"])
            approval_check = next(
                check
                for check in receipt_payload["checks"]
                if check["id"]
                == "approval_requires_completed_governance_ready_execution"
            )
            self.assertFalse(approval_check["passed"])

    def test_agentic_loop_governance_receipt_blocks_unavailable_approval(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            root = Path(tmp)
            blocked_plan = self.write_loop_plan(root / "plans" / "blocked.json", "loop-001", {})
            ledger = root / "ledger.json"
            receipt = root / "governance_approve.json"
            self.assertEqual(run_cli(["agentic-loop", "ledger", "--plan", str(blocked_plan), "--out", str(ledger)]), 0)

            code = run_cli(
                [
                    "agentic-loop",
                    "governance",
                    "--ledger",
                    str(ledger),
                    "--action",
                    "approve",
                    "--created-at",
                    "2026-07-03T00:00:00+00:00",
                    "--out",
                    str(receipt),
                ]
            )

            self.assertEqual(code, 1)
            payload = json.loads(receipt.read_text(encoding="utf-8"))
            self.assertFalse(payload["passed"])
            self.assertEqual(payload["readiness"], "blocked")
            self.assertEqual(payload["recommendation"], "fix_governance_inputs")
            self.assertFalse(payload["requested_action"]["available"])
            self.assertIn("latest_iteration_not_ready_for_governance_review", payload["requested_action"]["blocked_reasons"])
            self.assertIn("missing_promotion_decision", payload["requested_action"]["blocked_reasons"])
            self.assertIn("missing_promotion_ledger", payload["requested_action"]["blocked_reasons"])
            self.assertEqual(run_cli(["validate", "--agentic-loop-governance-receipt", str(receipt), "--strict"]), 0)
            self.assertEqual(run_cli(["schemas", "--check", str(receipt)]), 0)

    def test_agentic_loop_governance_receipt_blocks_forged_ledger_approval(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            root = Path(tmp)
            blocked_plan = self.write_loop_plan(root / "plans" / "blocked.json", "loop-001", {})
            ledger = root / "ledger.json"
            receipt = root / "governance_approve.json"
            summary = root / "summary.json"
            self.assertEqual(run_cli(["agentic-loop", "ledger", "--plan", str(blocked_plan), "--out", str(ledger)]), 0)
            payload = json.loads(ledger.read_text(encoding="utf-8"))
            payload["decision"]["recommended_governance_action"] = "approve"
            for row in payload["decision"]["governance_actions"]:
                if row["action"] == "approve":
                    row["available"] = True
                    row["blocked_reasons"] = []
                    row["blocked_reason_count"] = 0
                    row["summary"] = "forged approval"
            ledger.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            code = run_cli(["agentic-loop", "governance", "--ledger", str(ledger), "--action", "approve", "--out", str(receipt)])

            self.assertEqual(code, 1)
            receipt_payload = json.loads(receipt.read_text(encoding="utf-8"))
            checks = {row["id"]: row for row in receipt_payload["checks"]}
            self.assertFalse(receipt_payload["passed"])
            self.assertFalse(receipt_payload["source_ledger"]["passed"])
            self.assertTrue(receipt_payload["requested_action"]["available"])
            self.assertFalse(checks["source_ledger_passed"]["passed"])
            self.assertEqual(run_cli(["validate", "--agentic-loop-governance-receipt", str(receipt), "--out", str(summary)]), 1)
            errors = "\n".join(error for target in json.loads(summary.read_text(encoding="utf-8"))["targets"] for error in target["errors"])
            self.assertIn("agentic_loop_ledger.decision.recommended_governance_action must match latest iteration state.", errors)
            self.assertEqual(run_cli(["schemas", "--check", str(receipt)]), 0)

    def test_agentic_loop_governance_receipt_blocks_stale_source_plan(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            root = Path(tmp)
            ready_artifacts = self.write_ready_artifacts(root / "ready")
            plan = self.write_loop_plan(root / "ready" / "plan.json", "loop-001", ready_artifacts)
            ledger = root / "ledger.json"
            receipt = root / "governance_approve.json"
            summary = root / "summary.json"
            self.assertEqual(run_cli(["agentic-loop", "ledger", "--plan", str(plan), "--out", str(ledger)]), 0)
            self.assertEqual(run_cli(["validate", "--agentic-loop-ledger", str(ledger), "--strict"]), 0)
            plan_payload = json.loads(plan.read_text(encoding="utf-8"))
            plan_payload["objective"] = "tampered after ledger creation"
            plan.write_text(json.dumps(plan_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            code = run_cli(["agentic-loop", "governance", "--ledger", str(ledger), "--action", "approve", "--out", str(receipt)])

            self.assertEqual(code, 1)
            receipt_payload = json.loads(receipt.read_text(encoding="utf-8"))
            checks = {row["id"]: row for row in receipt_payload["checks"]}
            self.assertFalse(receipt_payload["passed"])
            self.assertFalse(receipt_payload["source_ledger"]["passed"])
            self.assertTrue(receipt_payload["requested_action"]["available"])
            self.assertFalse(checks["source_ledger_passed"]["passed"])
            self.assertEqual(run_cli(["validate", "--agentic-loop-governance-receipt", str(receipt), "--out", str(summary)]), 1)
            errors = "\n".join(error for target in json.loads(summary.read_text(encoding="utf-8"))["targets"] for error in target["errors"])
            self.assertIn("sha256 does not match the current file", errors)
            self.assertEqual(run_cli(["schemas", "--check", str(receipt)]), 0)

    def test_direct_governance_receipt_builder_requires_source_ledger_replay_proof(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            root = Path(tmp)
            ready_artifacts = self.write_ready_artifacts(root / "ready")
            plan = self.write_loop_plan(root / "ready" / "plan.json", "loop-001", ready_artifacts)
            ledger = root / "ledger.json"
            self.assertEqual(run_cli(["agentic-loop", "ledger", "--plan", str(plan), "--out", str(ledger)]), 0)
            plan_payload = json.loads(plan.read_text(encoding="utf-8"))
            plan_payload["objective"] = "tampered before direct receipt build"
            plan.write_text(json.dumps(plan_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            receipt = build_agentic_loop_governance_receipt(
                ledger_path=ledger,
                action="approve",
                out_path=root / "governance_approve.json",
            )

            checks = {row["id"]: row for row in receipt["checks"]}
            self.assertFalse(receipt["passed"])
            self.assertFalse(receipt["source_ledger"]["passed"])
            self.assertTrue(receipt["requested_action"]["available"])
            self.assertFalse(checks["source_ledger_passed"]["passed"])

    def test_validate_rejects_tampered_governance_receipt_action(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            root = Path(tmp)
            blocked_plan = self.write_loop_plan(root / "plans" / "blocked.json", "loop-001", {})
            ledger = root / "ledger.json"
            receipt = root / "governance_approve.json"
            summary = root / "summary.json"
            self.assertEqual(run_cli(["agentic-loop", "ledger", "--plan", str(blocked_plan), "--out", str(ledger)]), 0)
            self.assertEqual(
                run_cli(["agentic-loop", "governance", "--ledger", str(ledger), "--action", "approve", "--out", str(receipt)]),
                1,
            )
            payload = json.loads(receipt.read_text(encoding="utf-8"))
            payload["requested_action"]["available"] = True
            payload["requested_action"]["blocked_reasons"] = []
            payload["requested_action"]["blocked_reason_count"] = 0
            payload["checks"][-2]["passed"] = True
            payload["passed"] = True
            payload["readiness"] = "recorded"
            payload["recommendation"] = "record_approval_for_promotion_review"
            payload["decision"]["readiness"] = "ready"
            payload["decision"]["recommendation"] = "record_approval_for_promotion_review"
            receipt.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            code = run_cli(["validate", "--agentic-loop-governance-receipt", str(receipt), "--out", str(summary)])

            self.assertEqual(code, 1)
            errors = "\n".join(error for target in json.loads(summary.read_text(encoding="utf-8"))["targets"] for error in target["errors"])
            self.assertIn("agentic_loop_governance_receipt.checks must match current ledger action state.", errors)
            self.assertIn("agentic_loop_governance_receipt.requested_action.available must match current ledger action state.", errors)
            self.assertIn("agentic_loop_governance_receipt.decision must match current ledger action state.", errors)

    def test_validate_rejects_successful_governance_receipt_without_source_ledger(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            blocked_plan = self.write_loop_plan(root / "plans" / "blocked.json", "loop-001", {})
            ledger = root / "ledger.json"
            receipt = root / "governance_reject.json"
            summary = root / "summary.json"
            self.assertEqual(run_cli(["agentic-loop", "ledger", "--plan", str(blocked_plan), "--out", str(ledger)]), 0)
            self.assertEqual(run_cli(["agentic-loop", "governance", "--ledger", str(ledger), "--action", "reject", "--out", str(receipt)]), 0)
            payload = json.loads(receipt.read_text(encoding="utf-8"))
            payload["source_ledger"]["exists"] = False
            payload["source_ledger"]["kind"] = "missing"
            payload["source_ledger"]["sha256"] = None
            payload["source_ledger"]["size_bytes"] = None
            payload["checks"] = []
            payload["check_count"] = 0
            payload["failed_check_count"] = 0
            payload["passed"] = True
            payload["readiness"] = "recorded"
            payload["recommendation"] = "record_rejection"
            payload["decision"]["readiness"] = "ready"
            payload["decision"]["recommendation"] = "record_rejection"
            payload["decision"]["blocking_check_count"] = 0
            payload["decision"]["blocking_checks"] = []
            receipt.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            code = run_cli(["validate", "--agentic-loop-governance-receipt", str(receipt), "--out", str(summary)])

            self.assertEqual(code, 1)
            errors = "\n".join(error for target in json.loads(summary.read_text(encoding="utf-8"))["targets"] for error in target["errors"])
            self.assertIn("agentic_loop_governance_receipt.passed must be false when source ledger cannot be replayed.", errors)
            self.assertIn("agentic_loop_governance_receipt.readiness must be blocked when source ledger cannot be replayed.", errors)
            self.assertIn("agentic_loop_governance_receipt.checks must match fail-closed source-ledger checks.", errors)

    def test_missing_ledger_governance_receipt_round_trips_as_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            receipt = root / "missing_ledger_governance.json"

            code = run_cli(
                [
                    "agentic-loop",
                    "governance",
                    "--ledger",
                    str(root / "missing_ledger.json"),
                    "--action",
                    "reject",
                    "--created-at",
                    "2026-07-03T00:00:00+00:00",
                    "--out",
                    str(receipt),
                ]
            )

            self.assertEqual(code, 1)
            payload = json.loads(receipt.read_text(encoding="utf-8"))
            self.assertFalse(payload["passed"])
            self.assertEqual(payload["readiness"], "blocked")
            self.assertEqual(payload["recommendation"], "fix_governance_inputs")
            self.assertEqual(payload["requested_action"]["summary"], "Action reject is blocked because the source ledger did not list it.")
            self.assertIn("requested_action_not_listed_by_ledger", payload["requested_action"]["blocked_reasons"])
            self.assertEqual(payload["source_ledger"]["kind"], "missing")
            self.assertEqual(payload["source_ledger"]["path"], "<redacted:missing_ledger.json>")
            summary = root / "summary.json"
            self.assertEqual(
                run_cli(["validate", "--agentic-loop-governance-receipt", str(receipt), "--strict", "--out", str(summary)]),
                1,
            )
            errors = "\n".join(error for target in json.loads(summary.read_text(encoding="utf-8"))["targets"] for error in target["errors"])
            self.assertIn("agentic_loop_governance_receipt.source_ledger must resolve to a replayable source ledger.", errors)
            self.assertEqual(run_cli(["schemas", "--check", str(receipt)]), 0)

    def test_validate_rejects_noncanonical_missing_governance_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            receipt = root / "missing_ledger_governance.json"
            summary = root / "summary.json"
            self.assertEqual(
                run_cli(
                    [
                        "agentic-loop",
                        "governance",
                        "--ledger",
                        str(root / "missing_ledger.json"),
                        "--action",
                        "reject",
                        "--out",
                        str(receipt),
                    ]
                ),
                1,
            )
            payload = json.loads(receipt.read_text(encoding="utf-8"))
            payload["source_ledger"]["kind"] = "file"
            payload["source_ledger"]["path"] = str(root / "private" / "missing_ledger.json")
            receipt.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            code = run_cli(["validate", "--agentic-loop-governance-receipt", str(receipt), "--out", str(summary)])

            self.assertEqual(code, 1)
            errors = "\n".join(error for target in json.loads(summary.read_text(encoding="utf-8"))["targets"] for error in target["errors"])
            self.assertIn("agentic_loop_governance_receipt.source_ledger.path must be a safe relative path or redacted placeholder.", errors)
            self.assertIn("agentic_loop_governance_receipt.source_ledger.kind must be missing when exists is false.", errors)

    def test_validate_rejects_missing_source_that_points_at_existing_ledger(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            root = Path(tmp)
            blocked_plan = self.write_loop_plan(root / "plans" / "blocked.json", "loop-001", {})
            ledger = root / "ledger.json"
            receipt = root / "missing_ledger_governance.json"
            summary = root / "summary.json"
            self.assertEqual(run_cli(["agentic-loop", "ledger", "--plan", str(blocked_plan), "--out", str(ledger)]), 0)
            self.assertEqual(
                run_cli(["agentic-loop", "governance", "--ledger", str(root / "missing_ledger.json"), "--action", "reject", "--out", str(receipt)]),
                1,
            )
            payload = json.loads(receipt.read_text(encoding="utf-8"))
            payload["source_ledger"]["path"] = "ledger.json"
            receipt.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            code = run_cli(["validate", "--agentic-loop-governance-receipt", str(receipt), "--out", str(summary)])

            self.assertEqual(code, 1)
            errors = "\n".join(error for target in json.loads(summary.read_text(encoding="utf-8"))["targets"] for error in target["errors"])
            self.assertIn("agentic_loop_governance_receipt.source_ledger.exists must be true when path resolves to an existing file.", errors)

    def test_validate_rejects_unknown_governance_receipt_fields(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            root = Path(tmp)
            blocked_plan = self.write_loop_plan(root / "plans" / "blocked.json", "loop-001", {})
            ledger = root / "ledger.json"
            receipt = root / "governance_reject.json"
            summary = root / "summary.json"
            self.assertEqual(run_cli(["agentic-loop", "ledger", "--plan", str(blocked_plan), "--out", str(ledger)]), 0)
            self.assertEqual(run_cli(["agentic-loop", "governance", "--ledger", str(ledger), "--action", "reject", "--out", str(receipt)]), 0)
            payload = json.loads(receipt.read_text(encoding="utf-8"))
            payload["promotion_alias_moved"] = True
            payload["rollback_applied"] = True
            payload["requested_action"]["applies_rollback"] = True
            payload["source_ledger"]["private_path"] = "local/private/path"
            payload["checks"][0]["private_path"] = "local/private/path"
            payload["decision"]["blocking_checks"] = [
                {"id": "forged", "summary": "forged", "scope": {}, "private_path": "local/private/path"}
            ]
            receipt.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            code = run_cli(["validate", "--agentic-loop-governance-receipt", str(receipt), "--out", str(summary)])

            self.assertEqual(code, 1)
            errors = "\n".join(error for target in json.loads(summary.read_text(encoding="utf-8"))["targets"] for error in target["errors"])
            self.assertIn("agentic_loop_governance_receipt contains unknown field(s): ['promotion_alias_moved', 'rollback_applied'].", errors)
            self.assertIn("agentic_loop_governance_receipt.requested_action contains unknown field(s): ['applies_rollback'].", errors)
            self.assertIn("agentic_loop_governance_receipt.source_ledger contains unknown field(s): ['private_path'].", errors)
            self.assertIn("agentic_loop_governance_receipt.checks[0] contains unknown field(s): ['private_path'].", errors)
            self.assertIn(
                "agentic_loop_governance_receipt.decision.blocking_checks[0] contains unknown field(s): ['private_path'].",
                errors,
            )
            self.assertEqual(run_cli(["schemas", "--check", str(receipt)]), 1)

    def test_governance_receipt_schema_rejects_forged_source_ledger_side_effects(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            root = Path(tmp)
            blocked_plan = self.write_loop_plan(root / "plans" / "blocked.json", "loop-001", {})
            ledger = root / "ledger.json"
            receipt = root / "governance_reject.json"
            summary = root / "summary.json"
            self.assertEqual(run_cli(["agentic-loop", "ledger", "--plan", str(blocked_plan), "--out", str(ledger)]), 0)
            self.assertEqual(run_cli(["agentic-loop", "governance", "--ledger", str(ledger), "--action", "reject", "--out", str(receipt)]), 0)
            payload = json.loads(receipt.read_text(encoding="utf-8"))
            payload["source_ledger"]["execution_boundary"]["cloud_jobs_started"] = True
            payload["source_ledger"]["execution_boundary"]["model_downloads_started"] = True
            payload["source_ledger"]["execution_boundary"]["weights_updated_by_flight_recorder"] = True
            receipt.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            validation_code = run_cli(["validate", "--agentic-loop-governance-receipt", str(receipt), "--out", str(summary)])

            self.assertEqual(validation_code, 1)
            errors = "\n".join(error for target in json.loads(summary.read_text(encoding="utf-8"))["targets"] for error in target["errors"])
            self.assertIn("agentic_loop_governance_receipt.source_ledger.execution_boundary must match the current source ledger.", errors)
            self.assertEqual(run_cli(["schemas", "--check", str(receipt)]), 1)

    def test_validate_rejects_governance_receipt_numeric_type_coercions(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            root = Path(tmp)
            blocked_plan = self.write_loop_plan(root / "plans" / "blocked.json", "loop-001", {})
            ledger = root / "ledger.json"
            receipt = root / "governance_reject.json"
            summary = root / "summary.json"
            self.assertEqual(run_cli(["agentic-loop", "ledger", "--plan", str(blocked_plan), "--out", str(ledger)]), 0)
            self.assertEqual(run_cli(["agentic-loop", "governance", "--ledger", str(ledger), "--action", "reject", "--out", str(receipt)]), 0)
            payload = json.loads(receipt.read_text(encoding="utf-8"))
            payload["passed"] = 1
            payload["check_count"] = float(payload["check_count"])
            payload["failed_check_count"] = float(payload["failed_check_count"])
            payload["requested_action"]["blocked_reason_count"] = float(payload["requested_action"]["blocked_reason_count"])
            payload["decision"]["blocking_check_count"] = float(payload["decision"]["blocking_check_count"])
            receipt.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            code = run_cli(["validate", "--agentic-loop-governance-receipt", str(receipt), "--out", str(summary)])

            self.assertEqual(code, 1)
            errors = "\n".join(error for target in json.loads(summary.read_text(encoding="utf-8"))["targets"] for error in target["errors"])
            self.assertIn("agentic_loop_governance_receipt.passed must be a boolean.", errors)
            self.assertIn("agentic_loop_governance_receipt.check_count must be a non-negative integer.", errors)
            self.assertIn("agentic_loop_governance_receipt.failed_check_count must be a non-negative integer.", errors)
            self.assertIn("agentic_loop_governance_receipt.requested_action.blocked_reason_count must be a non-negative integer.", errors)
            self.assertIn("agentic_loop_governance_receipt.decision.blocking_check_count must be a non-negative integer.", errors)

    def test_validate_and_schema_reject_nested_governance_snapshot_fields(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            root = Path(tmp)
            blocked_plan = self.write_loop_plan(root / "plans" / "blocked.json", "loop-001", {})
            ledger = root / "ledger.json"
            receipt = root / "governance_reject.json"
            summary = root / "summary.json"
            self.assertEqual(run_cli(["agentic-loop", "ledger", "--plan", str(blocked_plan), "--out", str(ledger)]), 0)
            self.assertEqual(run_cli(["agentic-loop", "governance", "--ledger", str(ledger), "--action", "reject", "--out", str(receipt)]), 0)
            payload = json.loads(receipt.read_text(encoding="utf-8"))
            payload["decision"]["blocking_check_count"] = 1
            payload["decision"]["blocking_checks"] = [
                {"id": "forged", "summary": "forged", "scope": {"private_path": "local/private/path"}}
            ]
            payload["source_ledger"]["decision"]["private_path"] = "local/private/path"
            payload["source_ledger"]["readiness_digest"]["private_path"] = "local/private/path"
            payload["source_ledger"]["execution_boundary"]["private_path"] = "local/private/path"
            receipt.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            code = run_cli(["validate", "--agentic-loop-governance-receipt", str(receipt), "--out", str(summary)])

            self.assertEqual(code, 1)
            errors = "\n".join(error for target in json.loads(summary.read_text(encoding="utf-8"))["targets"] for error in target["errors"])
            self.assertIn(
                "agentic_loop_governance_receipt.decision.blocking_checks[0].scope contains unknown field(s): ['private_path'].",
                errors,
            )
            self.assertIn("agentic_loop_governance_receipt.source_ledger.decision contains unknown field(s): ['private_path'].", errors)
            self.assertIn("agentic_loop_governance_receipt.source_ledger.readiness_digest contains unknown field(s): ['private_path'].", errors)
            self.assertIn("agentic_loop_governance_receipt.source_ledger.execution_boundary contains unknown field(s): ['private_path'].", errors)
            self.assertEqual(run_cli(["schemas", "--check", str(receipt)]), 1)

    def test_validate_and_schema_reject_unreplayable_governance_action_fields(self):
        with tempfile.TemporaryDirectory() as source_tmp, tempfile.TemporaryDirectory(dir=Path.cwd()) as out_tmp:
            source_root = Path(source_tmp)
            out_root = Path(out_tmp)
            blocked_plan = self.write_loop_plan(source_root / "plans" / "blocked.json", "loop-001", {})
            ledger = source_root / "ledger.json"
            receipt = out_root / "governance_reject.json"
            summary = out_root / "summary.json"
            self.assertEqual(run_cli(["agentic-loop", "ledger", "--plan", str(blocked_plan), "--out", str(ledger)]), 0)
            self.assertEqual(run_cli(["agentic-loop", "governance", "--ledger", str(ledger), "--action", "reject", "--out", str(receipt)]), 1)
            payload = json.loads(receipt.read_text(encoding="utf-8"))
            payload["source_ledger"]["decision"]["governance_actions"][0]["private_path"] = "local/private/path"
            receipt.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            code = run_cli(["validate", "--agentic-loop-governance-receipt", str(receipt), "--out", str(summary)])

            self.assertEqual(code, 1)
            errors = "\n".join(error for target in json.loads(summary.read_text(encoding="utf-8"))["targets"] for error in target["errors"])
            self.assertIn(
                "agentic_loop_governance_receipt.source_ledger.decision.governance_actions[0] contains unknown field(s): ['private_path'].",
                errors,
            )
            self.assertEqual(run_cli(["schemas", "--check", str(receipt)]), 1)

    def test_validate_rejects_missing_governance_source_snapshots(self):
        with tempfile.TemporaryDirectory() as source_tmp, tempfile.TemporaryDirectory(dir=Path.cwd()) as out_tmp:
            source_root = Path(source_tmp)
            out_root = Path(out_tmp)
            blocked_plan = self.write_loop_plan(source_root / "plans" / "blocked.json", "loop-001", {})
            ledger = source_root / "ledger.json"
            receipt = out_root / "governance_reject.json"
            summary = out_root / "summary.json"
            self.assertEqual(run_cli(["agentic-loop", "ledger", "--plan", str(blocked_plan), "--out", str(ledger)]), 0)
            self.assertEqual(run_cli(["agentic-loop", "governance", "--ledger", str(ledger), "--action", "reject", "--out", str(receipt)]), 1)
            payload = json.loads(receipt.read_text(encoding="utf-8"))
            del payload["source_ledger"]["decision"]
            del payload["source_ledger"]["readiness_digest"]
            del payload["source_ledger"]["execution_boundary"]
            receipt.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            code = run_cli(["validate", "--agentic-loop-governance-receipt", str(receipt), "--out", str(summary)])

            self.assertEqual(code, 1)
            errors = "\n".join(error for target in json.loads(summary.read_text(encoding="utf-8"))["targets"] for error in target["errors"])
            self.assertIn("agentic_loop_governance_receipt.source_ledger.decision must be an object.", errors)
            self.assertIn("agentic_loop_governance_receipt.source_ledger.readiness_digest must be an object.", errors)
            self.assertIn("agentic_loop_governance_receipt.source_ledger.execution_boundary must be an object.", errors)
            self.assertEqual(run_cli(["schemas", "--check", str(receipt)]), 1)

    def test_schema_rejects_missing_governance_source_promotion_ledger_digest(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            root = Path(tmp)
            blocked_plan = self.write_loop_plan(root / "plans" / "blocked.json", "loop-001", {})
            ledger = root / "ledger.json"
            receipt = root / "governance_reject.json"
            summary = root / "summary.json"
            self.assertEqual(run_cli(["agentic-loop", "ledger", "--plan", str(blocked_plan), "--out", str(ledger)]), 0)
            self.assertEqual(run_cli(["agentic-loop", "governance", "--ledger", str(ledger), "--action", "reject", "--out", str(receipt)]), 0)
            payload = json.loads(receipt.read_text(encoding="utf-8"))
            del payload["source_ledger"]["readiness_digest"]["promotion_ledger_present"]
            receipt.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            self.assertEqual(run_cli(["schemas", "--check", str(receipt)]), 1)
            code = run_cli(["validate", "--agentic-loop-governance-receipt", str(receipt), "--out", str(summary)])

            self.assertEqual(code, 1)
            errors = "\n".join(error for target in json.loads(summary.read_text(encoding="utf-8"))["targets"] for error in target["errors"])
            self.assertIn("agentic_loop_governance_receipt.source_ledger.readiness_digest must match the current source ledger.", errors)

    def test_governance_receipt_preserve_paths_keeps_receipt_path_valid(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            root = Path(tmp)
            blocked_plan = self.write_loop_plan(root / "plans" / "blocked.json", "loop-001", {})
            ledger = root / "ledger.json"
            receipt = root / "governance_reject.json"
            ledger_arg = str(ledger.relative_to(Path.cwd()))
            self.assertEqual(run_cli(["agentic-loop", "ledger", "--plan", str(blocked_plan), "--out", str(ledger)]), 0)

            code = run_cli(["agentic-loop", "governance", "--ledger", ledger_arg, "--action", "reject", "--out", str(receipt), "--preserve-paths"])

            self.assertEqual(code, 0)
            payload = json.loads(receipt.read_text(encoding="utf-8"))
            self.assertEqual(payload["receipt_path"], str(receipt.relative_to(Path.cwd())))
            self.assertEqual(payload["source_ledger"]["path"], "ledger.json")
            self.assertEqual(run_cli(["validate", "--agentic-loop-governance-receipt", str(receipt), "--strict"]), 0)
            self.assertEqual(run_cli(["schemas", "--check", str(receipt)]), 0)

    def test_governance_receipt_preserve_paths_redacts_absolute_source_paths(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            root = Path(tmp)
            blocked_plan = self.write_loop_plan(root / "plans" / "blocked.json", "loop-001", {})
            ledger = root / "ledger.json"
            receipt = root / "governance_reject.json"
            summary = root / "summary.json"
            self.assertEqual(run_cli(["agentic-loop", "ledger", "--plan", str(blocked_plan), "--out", str(ledger)]), 0)

            code = run_cli(
                ["agentic-loop", "governance", "--ledger", str(ledger.resolve()), "--action", "reject", "--out", str(receipt), "--preserve-paths"]
            )

            self.assertEqual(code, 1)
            payload = json.loads(receipt.read_text(encoding="utf-8"))
            self.assertEqual(payload["source_ledger"]["path"], "<redacted:ledger.json>")
            self.assertFalse(payload["source_ledger"]["exists"])
            self.assertNotIn(str(root), json.dumps(payload, sort_keys=True))
            self.assertEqual(
                run_cli(["validate", "--agentic-loop-governance-receipt", str(receipt), "--strict", "--out", str(summary)]),
                1,
            )
            errors = "\n".join(error for target in json.loads(summary.read_text(encoding="utf-8"))["targets"] for error in target["errors"])
            self.assertIn("agentic_loop_governance_receipt.source_ledger must resolve to a replayable source ledger.", errors)
            self.assertEqual(run_cli(["schemas", "--check", str(receipt)]), 0)

    def test_governance_receipt_redacts_traversal_output_path(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            root = Path(tmp)
            blocked_plan = self.write_loop_plan(root / "plans" / "blocked.json", "loop-001", {})
            ledger = root / "ledger.json"
            receipt_arg = str(root.relative_to(Path.cwd()) / ".." / "governance_reject.json")
            receipt = Path("governance_reject.json")
            self.assertEqual(run_cli(["agentic-loop", "ledger", "--plan", str(blocked_plan), "--out", str(ledger)]), 0)

            code = run_cli(["agentic-loop", "governance", "--ledger", str(ledger), "--action", "reject", "--out", receipt_arg])

            self.assertEqual(code, 0)
            payload = json.loads(receipt.read_text(encoding="utf-8"))
            self.assertEqual(payload["receipt_path"], "<redacted:governance_reject.json>")
            self.assertNotIn("..", payload["receipt_path"])
            self.assertEqual(run_cli(["validate", "--agentic-loop-governance-receipt", str(receipt), "--strict"]), 0)
            self.assertEqual(run_cli(["schemas", "--check", str(receipt)]), 0)
            receipt.unlink()

    def test_governance_receipt_redacts_traversal_missing_ledger_path(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            root = Path(tmp)
            receipt = root / "missing_ledger_governance.json"

            code = run_cli(["agentic-loop", "governance", "--ledger", "../missing_ledger.json", "--action", "reject", "--out", str(receipt)])

            self.assertEqual(code, 1)
            payload = json.loads(receipt.read_text(encoding="utf-8"))
            self.assertEqual(payload["source_ledger"]["path"], "<redacted:missing_ledger.json>")
            self.assertFalse(payload["source_ledger"]["exists"])
            self.assertNotIn("..", payload["source_ledger"]["path"])
            self.assertEqual(run_cli(["schemas", "--check", str(receipt)]), 0)

    def test_governance_receipt_redacts_external_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            external_root = Path(tmp)
            blocked_plan = self.write_loop_plan(external_root / "plans" / "blocked.json", "loop-001", {})
            ledger = external_root / "ledger.json"
            receipt = external_root / "governance_reject.json"
            self.assertEqual(run_cli(["agentic-loop", "ledger", "--plan", str(blocked_plan), "--out", str(ledger)]), 0)

            self.assertEqual(run_cli(["agentic-loop", "governance", "--ledger", str(ledger), "--action", "reject", "--out", str(receipt)]), 0)

            payload = json.loads(receipt.read_text(encoding="utf-8"))
            self.assertEqual(payload["receipt_path"], "<redacted:governance_reject.json>")
            self.assertEqual(payload["source_ledger"]["path"], "ledger.json")
            self.assertNotIn("..", payload["source_ledger"]["path"])
            self.assertNotIn(str(external_root), json.dumps(payload, sort_keys=True))

    def test_governance_receipt_blocks_unreplayable_external_source(self):
        with tempfile.TemporaryDirectory() as source_tmp, tempfile.TemporaryDirectory(dir=Path.cwd()) as out_tmp:
            source_root = Path(source_tmp)
            out_root = Path(out_tmp)
            blocked_plan = self.write_loop_plan(source_root / "plans" / "blocked.json", "loop-001", {})
            ledger = source_root / "ledger.json"
            receipt = out_root / "governance_reject.json"
            self.assertEqual(run_cli(["agentic-loop", "ledger", "--plan", str(blocked_plan), "--out", str(ledger)]), 0)

            code = run_cli(["agentic-loop", "governance", "--ledger", str(ledger), "--action", "reject", "--out", str(receipt)])

            self.assertEqual(code, 1)
            payload = json.loads(receipt.read_text(encoding="utf-8"))
            self.assertFalse(payload["passed"])
            self.assertEqual(payload["source_ledger"]["path"], "<redacted:ledger.json>")
            self.assertFalse(payload["source_ledger"]["exists"])
            summary = out_root / "summary.json"
            self.assertEqual(
                run_cli(["validate", "--agentic-loop-governance-receipt", str(receipt), "--strict", "--out", str(summary)]),
                1,
            )
            errors = "\n".join(error for target in json.loads(summary.read_text(encoding="utf-8"))["targets"] for error in target["errors"])
            self.assertIn("agentic_loop_governance_receipt.source_ledger must resolve to a replayable source ledger.", errors)

    def test_validate_rejects_stale_governance_receipt_ledger_hash(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            root = Path(tmp)
            blocked_plan = self.write_loop_plan(root / "plans" / "blocked.json", "loop-001", {})
            ledger = root / "ledger.json"
            receipt = root / "governance_reject.json"
            summary = root / "summary.json"
            self.assertEqual(run_cli(["agentic-loop", "ledger", "--plan", str(blocked_plan), "--out", str(ledger)]), 0)
            self.assertEqual(run_cli(["agentic-loop", "governance", "--ledger", str(ledger), "--action", "reject", "--out", str(receipt)]), 0)
            ledger.write_text(ledger.read_text(encoding="utf-8") + "\n", encoding="utf-8")

            code = run_cli(["validate", "--agentic-loop-governance-receipt", str(receipt), "--out", str(summary)])

            self.assertEqual(code, 1)
            errors = "\n".join(error for target in json.loads(summary.read_text(encoding="utf-8"))["targets"] for error in target["errors"])
            self.assertIn("agentic_loop_governance_receipt.source_ledger.size_bytes does not match the current file.", errors)
            self.assertIn("agentic_loop_governance_receipt.source_ledger.sha256 does not match the current file.", errors)

    def test_validate_rejects_parent_symlink_governance_source_ledger(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            root = Path(tmp)
            source = root / "source"
            source.mkdir()
            blocked_plan = self.write_loop_plan(source / "plans" / "blocked.json", "loop-001", {})
            ledger = source / "ledger.json"
            receipt = root / "governance_reject.json"
            summary = root / "summary.json"
            self.assertEqual(run_cli(["agentic-loop", "ledger", "--plan", str(blocked_plan), "--out", str(ledger)]), 0)
            self.assertEqual(run_cli(["agentic-loop", "governance", "--ledger", str(ledger), "--action", "reject", "--out", str(receipt)]), 0)
            linked_source = root / "linked_source"
            linked_source.symlink_to(source, target_is_directory=True)
            payload = json.loads(receipt.read_text(encoding="utf-8"))
            payload["source_ledger"]["path"] = str(Path("linked_source") / "ledger.json")
            receipt.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            code = run_cli(["validate", "--agentic-loop-governance-receipt", str(receipt), "--out", str(summary)])

            self.assertEqual(code, 1)
            errors = "\n".join(error for target in json.loads(summary.read_text(encoding="utf-8"))["targets"] for error in target["errors"])
            self.assertIn(
                "agentic_loop_governance_receipt.source_ledger.path must resolve to a regular non-symlink source ledger.",
                errors,
            )

    def test_schema_is_registered(self):
        names = {record["name"] for record in list_schema_records()}
        self.assertIn("agentic_loop_ledger", names)
        self.assertIn("agentic_loop_governance_receipt", names)

    def write_loop_plan(self, path: Path, iteration_id: str, artifacts: dict[str, list[Path]]) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        plan = build_agentic_training_loop_plan(
            out_path=path,
            iteration_id=iteration_id,
            objective=f"Iteration {iteration_id}",
            artifact_paths=artifacts,
            budget={"max_cloud_cost_usd": 0, "max_gpu_hours": 0},
            created_at="2026-07-03T00:00:00+00:00",
        )
        write_agentic_training_loop_plan(path, plan)
        return path

    def write_ready_artifacts(self, root: Path) -> dict[str, list[Path]]:
        root.mkdir(parents=True, exist_ok=True)
        artifacts = copy_valid_loop_artifacts(root)
        artifacts.pop("promotion_rollback_receipt", None)
        return artifacts

    def write_json(self, path: Path, schema_version: str, source_artifacts: dict[str, dict[str, object]] | None = None) -> Path:
        payload = {
            "schema_version": schema_version,
            "passed": True,
            "readiness": "ready",
            "source_artifacts": source_artifacts or {},
        }
        if schema_version == "hfr.cloud_training_launch_receipt.v1":
            payload.update(
                {
                    "readiness": "dry_run_recorded",
                    "recommendation": "safe_to_archive_dry_run_receipt",
                    "launch": {
                        "mode": "dry_run",
                        "cloud_job_started": False,
                        "provider_job_id": None,
                        "provider_api_called": False,
                        "cost_incurred_usd": 0,
                    },
                    "execution_boundary": {
                        "live_requested": False,
                        "allow_live": False,
                        "credential_values_recorded": False,
                    },
                }
            )
        if schema_version == "hfr.cloud_training_status_receipt.v1":
            payload.update(
                {
                    "readiness": "status_recorded",
                    "recommendation": "archive_status_receipt",
                    "status": {
                        "provider_status": "not_started",
                        "terminal": True,
                        "cancel_requested": False,
                        "provider_cancel_called": False,
                        "provider_api_called": False,
                        "cost_incurred_usd": 0,
                    },
                    "execution_boundary": {"credential_values_recorded": False},
                }
            )
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return path



if __name__ == "__main__":
    unittest.main()
