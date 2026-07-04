import hashlib
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from flightrecorder.agentic_training_loop_plan import build_agentic_training_loop_plan, write_agentic_training_loop_plan
from flightrecorder.cli import main
from flightrecorder.schema_registry import list_schema_records


def run_cli(args):
    with redirect_stdout(StringIO()):
        return main(args)


class AgenticLoopLedgerTests(unittest.TestCase):
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
            digest = payload["readiness_digest"]
            self.assertEqual(digest["latest_iteration_id"], "loop-002")
            self.assertTrue(digest["ready_for_governance_review"])
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
            self.assertIn("model_grader_override_receipt", role_names)
            self.assertTrue(ready_record["cloud_training"]["status_receipt_present"])
            self.assertTrue(ready_record["cloud_training_receipt_state"]["fail_closed"])
            self.assertEqual(ready_record["cloud_training_receipt_state"]["launch_mode"], "dry_run")
            self.assertEqual(ready_record["cloud_training_receipt_state"]["status_provider_status"], "not_started")
            self.assertTrue(ready_record["cloud_training_lineage"]["passed"])
            self.assertEqual(ready_record["cloud_training_lineage"]["provider"]["pipeline_provider_id"], "modal")
            self.assertFalse(ready_record["cloud_training"]["provider_api_calls_started"])
            self.assertIn("external_eval_receipt", ready_record["evals"]["roles_present"])
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
            self.assertEqual(run_cli(["validate", "--agentic-loop-ledger", str(ledger), "--strict"]), 0)
            self.assertEqual(run_cli(["schemas", "--check", str(ledger)]), 0)

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
            ledger.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            code = run_cli(["validate", "--agentic-loop-ledger", str(ledger), "--out", str(summary)])

            self.assertEqual(code, 1)
            errors = "\n".join(error for target in json.loads(summary.read_text(encoding="utf-8"))["targets"] for error in target["errors"])
            self.assertIn("cloud_training.artifact_count must match ledger cloud training role counts", errors)
            self.assertIn("cloud_training.cloud_jobs_started must match ledger cloud training role counts", errors)
            self.assertIn("cloud_training_receipt_state.fail_closed must match source loop plan cloud training receipt artifacts", errors)
            self.assertIn("cloud_training_lineage must match the source loop plan cloud_training_lineage", errors)

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

    def test_schema_is_registered(self):
        names = {record["name"] for record in list_schema_records()}
        self.assertIn("agentic_loop_ledger", names)

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
        training_export = root / "training_export"
        training_export.mkdir()
        agentic_training_plan = self.write_json(root / "agentic_training_plan.json", "hfr.agentic_training_plan.v1")
        trainer_preflight = self.write_json(root / "trainer_preflight.json", "hfr.trainer_preflight.v1")
        trainer_launch_check = self.write_json(root / "trainer_launch_check.json", "hfr.trainer_launch_check.v1")
        cloud_artifact_manifest = self.write_provider_json(
            root / "cloud_training_artifact_manifest.json",
            "hfr.cloud_training_artifact_manifest.v1",
        )
        cloud_preflight = self.write_provider_json(
            root / "cloud_training_preflight.json",
            "hfr.cloud_training_preflight.v1",
            {
                "agentic_training_plan": self.artifact_ref(agentic_training_plan, "agentic_training_plan"),
                "trainer_preflight": self.artifact_ref(trainer_preflight, "trainer_preflight"),
                "trainer_launch_check": self.artifact_ref(trainer_launch_check, "trainer_launch_check"),
            },
        )
        cloud_launch_plan = self.write_provider_json(
            root / "cloud_training_launch_plan.json",
            "hfr.cloud_training_launch_plan.v1",
            {
                "preflight": self.artifact_ref(cloud_preflight, "cloud_training_preflight"),
                "artifact_manifest": self.artifact_ref(cloud_artifact_manifest, "cloud_training_artifact_manifest"),
            },
        )
        cloud_launch_receipt = self.write_json(
            root / "cloud_training_launch_receipt.json",
            "hfr.cloud_training_launch_receipt.v1",
            {"launch_plan": self.artifact_ref(cloud_launch_plan, "cloud_training_launch_plan")},
        )
        cloud_status_receipt = self.write_json(
            root / "cloud_training_status_receipt.json",
            "hfr.cloud_training_status_receipt.v1",
            {"launch_receipt": self.artifact_ref(cloud_launch_receipt, "cloud_training_launch_receipt")},
        )
        return {
            "agentic_rollout_plan": [self.write_json(root / "agentic_rollout_plan.json", "hfr.agentic_rollout_plan.v1")],
            "agentic_rollout_receipt": [self.write_json(root / "agentic_rollout_receipt.json", "hfr.agentic_rollout_receipt.v1")],
            "harness_result": [self.write_json(root / "harness_result.json", "hfr.harness_run_result.v1")],
            "evidence_bundle": [self.write_json(root / "evidence_bundle.json", "hfr.evidence_bundle.v1")],
            "rubric_spec": [self.write_json(root / "rubric_spec.json", "hfr.rubric_spec.v1")],
            "model_grader_dry_run": [self.write_json(root / "model_grader_dry_run.json", "hfr.model_grader_dry_run.v1")],
            "model_grader_override_receipt": [
                self.write_json(root / "model_grader_override_receipt.json", "hfr.model_grader_override_receipt.v1")
            ],
            "model_grader_gate": [self.write_json(root / "model_grader_gate.json", "hfr.model_grader_gate.v1")],
            "review_calibration": [self.write_json(root / "review_calibration.json", "hfr.review_calibration.v1")],
            "reviewed_gate": [self.write_json(root / "reviewed_gate.json", "hfr.reviewed_gate.v1")],
            "rejection_sampling_gate": [self.write_json(root / "rejection_sampling_gate.json", "hfr.rejection_sampling_gate.v1")],
            "dataset_curation_receipt": [self.write_json(root / "dataset_curation_receipt.json", "hfr.dataset_curation_receipt.v1")],
            "training_export": [training_export],
            "agentic_training_plan": [agentic_training_plan],
            "agentic_training_flow": [self.write_json(root / "agentic_training_flow.json", "hfr.agentic_training_flow.v1")],
            "cloud_training_provider_registry": [
                self.write_provider_registry(root / "cloud_training_provider_registry.json")
            ],
            "cloud_training_preflight": [cloud_preflight],
            "cloud_training_artifact_manifest": [cloud_artifact_manifest],
            "cloud_training_launch_plan": [cloud_launch_plan],
            "cloud_training_launch_receipt": [cloud_launch_receipt],
            "cloud_training_status_receipt": [cloud_status_receipt],
            "trainer_preflight": [trainer_preflight],
            "trainer_launch_check": [trainer_launch_check],
            "serving_lifecycle": [self.write_json(root / "serving_lifecycle.json", "hfr.serving_lifecycle.v1")],
            "heldout_manifest": [self.write_json(root / "heldout_manifest.json", "hfr.heldout_scenario_manifest.v1")],
            "external_eval_plan": [self.write_json(root / "external_eval_plan.json", "hfr.external_eval_adapters.v1")],
            "external_eval_receipt": [self.write_json(root / "external_eval_receipt.json", "hfr.external_eval_receipt.v1")],
            "eval_summary": [self.write_json(root / "eval_summary.json", "hfr.eval_summary.v1")],
            "improvement_plan": [self.write_json(root / "improvement_plan.json", "hfr.improvement_plan.v1")],
            "promotion_decision": [self.write_json(root / "promotion_decision.json", "hfr.promotion_decision.v1")],
            "promotion_ledger": [self.write_json(root / "promotion_ledger.json", "hfr.promotion_ledger.v1")],
            "next_iteration_schedule": [self.write_json(root / "next_iteration_schedule.json", "hfr.next_iteration_schedule.v1")],
            "action_ledger": [self.write_json(root / "action_ledger.json", "hfr.action_ledger.v1")],
        }

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

    def write_provider_registry(self, path: Path) -> Path:
        path.write_text(
            json.dumps(
                {
                    "schema_version": "hfr.cloud_training_provider_registry.v1",
                    "passed": True,
                    "readiness": "ready",
                    "providers": [{"id": "modal"}],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return path

    def write_provider_json(
        self,
        path: Path,
        schema_version: str,
        source_artifacts: dict[str, dict[str, object]] | None = None,
    ) -> Path:
        path.write_text(
            json.dumps(
                {
                    "schema_version": schema_version,
                    "passed": True,
                    "readiness": "ready",
                    "provider": {"id": "modal"},
                    "source_artifacts": source_artifacts or {},
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return path

    def artifact_ref(self, path: Path, role: str) -> dict[str, object]:
        return {
            "role": role,
            "path": path.name,
            "exists": True,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size_bytes": path.stat().st_size,
        }


if __name__ == "__main__":
    unittest.main()
