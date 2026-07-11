import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import flightrecorder.agentic_training_loop_plan as loop_plan_module
from flightrecorder.agentic_training_loop_plan import build_agentic_training_loop_plan
from flightrecorder.agentic_training_result import build_agentic_training_result
from flightrecorder.external_eval import (
    build_external_eval_plan,
    build_external_eval_receipt,
    write_external_eval_plan,
    write_external_eval_receipt,
)
from flightrecorder.external_eval_result import build_external_eval_result, write_external_eval_result
from flightrecorder.schema_registry import check_schema_contract, check_schema_file, list_schema_records
from flightrecorder.validation import validate_artifacts
from tests.agentic_loop_fixtures import copy_valid_loop_artifacts


ROOT = Path(__file__).resolve().parents[1]


def loop_candidate_model_id(artifacts: dict[str, list[Path]]) -> str | None:
    rows = artifacts.get("agentic_training_result")
    if not isinstance(rows, list) or len(rows) != 1:
        return None
    try:
        payload = json.loads(rows[0].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    registry_update = payload.get("registry_update") if isinstance(payload.get("registry_update"), dict) else {}
    value = registry_update.get("target_model_id")
    return value if isinstance(value, str) and value else None


def attach_cloud_training_completion(artifacts: dict[str, list[Path]]) -> None:
    existing = artifacts.get("cloud_training_completion_receipt")
    if isinstance(existing, list) and len(existing) == 1:
        return
    rows = artifacts.get("agentic_training_result")
    if isinstance(rows, list) and len(rows) == 1:
        parent = rows[0].parent
        for name in (
            "promotion_cloud_training_completion_receipt.json",
            "cloud_training_completion_receipt.json",
        ):
            candidate = parent / name
            if candidate.is_file():
                artifacts["cloud_training_completion_receipt"] = [candidate]
                return


def directory_tree_fingerprint(path: Path) -> dict[str, int | str]:
    digest = hashlib.sha256()
    file_count = 0
    size_bytes = 0
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file() and not candidate.is_symlink()):
        relative = item.relative_to(path)
        size = item.stat().st_size
        digest.update(str(relative).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(item.read_bytes()).hexdigest().encode("ascii"))
        digest.update(b"\0")
        file_count += 1
        size_bytes += size
    return {"sha256": digest.hexdigest(), "file_count": file_count, "size_bytes": size_bytes}


def write_external_result_fixture(
    artifacts: dict[str, list[Path]],
    *,
    execution_status: str,
    benchmark_passed: bool = True,
) -> tuple[Path, dict[str, object]]:
    result_dir = artifacts["external_eval_plan"][0].parent.resolve()
    heldout_path = artifacts["heldout_manifest"][0].resolve()
    heldout = json.loads(heldout_path.read_text(encoding="utf-8"))
    scenario_ids = heldout["scenario_ids"]
    if execution_status == "completed":
        runs = [
            {
                "scenario_id": scenario_id,
                "passed": benchmark_passed,
                "score": 100 if benchmark_passed else 0,
            }
            for scenario_id in scenario_ids
        ]
        raw_format = "hfr.run_suite.v1"
        normalizer_id = "hfr.local_mock.run_suite"
        raw = {
            "schema_version": "hfr.run_suite.v1",
            "scenarios_dir": "scenarios",
            "out_dir": "runs",
            "total": len(runs),
            "passed": sum(row["passed"] is True for row in runs),
            "failed": sum(row["passed"] is False for row in runs),
            "error_count": 0,
            "errors": [],
            "metrics": {},
            "runs": runs,
            "artifacts": {},
        }
        failure_class = "none"
        failure_message = ""
    else:
        raw_format = "aggregate_json"
        normalizer_id = "hfr.local_mock.aggregate_json"
        raw = {"error": "external runner exited before emitting per-case results"}
        failure_class = "runner_error"
        failure_message = "External runner exited before emitting per-case results."
    raw_path = result_dir / f"loop-{execution_status}-raw-result.json"
    raw_path.write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    out_path = result_dir / f"loop-{execution_status}-external-eval-result.json"
    result = build_external_eval_result(
        plan_path=artifacts["external_eval_plan"][0].resolve(),
        heldout_manifest_path=heldout_path,
        raw_result_path=raw_path.resolve(),
        runner_metadata_path=None,
        runner_observation={
            "runner_id": "loop-test-runner",
            "runner_version": "1",
            "started_at": "2026-07-03T00:00:00+00:00",
            "finished_at": "2026-07-03T00:01:00+00:00",
            "exit_code": 0 if execution_status == "completed" else 17,
            "cost": {"reported": True, "amount": 0, "currency": "USD"},
            "side_effects": {
                "network_access": "not_observed",
                "provider_api_calls": "not_observed",
                "model_downloads": "not_observed",
                "filesystem_writes": "observed",
                "credential_values_recorded": "not_observed",
            },
        },
        adapter_id="local_mock",
        execution_id=f"loop-{execution_status}",
        model_id="local/mock-candidate",
        normalizer_id=normalizer_id,
        normalizer_version="1",
        raw_format=raw_format,
        execution_status=execution_status,
        failure_class=failure_class,
        failure_message=failure_message,
        out_path=out_path,
        created_at="2026-07-03T00:02:00+00:00",
    )
    write_external_eval_result(result, out_path)
    return out_path, result


class AgenticTrainingLoopPlanTests(unittest.TestCase):
    def test_cloud_training_completion_state_requires_bound_success_evidence(self):
        launch_plan_sha = "1" * 64
        launch_receipt_sha = "2" * 64
        status_receipt_sha = "3" * 64
        training_result_sha = "4" * 64
        payload = {
            "schema_version": "hfr.cloud_training_completion_receipt.v1",
            "passed": True,
            "execution": {"status": "completed", "terminal": True},
            "identity": {
                "provider_id": "local_mock",
                "provider_job_id": "job-001",
                "execution_id": "execution-001",
                "candidate_model_id": "local/output-candidate",
                "output_artifact_manifest_sha256": training_result_sha,
                "output_artifact_set_sha256": "5" * 64,
            },
            "sources": {
                "launch_plan": {"sha256": launch_plan_sha},
                "launch_receipt": {"sha256": launch_receipt_sha},
                "status_receipt": {"sha256": status_receipt_sha},
                "output_artifact_manifest": {"sha256": training_result_sha},
            },
            "outputs": {
                "manifest_sha256": training_result_sha,
                "artifact_set_sha256": "5" * 64,
                "artifact_count": 3,
                "regular_artifact_count": 3,
                "output_artifact_count": 1,
                "candidate_model_id": "local/output-candidate",
            },
            "governance": {
                "readiness": "ready_for_review",
                "cloud_training_completion_claims_allowed": True,
            },
        }
        source = {
            "physical_exists": True,
            "regular_file": True,
            "parse_valid": True,
            "schema_valid": True,
            "semantic_valid": True,
            "stable": True,
            "ready": True,
            "payload": payload,
        }
        refs = {
            "cloud_training_launch_plan": [{"sha256": launch_plan_sha}],
            "cloud_training_launch_receipt": [{"sha256": launch_receipt_sha}],
            "cloud_training_status_receipt": [{"sha256": status_receipt_sha}],
            "agentic_training_result": [{"sha256": training_result_sha}],
        }
        with (
            patch.object(loop_plan_module, "inspect_artifact_source", return_value=source),
            patch.object(
                loop_plan_module,
                "_cloud_training_provider_lineage",
                return_value={"pipeline_provider_id": "local_mock"},
            ),
            patch.object(
                loop_plan_module,
                "_single_payload",
                return_value={"registry_update": {"target_model_id": "local/output-candidate"}},
            ) as training_result_payload,
        ):
            state = loop_plan_module._cloud_training_completion_state(
                {"cloud_training_completion_receipt": [Path("completion.json")]},
                refs,
                loop_candidate_model_id="local/output-candidate",
            )
            training_result_payload.return_value = {
                "registry_update": {"target_model_id": "local/different-output-candidate"}
            }
            mismatched_candidate_state = loop_plan_module._cloud_training_completion_state(
                {"cloud_training_completion_receipt": [Path("completion.json")]},
                refs,
                loop_candidate_model_id="local/output-candidate",
            )
            training_result_payload.return_value = {
                "registry_update": {"target_model_id": "local/output-candidate"}
            }
            payload["identity"]["output_artifact_set_sha256"] = "8" * 64
            mismatched_output_set_state = loop_plan_module._cloud_training_completion_state(
                {"cloud_training_completion_receipt": [Path("completion.json")]},
                refs,
                loop_candidate_model_id="local/output-candidate",
            )
            payload["identity"]["output_artifact_set_sha256"] = "5" * 64
            mismatched_loop_candidate_state = loop_plan_module._cloud_training_completion_state(
                {"cloud_training_completion_receipt": [Path("completion.json")]},
                refs,
                loop_candidate_model_id="local/different-loop-candidate",
            )

        self.assertTrue(state["integrity_passed"])
        self.assertTrue(state["successful"])
        self.assertTrue(state["source_bindings_complete"])
        self.assertTrue(state["candidate_matches_loop"])
        self.assertTrue(state["candidate_matches_training_result"])
        self.assertTrue(state["output_artifact_manifest_bound"])
        self.assertTrue(state["output_artifact_set_bound"])
        self.assertEqual(loop_plan_module._cloud_training_completion_status(state), "completed")
        self.assertFalse(mismatched_candidate_state["candidate_matches_training_result"])
        self.assertFalse(mismatched_candidate_state["successful"])
        self.assertFalse(mismatched_output_set_state["output_artifact_set_bound"])
        self.assertFalse(mismatched_output_set_state["successful"])
        self.assertFalse(mismatched_loop_candidate_state["candidate_matches_loop"])
        self.assertFalse(mismatched_loop_candidate_state["successful"])

    def test_cloud_training_completion_outcomes_fail_closed(self):
        for status, expected in (("failed", "failed"), ("incomplete", "incomplete"), ("unknown", "incomplete")):
            with self.subTest(status=status):
                state = {
                    "integrity_passed": True,
                    "execution_status": status,
                    "successful": False,
                }
                self.assertEqual(loop_plan_module._cloud_training_completion_status(state), expected)
        self.assertEqual(
            loop_plan_module._cloud_training_completion_status(
                {"integrity_passed": False, "execution_status": "failed", "successful": False}
            ),
            "incomplete",
        )

    def test_cloud_training_completion_lineage_reads_top_level_sources(self):
        completion_sha = "6" * 64
        launch_plan_sha = "7" * 64
        spec = next(
            row
            for row in loop_plan_module.CLOUD_TRAINING_LINEAGE_LINKS
            if row["id"] == "completion_receipt_links_launch_plan"
        )
        refs = {
            "cloud_training_completion_receipt": [
                {"sha256": completion_sha, "schema_version": "hfr.cloud_training_completion_receipt.v1"}
            ],
            "cloud_training_launch_plan": [
                {"sha256": launch_plan_sha, "schema_version": "hfr.cloud_training_launch_plan.v1"}
            ],
        }
        payload = {"sources": {"launch_plan": {"sha256": launch_plan_sha}}}

        with patch.object(loop_plan_module, "_first_payload", return_value=payload):
            link = loop_plan_module._cloud_training_lineage_link(refs, {}, spec)

        self.assertTrue(link["passed"])
        self.assertEqual(link["source_ref_sha256"], launch_plan_sha)

    def test_completion_evidence_does_not_change_handoff_lineage(self):
        def matched_link(_refs, _paths, spec):
            return {"id": spec["id"], "status": "matched", "passed": True}

        provider = {
            "provider_consistent": True,
            "registry_contains_pipeline_provider": True,
        }
        refs = {"agentic_training_result": [{}, {}]}
        with (
            patch.object(loop_plan_module, "_cloud_training_provider_lineage", return_value=provider),
            patch.object(loop_plan_module, "_cloud_training_lineage_link", side_effect=matched_link),
        ):
            pre_execution = loop_plan_module._cloud_training_lineage(refs, {})
            with_completion = loop_plan_module._cloud_training_lineage(
                {**refs, "cloud_training_completion_receipt": [{}]},
                {},
            )

        self.assertNotIn("agentic_training_result", pre_execution["duplicate_roles"])
        self.assertTrue(pre_execution["passed"])
        self.assertEqual(with_completion, pre_execution)

    def test_missing_completion_receipt_preserves_plan_readiness_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = self.write_loop_artifacts(root)
            artifacts.pop("cloud_training_completion_receipt", None)

            plan = build_agentic_training_loop_plan(
                out_path=root / "loop.json",
                iteration_id="completion-is-post-execution",
                artifact_paths=artifacts,
                created_at="2026-07-03T00:00:00+00:00",
            )

            self.assertEqual(plan["plan_readiness"], "ready_to_execute")
            self.assertEqual(plan["execution_completion"], "incomplete")
            self.assertEqual(plan["governance_readiness"], "blocked")
            self.assertFalse(plan["cloud_training_completion_state"]["successful"])
            self.assertIn("cloud_training_completion_receipt", plan["missing_phase_inputs"])
            self.assertFalse(
                any(
                    link["id"].startswith("completion_receipt_links_")
                    for link in plan["cloud_training_lineage"]["links"]
                )
            )
            completion_check = next(
                check for check in plan["checks"] if check["id"] == "external_cloud_training_completion_imported"
            )
            self.assertFalse(completion_check["passed"])
            self.assertTrue(check_schema_contract(plan, name_or_id="agentic_training_loop_plan")["passed"])

    def test_imported_completion_receipt_completes_external_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = self.write_loop_artifacts(root)
            attach_cloud_training_completion(artifacts)

            plan = build_agentic_training_loop_plan(
                out_path=root / "loop.json",
                iteration_id="completion-imported",
                candidate=loop_candidate_model_id(artifacts),
                artifact_paths=artifacts,
                created_at="2026-07-10T00:00:00+00:00",
            )

            state = plan["cloud_training_completion_state"]
            self.assertEqual(plan["plan_readiness"], "ready_to_execute")
            self.assertEqual(plan["execution_completion"], "completed")
            self.assertTrue(state["integrity_passed"])
            self.assertTrue(state["successful"])
            self.assertTrue(state["candidate_matches_training_result"])
            self.assertTrue(state["source_bindings_complete"])
            self.assertTrue(state["output_artifact_manifest_bound"])
            self.assertTrue(state["output_artifact_set_bound"])
            self.assertFalse(
                any(
                    link["id"].startswith("completion_receipt_links_")
                    for link in plan["cloud_training_lineage"]["links"]
                )
            )
            self.assertTrue(check_schema_contract(plan, name_or_id="agentic_training_loop_plan")["passed"])

    def test_multi_adapter_external_results_require_an_exact_completed_set(self):
        def inspection(adapter_id: str, status: str = "completed") -> dict[str, object]:
            return {
                "physical_exists": True,
                "parse_valid": True,
                "schema_valid": True,
                "semantic_valid": True,
                "payload": {
                    "identity": {"adapter_id": adapter_id},
                    "integrity": {"passed": True},
                    "execution": {"status": status},
                    "coverage": {"complete": status == "completed"},
                    "benchmark_outcome": {"status": "passed" if status == "completed" else "not_available"},
                    "governance": {
                        "readiness": "ready_for_review" if status == "completed" else "blocked",
                        "external_eval_claims_allowed": status == "completed",
                    },
                },
            }

        plan = {"selected_adapters": ["bfcl", "inspect_ai"]}
        artifact_paths = {
            "external_eval_plan": [Path("plan.json")],
            "external_eval_result": [Path("bfcl.json"), Path("inspect.json")],
        }
        inspections = {
            Path("bfcl.json"): inspection("bfcl"),
            Path("inspect.json"): inspection("inspect_ai"),
        }
        with (
            patch.object(loop_plan_module, "_single_payload", return_value=plan),
            patch.object(
                loop_plan_module,
                "inspect_artifact_source",
                side_effect=lambda path, role: inspections[path],
            ),
        ):
            self.assertEqual(
                loop_plan_module._execution_result_status(artifact_paths, "external_eval_result"),
                "completed",
            )
            self.assertEqual(
                loop_plan_module._execution_result_status(
                    {**artifact_paths, "external_eval_result": [Path("bfcl.json")]},
                    "external_eval_result",
                ),
                "incomplete",
            )
            inspections[Path("bfcl.json")] = inspection("bfcl", "failed")
            self.assertEqual(
                loop_plan_module._execution_result_status(
                    {**artifact_paths, "external_eval_result": [Path("bfcl.json")]},
                    "external_eval_result",
                ),
                "failed",
            )

    def test_dry_run_external_eval_receipt_cannot_complete_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = self.write_loop_artifacts(root)
            result_path, _ = write_external_result_fixture(artifacts, execution_status="completed")
            artifacts["external_eval_result"] = [result_path]
            artifacts.pop("external_eval_result")

            plan = build_agentic_training_loop_plan(
                out_path=root / "loop.json",
                iteration_id="dry-run-is-not-completion",
                artifact_paths=artifacts,
                created_at="2026-07-03T00:00:00+00:00",
            )

            self.assertEqual(plan["plan_readiness"], "ready_to_execute")
            self.assertEqual(plan["execution_completion"], "incomplete")
            self.assertEqual(plan["governance_readiness"], "blocked")
            self.assertEqual(plan["recommendation"], "execute_ready_plan")
            self.assertFalse(plan["passed"])
            self.assertEqual(plan["readiness"], "planned_fail_closed")
            self.assertIn("external_eval_result", plan["missing_phase_inputs"])
            heldout = next(phase for phase in plan["phases"] if phase["id"] == "heldout_eval")
            self.assertEqual(heldout["status"], "blocked")
            self.assertIn("external_eval_result", heldout["missing_required_artifacts"])
            heldout_check = next(check for check in plan["checks"] if check["id"] == "heldout_eval_is_fail_closed")
            self.assertFalse(heldout_check["passed"])
            self.assertFalse(heldout_check["actual"]["external_eval_result_completed"])

    def test_training_plan_and_dry_run_receipts_cannot_complete_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = self.write_loop_artifacts(root)
            artifacts.pop("agentic_training_result")

            plan = build_agentic_training_loop_plan(
                out_path=root / "loop.json",
                iteration_id="training-plan-is-not-completion",
                artifact_paths=artifacts,
                created_at="2026-07-03T00:00:00+00:00",
            )

            self.assertEqual(plan["execution_completion"], "incomplete")
            self.assertEqual(plan["recommendation"], "execute_ready_plan")
            training = next(phase for phase in plan["phases"] if phase["id"] == "external_trainer_execution")
            self.assertEqual(training["status"], "blocked")
            self.assertIn("agentic_training_result", training["missing_required_artifacts"])
            execution_check = next(
                check for check in plan["checks"] if check["id"] == "external_trainer_execution_completed"
            )
            self.assertFalse(execution_check["passed"])
            self.assertEqual(execution_check["actual"]["training_result_status"], "missing")

    def test_duplicate_external_eval_handoff_inputs_block_plan_readiness(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = self.write_loop_artifacts(root)
            artifacts["external_eval_plan"].append(artifacts["external_eval_plan"][0])

            plan = build_agentic_training_loop_plan(
                out_path=root / "loop.json",
                iteration_id="ambiguous-external-eval-plan",
                artifact_paths=artifacts,
                created_at="2026-07-03T00:00:00+00:00",
            )

            self.assertEqual(plan["plan_readiness"], "blocked")
            self.assertEqual(plan["governance_readiness"], "blocked")
            self.assertEqual(plan["recommendation"], "collect_missing_plan_evidence")
            handoff_check = next(
                check for check in plan["checks"] if check["id"] == "external_eval_handoff_is_preflighted"
            )
            self.assertFalse(handoff_check["passed"])
            self.assertEqual(handoff_check["actual"]["external_eval_plan_count"], 2)
            schema = check_schema_contract(plan)
            self.assertTrue(schema["passed"], schema["errors"])

    def test_external_eval_receipt_from_another_plan_blocks_plan_readiness(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = self.write_loop_artifacts(root)
            handoff_dir = artifacts["external_eval_plan"][0].parent.resolve()
            other_plan_path = handoff_dir / "other-external-eval-plan.json"
            other_plan = build_external_eval_plan(
                adapters=["local_mock"],
                scenario_manifest=artifacts["heldout_manifest"][0].resolve(),
                model_endpoint="local/other-candidate",
                model="local/other-candidate",
                allow_installed=True,
                output_base_dir=handoff_dir,
            )
            write_external_eval_plan(other_plan, other_plan_path)
            other_receipt_path = handoff_dir / "other-external-eval-receipt.json"
            other_receipt = build_external_eval_receipt(
                plan_path=other_plan_path,
                created_at="2026-07-03T00:00:00+00:00",
                output_base_dir=handoff_dir,
            )
            write_external_eval_receipt(other_receipt, other_receipt_path)
            artifacts["external_eval_receipt"] = [other_receipt_path]

            plan = build_agentic_training_loop_plan(
                out_path=root / "loop.json",
                iteration_id="mismatched-external-eval-receipt",
                artifact_paths=artifacts,
                created_at="2026-07-03T00:00:00+00:00",
            )

            self.assertEqual(plan["plan_readiness"], "blocked")
            handoff_check = next(
                check for check in plan["checks"] if check["id"] == "external_eval_handoff_is_preflighted"
            )
            self.assertFalse(handoff_check["passed"])
            self.assertTrue(handoff_check["actual"]["plan_heldout_manifest_bound"])
            self.assertFalse(handoff_check["actual"]["receipt_plan_bound"])
            self.assertEqual(handoff_check["actual"]["external_eval_handoff_status"], "mismatched")

    def test_duplicate_external_eval_results_cannot_complete_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = self.write_loop_artifacts(root)
            result_path, _ = write_external_result_fixture(artifacts, execution_status="completed")
            artifacts["external_eval_result"] = [result_path, result_path]

            plan = build_agentic_training_loop_plan(
                out_path=root / "loop.json",
                iteration_id="ambiguous-external-eval-result",
                artifact_paths=artifacts,
                created_at="2026-07-03T00:00:00+00:00",
            )

            self.assertEqual(plan["plan_readiness"], "ready_to_execute")
            self.assertEqual(plan["execution_completion"], "incomplete")
            self.assertEqual(plan["governance_readiness"], "blocked")
            heldout = next(phase for phase in plan["phases"] if phase["id"] == "heldout_eval")
            self.assertIn("external_eval_result", heldout["present_required_artifacts"])
            self.assertNotIn("external_eval_result", heldout["missing_required_artifacts"])
            self.assertEqual(heldout["non_completed_required_artifacts"], ["external_eval_result"])
            heldout_check = next(check for check in plan["checks"] if check["id"] == "heldout_eval_is_fail_closed")
            self.assertEqual(heldout_check["actual"]["external_eval_result_count"], 2)
            self.assertEqual(heldout_check["actual"]["external_eval_result_status"], "incomplete")

    def test_valid_external_eval_runner_failure_marks_execution_failed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = self.write_loop_artifacts(root)
            result_path, result = write_external_result_fixture(artifacts, execution_status="failed")
            self.assertTrue(result["integrity"]["passed"], result["integrity"])
            artifacts["external_eval_result"] = [result_path]

            plan = build_agentic_training_loop_plan(
                out_path=root / "loop.json",
                iteration_id="external-eval-runner-failed",
                artifact_paths=artifacts,
                created_at="2026-07-03T00:00:00+00:00",
            )

            self.assertEqual(plan["plan_readiness"], "ready_to_execute")
            self.assertEqual(plan["execution_completion"], "failed")
            self.assertEqual(plan["governance_readiness"], "blocked")
            self.assertEqual(plan["recommendation"], "investigate_failed_execution")
            heldout = next(phase for phase in plan["phases"] if phase["id"] == "heldout_eval")
            self.assertIn("external_eval_result", heldout["present_required_artifacts"])
            self.assertNotIn("external_eval_result", heldout["missing_required_artifacts"])
            self.assertEqual(heldout["non_completed_required_artifacts"], ["external_eval_result"])
            heldout_check = next(check for check in plan["checks"] if check["id"] == "heldout_eval_is_fail_closed")
            self.assertEqual(heldout_check["actual"]["external_eval_result_status"], "failed")

    def test_failed_benchmark_outcome_still_counts_as_completed_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = self.write_loop_artifacts(root)
            attach_cloud_training_completion(artifacts)
            result_path, result = write_external_result_fixture(
                artifacts,
                execution_status="completed",
                benchmark_passed=False,
            )
            self.assertTrue(result["integrity"]["passed"], result["integrity"])
            self.assertEqual(result["benchmark_outcome"]["status"], "failed")
            artifacts["external_eval_result"] = [result_path]

            plan = build_agentic_training_loop_plan(
                out_path=root / "loop.json",
                iteration_id="external-eval-benchmark-failed",
                candidate=loop_candidate_model_id(artifacts),
                artifact_paths=artifacts,
                created_at="2026-07-03T00:00:00+00:00",
            )

            self.assertEqual(plan["plan_readiness"], "ready_to_execute")
            self.assertEqual(plan["execution_completion"], "completed")
            self.assertEqual(plan["governance_readiness"], "blocked")
            self.assertEqual(plan["recommendation"], "resolve_governance_blockers")
            heldout = next(phase for phase in plan["phases"] if phase["id"] == "heldout_eval")
            self.assertNotIn("external_eval_result", heldout["non_completed_required_artifacts"])
            heldout_check = next(check for check in plan["checks"] if check["id"] == "heldout_eval_is_fail_closed")
            self.assertEqual(heldout_check["actual"]["external_eval_result_status"], "completed")
            self.assertFalse(heldout_check["actual"]["eval_summary_result_bound"])

    def test_valid_classified_training_failure_marks_execution_failed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = self.write_loop_artifacts(root)
            result_dir = artifacts["agentic_training_result"][0].parent
            failure_log = result_dir / "trainer-failure.log"
            failure_log.write_text("external trainer ran out of memory\n", encoding="utf-8")
            failed_result_path = result_dir / "failed-training-result.json"
            failed_result = build_agentic_training_result(
                plan_path=artifacts["agentic_training_plan"][0].resolve(),
                runtime_preflight_path=artifacts["agentic_training_runtime_preflight"][0].resolve(),
                agentic_training_flow_path=artifacts["agentic_training_flow"][0].resolve(),
                out_path=failed_result_path.resolve(),
                status="failed",
                failure_class="out_of_memory",
                failure_message="External trainer exhausted its assigned device memory.",
                artifacts={"log": [failure_log.resolve()]},
                created_at="2026-07-03T00:00:00+00:00",
            )
            self.assertTrue(failed_result["passed"], failed_result["blocked_reasons"])
            failed_result_path.write_text(json.dumps(failed_result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            artifacts["agentic_training_result"] = [failed_result_path.resolve()]

            plan = build_agentic_training_loop_plan(
                out_path=root / "loop.json",
                iteration_id="classified-training-failure",
                artifact_paths=artifacts,
                created_at="2026-07-03T00:00:00+00:00",
            )

            self.assertEqual(plan["plan_readiness"], "ready_to_execute")
            self.assertEqual(plan["execution_completion"], "failed")
            self.assertEqual(plan["governance_readiness"], "blocked")
            self.assertEqual(plan["recommendation"], "investigate_failed_execution")
            self.assertFalse(plan["passed"])
            training = next(phase for phase in plan["phases"] if phase["id"] == "external_trainer_execution")
            self.assertEqual(training["status"], "blocked")
            self.assertIn("agentic_training_result", training["present_required_artifacts"])
            self.assertNotIn("agentic_training_result", training["missing_required_artifacts"])
            self.assertEqual(
                training["non_completed_required_artifacts"],
                ["cloud_training_completion_receipt", "agentic_training_result"],
            )
            self.assertNotIn("agentic_training_result", plan["missing_phase_inputs"])
            execution_check = next(
                check for check in plan["checks"] if check["id"] == "external_trainer_execution_completed"
            )
            self.assertFalse(execution_check["passed"])
            self.assertEqual(execution_check["actual"]["training_result_status"], "failed")

    def test_validator_rejects_refreshed_hash_for_semantically_invalid_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = copy_valid_loop_artifacts(root)
            fixture_root = root / "loop_fixture" / "examples" / "agentic_training"
            loop_plan = fixture_root / "loop_plan.json"
            heldout_manifest = artifacts["heldout_manifest"][0]
            heldout_manifest.write_text("{}\n", encoding="utf-8")
            plan = json.loads(loop_plan.read_text(encoding="utf-8"))
            ref = plan["source_artifacts"]["heldout_manifest"][0]
            ref["size_bytes"] = heldout_manifest.stat().st_size
            ref["sha256"] = hashlib.sha256(heldout_manifest.read_bytes()).hexdigest()
            loop_plan.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            validation = validate_artifacts(agentic_training_loop_plan_paths=[loop_plan], strict=True)

            self.assertFalse(validation["passed"], validation)
            errors = "\n".join(error for target in validation["targets"] for error in target["errors"])
            self.assertIn(
                "referenced heldout_manifest artifact is not semantically ready",
                errors,
            )

    def test_phase_readiness_rejects_invalid_required_source_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = self.write_loop_artifacts(root)
            invalid = root / "invalid_rubric_spec.json"
            invalid.write_text('{"schema_version":"hfr.rubric_spec.v1"}\n', encoding="utf-8")
            artifacts["rubric_spec"] = [invalid]

            plan = build_agentic_training_loop_plan(
                out_path=root / "loop.json",
                iteration_id="invalid-required-source",
                artifact_paths=artifacts,
                created_at="2026-07-03T00:00:00+00:00",
            )

            phase = next(row for row in plan["phases"] if row["id"] == "rubric_model_grader_review")
            self.assertFalse(plan["passed"])
            self.assertEqual(plan["readiness"], "planned_fail_closed")
            self.assertEqual(phase["status"], "blocked")
            self.assertIn("rubric_spec", phase["missing_required_artifacts"])
            self.assertNotIn("rubric_spec", phase["present_required_artifacts"])

    def test_rollout_phase_rejects_schema_valid_failed_harness_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = self.write_loop_artifacts(root)
            result_path = artifacts["harness_result"][0]
            result = json.loads(result_path.read_text(encoding="utf-8"))
            result["scorecard"]["passed"] = False
            result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            plan = build_agentic_training_loop_plan(
                out_path=root / "loop.json",
                iteration_id="failed-harness-result",
                artifact_paths=artifacts,
                created_at="2026-07-03T00:00:00+00:00",
            )

            phase = next(row for row in plan["phases"] if row["id"] == "rollout_collection")
            self.assertEqual(plan["readiness"], "planned_fail_closed")
            self.assertEqual(phase["status"], "blocked")
            self.assertIn("harness_result", phase["missing_required_artifacts"])

    def test_committed_example_loop_plan_replays_fail_closed_sources(self):
        plan_path = ROOT / "examples" / "agentic_training" / "loop_plan.json"
        rollout_plan_path = ROOT / "examples" / "agentic_training" / "rollouts" / "rollout_plan.json"
        rollout_receipt_path = ROOT / "examples" / "agentic_training" / "rollouts" / "rollout_receipt.json"
        harness_result_path = ROOT / "examples" / "agentic_training" / "evidence_handoff" / "harness_handoff" / "harness_result.json"
        evidence_bundle_path = ROOT / "examples" / "agentic_training" / "evidence_handoff" / "evidence_bundle.json"
        reviewed_gate_path = ROOT / "examples" / "agentic_training" / "model_grader" / "reviewed_gate.json"
        rejection_sampling_gate_path = ROOT / "examples" / "agentic_training" / "rejection_sampling_gate.json"
        dataset_curation_receipt_path = ROOT / "examples" / "agentic_training" / "dataset_curation_receipt.json"
        trainer_preflight_path = (
            ROOT
            / "examples"
            / "agentic_training"
            / "cloud_training"
            / "sources"
            / "trainer_preflight.json"
        )
        trainer_launch_check_path = (
            ROOT
            / "examples"
            / "agentic_training"
            / "cloud_training"
            / "sources"
            / "trainer_launch_check.json"
        )
        provider_registry_path = ROOT / "examples" / "agentic_training" / "cloud_training" / "provider_registry.json"
        cloud_preflight_path = ROOT / "examples" / "agentic_training" / "cloud_training" / "preflight.json"
        cloud_launch_receipt_path = ROOT / "examples" / "agentic_training" / "cloud_training" / "launch_receipt.json"
        serving_lifecycle_path = ROOT / "examples" / "agentic_training" / "serving_lifecycle" / "managed_mock" / "serving_lifecycle.json"
        heldout_manifest_path = ROOT / "examples" / "agentic_training" / "heldout_eval" / "heldout_manifest.json"
        external_eval_plan_path = ROOT / "examples" / "agentic_training" / "heldout_eval" / "external_eval_plan.json"
        external_eval_receipt_path = ROOT / "examples" / "agentic_training" / "heldout_eval" / "external_eval_receipt.json"
        external_eval_result_path = ROOT / "examples" / "agentic_training" / "heldout_eval" / "external_eval_result.json"
        eval_summary_path = ROOT / "examples" / "agentic_training" / "heldout_eval" / "eval_summary.json"
        model_grader_gate_path = ROOT / "examples" / "agentic_training" / "model_grader" / "passing_gate.json"
        action_ledger_path = ROOT / "examples" / "agentic_training" / "iteration_ledgers" / "action_ledger.json"
        improvement_ledger_path = ROOT / "examples" / "agentic_training" / "iteration_ledgers" / "improvement_ledger.json"
        promotion_decision_path = ROOT / "examples" / "agentic_training" / "promotion_governance" / "promotion_decision.json"
        promotion_ledger_path = ROOT / "examples" / "agentic_training" / "promotion_governance" / "promotion_ledger.json"
        promotion_cards_path = ROOT / "examples" / "agentic_training" / "promotion_governance" / "promotion_cards"
        promotion_alias_apply_path = ROOT / "examples" / "agentic_training" / "promotion_governance" / "promotion_alias_apply.json"
        promotion_rollback_receipt_path = (
            ROOT / "examples" / "agentic_training" / "promotion_governance" / "promotion_rollback_receipt.json"
        )
        promotion_release_record_path = (
            ROOT / "examples" / "agentic_training" / "promotion_governance" / "promotion_release_record.json"
        )
        promotion_archive_path = ROOT / "examples" / "agentic_training" / "promotion_governance" / "promotion_archive"
        plan = json.loads(plan_path.read_text(encoding="utf-8"))

        expected_refs = {
            "agentic_rollout_plan": ("rollouts/rollout_plan.json", rollout_plan_path),
            "agentic_rollout_receipt": ("rollouts/rollout_receipt.json", rollout_receipt_path),
            "harness_result": ("evidence_handoff/harness_handoff/harness_result.json", harness_result_path),
            "evidence_bundle": ("evidence_handoff/evidence_bundle.json", evidence_bundle_path),
            "reviewed_gate": ("model_grader/reviewed_gate.json", reviewed_gate_path),
            "rejection_sampling_gate": ("rejection_sampling_gate.json", rejection_sampling_gate_path),
            "dataset_curation_receipt": ("dataset_curation_receipt.json", dataset_curation_receipt_path),
            "trainer_preflight": (
                "cloud_training/sources/trainer_preflight.json",
                trainer_preflight_path,
            ),
            "trainer_launch_check": (
                "cloud_training/sources/trainer_launch_check.json",
                trainer_launch_check_path,
            ),
            "cloud_training_provider_registry": ("cloud_training/provider_registry.json", provider_registry_path),
            "cloud_training_preflight": ("cloud_training/preflight.json", cloud_preflight_path),
            "cloud_training_launch_receipt": ("cloud_training/launch_receipt.json", cloud_launch_receipt_path),
            "serving_lifecycle": ("serving_lifecycle/managed_mock/serving_lifecycle.json", serving_lifecycle_path),
            "heldout_manifest": ("heldout_eval/heldout_manifest.json", heldout_manifest_path),
            "external_eval_plan": ("heldout_eval/external_eval_plan.json", external_eval_plan_path),
            "external_eval_receipt": ("heldout_eval/external_eval_receipt.json", external_eval_receipt_path),
            "external_eval_result": ("heldout_eval/external_eval_result.json", external_eval_result_path),
            "eval_summary": ("heldout_eval/eval_summary.json", eval_summary_path),
            "model_grader_gate": ("model_grader/passing_gate.json", model_grader_gate_path),
            "action_ledger": ("iteration_ledgers/action_ledger.json", action_ledger_path),
            "improvement_ledger": ("iteration_ledgers/improvement_ledger.json", improvement_ledger_path),
            "promotion_decision": ("promotion_governance/promotion_decision.json", promotion_decision_path),
            "promotion_ledger": ("promotion_governance/promotion_ledger.json", promotion_ledger_path),
            "promotion_cards": ("promotion_governance/promotion_cards", promotion_cards_path),
            "promotion_alias_apply": ("promotion_governance/promotion_alias_apply.json", promotion_alias_apply_path),
            "promotion_rollback_receipt": (
                "promotion_governance/promotion_rollback_receipt.json",
                promotion_rollback_receipt_path,
            ),
            "promotion_release_record": (
                "promotion_governance/promotion_release_record.json",
                promotion_release_record_path,
            ),
            "promotion_archive": ("promotion_governance/promotion_archive", promotion_archive_path),
        }
        for role, (expected_path, source_path) in expected_refs.items():
            ref = plan["source_artifacts"][role][0]
            self.assertEqual(ref["path"], expected_path)
            if source_path.is_dir():
                expected_tree = directory_tree_fingerprint(source_path)
                self.assertEqual(ref["kind"], "directory")
                self.assertTrue(ref["exists"])
                self.assertEqual(ref["file_count"], expected_tree["file_count"])
                self.assertEqual(ref["size_bytes"], expected_tree["size_bytes"])
                self.assertEqual(ref["sha256"], expected_tree["sha256"])
                self.assertFalse(ref["contains_symlinks"])
            else:
                self.assertIsNone(ref.get("file_count"))
                self.assertIsNone(ref.get("contains_symlinks"))
                self.assertEqual(ref["size_bytes"], source_path.stat().st_size)
                self.assertEqual(ref["sha256"], hashlib.sha256(source_path.read_bytes()).hexdigest())
        self.assertTrue(plan["passed"])
        self.assertEqual(plan["readiness"], "ready_for_governance_review")
        self.assertEqual(plan["artifact_count"], 42)
        self.assertEqual(plan["plan_readiness"], "ready_to_execute")
        self.assertEqual(plan["execution_completion"], "completed")
        self.assertEqual(plan["governance_readiness"], "ready_for_review")
        self.assertEqual(plan["missing_phase_inputs"], [])
        self.assertNotIn("agentic_rollout_plan", plan["missing_phase_inputs"])
        self.assertNotIn("agentic_rollout_receipt", plan["missing_phase_inputs"])
        self.assertNotIn("harness_result", plan["missing_phase_inputs"])
        self.assertNotIn("evidence_bundle", plan["missing_phase_inputs"])
        self.assertNotIn("reviewed_gate", plan["missing_phase_inputs"])
        self.assertNotIn("rejection_sampling_gate", plan["missing_phase_inputs"])
        self.assertNotIn("dataset_curation_receipt", plan["missing_phase_inputs"])
        self.assertNotIn("training_export", plan["missing_phase_inputs"])
        self.assertNotIn("trainer_preflight", plan["missing_phase_inputs"])
        self.assertNotIn("trainer_launch_check", plan["missing_phase_inputs"])
        self.assertNotIn("heldout_manifest", plan["missing_phase_inputs"])
        self.assertNotIn("external_eval_plan", plan["missing_phase_inputs"])
        self.assertNotIn("external_eval_receipt", plan["missing_phase_inputs"])
        self.assertNotIn("eval_summary", plan["missing_phase_inputs"])
        self.assertNotIn("serving_lifecycle", plan["missing_phase_inputs"])
        self.assertNotIn("promotion_decision", plan["missing_phase_inputs"])
        self.assertNotIn("promotion_ledger", plan["missing_phase_inputs"])
        self.assertEqual(
            {check["id"] for check in plan["checks"] if not check["passed"]},
            set(),
        )
        self.assertNotIn("rollout_receipt_required_before_review", {check["id"] for check in plan["checks"] if not check["passed"]})
        self.assertNotIn("uncalibrated_labels_block_training_data", {check["id"] for check in plan["checks"] if not check["passed"]})
        self.assertNotIn("dataset_curation_receipt_required_for_trainer_handoff", {check["id"] for check in plan["checks"] if not check["passed"]})
        self.assertNotIn("cloud_training_lineage_bound_for_provider_handoff", {check["id"] for check in plan["checks"] if not check["passed"]})
        training_export_ref = plan["source_artifacts"]["training_export"][0]
        self.assertEqual(training_export_ref["path"], "training_export")
        self.assertEqual(training_export_ref["kind"], "directory")
        self.assertTrue(training_export_ref["exists"])
        phases = {phase["id"]: phase for phase in plan["phases"]}
        self.assertEqual(
            set(phases["rollout_collection"]["present_required_artifacts"]),
            {"agentic_rollout_plan", "agentic_rollout_receipt", "harness_result"},
        )
        self.assertEqual(phases["rollout_collection"]["missing_required_artifacts"], [])
        self.assertEqual(phases["evidence_scoring"]["status"], "ready")
        self.assertEqual(phases["evidence_scoring"]["missing_required_artifacts"], [])
        self.assertEqual(phases["rubric_model_grader_review"]["status"], "ready")
        self.assertEqual(phases["rejection_sampling"]["status"], "ready")
        self.assertEqual(phases["dataset_curation"]["status"], "ready")
        self.assertEqual(
            set(phases["dataset_curation"]["present_required_artifacts"]),
            {"rejection_sampling_gate", "dataset_curation_receipt", "training_export"},
        )
        self.assertEqual(phases["dataset_curation"]["missing_required_artifacts"], [])
        self.assertEqual(phases["external_trainer_execution"]["status"], "ready")
        self.assertIn("trainer_preflight", phases["external_trainer_execution"]["present_required_artifacts"])
        self.assertIn("trainer_launch_check", phases["external_trainer_execution"]["present_required_artifacts"])
        self.assertEqual(phases["serving_checks"]["status"], "ready")
        self.assertEqual(phases["serving_checks"]["present_required_artifacts"], ["serving_lifecycle"])
        self.assertEqual(phases["serving_checks"]["missing_required_artifacts"], [])
        self.assertEqual(phases["heldout_eval"]["status"], "ready")
        self.assertEqual(phases["heldout_eval"]["missing_required_artifacts"], [])
        self.assertEqual(phases["improvement_planning"]["status"], "ready")
        self.assertEqual(phases["governance_decision"]["status"], "ready")
        self.assertEqual(phases["governance_decision"]["present_required_artifacts"], ["promotion_decision"])
        self.assertEqual(phases["promotion_or_rollback"]["status"], "ready")
        self.assertEqual(phases["promotion_or_rollback"]["present_required_artifacts"], ["promotion_ledger"])
        self.assertEqual(phases["next_iteration"]["status"], "ready")
        self.assertFalse(plan["cloud_training"]["cloud_jobs_started"])
        self.assertFalse(plan["cloud_training"]["provider_api_calls_started"])
        self.assertTrue(plan["cloud_training_lineage"]["passed"])
        self.assertEqual(plan["cloud_training_lineage"]["missing_link_count"], 0)
        self.assertTrue(plan["cloud_training_receipt_state"]["fail_closed"])
        self.assertEqual(plan["cloud_training_receipt_state"]["cost_incurred_usd"], 0)
        self.assertEqual(plan["external_eval_receipt_state"]["adapter_count"], 1)
        self.assertEqual(plan["external_eval_receipt_state"]["ready_adapter_count"], 1)
        self.assertTrue(plan["external_eval_receipt_state"]["receipts_passed"])
        self.assertTrue(plan["external_eval_receipt_state"]["fail_closed"])
        self.assertFalse(plan["external_eval_receipt_state"]["live_benchmarks_started"])
        self.assertFalse(plan["external_eval_receipt_state"]["provider_api_calls_started"])
        heldout_check = next(check for check in plan["checks"] if check["id"] == "heldout_eval_is_fail_closed")
        self.assertTrue(heldout_check["passed"])
        self.assertTrue(heldout_check["actual"]["eval_summary_valid"])
        self.assertTrue(heldout_check["actual"]["eval_summary_passed"])
        self.assertFalse(plan["execution_boundary"]["cloud_jobs_started"])
        self.assertFalse(plan["execution_boundary"]["paid_model_grader_calls_started"])
        self.assertFalse(plan["execution_boundary"]["weights_updated_by_flight_recorder"])
        schema = check_schema_file(plan_path)
        self.assertTrue(schema["passed"], schema["errors"])
        validation = validate_artifacts(agentic_training_loop_plan_paths=[plan_path], strict=True)
        self.assertTrue(validation["passed"], validation)

    def test_complete_receipt_set_is_ready_for_governance_review(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = self.write_loop_artifacts(root)
            attach_cloud_training_completion(artifacts)

            plan = build_agentic_training_loop_plan(
                out_path=root / "loop.json",
                iteration_id="loop-001",
                objective="Close the held-out tool-use regression.",
                baseline="local/baseline",
                candidate=loop_candidate_model_id(artifacts),
                teacher="local/teacher",
                artifact_paths=artifacts,
                budget={"max_rollouts": 20, "max_cloud_cost_usd": 0, "max_gpu_hours": 0},
                provider_constraints={"providers": ["mock"], "regions": ["local"], "gpu_classes": ["none"]},
                created_at="2026-07-03T00:00:00+00:00",
            )

            self.assertTrue(plan["passed"], plan["blocked_reasons"])
            self.assertEqual(plan["plan_readiness"], "ready_to_execute")
            self.assertEqual(plan["execution_completion"], "completed")
            self.assertEqual(plan["governance_readiness"], "ready_for_review")
            self.assertEqual(plan["readiness"], "ready_for_governance_review")
            self.assertEqual(plan["recommendation"], "submit_for_governance_review")
            self.assertFalse(plan["execution_boundary"]["cloud_jobs_started"])
            self.assertFalse(plan["handoff_contract"]["default_live_execution_allowed"])
            self.assertEqual(plan["cloud_training"]["missing_artifacts"], [])
            self.assertTrue(plan["cloud_training_lineage"]["passed"])
            self.assertEqual(plan["cloud_training_lineage"]["matched_link_count"], plan["cloud_training_lineage"]["required_link_count"])
            self.assertEqual(plan["cloud_training_lineage"]["provider"]["pipeline_provider_id"], "modal")
            self.assertFalse(plan["cloud_training"]["provider_api_calls_started"])
            self.assertTrue(plan["cloud_training_receipt_state"]["fail_closed"])
            self.assertEqual(plan["cloud_training_receipt_state"]["launch_mode"], "dry_run")
            self.assertEqual(plan["cloud_training_receipt_state"]["status_provider_status"], "not_started")
            self.assertTrue(plan["external_eval_receipt_state"]["receipts_passed"])
            self.assertTrue(plan["external_eval_receipt_state"]["fail_closed"])
            self.assertEqual(plan["external_eval_receipt_state"]["launch_mode"], "dry_run")
            self.assertEqual(plan["external_eval_receipt_state"]["adapter_count"], 1)
            self.assertEqual(plan["missing_phase_inputs"], [])
            self.assertTrue(all(phase["status"] != "blocked" for phase in plan["phases"]))
            self.assertEqual(plan["artifact_count"], sum(len(paths) for paths in artifacts.values()))
            schema = check_schema_contract(plan)
            self.assertTrue(schema["passed"], schema["errors"])

    def test_missing_receipts_keep_loop_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            agentic_plan = self.write_json(root / "agentic_training_plan.json", "hfr.agentic_training_plan.v1")

            plan = build_agentic_training_loop_plan(
                out_path=root / "loop.json",
                iteration_id="loop-002",
                artifact_paths={"agentic_training_plan": [agentic_plan]},
                created_at="2026-07-03T00:00:00+00:00",
            )

            self.assertFalse(plan["passed"])
            self.assertEqual(plan["plan_readiness"], "blocked")
            self.assertEqual(plan["execution_completion"], "incomplete")
            self.assertEqual(plan["governance_readiness"], "blocked")
            self.assertEqual(plan["readiness"], "planned_fail_closed")
            self.assertEqual(plan["recommendation"], "collect_missing_plan_evidence")
            self.assertIn("agentic_rollout_receipt", plan["missing_phase_inputs"])
            self.assertIn("rejection_sampling_gate", plan["missing_phase_inputs"])
            self.assertIn("dataset_curation_receipt", plan["missing_phase_inputs"])
            self.assertIn("trainer_preflight", plan["missing_phase_inputs"])
            self.assertIn("cloud_training_preflight", plan["missing_phase_inputs"])
            self.assertIn("uncalibrated_labels_block_training_data", {check["id"] for check in plan["checks"] if not check["passed"]})
            self.assertIn("rollout_receipt_required_before_review", {check["id"] for check in plan["checks"] if not check["passed"]})
            self.assertIn("cloud_training_receipts_bound_for_provider_handoff", {check["id"] for check in plan["checks"] if not check["passed"]})
            self.assertIn("cloud_training_lineage_bound_for_provider_handoff", {check["id"] for check in plan["checks"] if not check["passed"]})
            schema = check_schema_contract(plan)
            self.assertTrue(schema["passed"], schema["errors"])

    def test_eval_summary_is_required_by_heldout_eval_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = self.write_loop_artifacts(root)
            artifacts.pop("eval_summary")

            plan = build_agentic_training_loop_plan(
                out_path=root / "loop.json",
                iteration_id="loop-missing-eval-summary",
                objective="Require eval summary before governance review.",
                artifact_paths=artifacts,
                budget={"max_cloud_cost_usd": 0, "max_gpu_hours": 0},
                provider_constraints={"providers": ["mock"], "regions": ["local"], "gpu_classes": ["none"]},
                created_at="2026-07-03T00:00:00+00:00",
            )

            self.assertFalse(plan["passed"])
            self.assertEqual(plan["readiness"], "planned_fail_closed")
            self.assertIn("eval_summary", plan["missing_phase_inputs"])
            heldout_check = next(check for check in plan["checks"] if check["id"] == "heldout_eval_is_fail_closed")
            self.assertFalse(heldout_check["passed"])
            self.assertFalse(heldout_check["actual"]["eval_summary_present"])
            self.assertTrue(heldout_check["expected"]["eval_summary_present"])
            schema = check_schema_contract(plan)
            self.assertTrue(schema["passed"], schema["errors"])
            loop_plan = root / "loop.json"
            loop_plan.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            validation = validate_artifacts(agentic_training_loop_plan_paths=[loop_plan], strict=True)
            self.assertTrue(validation["passed"], validation)

    def test_blocked_external_eval_receipt_keeps_loop_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = self.write_loop_artifacts(root)
            receipt = artifacts["external_eval_receipt"][0]
            payload = json.loads(receipt.read_text(encoding="utf-8"))
            payload["passed"] = False
            payload["readiness"] = "blocked"
            payload["recommendation"] = "keep_external_eval_claims_disabled"
            payload["ready_adapter_count"] = 0
            receipt.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            plan = build_agentic_training_loop_plan(
                out_path=root / "loop.json",
                iteration_id="loop-blocked-external-eval",
                artifact_paths=artifacts,
                created_at="2026-07-03T00:00:00+00:00",
            )

            self.assertFalse(plan["passed"])
            self.assertEqual(plan["readiness"], "planned_fail_closed")
            self.assertFalse(plan["external_eval_receipt_state"]["receipts_passed"])
            self.assertTrue(plan["external_eval_receipt_state"]["fail_closed"])
            heldout_check = next(check for check in plan["checks"] if check["id"] == "heldout_eval_is_fail_closed")
            self.assertFalse(heldout_check["passed"])
            self.assertFalse(heldout_check["actual"]["external_eval_receipts_passed"])
            schema = check_schema_contract(plan)
            self.assertTrue(schema["passed"], schema["errors"])

    def test_external_eval_receipt_side_effects_keep_loop_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = self.write_loop_artifacts(root)
            receipt = artifacts["external_eval_receipt"][0]
            payload = json.loads(receipt.read_text(encoding="utf-8"))
            payload["launch"]["live_benchmarks_started"] = True
            payload["launch"]["provider_api_called"] = True
            payload["launch"]["model_downloads_started"] = True
            payload["launch"]["cost_incurred_usd"] = 2
            payload["execution_boundary"]["credential_values_recorded"] = True
            receipt.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            loop_plan = root / "loop.json"
            plan = build_agentic_training_loop_plan(
                out_path=loop_plan,
                iteration_id="loop-external-eval-side-effects",
                artifact_paths=artifacts,
                created_at="2026-07-03T00:00:00+00:00",
            )

            self.assertFalse(plan["passed"])
            self.assertFalse(plan["external_eval_receipt_state"]["fail_closed"])
            self.assertTrue(plan["external_eval_receipt_state"]["live_benchmarks_started"])
            self.assertTrue(plan["external_eval_receipt_state"]["provider_api_calls_started"])
            self.assertTrue(plan["external_eval_receipt_state"]["model_downloads_started"])
            self.assertTrue(plan["external_eval_receipt_state"]["credential_values_recorded"])
            self.assertEqual(plan["external_eval_receipt_state"]["cost_incurred_usd"], 2)
            self.assertIn(
                "heldout_eval_is_fail_closed",
                {check["id"] for check in plan["checks"] if not check["passed"]},
            )

            forged = json.loads(json.dumps(plan))
            forged["external_eval_receipt_state"]["fail_closed"] = True
            forged["external_eval_receipt_state"]["live_benchmarks_started"] = False
            forged["external_eval_receipt_state"]["provider_api_calls_started"] = False
            forged["external_eval_receipt_state"]["model_downloads_started"] = False
            forged["external_eval_receipt_state"]["credential_values_recorded"] = False
            forged["external_eval_receipt_state"]["cost_incurred_usd"] = 0
            for check in forged["checks"]:
                if check["id"] == "heldout_eval_is_fail_closed":
                    check["passed"] = True
            loop_plan.write_text(json.dumps(forged, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            validation = validate_artifacts(agentic_training_loop_plan_paths=[loop_plan], strict=True)

            self.assertFalse(validation["passed"], validation)
            errors = "\n".join(error for target in validation["targets"] for error in target["errors"])
            self.assertIn(
                "agentic_training_loop_plan.external_eval_receipt_state.live_benchmarks_started must match external eval receipt artifacts.",
                errors,
            )
            self.assertIn(
                "agentic_training_loop_plan.external_eval_receipt_state.provider_api_calls_started must match external eval receipt artifacts.",
                errors,
            )
            self.assertIn(
                "agentic_training_loop_plan.external_eval_receipt_state.cost_incurred_usd must match external eval receipt artifacts.",
                errors,
            )
            self.assertIn(
                "agentic_training_loop_plan.checks.heldout_eval_is_fail_closed.passed must match external eval receipt and eval summary state.",
                errors,
            )

    def test_invalid_eval_summary_cannot_unlock_loop_readiness(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = self.write_loop_artifacts(root)
            eval_summary_path = artifacts["eval_summary"][0]
            eval_summary = json.loads(eval_summary_path.read_text(encoding="utf-8"))
            del eval_summary["arms"]
            eval_summary_path.write_text(json.dumps(eval_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            loop_plan = root / "loop.json"

            plan = build_agentic_training_loop_plan(
                out_path=loop_plan,
                iteration_id="loop-invalid-eval-summary",
                artifact_paths=artifacts,
                created_at="2026-07-03T00:00:00+00:00",
            )
            self.assertFalse(plan["passed"])
            loop_plan.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            validation = validate_artifacts(agentic_training_loop_plan_paths=[loop_plan], strict=True)

            self.assertTrue(validation["passed"], validation)
            heldout_check = next(check for check in plan["checks"] if check["id"] == "heldout_eval_is_fail_closed")
            self.assertFalse(heldout_check["passed"])
            self.assertFalse(heldout_check["actual"]["eval_summary_valid"])

    def test_failed_eval_summary_cannot_unlock_loop_readiness(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = self.write_loop_artifacts(root)
            eval_summary_path = artifacts["eval_summary"][0]
            eval_summary = json.loads(eval_summary_path.read_text(encoding="utf-8"))
            eval_summary["passed"] = False
            eval_summary["governance_ready"] = False
            eval_summary_path.write_text(json.dumps(eval_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            plan = build_agentic_training_loop_plan(
                out_path=root / "loop.json",
                iteration_id="loop-failed-eval-summary",
                artifact_paths=artifacts,
                created_at="2026-07-03T00:00:00+00:00",
            )

            self.assertFalse(plan["passed"])
            heldout_check = next(check for check in plan["checks"] if check["id"] == "heldout_eval_is_fail_closed")
            self.assertFalse(heldout_check["passed"])
            self.assertTrue(heldout_check["actual"]["eval_summary_valid"])
            self.assertFalse(heldout_check["actual"]["eval_summary_passed"])

    def test_invalid_promotion_ledger_cannot_unlock_loop_governance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = self.write_loop_artifacts(root)
            ledger_path = artifacts["promotion_ledger"][0]
            ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
            del ledger["records"]
            ledger_path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            loop_plan = root / "loop.json"

            plan = build_agentic_training_loop_plan(
                out_path=loop_plan,
                iteration_id="loop-invalid-promotion-ledger",
                artifact_paths=artifacts,
                created_at="2026-07-03T00:00:00+00:00",
            )
            self.assertFalse(plan["passed"])
            self.assertIn("promotion_ledger", plan["missing_phase_inputs"])
            loop_plan.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            validation = validate_artifacts(agentic_training_loop_plan_paths=[loop_plan], strict=True)

            self.assertTrue(validation["passed"], validation)

    def test_invalid_promotion_decision_cannot_unlock_loop_governance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = self.write_loop_artifacts(root)
            decision_path = artifacts["promotion_decision"][0]
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            del decision["checks"]
            decision_path.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            loop_plan = root / "loop.json"

            plan = build_agentic_training_loop_plan(
                out_path=loop_plan,
                iteration_id="loop-invalid-promotion-decision",
                artifact_paths=artifacts,
                created_at="2026-07-03T00:00:00+00:00",
            )
            self.assertFalse(plan["passed"])
            self.assertIn("promotion_decision", plan["missing_phase_inputs"])
            loop_plan.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            validation = validate_artifacts(agentic_training_loop_plan_paths=[loop_plan], strict=True)

            self.assertTrue(validation["passed"], validation)

    def test_public_unsafe_promotion_decision_path_cannot_unlock_loop_governance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = self.write_loop_artifacts(root)
            decision_path = artifacts["promotion_decision"][0]
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            source_path = root / "promotion_decision_sources" / "evidence_bundle.json"
            decision["artifacts"]["evidence_bundle"]["path"] = str(source_path.resolve())
            decision_path.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            loop_plan = root / "loop.json"

            plan = build_agentic_training_loop_plan(
                out_path=loop_plan,
                iteration_id="loop-public-unsafe-promotion-decision",
                artifact_paths=artifacts,
                created_at="2026-07-03T00:00:00+00:00",
            )
            self.assertFalse(plan["passed"])
            self.assertIn("promotion_decision", plan["missing_phase_inputs"])
            loop_plan.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            validation = validate_artifacts(agentic_training_loop_plan_paths=[loop_plan], strict=True)

            self.assertTrue(validation["passed"], validation)

    def test_public_path_like_promotion_decision_prose_does_not_block_loop_governance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = self.write_loop_artifacts(root)
            attach_cloud_training_completion(artifacts)
            decision_path = artifacts["promotion_decision"][0]
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            decision["notes"].append("Operator note: expected drift is ~5%; /tmp is mentioned as prose, not an artifact path.")
            decision_path.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            loop_plan = root / "loop.json"

            plan = build_agentic_training_loop_plan(
                out_path=loop_plan,
                iteration_id="loop-path-like-prose",
                candidate=loop_candidate_model_id(artifacts),
                artifact_paths=artifacts,
                created_at="2026-07-03T00:00:00+00:00",
            )
            self.assertTrue(plan["passed"], plan["blocked_reasons"])
            loop_plan.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            validation = validate_artifacts(agentic_training_loop_plan_paths=[loop_plan], strict=True)

            self.assertTrue(validation["passed"], validation)

    def test_forged_external_eval_receipt_keeps_loop_state_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = self.write_loop_artifacts(root)
            receipt_path = artifacts["external_eval_receipt"][0]
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            receipt["source_plan"]["sha256"] = "0" * 64
            receipt["passed"] = True
            receipt["readiness"] = "dry_run_recorded"
            receipt["recommendation"] = "archive_external_eval_dry_run"
            receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            loop_plan = root / "loop.json"

            plan = build_agentic_training_loop_plan(
                out_path=loop_plan,
                iteration_id="loop-forged-external-eval",
                artifact_paths=artifacts,
                created_at="2026-07-03T00:00:00+00:00",
            )
            self.assertFalse(plan["passed"])
            self.assertFalse(plan["external_eval_receipt_state"]["receipts_passed"])
            self.assertEqual(plan["external_eval_receipt_state"]["receipt_passed_count"], 0)
            loop_plan.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            validation = validate_artifacts(agentic_training_loop_plan_paths=[loop_plan], strict=True)
            receipt_validation = validate_artifacts(external_eval_receipt_paths=[receipt_path], strict=True)

            self.assertTrue(validation["passed"], validation)
            self.assertFalse(receipt_validation["passed"], receipt_validation)

    def test_cli_writes_schema_checkable_and_validatable_loop_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = self.write_loop_artifacts(root)
            out = root / "loop.json"

            command = [
                sys.executable,
                "-m",
                "flightrecorder",
                "agentic-loop",
                "plan",
                "--iteration-id",
                "loop-cli",
                "--objective",
                "CLI closed-loop smoke",
                "--baseline",
                "local/baseline",
                "--candidate",
                "local/candidate",
                "--provider",
                "mock",
                "--region",
                "local",
                "--gpu-class",
                "none",
                "--budget",
                "max_cloud_cost_usd=0",
                "--out",
                str(out),
            ]
            for option, role in (
                ("--agentic-rollout-plan", "agentic_rollout_plan"),
                ("--agentic-rollout-receipt", "agentic_rollout_receipt"),
                ("--harness-result", "harness_result"),
                ("--evidence-bundle", "evidence_bundle"),
                ("--rubric-spec", "rubric_spec"),
                ("--model-grader-dry-run", "model_grader_dry_run"),
                ("--model-grader-disagreement-queue", "model_grader_disagreement_queue"),
                ("--model-grader-override-receipt", "model_grader_override_receipt"),
                ("--model-grader-gate", "model_grader_gate"),
                ("--review-calibration", "review_calibration"),
                ("--reviewed-gate", "reviewed_gate"),
                ("--rejection-sampling-gate", "rejection_sampling_gate"),
                ("--dataset-curation-receipt", "dataset_curation_receipt"),
                ("--training-export", "training_export"),
                ("--agentic-training-plan", "agentic_training_plan"),
                ("--agentic-training-flow", "agentic_training_flow"),
                ("--cloud-training-provider-registry", "cloud_training_provider_registry"),
                ("--cloud-training-preflight", "cloud_training_preflight"),
                ("--cloud-training-artifact-manifest", "cloud_training_artifact_manifest"),
                ("--cloud-training-launch-plan", "cloud_training_launch_plan"),
                ("--cloud-training-launch-receipt", "cloud_training_launch_receipt"),
                ("--cloud-training-status-receipt", "cloud_training_status_receipt"),
                ("--trainer-preflight", "trainer_preflight"),
                ("--trainer-launch-check", "trainer_launch_check"),
                ("--serving-lifecycle", "serving_lifecycle"),
                ("--heldout-manifest", "heldout_manifest"),
                ("--external-eval-plan", "external_eval_plan"),
                ("--external-eval-receipt", "external_eval_receipt"),
                ("--eval-summary", "eval_summary"),
                ("--improvement-plan", "improvement_plan"),
                ("--promotion-decision", "promotion_decision"),
                ("--promotion-ledger", "promotion_ledger"),
                ("--promotion-cards", "promotion_cards"),
                ("--promotion-alias-apply", "promotion_alias_apply"),
                ("--promotion-rollback-receipt", "promotion_rollback_receipt"),
                ("--promotion-release-record", "promotion_release_record"),
                ("--promotion-archive", "promotion_archive"),
                ("--next-iteration-schedule", "next_iteration_schedule"),
                ("--action-ledger", "action_ledger"),
            ):
                command.extend([option, str(artifacts[role][0])])

            completed = subprocess.run(
                command,
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr + completed.stdout)
            schema = check_schema_file(out)
            self.assertTrue(schema["passed"], schema["errors"])
            validation = validate_artifacts(agentic_training_loop_plan_paths=[out], strict=True)
            self.assertTrue(validation["passed"], validation)

            validate_completed = subprocess.run(
                [sys.executable, "-m", "flightrecorder", "validate", "--agentic-loop-plan", str(out), "--strict"],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(validate_completed.returncode, 0, validate_completed.stderr + validate_completed.stdout)

    def test_validate_rejects_forged_loop_plan_side_effect_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = self.write_loop_artifacts(root)
            loop_plan = root / "loop.json"
            plan = build_agentic_training_loop_plan(
                out_path=loop_plan,
                iteration_id="loop-forged-side-effects",
                artifact_paths=artifacts,
                budget={"max_cloud_cost_usd": 0, "max_gpu_hours": 0},
                provider_constraints={"providers": ["mock"], "regions": ["local"], "gpu_classes": ["none"]},
                created_at="2026-07-03T00:00:00+00:00",
            )
            schema = check_schema_contract(plan, name_or_id="agentic_training_loop_plan")
            self.assertTrue(schema["passed"], schema["errors"])

            forged = json.loads(json.dumps(plan))
            forged["cloud_job_url"] = "redacted-cloud-job-url"
            forged["budget"]["provider_billing_account"] = "redacted-account"
            forged["participants"]["api_key_env"] = "RED_ACTED"
            forged["provider_constraints"]["credential_value"] = "redacted-secret"
            forged["artifact_role_counts"][0]["download_url"] = "redacted-download-url"
            forged["checks"][0]["provider_call"] = "forged"
            forged["source_artifacts"]["agentic_training_plan"][0]["signed_url"] = "redacted-signed-url"
            forged["phases"][0]["live_execution_started"] = True
            forged["cloud_training"]["provider_call_receipt"] = "forged"
            forged["cloud_training_receipt_state"]["provider_console_url"] = "redacted-provider-console"
            forged["cloud_training_lineage"]["provider"]["credential_value"] = "redacted-secret"
            forged["cloud_training_lineage"]["role_counts"][0]["provider_call"] = "forged"
            forged["cloud_training_lineage"]["links"][0]["provider_trace_url"] = "redacted-trace-url"
            forged["external_eval_receipt_state"]["benchmark_job_id"] = "bench-live"
            forged["execution_boundary"]["provider_console_url"] = "redacted-provider-console"
            forged["handoff_contract"]["credential_hint"] = "redacted-secret"
            forged["next_iteration"]["auto_schedule_started"] = True

            forged_schema = check_schema_contract(forged, name_or_id="agentic_training_loop_plan")
            self.assertFalse(forged_schema["passed"])
            schema_errors = "\n".join(forged_schema["errors"])
            for field_name in (
                "cloud_job_url",
                "provider_billing_account",
                "api_key_env",
                "credential_value",
                "download_url",
                "provider_call",
                "signed_url",
                "live_execution_started",
                "provider_call_receipt",
                "provider_console_url",
                "provider_trace_url",
                "benchmark_job_id",
                "credential_hint",
                "auto_schedule_started",
            ):
                self.assertIn(field_name, schema_errors)

            loop_plan.write_text(json.dumps(forged, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            validation = validate_artifacts(agentic_training_loop_plan_paths=[loop_plan], strict=True)

            self.assertFalse(validation["passed"], validation)
            errors = "\n".join(error for target in validation["targets"] for error in target["errors"])
            self.assertIn("agentic_training_loop_plan contains unknown field(s): ['cloud_job_url'].", errors)
            self.assertIn("agentic_training_loop_plan.budget contains unknown field(s): ['provider_billing_account'].", errors)
            self.assertIn("agentic_training_loop_plan.participants contains unknown field(s): ['api_key_env'].", errors)
            self.assertIn("agentic_training_loop_plan.provider_constraints contains unknown field(s): ['credential_value'].", errors)
            self.assertIn("agentic_training_loop_plan.artifact_role_counts[0] contains unknown field(s): ['download_url'].", errors)
            self.assertIn("agentic_training_loop_plan.checks[0] contains unknown field(s): ['provider_call'].", errors)
            self.assertIn(
                "agentic_training_loop_plan.source_artifacts.agentic_training_plan[0] contains unknown field(s): ['signed_url'].",
                errors,
            )
            self.assertIn("agentic_training_loop_plan.phases[0] contains unknown field(s): ['live_execution_started'].", errors)
            self.assertIn("agentic_training_loop_plan.cloud_training contains unknown field(s): ['provider_call_receipt'].", errors)
            self.assertIn(
                "agentic_training_loop_plan.cloud_training_receipt_state contains unknown field(s): ['provider_console_url'].",
                errors,
            )
            self.assertIn(
                "agentic_training_loop_plan.cloud_training_lineage.provider contains unknown field(s): ['credential_value'].",
                errors,
            )
            self.assertIn(
                "agentic_training_loop_plan.cloud_training_lineage.role_counts[0] contains unknown field(s): ['provider_call'].",
                errors,
            )
            self.assertIn(
                "agentic_training_loop_plan.cloud_training_lineage.links[0] contains unknown field(s): ['provider_trace_url'].",
                errors,
            )
            self.assertIn(
                "agentic_training_loop_plan.external_eval_receipt_state contains unknown field(s): ['benchmark_job_id'].",
                errors,
            )
            self.assertIn(
                "agentic_training_loop_plan.execution_boundary contains unknown field(s): ['provider_console_url'].",
                errors,
            )
            self.assertIn("agentic_training_loop_plan.handoff_contract contains unknown field(s): ['credential_hint'].", errors)
            self.assertIn("agentic_training_loop_plan.next_iteration contains unknown field(s): ['auto_schedule_started'].", errors)

    def test_validate_rejects_stale_or_moved_loop_plan_source_refs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            agentic_plan = self.write_json(root / "agentic_training_plan.json", "hfr.agentic_training_plan.v1")
            loop_plan = root / "loop.json"
            plan = build_agentic_training_loop_plan(
                out_path=loop_plan,
                iteration_id="loop-stale-source",
                artifact_paths={"agentic_training_plan": [agentic_plan]},
                created_at="2026-07-03T00:00:00+00:00",
            )
            loop_plan.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            validation = validate_artifacts(agentic_training_loop_plan_paths=[loop_plan], strict=True)
            self.assertTrue(validation["passed"], validation)

            copied_plan = root / "copy" / "loop.json"
            copied_plan.parent.mkdir()
            copied_plan.write_text(loop_plan.read_text(encoding="utf-8"), encoding="utf-8")
            copied_validation = validate_artifacts(agentic_training_loop_plan_paths=[copied_plan], strict=True)
            self.assertFalse(copied_validation["passed"])
            copied_errors = "\n".join(error for target in copied_validation["targets"] for error in target["errors"])
            self.assertIn("agentic_training_loop_plan.source_artifacts.agentic_training_plan[0].path does not resolve to an existing file.", copied_errors)

            payload = json.loads(agentic_plan.read_text(encoding="utf-8"))
            payload["stale_after_plan_write"] = True
            agentic_plan.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            stale_validation = validate_artifacts(agentic_training_loop_plan_paths=[loop_plan], strict=True)
            self.assertFalse(stale_validation["passed"])
            stale_errors = "\n".join(error for target in stale_validation["targets"] for error in target["errors"])
            self.assertIn("agentic_training_loop_plan.source_artifacts.agentic_training_plan[0].size_bytes does not match the current file.", stale_errors)
            self.assertIn("agentic_training_loop_plan.source_artifacts.agentic_training_plan[0].sha256 does not match the current file.", stale_errors)

    def test_validate_rejects_stale_loop_plan_directory_source_refs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            training_export = copy_valid_loop_artifacts(root)["training_export"][0]
            loop_plan = root / "loop.json"
            plan = build_agentic_training_loop_plan(
                out_path=loop_plan,
                iteration_id="loop-stale-directory-source",
                artifact_paths={"training_export": [training_export]},
                created_at="2026-07-03T00:00:00+00:00",
            )
            loop_plan.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            ref = plan["source_artifacts"]["training_export"][0]
            expected_tree = directory_tree_fingerprint(training_export)
            self.assertEqual(ref["kind"], "directory")
            self.assertEqual(ref["file_count"], expected_tree["file_count"])
            self.assertEqual(ref["size_bytes"], expected_tree["size_bytes"])
            self.assertEqual(ref["sha256"], expected_tree["sha256"])
            self.assertFalse(ref["contains_symlinks"])
            validation = validate_artifacts(agentic_training_loop_plan_paths=[loop_plan], strict=True)
            self.assertTrue(validation["passed"], validation)

            episodes = training_export / "episodes.jsonl"
            episodes.write_text(episodes.read_text(encoding="utf-8") + '{"id":"new"}\n', encoding="utf-8")
            (training_export / "new_metadata.json").write_text("{}\n", encoding="utf-8")
            stale_validation = validate_artifacts(agentic_training_loop_plan_paths=[loop_plan], strict=True)
            self.assertFalse(stale_validation["passed"])
            stale_errors = "\n".join(error for target in stale_validation["targets"] for error in target["errors"])
            self.assertIn("agentic_training_loop_plan.source_artifacts.training_export[0].file_count does not match the current directory.", stale_errors)
            self.assertIn("agentic_training_loop_plan.source_artifacts.training_export[0].size_bytes does not match the current directory.", stale_errors)
            self.assertIn("agentic_training_loop_plan.source_artifacts.training_export[0].sha256 does not match the current directory.", stale_errors)

    def test_validate_rejects_symlink_descendants_in_loop_plan_directory_source_refs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            training_export = copy_valid_loop_artifacts(root)["training_export"][0]
            source_file = training_export / "episodes.jsonl"
            loop_plan = root / "loop.json"
            plan = build_agentic_training_loop_plan(
                out_path=loop_plan,
                iteration_id="loop-directory-symlink-source",
                artifact_paths={"training_export": [training_export]},
                created_at="2026-07-03T00:00:00+00:00",
            )
            loop_plan.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            validation = validate_artifacts(agentic_training_loop_plan_paths=[loop_plan], strict=True)
            self.assertTrue(validation["passed"], validation)

            symlink_path = training_export / "episodes-link.jsonl"
            try:
                symlink_path.symlink_to(source_file)
            except (NotImplementedError, OSError) as exc:
                self.skipTest(f"symlink unavailable: {exc}")
            stale_validation = validate_artifacts(agentic_training_loop_plan_paths=[loop_plan], strict=True)
            self.assertFalse(stale_validation["passed"])
            stale_errors = "\n".join(error for target in stale_validation["targets"] for error in target["errors"])
            self.assertIn(
                "agentic_training_loop_plan.source_artifacts.training_export[0].path must not contain symlink descendants.",
                stale_errors,
            )

    def test_validate_rejects_symlink_loop_plan_source_ref(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            agentic_plan = self.write_json(root / "agentic_training_plan.json", "hfr.agentic_training_plan.v1")
            loop_plan = root / "loop.json"
            plan = build_agentic_training_loop_plan(
                out_path=loop_plan,
                iteration_id="loop-symlink-source",
                artifact_paths={"agentic_training_plan": [agentic_plan]},
                created_at="2026-07-03T00:00:00+00:00",
            )
            link_path = root / "agentic_training_plan_link.json"
            try:
                link_path.symlink_to(agentic_plan)
            except (NotImplementedError, OSError) as exc:
                self.skipTest(f"symlink unavailable: {exc}")
            plan["source_artifacts"]["agentic_training_plan"][0]["path"] = "agentic_training_plan_link.json"
            loop_plan.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            validation = validate_artifacts(agentic_training_loop_plan_paths=[loop_plan], strict=True)

            self.assertFalse(validation["passed"], validation)
            errors = "\n".join(error for target in validation["targets"] for error in target["errors"])
            self.assertIn(
                "agentic_training_loop_plan.source_artifacts.agentic_training_plan[0].path must resolve to a regular non-symlink file.",
                errors,
            )

    def test_validate_rejects_symlink_parent_loop_plan_source_ref(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            agentic_plan = self.write_json(root / "agentic_training_plan.json", "hfr.agentic_training_plan.v1")
            loop_plan = root / "loop.json"
            plan = build_agentic_training_loop_plan(
                out_path=loop_plan,
                iteration_id="loop-symlink-parent-source",
                artifact_paths={"agentic_training_plan": [agentic_plan]},
                created_at="2026-07-03T00:00:00+00:00",
            )
            linked_target = root / "linked_target"
            linked_target.mkdir()
            (linked_target / "agentic_training_plan.json").write_text(agentic_plan.read_text(encoding="utf-8"), encoding="utf-8")
            linked_parent = root / "linked_artifacts"
            try:
                linked_parent.symlink_to(linked_target, target_is_directory=True)
            except (NotImplementedError, OSError) as exc:
                self.skipTest(f"symlink unavailable: {exc}")
            plan["source_artifacts"]["agentic_training_plan"][0]["path"] = "linked_artifacts/agentic_training_plan.json"
            loop_plan.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            validation = validate_artifacts(agentic_training_loop_plan_paths=[loop_plan], strict=True)

            self.assertFalse(validation["passed"], validation)
            errors = "\n".join(error for target in validation["targets"] for error in target["errors"])
            self.assertIn(
                "agentic_training_loop_plan.source_artifacts.agentic_training_plan[0].path must resolve to a regular non-symlink file.",
                errors,
            )

    def test_loop_plan_source_payload_readers_skip_symlinked_parent_refs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = self.write_loop_artifacts(root)
            loop_plan = root / "loop.json"
            plan = build_agentic_training_loop_plan(
                out_path=loop_plan,
                iteration_id="loop-symlink-payload-reader",
                artifact_paths=artifacts,
                created_at="2026-07-03T00:00:00+00:00",
            )
            launch_receipt = artifacts["cloud_training_launch_receipt"][0]
            linked_parent = root / "linked_artifacts"
            try:
                linked_parent.symlink_to(launch_receipt.parent, target_is_directory=True)
            except (NotImplementedError, OSError) as exc:
                self.skipTest(f"symlink unavailable: {exc}")
            plan["source_artifacts"]["cloud_training_launch_receipt"][0]["path"] = str(
                Path("linked_artifacts") / launch_receipt.name
            )
            loop_plan.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            validation = validate_artifacts(agentic_training_loop_plan_paths=[loop_plan], strict=True)

            self.assertFalse(validation["passed"], validation)
            errors = "\n".join(error for target in validation["targets"] for error in target["errors"])
            self.assertIn(
                "agentic_training_loop_plan.source_artifacts.cloud_training_launch_receipt[0].path must resolve to a regular non-symlink file.",
                errors,
            )
            self.assertIn(
                "agentic_training_loop_plan.cloud_training_receipt_state.launch_receipt_count must match cloud training receipt artifacts.",
                errors,
            )

    def test_loop_plan_refs_are_relative_to_output_directory_for_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = root / "runs"
            runs.mkdir()
            self.write_json(runs / "agentic_training_plan.json", "hfr.agentic_training_plan.v1")
            previous_cwd = Path.cwd()
            try:
                os.chdir(root)
                plan = build_agentic_training_loop_plan(
                    out_path=runs / "agentic_training_loop_plan.json",
                    iteration_id="loop-documented-paths",
                    artifact_paths={"agentic_training_plan": [Path("runs/agentic_training_plan.json")]},
                    created_at="2026-07-03T00:00:00+00:00",
                )
            finally:
                os.chdir(previous_cwd)

            ref = plan["source_artifacts"]["agentic_training_plan"][0]
            self.assertEqual(ref["path"], "agentic_training_plan.json")
            loop_plan = runs / "agentic_training_loop_plan.json"
            loop_plan.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            validation = validate_artifacts(agentic_training_loop_plan_paths=[loop_plan], strict=True)
            self.assertTrue(validation["passed"], validation)

    def test_validate_rejects_tampered_cloud_training_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = self.write_loop_artifacts(root)
            loop_plan = root / "loop.json"
            plan = build_agentic_training_loop_plan(
                out_path=loop_plan,
                iteration_id="loop-cloud-tamper",
                artifact_paths=artifacts,
                created_at="2026-07-03T00:00:00+00:00",
            )
            plan["cloud_training"]["artifact_count"] = 0
            plan["cloud_training"]["provider_api_calls_started"] = True
            plan["cloud_training_receipt_state"]["fail_closed"] = False
            plan["cloud_training_lineage"]["matched_link_count"] = 0
            loop_plan.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            validation = validate_artifacts(agentic_training_loop_plan_paths=[loop_plan], strict=True)

            self.assertFalse(validation["passed"], validation)
            errors = "\n".join(error for target in validation["targets"] for error in target["errors"])
            self.assertIn("agentic_training_loop_plan.cloud_training.artifact_count must match cloud training source artifacts.", errors)
            self.assertIn("agentic_training_loop_plan.cloud_training.provider_api_calls_started must match cloud training source artifacts.", errors)
            self.assertIn("agentic_training_loop_plan.cloud_training_receipt_state.fail_closed must match cloud training receipt artifacts.", errors)
            self.assertIn("agentic_training_loop_plan.cloud_training_lineage.matched_link_count must match cloud training source lineage.", errors)

    def test_launch_receipt_side_effects_keep_loop_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = self.write_loop_artifacts(root)
            receipt = artifacts["cloud_training_launch_receipt"][0]
            payload = json.loads(receipt.read_text(encoding="utf-8"))
            payload["launch"]["cloud_job_started"] = True
            payload["launch"]["provider_api_called"] = True
            payload["launch"]["cost_incurred_usd"] = 2
            receipt.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            plan = build_agentic_training_loop_plan(
                out_path=root / "loop.json",
                iteration_id="loop-cloud-side-effects",
                artifact_paths=artifacts,
                created_at="2026-07-03T00:00:00+00:00",
            )

            self.assertFalse(plan["passed"])
            self.assertFalse(plan["cloud_training_receipt_state"]["fail_closed"])
            self.assertTrue(plan["cloud_training_receipt_state"]["cloud_jobs_started"])
            self.assertTrue(plan["cloud_training_receipt_state"]["provider_api_calls_started"])
            self.assertEqual(plan["cloud_training_receipt_state"]["cost_incurred_usd"], 2)
            self.assertIn(
                "cloud_training_receipts_are_side_effect_free",
                {check["id"] for check in plan["checks"] if not check["passed"]},
            )
            schema = check_schema_contract(plan)
            self.assertTrue(schema["passed"], schema["errors"])

    def test_unlinked_cloud_training_receipts_keep_loop_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = self.write_loop_artifacts(root)
            receipt = artifacts["cloud_training_launch_receipt"][0]
            payload = json.loads(receipt.read_text(encoding="utf-8"))
            payload["source_artifacts"]["launch_plan"]["sha256"] = "0" * 64
            receipt.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            plan = build_agentic_training_loop_plan(
                out_path=root / "loop.json",
                iteration_id="loop-cloud-unlinked",
                artifact_paths=artifacts,
                created_at="2026-07-03T00:00:00+00:00",
            )

            self.assertFalse(plan["passed"])
            self.assertEqual(plan["readiness"], "planned_fail_closed")
            self.assertFalse(plan["cloud_training_receipt_state"]["launch_receipt_passed"])
            self.assertFalse(plan["cloud_training_receipt_state"]["receipts_passed"])
            self.assertFalse(plan["cloud_training_lineage"]["passed"])
            self.assertIn("launch_receipt_links_launch_plan", plan["cloud_training_lineage"]["mismatched_links"])
            self.assertIn(
                "cloud_training_lineage_bound_for_provider_handoff",
                {check["id"] for check in plan["checks"] if not check["passed"]},
            )
            schema = check_schema_contract(plan)
            self.assertTrue(schema["passed"], schema["errors"])
            loop_plan = root / "loop.json"
            loop_plan.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            validation = validate_artifacts(agentic_training_loop_plan_paths=[loop_plan], strict=True)
            receipt_validation = validate_artifacts(cloud_training_launch_receipt_paths=[receipt], strict=True)
            self.assertTrue(validation["passed"], validation)
            self.assertFalse(receipt_validation["passed"], receipt_validation)

    def test_duplicate_cloud_training_lineage_roles_keep_loop_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = self.write_loop_artifacts(root)
            duplicate_receipt = root / "cloud_training_launch_receipt_duplicate.json"
            duplicate_receipt.write_text(
                artifacts["cloud_training_launch_receipt"][0].read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            artifacts["cloud_training_launch_receipt"].append(duplicate_receipt)

            loop_plan = root / "loop.json"
            plan = build_agentic_training_loop_plan(
                out_path=loop_plan,
                iteration_id="loop-cloud-duplicate",
                artifact_paths=artifacts,
                created_at="2026-07-03T00:00:00+00:00",
            )
            loop_plan.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            self.assertFalse(plan["passed"])
            self.assertEqual(plan["readiness"], "planned_fail_closed")
            self.assertEqual(plan["cloud_training_lineage"]["duplicate_roles"], ["cloud_training_launch_receipt"])
            self.assertIn("launch_receipt_links_launch_plan", plan["cloud_training_lineage"]["ambiguous_links"])
            validation = validate_artifacts(agentic_training_loop_plan_paths=[loop_plan], strict=True)
            self.assertTrue(validation["passed"], validation)

            plan["cloud_training_lineage"]["duplicate_role_count"] = 0
            plan["cloud_training_lineage"]["duplicate_roles"] = []
            plan["cloud_training_lineage"]["ambiguous_link_count"] = 0
            plan["cloud_training_lineage"]["ambiguous_links"] = []
            loop_plan.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            tampered = validate_artifacts(agentic_training_loop_plan_paths=[loop_plan], strict=True)
            self.assertFalse(tampered["passed"], tampered)
            errors = "\n".join(error for target in tampered["targets"] for error in target["errors"])
            self.assertIn("cloud_training_lineage.duplicate_role_count must match cloud training source lineage", errors)
            self.assertIn("cloud_training_lineage.ambiguous_link_count must match cloud training source lineage", errors)

    def test_duplicate_cloud_receipt_side_effects_are_not_hidden_by_first_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = self.write_loop_artifacts(root)
            duplicate_receipt = root / "cloud_training_launch_receipt_duplicate.json"
            duplicate_payload = json.loads(artifacts["cloud_training_launch_receipt"][0].read_text(encoding="utf-8"))
            duplicate_payload["execution_boundary"]["provider_api_called"] = True
            duplicate_payload["execution_boundary"]["cloud_job_started"] = True
            duplicate_payload["execution_boundary"]["live_requested"] = True
            duplicate_payload["execution_boundary"]["cloud_cost_incurred_usd"] = 3
            duplicate_receipt.write_text(json.dumps(duplicate_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            artifacts["cloud_training_launch_receipt"].append(duplicate_receipt)

            loop_plan = root / "loop.json"
            plan = build_agentic_training_loop_plan(
                out_path=loop_plan,
                iteration_id="loop-cloud-duplicate-side-effects",
                artifact_paths=artifacts,
                created_at="2026-07-03T00:00:00+00:00",
            )

            self.assertFalse(plan["cloud_training_receipt_state"]["fail_closed"])
            self.assertTrue(plan["cloud_training_receipt_state"]["provider_api_calls_started"])
            self.assertTrue(plan["cloud_training_receipt_state"]["cloud_jobs_started"])
            self.assertEqual(plan["cloud_training_receipt_state"]["launch_receipt_count"], 2)
            self.assertEqual(plan["cloud_training_receipt_state"]["cost_incurred_usd"], 3)
            self.assertTrue(plan["cloud_training_receipt_state"]["live_launch_requested"])
            for check in plan["checks"]:
                if check["id"] == "cloud_training_receipts_are_side_effect_free":
                    check["passed"] = True
            plan["cloud_training"]["provider_api_calls_started"] = False
            plan["cloud_training"]["cloud_jobs_started"] = False
            plan["cloud_training_receipt_state"]["provider_api_calls_started"] = False
            plan["cloud_training_receipt_state"]["cloud_jobs_started"] = False
            plan["cloud_training_receipt_state"]["cost_incurred_usd"] = 0
            plan["cloud_training_receipt_state"]["live_launch_requested"] = False
            plan["cloud_training_receipt_state"]["fail_closed"] = True
            loop_plan.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            validation = validate_artifacts(agentic_training_loop_plan_paths=[loop_plan], strict=True)

            self.assertFalse(validation["passed"], validation)
            errors = "\n".join(error for target in validation["targets"] for error in target["errors"])
            self.assertIn("agentic_training_loop_plan.cloud_training.provider_api_calls_started must match cloud training source artifacts.", errors)
            self.assertIn("agentic_training_loop_plan.cloud_training.cloud_jobs_started must match cloud training source artifacts.", errors)
            self.assertIn(
                "agentic_training_loop_plan.cloud_training_receipt_state.provider_api_calls_started must match cloud training receipt artifacts.",
                errors,
            )
            self.assertIn("agentic_training_loop_plan.cloud_training_receipt_state.cost_incurred_usd must match cloud training receipt artifacts.", errors)
            self.assertIn("agentic_training_loop_plan.cloud_training_receipt_state.fail_closed must match cloud training receipt artifacts.", errors)
            self.assertIn(
                "agentic_training_loop_plan.checks.cloud_training_receipts_are_side_effect_free.passed must match receipt state.",
                errors,
            )

    def test_live_launch_request_alone_keeps_loop_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = self.write_loop_artifacts(root)
            launch_receipt = artifacts["cloud_training_launch_receipt"][0]
            launch_payload = json.loads(launch_receipt.read_text(encoding="utf-8"))
            launch_payload["launch"]["mode"] = "live"
            launch_receipt.write_text(json.dumps(launch_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            loop_plan = root / "loop.json"
            plan = build_agentic_training_loop_plan(
                out_path=loop_plan,
                iteration_id="loop-cloud-live-only",
                artifact_paths=artifacts,
                created_at="2026-07-03T00:00:00+00:00",
            )

            self.assertFalse(plan["passed"])
            self.assertTrue(plan["cloud_training_receipt_state"]["live_launch_requested"])
            self.assertFalse(plan["cloud_training_receipt_state"]["provider_api_calls_started"])
            self.assertFalse(plan["cloud_training_receipt_state"]["cloud_jobs_started"])
            self.assertEqual(plan["cloud_training_receipt_state"]["cost_incurred_usd"], 0)
            self.assertFalse(plan["cloud_training_receipt_state"]["fail_closed"])
            forged_readiness = json.loads(json.dumps(plan))
            forged_readiness["readiness"] = "ready_for_governance_review"
            forged_readiness["recommendation"] = "submit_for_governance_review"
            loop_plan.write_text(json.dumps(forged_readiness, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            validation = validate_artifacts(agentic_training_loop_plan_paths=[loop_plan], strict=True)

            self.assertFalse(validation["passed"], validation)
            errors = "\n".join(error for target in validation["targets"] for error in target["errors"])
            self.assertIn("agentic_training_loop_plan.readiness expected 'planned_fail_closed'", errors)
            self.assertIn(
                "agentic_training_loop_plan.recommendation expected 'collect_missing_plan_evidence'",
                errors,
            )
            for check in plan["checks"]:
                if check["id"] == "cloud_training_receipts_are_side_effect_free":
                    check["passed"] = True
            plan["cloud_training_receipt_state"]["live_launch_requested"] = False
            plan["cloud_training_receipt_state"]["fail_closed"] = True
            loop_plan.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            validation = validate_artifacts(agentic_training_loop_plan_paths=[loop_plan], strict=True)

            self.assertFalse(validation["passed"], validation)
            errors = "\n".join(error for target in validation["targets"] for error in target["errors"])
            self.assertIn(
                "agentic_training_loop_plan.cloud_training_receipt_state.live_launch_requested must match cloud training receipt artifacts.",
                errors,
            )
            self.assertIn(
                "agentic_training_loop_plan.cloud_training_receipt_state.fail_closed must match cloud training receipt artifacts.",
                errors,
            )
            self.assertIn(
                "agentic_training_loop_plan.checks.cloud_training_receipts_are_side_effect_free.passed must match receipt state.",
                errors,
            )

    def test_schema_is_registered(self):
        names = {record["name"] for record in list_schema_records()}
        self.assertIn("agentic_training_loop_plan", names)

    def write_loop_artifacts(self, root: Path) -> dict[str, list[Path]]:
        return copy_valid_loop_artifacts(root)

    def write_json(self, path: Path, schema_version: str, source_artifacts: dict[str, dict[str, object]] | None = None) -> Path:
        if schema_version == "hfr.agentic_training_plan.v1":
            shutil.copyfile(ROOT / "examples" / "agentic_training" / "plans" / "sft_then_dpo_plan.json", path)
            return path
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
