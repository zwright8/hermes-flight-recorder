from __future__ import annotations

import json
import hashlib
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path

from flightrecorder.loop_controller import (
    CommandControllerAdapter,
    ControllerError,
    InMemoryControllerAdapter,
    build_controller_plan as _build_controller_plan,
    run_controller,
)
from flightrecorder.loop_controller import _new_phase_record
from flightrecorder.schema_registry import check_schema_contract


DEADLINE_AT = "2099-01-01T00:00:00+00:00"


def build_controller_plan(**kwargs):
    return _build_controller_plan(deadline_at=DEADLINE_AT, **kwargs)


class LoopControllerTests(unittest.TestCase):
    @staticmethod
    def _refingerprint(plan: dict[str, object]) -> None:
        payload = {key: value for key, value in plan.items() if key != "plan_fingerprint"}
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        plan["plan_fingerprint"] = hashlib.sha256(encoded).hexdigest()

    def test_command_adapter_measures_duration_without_persisting_output(self) -> None:
        result = CommandControllerAdapter(allow_external=False).execute(
            {
                "id": "collect",
                "command": [sys.executable, "-c", "print('visible@example.test')"],
                "timeout_seconds": 5,
                "external_side_effect": False,
            },
            idempotency_key="test-command-duration",
            context={},
        )
        self.assertTrue(result["passed"])
        self.assertGreater(result["duration_seconds"], 0)
        self.assertRegex(result["stdout_sha256"], r"^[0-9a-f]{64}$")
        self.assertNotIn("stdout_preview", result)
        self.assertNotIn("stderr_preview", result)

    def test_reconciliation_uses_remaining_duration_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result_path = Path(temp_dir) / "result.json"
            phase = {
                "id": "train",
                "external_side_effect": True,
                "timeout_seconds": 5,
                "result_path": str(result_path),
                "reconcile_command": [sys.executable, "-c", "import time; time.sleep(0.2)"],
            }
            with self.assertRaises(subprocess.TimeoutExpired):
                CommandControllerAdapter(allow_external=True).reconcile(
                    phase,
                    idempotency_key="duration-bound-reconcile",
                    context={"fencing_token": 1, "timeout_seconds": 0.05},
                )

    def test_identical_plan_inputs_produce_identical_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            arguments = {
                "controller_id": "loop-deterministic-plan",
                "artifact_dir": Path(temp_dir) / "artifacts",
                "candidate_model": "candidate-v2",
                "champion_model": "champion-v1",
                "canary_percentages": [1, 10],
                "budget": {"max_cost_usd": 1.0, "max_duration_seconds": 60, "max_attempts": 30},
            }
            self.assertEqual(build_controller_plan(**arguments), build_controller_plan(**arguments))

    def test_concurrent_owner_cannot_overwrite_active_controller_lock(self) -> None:
        class BlockingAdapter(InMemoryControllerAdapter):
            def __init__(self) -> None:
                super().__init__()
                self.started = threading.Event()
                self.release = threading.Event()

            def execute(self, phase, *, idempotency_key, context):
                if phase["id"] == "collect":
                    self.started.set()
                    self.release.wait(timeout=5)
                return super().execute(phase, idempotency_key=idempotency_key, context=context)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            plan = build_controller_plan(
                controller_id="loop-lock",
                artifact_dir=root / "artifacts",
                candidate_model="candidate-v2",
                champion_model="champion-v1",
                canary_percentages=[],
                budget={"max_cost_usd": 1.0, "max_duration_seconds": 60, "max_attempts": 30},
            )
            approvals = {phase["id"]: plan["plan_fingerprint"] for phase in plan["phases"] if phase["requires_approval"]}
            adapter = BlockingAdapter()
            result: dict[str, object] = {}

            def run_first() -> None:
                result["state"] = run_controller(
                    plan,
                    state_path=root / "state.json",
                    adapter=adapter,
                    approvals=approvals,
                    owner_id="worker-a",
                )

            thread = threading.Thread(target=run_first)
            thread.start()
            self.assertTrue(adapter.started.wait(timeout=2))
            with self.assertRaisesRegex(ControllerError, "locked by another owner"):
                run_controller(
                    plan,
                    state_path=root / "state.json",
                    adapter=InMemoryControllerAdapter(),
                    approvals=approvals,
                    owner_id="worker-b",
                )
            adapter.release.set()
            thread.join(timeout=5)
            self.assertFalse(thread.is_alive())
            self.assertEqual(result["state"]["status"], "complete")

    def test_complete_loop_is_idempotent_and_receipt_backed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            plan = build_controller_plan(
                controller_id="loop-1",
                artifact_dir=root / "artifacts",
                candidate_model="candidate-v2",
                champion_model="champion-v1",
                canary_percentages=[1, 5, 25, 100],
                budget={"max_cost_usd": 10.0, "max_duration_seconds": 3600, "max_attempts": 50},
            )
            adapter = InMemoryControllerAdapter()
            approvals = {phase["id"]: plan["plan_fingerprint"] for phase in plan["phases"] if phase["requires_approval"]}
            self.assertTrue(check_schema_contract(plan, name_or_id="agentic_loop_controller_plan")["passed"])
            state = run_controller(
                plan,
                state_path=root / "controller_state.json",
                adapter=adapter,
                approvals=approvals,
                owner_id="worker-a",
            )
            self.assertEqual(state["status"], "complete")
            self.assertEqual(state["active_model"], "candidate-v2")
            self.assertTrue(state["next_iteration"]["scheduled"])
            self.assertEqual(set(adapter.call_counts.values()), {1})
            self.assertTrue(all(Path(row["receipt_path"]).is_file() for row in state["phase_records"]))
            self.assertTrue(check_schema_contract(state, name_or_id="agentic_loop_controller_state")["passed"])
            for row in state["phase_records"]:
                receipt = json.loads(Path(row["receipt_path"]).read_text(encoding="utf-8"))
                self.assertTrue(check_schema_contract(receipt, name_or_id="agentic_loop_phase_receipt")["passed"])

            second = run_controller(
                plan,
                state_path=root / "controller_state.json",
                adapter=adapter,
                approvals=approvals,
                owner_id="worker-b",
            )
            self.assertEqual(second, state)
            self.assertEqual(set(adapter.call_counts.values()), {1})

    def test_crash_resume_reuses_idempotency_key_without_duplicate_side_effect(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            plan = build_controller_plan(
                controller_id="loop-retry",
                artifact_dir=root / "artifacts",
                candidate_model="candidate-v2",
                champion_model="champion-v1",
                canary_percentages=[],
                budget={"max_cost_usd": 10.0, "max_duration_seconds": 3600, "max_attempts": 50},
                max_retries=2,
            )
            adapter = InMemoryControllerAdapter(fail_once={"train"})
            approvals = {phase["id"]: plan["plan_fingerprint"] for phase in plan["phases"] if phase["requires_approval"]}
            first = run_controller(
                plan,
                state_path=root / "state.json",
                adapter=adapter,
                approvals=approvals,
                owner_id="worker-a",
            )
            self.assertEqual(first["status"], "retryable_failure")
            train_before = next(row for row in first["phase_records"] if row["phase_id"] == "train")
            self.assertEqual(train_before["attempt_count"], 1)

            completed = run_controller(
                plan,
                state_path=root / "state.json",
                adapter=adapter,
                approvals=approvals,
                owner_id="worker-b",
            )
            self.assertEqual(completed["status"], "complete")
            train_after = next(row for row in completed["phase_records"] if row["phase_id"] == "train")
            self.assertEqual(train_after["attempt_count"], 2)
            self.assertEqual(train_after["idempotency_key"], train_before["idempotency_key"])
            self.assertEqual(adapter.side_effect_counts[train_after["idempotency_key"]], 1)

    def test_external_phase_requires_plan_bound_approval(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            plan = build_controller_plan(
                controller_id="loop-approval",
                artifact_dir=root / "artifacts",
                candidate_model="candidate-v2",
                champion_model="champion-v1",
                canary_percentages=[],
                budget={"max_cost_usd": 10.0, "max_duration_seconds": 3600, "max_attempts": 50},
            )
            state = run_controller(
                plan,
                state_path=root / "state.json",
                adapter=InMemoryControllerAdapter(),
                approvals={},
                owner_id="worker-a",
            )
            self.assertEqual(state["status"], "approval_required")
            self.assertEqual(state["current_phase"], "train")
            self.assertIn("train", state["pending_approvals"])

    def test_canary_guardrail_failure_rolls_back_and_verifies_champion(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            plan = build_controller_plan(
                controller_id="loop-rollback",
                artifact_dir=root / "artifacts",
                candidate_model="candidate-v2",
                champion_model="champion-v1",
                canary_percentages=[1, 5],
                budget={"max_cost_usd": 10.0, "max_duration_seconds": 3600, "max_attempts": 50},
                canary_guardrails={
                    "min_task_success_rate": 0.8,
                    "max_critical_failures": 0,
                    "max_cost_delta": 0.2,
                    "max_latency_delta": 0.2,
                },
            )
            adapter = InMemoryControllerAdapter(
                outcomes={
                    "canary_005": {
                        "passed": True,
                        "metrics": {
                            "task_success_rate": 0.7,
                            "critical_failures": 0,
                            "cost_delta": 0.0,
                            "latency_delta": 0.0,
                        },
                    }
                }
            )
            approvals = {phase["id"]: plan["plan_fingerprint"] for phase in plan["phases"] if phase["requires_approval"]}
            approvals["rollback"] = plan["plan_fingerprint"]
            approvals["post_rollback_smoke"] = plan["plan_fingerprint"]
            state = run_controller(
                plan,
                state_path=root / "state.json",
                adapter=adapter,
                approvals=approvals,
                owner_id="worker-a",
            )
            self.assertEqual(state["status"], "rolled_back")
            self.assertEqual(state["active_model"], "champion-v1")
            self.assertTrue(state["rollback"]["post_rollback_smoke_passed"])
            self.assertTrue(state["next_iteration"]["scheduled"])
            self.assertEqual(adapter.call_counts["rollback"], 1)
            self.assertEqual(adapter.call_counts["post_rollback_smoke"], 1)
            self.assertEqual(adapter.call_counts["failure_analysis"], 1)
            self.assertEqual(adapter.call_counts["intervention_route"], 1)
            self.assertTrue(state["failure_analysis"]["analysis_passed"])
            self.assertTrue(state["failure_analysis"]["intervention_route_passed"])
            self.assertEqual(state["failure_analysis"]["selected_intervention"], "prompt_policy")
            self.assertEqual(
                state["failure_analysis"]["work_item"]["intervention"],
                state["failure_analysis"]["selected_intervention"],
            )
            routed = {row["phase_id"] for row in state["phase_records"]}
            self.assertTrue({"failure_analysis", "intervention_route"}.issubset(routed))

    def test_plan_overrides_cannot_weaken_execution_safety(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            common = {
                "controller_id": "loop-unsafe-override",
                "artifact_dir": Path(temp_dir),
                "candidate_model": "candidate-v2",
                "champion_model": "champion-v1",
                "canary_percentages": [1],
                "budget": {"max_cost_usd": 1.0, "max_duration_seconds": 10, "max_attempts": 20},
            }
            for field, value in (
                ("requires_approval", False),
                ("external_side_effect", False),
                ("depends_on", []),
                ("id", "collect"),
            ):
                with self.subTest(field=field):
                    with self.assertRaisesRegex(ControllerError, "cannot change"):
                        build_controller_plan(**common, phase_overrides={"train": {field: value}})
            plan = build_controller_plan(**common)
            shadow = next(phase for phase in plan["phases"] if phase["id"] == "shadow")
            self.assertTrue(shadow["requires_approval"])
            self.assertTrue(shadow["external_side_effect"])

    def test_refingerprinted_plan_cannot_bypass_semantic_safety(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            plan = build_controller_plan(
                controller_id="loop-forged",
                artifact_dir=root / "artifacts",
                candidate_model="candidate-v2",
                champion_model="champion-v1",
                canary_percentages=[1],
                budget={"max_cost_usd": 1.0, "max_duration_seconds": 60, "max_attempts": 30},
            )
            train = next(phase for phase in plan["phases"] if phase["id"] == "train")
            train["requires_approval"] = False
            self._refingerprint(plan)
            with self.assertRaisesRegex(ControllerError, "unsafe approval"):
                run_controller(
                    plan,
                    state_path=root / "state.json",
                    adapter=InMemoryControllerAdapter(),
                    approvals={},
                    owner_id="worker-a",
                )

    def test_external_command_requires_durable_reconciliation_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ControllerError, "result_path and reconcile_command"):
                build_controller_plan(
                    controller_id="loop-command-contract",
                    artifact_dir=Path(temp_dir),
                    candidate_model="candidate-v2",
                    champion_model="champion-v1",
                    canary_percentages=[],
                    budget={"max_cost_usd": 1.0, "max_duration_seconds": 10, "max_attempts": 20},
                    phase_overrides={"train": {"command": [sys.executable, "-c", "pass"]}},
                )

    def test_command_adapter_durable_result_survives_process_restart(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            result_path = root / "result.json"
            phase = {
                "id": "collect",
                "command": [sys.executable, "-c", "pass"],
                "result_path": str(result_path),
                "timeout_seconds": 5,
                "external_side_effect": False,
            }
            first = CommandControllerAdapter(allow_external=False).execute(
                phase,
                idempotency_key="durable-command",
                context={"fencing_token": 3},
            )
            durable_path = root / ".result.json.hfr-adapter-result.json"
            self.assertTrue(durable_path.is_file())
            second = CommandControllerAdapter(allow_external=False).reconcile(
                phase,
                idempotency_key="durable-command",
                context={"fencing_token": 4},
            )
            self.assertEqual(second, first)

    def test_inflight_resume_reconciles_original_fencing_token(self) -> None:
        class ReconcilingAdapter(InMemoryControllerAdapter):
            def __init__(self) -> None:
                super().__init__()
                self.reconciled_fencing_token = None

            def reconcile(self, phase, *, idempotency_key, context):
                self.reconciled_fencing_token = context["fencing_token"]
                return {
                    "passed": True,
                    "cost_usd": 0.0,
                    "duration_seconds": 0.01,
                    "metrics": {},
                    "output_artifacts": [],
                }

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            plan = build_controller_plan(
                controller_id="loop-fencing",
                artifact_dir=root / "artifacts",
                candidate_model="candidate-v2",
                champion_model="champion-v1",
                canary_percentages=[],
                budget={"max_cost_usd": 1.0, "max_duration_seconds": 60, "max_attempts": 30},
            )
            state_path = root / "state.json"
            initial = run_controller(
                plan,
                state_path=state_path,
                adapter=InMemoryControllerAdapter(),
                approvals={},
                owner_id="worker-a",
                max_steps=1,
            )
            self.assertEqual(initial["lease"]["fencing_token"], 1)
            govern = next(phase for phase in plan["phases"] if phase["id"] == "govern_curate")
            record = _new_phase_record(plan, govern, initial)
            record.update({"status": "running", "operation_status": "prepared", "fencing_token": 1, "attempt_count": 1})
            initial["phase_records"].append(record)
            initial["total_attempts"] += 1
            state_path.write_text(json.dumps(initial), encoding="utf-8")
            adapter = ReconcilingAdapter()
            resumed = run_controller(
                plan,
                state_path=state_path,
                adapter=adapter,
                approvals={},
                owner_id="worker-b",
                max_steps=1,
            )
            self.assertEqual(adapter.reconciled_fencing_token, 1)
            self.assertEqual(resumed["lease"]["fencing_token"], 2)
            self.assertEqual(next(row for row in resumed["phase_records"] if row["phase_id"] == "govern_curate")["status"], "completed")

    def test_inflight_operation_is_not_reexecuted_without_reconciliation(self) -> None:
        class UnknownAdapter(InMemoryControllerAdapter):
            def reconcile(self, phase, *, idempotency_key, context):
                return None

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            plan = build_controller_plan(
                controller_id="loop-no-reconcile",
                artifact_dir=root / "artifacts",
                candidate_model="candidate-v2",
                champion_model="champion-v1",
                canary_percentages=[],
                budget={"max_cost_usd": 1.0, "max_duration_seconds": 60, "max_attempts": 30},
            )
            state_path = root / "state.json"
            initial = run_controller(
                plan,
                state_path=state_path,
                adapter=InMemoryControllerAdapter(),
                approvals={},
                owner_id="worker-a",
                max_steps=1,
            )
            govern = next(phase for phase in plan["phases"] if phase["id"] == "govern_curate")
            record = _new_phase_record(plan, govern, initial)
            record.update({"status": "running", "operation_status": "prepared", "fencing_token": 1, "attempt_count": 1})
            initial["phase_records"].append(record)
            initial["total_attempts"] += 1
            state_path.write_text(json.dumps(initial), encoding="utf-8")
            adapter = UnknownAdapter()
            resumed = run_controller(
                plan,
                state_path=state_path,
                adapter=adapter,
                approvals={},
                owner_id="worker-b",
            )
            self.assertEqual(resumed["status"], "reconciliation_required")
            self.assertEqual(adapter.call_counts["govern_curate"], 0)

    def test_budget_stops_before_unapproved_cost(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            plan = build_controller_plan(
                controller_id="loop-budget",
                artifact_dir=root / "artifacts",
                candidate_model="candidate-v2",
                champion_model="champion-v1",
                canary_percentages=[],
                budget={"max_cost_usd": 0.0, "max_duration_seconds": 3600, "max_attempts": 50},
                phase_overrides={"collect": {"estimated_cost_usd": 0.01}},
            )
            adapter = InMemoryControllerAdapter(outcomes={"collect": {"passed": True, "cost_usd": 0.01}})
            state = run_controller(
                plan,
                state_path=root / "state.json",
                adapter=adapter,
                approvals={},
                owner_id="worker-a",
            )
            self.assertEqual(state["status"], "budget_exhausted")
            self.assertEqual(state["current_phase"], "collect")
            self.assertEqual(adapter.call_counts["collect"], 0)

    def test_duration_budget_clamps_phase_timeout_and_rejects_overrun(self) -> None:
        class CapturingAdapter(InMemoryControllerAdapter):
            def __init__(self) -> None:
                super().__init__(
                    outcomes={
                        "collect": {"duration_seconds": 0.9},
                        "govern_curate": {"duration_seconds": 0.9},
                    }
                )
                self.timeouts: dict[str, float] = {}

            def execute(self, phase, *, idempotency_key, context):
                self.timeouts[phase["id"]] = context["timeout_seconds"]
                return super().execute(phase, idempotency_key=idempotency_key, context=context)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            plan = build_controller_plan(
                controller_id="loop-duration-budget",
                artifact_dir=root / "artifacts",
                candidate_model="candidate-v2",
                champion_model="champion-v1",
                canary_percentages=[],
                budget={"max_cost_usd": 1.0, "max_duration_seconds": 1.0, "max_attempts": 30},
            )
            self.assertGreater(plan["deadline_at"], "")
            adapter = CapturingAdapter()
            state = run_controller(
                plan,
                state_path=root / "state.json",
                adapter=adapter,
                approvals={},
                owner_id="worker-a",
            )
            self.assertLessEqual(adapter.timeouts["govern_curate"], 0.1)
            self.assertLessEqual(state["elapsed_duration_seconds"], 1.0)
            self.assertEqual(state["status"], "repair_required")

    def test_execution_exception_is_analyzed_and_routed(self) -> None:
        class FailingAdapter(InMemoryControllerAdapter):
            def execute(self, phase, *, idempotency_key, context):
                if phase["id"] == "collect":
                    raise RuntimeError("provider token=private-value failed")
                return super().execute(phase, idempotency_key=idempotency_key, context=context)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            plan = build_controller_plan(
                controller_id="loop-exception-route",
                artifact_dir=root / "artifacts",
                candidate_model="candidate-v2",
                champion_model="champion-v1",
                canary_percentages=[],
                budget={"max_cost_usd": 1.0, "max_duration_seconds": 60, "max_attempts": 30},
            )
            adapter = FailingAdapter()
            state = run_controller(
                plan,
                state_path=root / "state.json",
                adapter=adapter,
                approvals={},
                owner_id="worker-a",
            )
            self.assertEqual(state["status"], "repair_required")
            self.assertEqual(state["failure_analysis"]["selected_intervention"], "parser_runtime")
            self.assertEqual(adapter.call_counts["failure_analysis"], 1)
            self.assertEqual(adapter.call_counts["intervention_route"], 1)
            collect = next(row for row in state["phase_records"] if row["phase_id"] == "collect")
            self.assertNotIn("private-value", collect["last_error"])

    def test_failure_analysis_can_change_intervention_route(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            analyzed_cluster = {
                "cluster_id": "analyzed-tool-schema",
                "failure_modes": ["invalid_tool_arguments"],
                "severity": "high",
                "confidence": 0.99,
                "frequency": 3,
                "affected_task_families": ["mail"],
                "affected_tools": ["mail.send"],
                "affected_policies": [],
                "evidence_refs": [{"artifact": "analysis.json", "sha256": "a" * 64}],
            }
            plan = build_controller_plan(
                controller_id="loop-analysis-route",
                artifact_dir=root / "artifacts",
                candidate_model="candidate-v2",
                champion_model="champion-v1",
                canary_percentages=[],
                budget={"max_cost_usd": 1.0, "max_duration_seconds": 60, "max_attempts": 30},
            )
            adapter = InMemoryControllerAdapter(
                outcomes={
                    "evaluate": {"passed": False},
                    "failure_analysis": {"passed": True, "failure_cluster": analyzed_cluster},
                }
            )
            approvals = {
                phase["id"]: plan["plan_fingerprint"]
                for phase in plan["phases"]
                if phase["requires_approval"]
            }
            state = run_controller(
                plan,
                state_path=root / "state.json",
                adapter=adapter,
                approvals=approvals,
                owner_id="worker-a",
            )
            self.assertEqual(state["status"], "repair_required")
            self.assertEqual(state["failure_analysis"]["failure_cluster"], analyzed_cluster)
            self.assertEqual(state["failure_analysis"]["selected_intervention"], "tool_schema")

    def test_terminal_state_and_receipts_are_replayed_before_return(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            plan = build_controller_plan(
                controller_id="loop-state-replay",
                artifact_dir=root / "artifacts",
                candidate_model="candidate-v2",
                champion_model="champion-v1",
                canary_percentages=[],
                budget={"max_cost_usd": 1.0, "max_duration_seconds": 60, "max_attempts": 30},
            )
            approvals = {
                phase["id"]: plan["plan_fingerprint"]
                for phase in plan["phases"]
                if phase["requires_approval"]
            }
            state_path = root / "state.json"
            state = run_controller(
                plan,
                state_path=state_path,
                adapter=InMemoryControllerAdapter(),
                approvals=approvals,
                owner_id="worker-a",
            )
            state["phase_records"][-1]["receipt_sha256"] = "f" * 64
            state_path.write_text(json.dumps(state), encoding="utf-8")
            with self.assertRaisesRegex(ControllerError, "receipt hash does not match"):
                run_controller(
                    plan,
                    state_path=state_path,
                    adapter=InMemoryControllerAdapter(),
                    approvals=approvals,
                    owner_id="worker-b",
                )

    def test_nonterminal_active_model_cannot_be_forged(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            plan = build_controller_plan(
                controller_id="loop-active-model-replay",
                artifact_dir=root / "artifacts",
                candidate_model="candidate-v2",
                champion_model="champion-v1",
                canary_percentages=[],
                budget={"max_cost_usd": 1.0, "max_duration_seconds": 60, "max_attempts": 30},
            )
            state_path = root / "state.json"
            state = run_controller(
                plan,
                state_path=state_path,
                adapter=InMemoryControllerAdapter(),
                approvals={},
                owner_id="worker-a",
                max_steps=1,
            )
            self.assertEqual(state["status"], "paused")
            state["active_model"] = "attacker-controlled"
            state_path.write_text(json.dumps(state), encoding="utf-8")
            with self.assertRaisesRegex(ControllerError, "active_model does not replay"):
                run_controller(
                    plan,
                    state_path=state_path,
                    adapter=InMemoryControllerAdapter(),
                    approvals={},
                    owner_id="worker-b",
                )

    def test_plan_rejects_invalid_canary_order(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(ControllerError):
                build_controller_plan(
                    controller_id="loop-invalid",
                    artifact_dir=Path(temp_dir),
                    candidate_model="candidate-v2",
                    champion_model="champion-v1",
                    canary_percentages=[5, 1],
                    budget={"max_cost_usd": 1.0, "max_duration_seconds": 10, "max_attempts": 5},
                )


if __name__ == "__main__":
    unittest.main()
