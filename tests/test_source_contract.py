import json
import shutil
import tempfile
import unittest
from pathlib import Path

from flightrecorder import validation
from flightrecorder.agentic_training_loop_plan import PHASES
from flightrecorder.schema_registry import check_schema_contract
from flightrecorder.source_contract import (
    _DIRECTORY_MANIFESTS,
    _GATE_CONTRACT_ROLES,
    _SEMANTIC_VALIDATOR_NAMES,
    inspect_artifact_source,
)


ROOT = Path(__file__).resolve().parents[1]


class SourceContractTests(unittest.TestCase):
    def test_all_loop_readiness_roles_have_semantic_contracts(self):
        required_roles = {role for phase in PHASES for role in phase["required"]}
        downstream_roles = {
            "action_ledger",
            "agentic_loop_ledger",
            "agentic_rollout_plan",
            "agentic_rollout_receipt",
            "agentic_training_runtime_preflight",
            "improvement_ledger",
            "rejection_sampling_gate",
            "review_calibration",
            "reviewed_gate",
            "training_export",
        }
        special_roles = {
            "training_export",
            *_DIRECTORY_MANIFESTS,
            *_GATE_CONTRACT_ROLES,
        }

        self.assertEqual(
            (required_roles | downstream_roles) - set(_SEMANTIC_VALIDATOR_NAMES) - special_roles,
            set(),
        )
        for validator_name in _SEMANTIC_VALIDATOR_NAMES.values():
            self.assertTrue(callable(getattr(validation, validator_name, None)), validator_name)

    def test_runtime_preflight_dispatches_to_full_semantic_validator(self):
        with tempfile.TemporaryDirectory() as tmp:
            examples = Path(tmp) / "examples"
            shutil.copytree(ROOT / "examples", examples)
            preflight_path = examples / "agentic_training" / "runtime_preflight" / "ready.json"
            preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
            preflight["checks"][0]["passed"] = False
            preflight_path.write_text(json.dumps(preflight, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            schema = check_schema_contract(preflight, name_or_id="agentic_training_runtime_preflight")
            source = inspect_artifact_source(preflight_path, "agentic_training_runtime_preflight")

            self.assertTrue(schema["passed"], schema["errors"])
            self.assertTrue(source["schema_valid"])
            self.assertFalse(source["semantic_valid"])
            self.assertFalse(source["ready"])

    def test_schema_valid_reviewed_gate_with_forged_nested_check_is_not_ready(self):
        with tempfile.TemporaryDirectory() as tmp:
            gate_path = Path(tmp) / "reviewed_gate.json"
            gate = json.loads(
                (ROOT / "examples" / "agentic_training" / "model_grader" / "reviewed_gate.json").read_text(
                    encoding="utf-8"
                )
            )
            gate["checks"][0]["passed"] = False
            gate_path.write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            schema = check_schema_contract(gate, name_or_id="reviewed_gate")
            source = inspect_artifact_source(gate_path, "reviewed_gate")

            self.assertTrue(schema["passed"], schema["errors"])
            self.assertTrue(source["schema_valid"])
            self.assertFalse(source["semantic_valid"])
            self.assertFalse(source["ready"])

    def test_schema_valid_rollout_plan_with_forged_nested_check_is_not_ready(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rollout_root = root / "rollouts"
            shutil.copytree(ROOT / "examples" / "agentic_training" / "rollouts", rollout_root)
            plan_path = rollout_root / "rollout_plan.json"
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            plan["checks"][0]["passed"] = False
            plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            schema = check_schema_contract(plan, name_or_id="agentic_rollout_plan")
            source = inspect_artifact_source(plan_path, "agentic_rollout_plan")

            self.assertTrue(schema["passed"], schema["errors"])
            self.assertTrue(source["schema_valid"])
            self.assertFalse(source["semantic_valid"])
            self.assertFalse(source["ready"])

    def test_schema_valid_action_ledger_with_forged_metrics_is_not_ready(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            examples = root / "examples"
            shutil.copytree(ROOT / "examples", examples)
            ledger_path = examples / "agentic_training" / "iteration_ledgers" / "action_ledger.json"
            ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
            ledger["metrics"]["action_count"] = 0
            ledger_path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            schema = check_schema_contract(ledger, name_or_id="action_ledger")
            source = inspect_artifact_source(ledger_path, "action_ledger")

            self.assertTrue(schema["passed"], schema["errors"])
            self.assertTrue(source["schema_valid"])
            self.assertFalse(source["semantic_valid"])
            self.assertFalse(source["ready"])


if __name__ == "__main__":
    unittest.main()
