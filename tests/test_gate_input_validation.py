import json
import unittest
from pathlib import Path

from flightrecorder.action_gate import evaluate_action_ledger_gate
from flightrecorder.improvement_gate import evaluate_improvement_ledger_gate
from flightrecorder.promotion_gate import evaluate_promotion_ledger_gate
from flightrecorder.suite_gate import evaluate_suite_gate

ROOT = Path(__file__).resolve().parents[1]


def _load(relative_path: str) -> dict:
    return json.loads((ROOT / relative_path).read_text(encoding="utf-8"))


class GateInputValidationTests(unittest.TestCase):
    def test_suite_gate_rejects_schema_invalid_summary(self):
        with self.assertRaisesRegex(ValueError, "run_suite schema"):
            evaluate_suite_gate({}, suite_summary_path="suite_summary.json")

    def test_action_gate_rejects_schema_invalid_ledger(self):
        with self.assertRaisesRegex(ValueError, "action_ledger schema"):
            evaluate_action_ledger_gate(
                {"schema_version": "hfr.action_ledger.v1", "metrics": {}, "entries": []},
                action_ledger_path="action_ledger.json",
                max_open_actions=0,
            )

    def test_improvement_gate_rejects_schema_invalid_ledger(self):
        with self.assertRaisesRegex(ValueError, "improvement_ledger schema"):
            evaluate_improvement_ledger_gate(
                {"schema_version": "hfr.improvement_ledger.v1", "metrics": {}, "entries": []},
                improvement_ledger_path="improvement_ledger.json",
                max_open_work_items=0,
            )

    def test_promotion_gate_rejects_schema_invalid_ledger(self):
        with self.assertRaisesRegex(ValueError, "promotion_ledger schema"):
            evaluate_promotion_ledger_gate(
                {"schema_version": "hfr.promotion_ledger.v1", "metrics": {}, "records": []},
                promotion_ledger_path="promotion_ledger.json",
                max_blocked_count=0,
            )

    def test_suite_gate_rejects_schema_valid_forged_aggregate_metrics(self):
        path = ROOT / "examples" / "agentic_training" / "evidence_handoff" / "suite_summary.json"
        summary = _load("examples/agentic_training/evidence_handoff/suite_summary.json")
        summary["metrics"]["passed"] += 1

        with self.assertRaisesRegex(ValueError, "semantic validation"):
            evaluate_suite_gate(summary, suite_summary_path=path)

    def test_suite_gate_rejects_schema_valid_forged_task_family_metrics(self):
        path = ROOT / "examples" / "agentic_training" / "evidence_handoff" / "suite_summary.json"
        summary = _load("examples/agentic_training/evidence_handoff/suite_summary.json")
        summary["metrics"]["task_families"][0]["total"] += 1

        with self.assertRaisesRegex(ValueError, "semantic validation"):
            evaluate_suite_gate(summary, suite_summary_path=path)

    def test_suite_gate_rejects_deleted_task_family_metrics(self):
        path = ROOT / "examples" / "agentic_training" / "evidence_handoff" / "suite_summary.json"
        summary = _load("examples/agentic_training/evidence_handoff/suite_summary.json")
        summary["metrics"]["task_families"] = []

        with self.assertRaisesRegex(ValueError, "semantic validation"):
            evaluate_suite_gate(summary, suite_summary_path=path)

    def test_action_gate_rejects_schema_valid_forged_aggregate_metrics(self):
        path = ROOT / "examples" / "agentic_training" / "iteration_ledgers" / "action_ledger.json"
        ledger = _load("examples/agentic_training/iteration_ledgers/action_ledger.json")
        ledger["metrics"]["open_action_count"] += 1

        with self.assertRaisesRegex(ValueError, "semantic validation"):
            evaluate_action_ledger_gate(ledger, action_ledger_path=path, max_open_actions=999)

    def test_improvement_gate_rejects_schema_valid_forged_aggregate_metrics(self):
        path = ROOT / "examples" / "agentic_training" / "iteration_ledgers" / "improvement_ledger.json"
        ledger = _load("examples/agentic_training/iteration_ledgers/improvement_ledger.json")
        ledger["metrics"]["open_work_item_count"] += 1

        with self.assertRaisesRegex(ValueError, "semantic validation"):
            evaluate_improvement_ledger_gate(ledger, improvement_ledger_path=path, max_open_work_items=999)

    def test_promotion_gate_rejects_schema_valid_forged_aggregate_metrics(self):
        path = ROOT / "examples" / "agentic_training" / "promotion_governance" / "promotion_ledger.json"
        ledger = _load("examples/agentic_training/promotion_governance/promotion_ledger.json")
        ledger["metrics"]["allowed_count"] += 1

        with self.assertRaisesRegex(ValueError, "semantic validation"):
            evaluate_promotion_ledger_gate(ledger, promotion_ledger_path=path, max_blocked_count=999)


if __name__ == "__main__":
    unittest.main()
