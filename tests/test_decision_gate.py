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


class DecisionGateTests(unittest.TestCase):
    def test_gate_decision_allows_expected_recommendation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "action_ledger_gate.json"
            decision_gate = root / "decision_gate.json"
            source.write_text(
                json.dumps(
                    {
                        "schema_version": "hfr.action_ledger_gate.v1",
                        "passed": True,
                        "decision": {
                            "readiness": "ready",
                            "recommendation": "promote_iteration",
                            "summary": "ok",
                            "blocking_check_count": 0,
                            "key_metrics": {"recurring_action_count": 0},
                        },
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )

            code = run_cli(
                [
                    "gate-decision",
                    "--artifact",
                    str(source),
                    "--expect-recommendation",
                    "promote_iteration",
                    "--expect-readiness",
                    "ready",
                    "--require-passed",
                    "--preserve-paths",
                    "--out",
                    str(decision_gate),
                ]
            )

            self.assertEqual(code, 0)
            gate = json.loads(decision_gate.read_text(encoding="utf-8"))
            self.assertEqual(gate["schema_version"], "hfr.decision_gate.v1")
            self.assertTrue(gate["passed"])
            self.assertEqual(gate["recommendation"], "allow_promotion")
            self.assertEqual(gate["artifact"], str(source))
            self.assertEqual(gate["source_artifact"]["path"], str(source))
            self.assertTrue(gate["source_artifact"]["exists"])
            self.assertEqual(len(gate["source_artifact"]["sha256"]), 64)
            self.assertEqual(gate["source_decision"]["recommendation"], "promote_iteration")
            self.assertEqual(gate["source_decision"]["key_metrics"]["recurring_action_count"], 0)
            self.assertEqual(run_cli(["validate", "--decision-gate", str(decision_gate), "--strict"]), 0)

            tampered = json.loads(json.dumps(gate))
            tampered["source_decision"]["recommendation"] = "block_iteration"
            decision_gate.write_text(json.dumps(tampered, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            self.assertEqual(run_cli(["validate", "--decision-gate", str(decision_gate)]), 1)

            decision_gate.write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            self.assertEqual(run_cli(["validate", "--decision-gate", str(decision_gate), "--strict"]), 0)

            source.write_text(source.read_text(encoding="utf-8").replace("promote_iteration", "block_iteration"), encoding="utf-8")
            self.assertEqual(run_cli(["validate", "--decision-gate", str(decision_gate)]), 1)

    def test_gate_decision_blocks_unexpected_recommendation_but_validates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "action_ledger_gate.json"
            decision_gate = root / "decision_gate.json"
            source.write_text(
                json.dumps(
                    {
                        "schema_version": "hfr.action_ledger_gate.v1",
                        "passed": False,
                        "decision": {
                            "readiness": "blocked",
                            "recommendation": "block_iteration",
                            "summary": "blocked",
                            "blocking_check_count": 1,
                            "key_metrics": {"recurring_action_count": 3},
                        },
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )

            code = run_cli(
                [
                    "gate-decision",
                    "--artifact",
                    str(source),
                    "--expect-recommendation",
                    "promote_iteration",
                    "--expect-readiness",
                    "ready",
                    "--require-passed",
                    "--out",
                    str(decision_gate),
                ]
            )

            self.assertEqual(code, 1)
            gate = json.loads(decision_gate.read_text(encoding="utf-8"))
            self.assertFalse(gate["passed"])
            self.assertEqual(gate["recommendation"], "block_promotion")
            self.assertEqual(len(gate["source_artifact"]["sha256"]), 64)
            self.assertEqual(gate["source_decision"]["recommendation"], "block_iteration")
            self.assertEqual(run_cli(["validate", "--decision-gate", str(decision_gate), "--strict"]), 0)

            gate["failed_check_count"] = 0
            decision_gate.write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            self.assertEqual(run_cli(["validate", "--decision-gate", str(decision_gate)]), 1)


if __name__ == "__main__":
    unittest.main()
