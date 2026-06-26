import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

from flightrecorder.cli import main

ROOT = Path(__file__).resolve().parents[1]


def run_cli(args):
    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        return main(args)


class PromotionLedgerTests(unittest.TestCase):
    def test_promotion_ledger_tracks_allow_and_block_decisions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            allow_source = root / "allow_action_ledger_gate.json"
            block_source = root / "block_action_ledger_gate.json"
            allow_gate = root / "allow_decision_gate.json"
            block_gate = root / "block_decision_gate.json"
            ledger_path = root / "promotion_ledger.json"

            allow_source.write_text(
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
            block_source.write_text(
                json.dumps(
                    {
                        "schema_version": "hfr.action_ledger_gate.v1",
                        "passed": False,
                        "decision": {
                            "readiness": "blocked",
                            "recommendation": "block_iteration",
                            "summary": "repair pressure remains",
                            "blocking_check_count": 1,
                            "key_metrics": {"recurring_action_count": 2},
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
                        str(allow_source),
                        "--expect-recommendation",
                        "promote_iteration",
                        "--expect-readiness",
                        "ready",
                        "--require-passed",
                        "--preserve-paths",
                        "--out",
                        str(allow_gate),
                    ]
                ),
                0,
            )
            self.assertEqual(
                run_cli(
                    [
                        "gate-decision",
                        "--artifact",
                        str(block_source),
                        "--expect-recommendation",
                        "promote_iteration",
                        "--expect-readiness",
                        "ready",
                        "--require-passed",
                        "--preserve-paths",
                        "--out",
                        str(block_gate),
                    ]
                ),
                1,
            )

            code = run_cli(
                [
                    "promotion-ledger",
                    "--decision-gate",
                    str(allow_gate),
                    "--decision-gate",
                    str(block_gate),
                    "--preserve-paths",
                    "--out",
                    str(ledger_path),
                ]
            )

            self.assertEqual(code, 0)
            ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
            self.assertEqual(ledger["schema_version"], "hfr.promotion_ledger.v1")
            self.assertTrue(ledger["passed"])
            self.assertEqual(ledger["decision_count"], 2)
            self.assertEqual(ledger["metrics"]["decision_count"], 2)
            self.assertEqual(ledger["metrics"]["allowed_count"], 1)
            self.assertEqual(ledger["metrics"]["blocked_count"], 1)
            self.assertEqual(ledger["metrics"]["latest_recommendation"], "block_promotion")
            self.assertEqual(ledger["metrics"]["latest_readiness"], "blocked")
            self.assertFalse(ledger["metrics"]["latest_passed"])
            self.assertEqual(ledger["metrics"]["consecutive_allowed_count"], 0)
            self.assertEqual(ledger["metrics"]["consecutive_blocked_count"], 1)
            self.assertEqual(ledger["metrics"]["unique_source_artifact_count"], 2)
            self.assertEqual(
                ledger["metrics"]["recommendation_counts"],
                [{"count": 1, "id": "allow_promotion"}, {"count": 1, "id": "block_promotion"}],
            )
            self.assertEqual(
                ledger["metrics"]["source_recommendation_counts"],
                [{"count": 1, "id": "block_iteration"}, {"count": 1, "id": "promote_iteration"}],
            )
            self.assertEqual(ledger["records"][0]["source"]["recommendation"], "promote_iteration")
            self.assertEqual(ledger["records"][1]["source"]["recommendation"], "block_iteration")
            self.assertEqual(len(ledger["records"][0]["sha256"]), 64)
            self.assertEqual(len(ledger["records"][0]["source"]["artifact_sha256"]), 64)
            self.assertEqual(run_cli(["validate", "--promotion-ledger", str(ledger_path), "--strict"]), 0)

            ledger["metrics"]["allowed_count"] = 99
            ledger_path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            self.assertEqual(run_cli(["validate", "--promotion-ledger", str(ledger_path)]), 1)

    def test_gate_promotion_ledger_allows_clean_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "action_ledger_gate.json"
            decision_gate = root / "decision_gate.json"
            ledger_path = root / "promotion_ledger.json"
            gate_path = root / "promotion_ledger_gate.json"
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
            self.assertEqual(
                run_cli(
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
                ),
                0,
            )
            run_cli(
                [
                    "promotion-ledger",
                    "--decision-gate",
                    str(decision_gate),
                    "--preserve-paths",
                    "--out",
                    str(ledger_path),
                ]
            )

            code = run_cli(
                [
                    "gate-promotion-ledger",
                    "--promotion-ledger",
                    str(ledger_path),
                    "--policy",
                    str(ROOT / "examples" / "promotion_ledger_gate_policy.demo.json"),
                    "--out",
                    str(gate_path),
                ]
            )

            self.assertEqual(code, 0)
            gate = json.loads(gate_path.read_text(encoding="utf-8"))
            self.assertEqual(gate["schema_version"], "hfr.promotion_ledger_gate.v1")
            self.assertTrue(gate["passed"])
            self.assertEqual(gate["decision"]["recommendation"], "promote_iteration")
            self.assertEqual(gate["decision"]["readiness"], "ready")
            self.assertEqual(gate["metrics"]["blocked_rate"], 0.0)
            self.assertEqual(gate["metrics"]["failed_decision_count"], 0)
            self.assertEqual(gate["policy"]["schema_version"], "hfr.promotion_ledger_gate.policy.v1")
            self.assertEqual(gate["policy"]["effective"]["require_latest_recommendation"], "allow_promotion")
            self.assertTrue(gate["policy"]["effective"]["require_latest_passed"])
            self.assertEqual(run_cli(["validate", "--promotion-ledger-gate", str(gate_path), "--strict"]), 0)

            gate["metrics"]["blocked_rate"] = 1.0
            gate_path.write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            self.assertEqual(run_cli(["validate", "--promotion-ledger-gate", str(gate_path)]), 1)

    def test_gate_promotion_ledger_blocks_bad_latest_decision(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "blocked_action_ledger_gate.json"
            decision_gate = root / "decision_gate.json"
            ledger_path = root / "promotion_ledger.json"
            gate_path = root / "promotion_ledger_gate.json"
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
                            "key_metrics": {"recurring_action_count": 5},
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
                ),
                1,
            )
            run_cli(
                [
                    "promotion-ledger",
                    "--decision-gate",
                    str(decision_gate),
                    "--preserve-paths",
                    "--out",
                    str(ledger_path),
                ]
            )

            code = run_cli(
                [
                    "gate-promotion-ledger",
                    "--promotion-ledger",
                    str(ledger_path),
                    "--min-decisions",
                    "1",
                    "--require-latest-recommendation",
                    "allow_promotion",
                    "--require-latest-passed",
                    "--max-blocked-count",
                    "0",
                    "--max-consecutive-blocked",
                    "0",
                    "--max-failed-decisions",
                    "0",
                    "--forbid-source-recommendation",
                    "block_iteration",
                    "--out",
                    str(gate_path),
                ]
            )

            self.assertEqual(code, 1)
            gate = json.loads(gate_path.read_text(encoding="utf-8"))
            self.assertFalse(gate["passed"])
            self.assertEqual(gate["decision"]["recommendation"], "block_iteration")
            self.assertEqual(gate["decision"]["readiness"], "blocked")
            self.assertEqual(gate["metrics"]["blocked_rate"], 1.0)
            self.assertEqual(gate["metrics"]["failed_decision_count"], 1)
            failed_checks = {check["id"] for check in gate["checks"] if not check["passed"]}
            self.assertIn("require_latest_recommendation", failed_checks)
            self.assertIn("require_latest_passed", failed_checks)
            self.assertIn("max_blocked_count", failed_checks)
            self.assertIn("max_consecutive_blocked", failed_checks)
            self.assertIn("max_failed_decisions", failed_checks)
            self.assertIn("forbid_source_recommendation", failed_checks)
            self.assertEqual(run_cli(["validate", "--promotion-ledger-gate", str(gate_path), "--strict"]), 0)

    def test_promotion_ledger_rejects_wrong_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            wrong_gate = root / "not_a_decision_gate.json"
            wrong_gate.write_text(json.dumps({"schema_version": "hfr.not_a_decision_gate.v1"}) + "\n", encoding="utf-8")

            with self.assertRaises(SystemExit) as raised:
                run_cli(["promotion-ledger", "--decision-gate", str(wrong_gate)])

            self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
