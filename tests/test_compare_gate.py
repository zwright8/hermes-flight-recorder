import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

from flightrecorder.cli import main


ROOT = Path(__file__).resolve().parents[1]


def run_cli(args):
    with redirect_stdout(StringIO()):
        return main(args)


class CompareGateTests(unittest.TestCase):
    def test_gate_compare_export_accepts_candidate_improvement_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            baseline, candidate = self._paired_email_dirs(Path(tmp))
            compare_export = Path(tmp) / "compare_rl"
            gate_path = Path(tmp) / "gate.json"
            run_cli(
                [
                    "export-compare-rl",
                    "--baseline",
                    str(baseline),
                    "--candidate",
                    str(candidate),
                    "--out",
                    str(compare_export),
                ]
            )

            code = run_cli(
                [
                    "gate-compare-export",
                    "--compare-export",
                    str(compare_export),
                    "--policy",
                    str(ROOT / "examples" / "compare_gate_policy.demo.json"),
                    "--out",
                    str(gate_path),
                ]
            )

            self.assertEqual(code, 0)
            gate = json.loads(gate_path.read_text(encoding="utf-8"))
            self.assertEqual(gate["schema_version"], "hfr.compare_gate.v1")
            self.assertTrue(gate["passed"])
            self.assertEqual(gate["metrics"]["candidate_win_count"], 1)
            self.assertEqual(gate["metrics"]["baseline_win_count"], 0)
            self.assertEqual(gate["metrics"]["task_completion_improvement_count"], 1)
            self.assertEqual(gate["metrics"]["task_completion_regression_count"], 0)
            self.assertEqual(gate["policy"]["schema_version"], "hfr.compare_gate.policy.v1")
            self.assertIn("email_reply_completion", gate["metrics"]["candidate_win_scenarios"])
            self.assertIn("email_reply_completion", gate["metrics"]["task_completion_improvement_scenarios"])
            families = {row["task_family"]: row for row in gate["metrics"]["task_families"]}
            self.assertEqual(families["email_reply_completion"]["pair_count"], 1)
            self.assertEqual(families["email_reply_completion"]["candidate_win_count"], 1)
            self.assertEqual(families["email_reply_completion"]["task_completion_improvement_count"], 1)
            self.assertEqual(families["email_reply_completion"]["task_completion_regression_count"], 0)
            self.assertIn("required_actions", families["email_reply_completion"]["fixed_rule_counts"])
            self.assertEqual(gate["policy"]["effective"]["min_task_completion_improvements"], 1)
            self.assertEqual(gate["policy"]["effective"]["max_task_completion_regressions"], 0)
            self.assertEqual(
                gate["policy"]["effective"]["require_task_completion_improvement_scenarios"],
                ["email_reply_completion"],
            )
            self.assertEqual(
                gate["policy"]["effective"]["forbid_task_completion_regression_scenarios"],
                ["email_reply_completion"],
            )
            self.assertEqual(gate["policy"]["effective"]["task_family_gates"][0]["task_family"], "email_reply_completion")
            self.assertTrue(any(check.get("scope", {}).get("task_family") == "email_reply_completion" for check in gate["checks"]))

    def test_gate_compare_export_fails_strict_thresholds(self):
        with tempfile.TemporaryDirectory() as tmp:
            baseline, candidate = self._paired_email_dirs(Path(tmp))
            compare_export = Path(tmp) / "compare_rl"
            gate_path = Path(tmp) / "gate.json"
            run_cli(["export-compare-rl", "--baseline", str(baseline), "--candidate", str(candidate), "--out", str(compare_export)])

            code = run_cli(
                [
                    "gate-compare-export",
                    "--compare-export",
                    str(compare_export),
                    "--min-candidate-wins",
                    "2",
                    "--require-rule-fix",
                    "missing_rule",
                    "--out",
                    str(gate_path),
                ]
            )

            self.assertEqual(code, 1)
            gate = json.loads(gate_path.read_text(encoding="utf-8"))
            self.assertFalse(gate["passed"])
            failed_ids = [check["id"] for check in gate["checks"] if not check["passed"]]
            self.assertIn("min_candidate_wins", failed_ids)
            self.assertIn("require_rule_fix", failed_ids)

    def test_gate_compare_export_blocks_task_completion_regressions(self):
        with tempfile.TemporaryDirectory() as tmp:
            baseline, candidate = self._paired_email_dirs(Path(tmp))
            compare_export = Path(tmp) / "compare_rl"
            gate_path = Path(tmp) / "gate.json"
            run_cli(["export-compare-rl", "--baseline", str(candidate), "--candidate", str(baseline), "--out", str(compare_export)])

            code = run_cli(
                [
                    "gate-compare-export",
                    "--compare-export",
                    str(compare_export),
                    "--min-task-completion-improvements",
                    "1",
                    "--max-task-completion-regressions",
                    "0",
                    "--forbid-task-completion-regression-scenario",
                    "email_reply_completion",
                    "--out",
                    str(gate_path),
                ]
            )

            self.assertEqual(code, 1)
            gate = json.loads(gate_path.read_text(encoding="utf-8"))
            self.assertFalse(gate["passed"])
            self.assertEqual(gate["metrics"]["candidate_win_count"], 0)
            self.assertEqual(gate["metrics"]["baseline_win_count"], 1)
            self.assertEqual(gate["metrics"]["task_completion_improvement_count"], 0)
            self.assertEqual(gate["metrics"]["task_completion_regression_count"], 1)
            self.assertIn("email_reply_completion", gate["metrics"]["task_completion_regression_scenarios"])
            failed_ids = [check["id"] for check in gate["checks"] if not check["passed"]]
            self.assertIn("min_task_completion_improvements", failed_ids)
            self.assertIn("max_task_completion_regressions", failed_ids)
            self.assertIn("forbid_task_completion_regression_scenario", failed_ids)

    def test_gate_compare_export_blocks_task_family_regressions_from_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            baseline, candidate = self._paired_email_dirs(Path(tmp))
            compare_export = Path(tmp) / "compare_rl"
            gate_path = Path(tmp) / "gate.json"
            policy = Path(tmp) / "policy.json"
            policy.write_text(
                json.dumps(
                    {
                        "schema_version": "hfr.compare_gate.policy.v1",
                        "task_family_gates": [
                            {
                                "task_family": "email_reply_completion",
                                "min_candidate_wins": 1,
                                "min_task_completion_improvements": 1,
                                "max_baseline_wins": 0,
                                "max_task_completion_regressions": 0,
                                "require_rule_fixes": ["required_actions"],
                            }
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            run_cli(["export-compare-rl", "--baseline", str(candidate), "--candidate", str(baseline), "--out", str(compare_export)])

            code = run_cli(
                [
                    "gate-compare-export",
                    "--compare-export",
                    str(compare_export),
                    "--policy",
                    str(policy),
                    "--out",
                    str(gate_path),
                ]
            )

            self.assertEqual(code, 1)
            gate = json.loads(gate_path.read_text(encoding="utf-8"))
            failed_family_checks = [
                (check["id"], check.get("scope", {}).get("task_family"))
                for check in gate["checks"]
                if not check["passed"] and check.get("scope", {}).get("task_family")
            ]
            self.assertIn(("task_family_min_candidate_wins", "email_reply_completion"), failed_family_checks)
            self.assertIn(("task_family_min_task_completion_improvements", "email_reply_completion"), failed_family_checks)
            self.assertIn(("task_family_max_baseline_wins", "email_reply_completion"), failed_family_checks)
            self.assertIn(("task_family_max_task_completion_regressions", "email_reply_completion"), failed_family_checks)
            self.assertIn(("task_family_require_rule_fix", "email_reply_completion"), failed_family_checks)

    def test_gate_compare_export_can_block_contract_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            baseline, candidate = self._paired_email_dirs(Path(tmp))
            compare_export = Path(tmp) / "compare_rl"
            gate_path = Path(tmp) / "gate.json"
            run_cli(["export-compare-rl", "--baseline", str(baseline), "--candidate", str(candidate), "--out", str(compare_export)])

            code = run_cli(
                [
                    "gate-compare-export",
                    "--compare-export",
                    str(compare_export),
                    "--max-contract-drifts",
                    "0",
                    "--max-unverified-contracts",
                    "0",
                    "--out",
                    str(gate_path),
                ]
            )

            self.assertEqual(code, 1)
            gate = json.loads(gate_path.read_text(encoding="utf-8"))
            self.assertEqual(gate["metrics"]["contract_drift_count"], 1)
            self.assertEqual(gate["metrics"]["unverified_contract_count"], 0)
            failed_ids = [check["id"] for check in gate["checks"] if not check["passed"]]
            self.assertIn("max_contract_drifts", failed_ids)

    def test_gate_compare_export_rejects_malformed_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            baseline, candidate = self._paired_email_dirs(Path(tmp))
            compare_export = Path(tmp) / "compare_rl"
            policy = Path(tmp) / "bad_policy.json"
            run_cli(["export-compare-rl", "--baseline", str(baseline), "--candidate", str(candidate), "--out", str(compare_export)])
            policy.write_text(
                json.dumps(
                    {
                        "schema_version": "hfr.compare_gate.policy.v1",
                        "min_pairs": -1,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    main(["gate-compare-export", "--compare-export", str(compare_export), "--policy", str(policy)])

            self.assertEqual(raised.exception.code, 2)

    def test_gate_compare_export_rejects_malformed_task_family_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            baseline, candidate = self._paired_email_dirs(Path(tmp))
            compare_export = Path(tmp) / "compare_rl"
            policy = Path(tmp) / "bad_policy.json"
            run_cli(["export-compare-rl", "--baseline", str(baseline), "--candidate", str(candidate), "--out", str(compare_export)])
            policy.write_text(
                json.dumps(
                    {
                        "schema_version": "hfr.compare_gate.policy.v1",
                        "task_family_gates": [
                            {
                                "task_family": "email_reply_completion",
                                "min_task_completion_improvements": -1,
                            }
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    main(["gate-compare-export", "--compare-export", str(compare_export), "--policy", str(policy)])

            self.assertEqual(raised.exception.code, 2)

    def _paired_email_dirs(self, root: Path) -> tuple[Path, Path]:
        baseline = root / "baseline"
        candidate = root / "candidate"
        run_cli(
            [
                "run",
                "--scenario",
                str(ROOT / "scenarios" / "email_reply_completion_bad.json"),
                "--out",
                str(baseline / "email_reply_completion"),
            ]
        )
        run_cli(
            [
                "run",
                "--scenario",
                str(ROOT / "scenarios" / "email_reply_completion_good.json"),
                "--out",
                str(candidate / "email_reply_completion"),
            ]
        )
        for side in (baseline, candidate):
            score_path = side / "email_reply_completion" / "scorecard.json"
            scorecard = json.loads(score_path.read_text(encoding="utf-8"))
            scorecard["scenario_id"] = "email_reply_completion"
            scorecard["scenario_title"] = "Email Reply Completion"
            score_path.write_text(json.dumps(scorecard, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return baseline, candidate


if __name__ == "__main__":
    unittest.main()
