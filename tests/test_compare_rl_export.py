import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from flightrecorder.cli import main


ROOT = Path(__file__).resolve().parents[1]


def run_cli(args):
    with redirect_stdout(StringIO()):
        return main(args)


class CompareRlExportTests(unittest.TestCase):
    def test_export_compare_rl_writes_candidate_improvement_pair(self):
        with tempfile.TemporaryDirectory() as tmp:
            baseline, candidate = self._paired_email_dirs(Path(tmp))
            out = Path(tmp) / "compare_rl"

            code = run_cli(
                [
                    "export-compare-rl",
                    "--baseline",
                    str(baseline),
                    "--candidate",
                    str(candidate),
                    "--out",
                    str(out),
                    "--metadata",
                    "candidate=email-fix",
                ]
            )
            validate_code = run_cli(["validate", "--compare-export", str(out), "--strict"])

            self.assertEqual(code, 0)
            self.assertEqual(validate_code, 0)
            manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
            pairs = self._read_jsonl(out / "improvement_pairs.jsonl")
            dpo = self._read_jsonl(out / "improvement_dpo.jsonl")
            card = (out / "IMPROVEMENT_CARD.md").read_text(encoding="utf-8")

            self.assertEqual(manifest["schema_version"], "hfr.compare_rl.manifest.v1")
            self.assertEqual(manifest["pair_count"], 1)
            self.assertEqual(manifest["candidate_win_count"], 1)
            self.assertEqual(manifest["baseline_win_count"], 0)
            self.assertEqual(manifest["contract_scope"], "scenario")
            self.assertEqual(manifest["contract_drift_count"], 1)
            self.assertEqual(manifest["unverified_contract_count"], 0)
            self.assertEqual(manifest["metadata"]["candidate"], "email-fix")
            self.assertEqual(pairs[0]["schema_version"], "hfr.compare_rl.pair.v1")
            self.assertEqual(pairs[0]["scenario_id"], "email_reply_completion")
            self.assertEqual(pairs[0]["chosen_side"], "candidate")
            self.assertEqual(pairs[0]["rejected_side"], "baseline")
            self.assertEqual(pairs[0]["candidate_score_delta"], 90)
            self.assertEqual(pairs[0]["contract_fingerprint_status"], "drifted")
            self.assertEqual(pairs[0]["contract_fingerprint_scope"], "scenario")
            self.assertIn("scenario_sha256_changed", pairs[0]["contract_fingerprint_reasons"])
            self.assertNotIn("source_trace_sha256_changed", pairs[0]["contract_fingerprint_reasons"])
            self.assertEqual(pairs[0]["candidate"]["source_fingerprint_status"], "verified")
            self.assertEqual(pairs[0]["candidate"]["task_completion"]["status"], "complete")
            self.assertEqual(pairs[0]["baseline"]["task_completion"]["status"], "incomplete")
            self.assertEqual(pairs[0]["chosen"]["task_completion"]["status"], "complete")
            self.assertEqual(pairs[0]["rejected"]["task_completion"]["status"], "incomplete")
            self.assertIn("required_actions", pairs[0]["rule_fixes"])
            self.assertEqual(dpo[0]["schema_version"], "hfr.compare_rl.dpo.v1")
            self.assertEqual(dpo[0]["contract_fingerprint_status"], "drifted")
            self.assertEqual(dpo[0]["contract_fingerprint_scope"], "scenario")
            self.assertEqual(dpo[0]["chosen_task_completion_status"], "complete")
            self.assertEqual(dpo[0]["rejected_task_completion_status"], "incomplete")
            self.assertTrue(dpo[0]["chosen_task_completion_passed"])
            self.assertFalse(dpo[0]["rejected_task_completion_passed"])
            self.assertIn("task_completion complete checks=4/4", dpo[0]["chosen"])
            self.assertIn("task_completion incomplete checks=1/4", dpo[0]["rejected"])
            self.assertIn("tool_result gmail_send ok", dpo[0]["chosen"])
            self.assertNotIn("tool_result gmail_send ok", dpo[0]["rejected"])
            self.assertNotEqual(dpo[0]["chosen"], dpo[0]["rejected"])
            self.assertIn("# Flight Recorder Improvement Pair Card", card)

    def test_export_compare_rl_can_require_strict_trace_fixture_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            baseline, candidate = self._paired_email_dirs(Path(tmp))
            out = Path(tmp) / "compare_rl"

            code = run_cli(
                [
                    "export-compare-rl",
                    "--baseline",
                    str(baseline),
                    "--candidate",
                    str(candidate),
                    "--out",
                    str(out),
                    "--contract-scope",
                    "scenario-and-trace",
                ]
            )

            self.assertEqual(code, 0)
            manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
            pair = self._read_jsonl(out / "improvement_pairs.jsonl")[0]
            self.assertEqual(manifest["contract_scope"], "scenario-and-trace")
            self.assertEqual(pair["contract_fingerprint_scope"], "scenario-and-trace")
            self.assertIn("source_trace_sha256_changed", pair["contract_fingerprint_reasons"])

    def test_validate_rejects_compare_dpo_that_does_not_match_pair(self):
        with tempfile.TemporaryDirectory() as tmp:
            baseline, candidate = self._paired_email_dirs(Path(tmp))
            out = Path(tmp) / "compare_rl"
            summary_path = Path(tmp) / "validation.json"
            run_cli(["export-compare-rl", "--baseline", str(baseline), "--candidate", str(candidate), "--out", str(out)])
            dpo_path = out / "improvement_dpo.jsonl"
            row = json.loads(dpo_path.read_text(encoding="utf-8").splitlines()[0])
            row["chosen"] = "stale row"
            dpo_path.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")

            code = run_cli(["validate", "--compare-export", str(out), "--out", str(summary_path)])

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in summary["targets"] for error in target["errors"])
            self.assertIn("improvement_dpo[0].chosen", errors)

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

    def _read_jsonl(self, path: Path) -> list[dict]:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


if __name__ == "__main__":
    unittest.main()
