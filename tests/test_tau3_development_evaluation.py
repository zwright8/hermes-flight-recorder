from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from flightrecorder.schema_registry import check_schema_file
from flightrecorder.tau3_development_evaluation import (
    build_tau3_development_evaluation,
)
from tests.test_tau3_candidate_selection import (
    _benchmark_manifest,
    _candidate_entry,
)

ROOT = Path(__file__).resolve().parents[1]


class Tau3DevelopmentEvaluationTests(unittest.TestCase):
    def test_builds_paired_qualification_evidence_from_real_manifests(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = _benchmark_manifest(
                root,
                "base",
                reward=0.0,
                db_match=False,
            )
            candidate = _candidate_entry(
                root,
                "candidate-a",
                reward=1.0,
                db_match=True,
            )

            result = build_tau3_development_evaluation(
                reference_root=root,
                out_dir=root / "qualification",
                candidate_id="candidate-a",
                base_manifest=base,
                candidate_manifest=candidate.development_manifest_path,
                training_receipt=candidate.training_receipt_path,
                candidate_identity=candidate.candidate_identity_path,
                created_at="2026-07-25T00:00:00Z",
                bootstrap_samples=200,
            )

            self.assertTrue(result["passed"], result)
            evaluation_path = root / "qualification" / "development-evaluation.json"
            scorecard_path = root / "qualification" / "development-scorecard.json"
            self.assertTrue(
                check_schema_file(
                    evaluation_path,
                    "tau3_development_evaluation",
                )["passed"]
            )
            self.assertTrue(
                check_schema_file(
                    scorecard_path,
                    "tau3_development_scorecard",
                )["passed"]
            )
            evaluation = json.loads(
                evaluation_path.read_text(encoding="utf-8")
            )
            scorecard = json.loads(
                scorecard_path.read_text(encoding="utf-8")
            )
            self.assertEqual(len(evaluation["development_trials"]), 12)
            self.assertEqual(
                evaluation["metrics"]["macro_pass1"],
                {"adapter": 1.0, "base": 0.0},
            )
            self.assertEqual(
                scorecard["development_evaluation"]["path"],
                "qualification/development-evaluation.json",
            )
            self.assertEqual(
                scorecard["bindings"],
                evaluation["bindings"],
            )
            self.assertNotIn(str(root), json.dumps(evaluation))

    def test_below_minimum_gain_emits_blocked_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = _benchmark_manifest(
                root,
                "base",
                reward=1.0,
                db_match=True,
            )
            candidate = _candidate_entry(
                root,
                "candidate-a",
                reward=1.0,
                db_match=True,
            )

            result = build_tau3_development_evaluation(
                reference_root=root,
                out_dir=root / "qualification",
                candidate_id="candidate-a",
                base_manifest=base,
                candidate_manifest=candidate.development_manifest_path,
                training_receipt=candidate.training_receipt_path,
                candidate_identity=candidate.candidate_identity_path,
                bootstrap_samples=200,
            )

            self.assertFalse(result["passed"])
            self.assertIn(
                "minimum_macro_gain",
                result["blocking_reasons"],
            )
            scorecard = json.loads(
                (
                    root
                    / "qualification"
                    / "development-scorecard.json"
                ).read_text(encoding="utf-8")
            )
            self.assertFalse(scorecard["passed"])
            self.assertEqual(scorecard["blockers"]["threshold"], 1)

    def test_cli_writes_the_same_registered_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = _benchmark_manifest(
                root,
                "base",
                reward=0.0,
                db_match=False,
            )
            candidate = _candidate_entry(
                root,
                "candidate-a",
                reward=1.0,
                db_match=True,
            )
            proc = subprocess.run(
                [
                    sys.executable,
                    str(
                        ROOT
                        / "scripts"
                        / "build_tau3_development_evaluation.py"
                    ),
                    "--reference-root",
                    str(root),
                    "--out",
                    str(root / "qualification"),
                    "--candidate-id",
                    "candidate-a",
                    "--base-manifest",
                    str(base),
                    "--candidate-manifest",
                    str(candidate.development_manifest_path),
                    "--training-receipt",
                    str(candidate.training_receipt_path),
                    "--candidate-identity",
                    str(candidate.candidate_identity_path),
                    "--bootstrap-samples",
                    "200",
                ],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            self.assertEqual(proc.returncode, 0, proc.stderr + proc.stdout)
            self.assertTrue(
                (
                    root
                    / "qualification"
                    / "development-scorecard.json"
                ).is_file()
            )


if __name__ == "__main__":
    unittest.main()
