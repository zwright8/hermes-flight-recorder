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


class Goal3HandoffTests(unittest.TestCase):
    def test_goal3_handoff_builds_export_gate_preflight_and_evidence_bundle(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "handoff"

            code = run_cli(
                [
                    "goal3-handoff",
                    "--scenarios",
                    str(ROOT / "scenarios"),
                    "--out",
                    str(out),
                    "--policy",
                    str(ROOT / "examples" / "training_gate_policy.demo.json"),
                    "--trainer-command",
                    "python train.py --dataset training_export",
                    "--metadata",
                    "launcher=dry-run",
                ]
            )

            self.assertEqual(code, 0)
            handoff = json.loads((out / "goal3_handoff.json").read_text(encoding="utf-8"))
            self.assertEqual(handoff["schema_version"], "hfr.goal3_handoff.v1")
            self.assertTrue(handoff["passed"])
            self.assertEqual(handoff["recommendation"], "handoff_ready")
            self.assertRegex(handoff["dataset_version"], r"^hfrds-[0-9a-f]+$")
            self.assertEqual(
                {stage["id"] for stage in handoff["stages"]},
                {"run_suite", "training_export", "validation", "evidence_bundle", "training_gate", "trainer_preflight"},
            )
            self.assertTrue(all(stage["passed"] for stage in handoff["stages"]))
            self.assertEqual(handoff["artifacts"]["training_export"], "<redacted:training_export>")
            self.assertEqual(handoff["artifacts"]["evidence_bundle"], "<redacted:evidence_bundle.json>")
            self.assertEqual(handoff["metadata"]["launcher"], "dry-run")
            self.assertNotIn(str(tmp), json.dumps(handoff))
            self.assertTrue((out / "training_export" / "manifest.json").exists())
            self.assertTrue((out / "validation.json").exists())
            self.assertTrue((out / "training_gate.json").exists())
            self.assertTrue((out / "trainer_preflight.json").exists())
            self.assertTrue((out / "runs" / "evidence_bundle.json").exists())
            self.assertEqual(run_cli(["schemas", "--check", str(out / "goal3_handoff.json")]), 0)
            self.assertEqual(run_cli(["validate", "--trainer-preflight", str(out / "trainer_preflight.json"), "--strict"]), 0)


if __name__ == "__main__":
    unittest.main()
