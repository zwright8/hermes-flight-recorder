import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from flightrecorder.cli import main
from flightrecorder.external_eval import build_external_eval_plan, write_external_eval_plan


def run_cli(args):
    with redirect_stdout(StringIO()):
        return main(args)


class ExternalEvalPlanTests(unittest.TestCase):
    def test_external_eval_plan_cli_fails_closed_but_validates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = _scenario_manifest(root / "heldout.json")
            out = root / "external_eval_plan.json"

            code = run_cli(["external-eval-plan", "--scenario-manifest", str(manifest), "--out", str(out)])
            validate_code = run_cli(["validate", "--external-eval-plan", str(out), "--strict"])

            self.assertEqual(code, 1)
            self.assertEqual(validate_code, 0)
            plan = _read_json(out)
            self.assertFalse(plan["ready"])
            self.assertEqual(plan["adapter_count"], 4)
            self.assertEqual(plan["ready_adapter_count"], 0)
            self.assertIn("no_ready_external_adapters", plan["blocking_reasons"])
            self.assertIn("adapter_disabled_until_allow_installed", plan["blocking_reasons"])
            self.assertEqual(len(plan["inputs"]["scenario_manifest"]["sha256"]), 64)
            self.assertFalse(plan["governance_handoff"]["external_eval_claims_allowed"])

    def test_external_eval_plan_ready_path_with_mocked_dependency(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = _scenario_manifest(root / "heldout.json")
            out = root / "external_eval_plan.json"

            with patch(
                "flightrecorder.external_eval._dependency_status",
                return_value={"available": True, "imports": {"inspect_ai": True}, "commands": {"inspect": True}},
            ):
                plan = build_external_eval_plan(
                    adapters=["inspect_ai"],
                    scenario_manifest=manifest,
                    model_endpoint="http://127.0.0.1:8000/v1",
                    inspect_task_set="heldout-inspect",
                    sandbox_policy="locked-network",
                    allow_installed=True,
                )
            write_external_eval_plan(plan, out)
            validate_code = run_cli(["validate", "--external-eval-plan", str(out), "--strict"])

            self.assertEqual(validate_code, 0)
            self.assertTrue(plan["ready"])
            self.assertEqual(plan["ready_adapter_count"], 1)
            self.assertEqual(plan["blocking_reasons"], [])
            self.assertTrue(plan["adapters"][0]["ready"])
            self.assertEqual(plan["adapters"][0]["blocking_reasons"], [])
            self.assertTrue(plan["governance_handoff"]["external_eval_claims_allowed"])

    def test_eval_summary_surfaces_external_adapter_blockers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = _scenario_manifest(root / "heldout.json")
            plan_path = root / "external_eval_plan.json"
            out = root / "eval_summary.json"
            run_cli(["external-eval-plan", "--scenario-manifest", str(manifest), "--out", str(plan_path)])

            code = run_cli(["eval-summary", "--external-adapter-plan", f"external={plan_path}", "--out", str(out)])
            validate_code = run_cli(["validate", "--eval-summary", str(out), "--strict"])

            self.assertEqual(code, 1)
            self.assertEqual(validate_code, 0)
            summary = _read_json(out)
            self.assertFalse(summary["passed"])
            self.assertEqual(summary["external_adapter_plan_count"], 1)
            self.assertFalse(summary["external_adapter_plans"][0]["ready"])
            self.assertTrue(any(risk["source"] == "external_adapter_plan" for risk in summary["risks"]))

    def test_validate_rejects_ready_plan_with_missing_adapter_blockers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = _scenario_manifest(root / "heldout.json")
            out = root / "external_eval_plan.json"
            validation = root / "validation.json"
            plan = build_external_eval_plan(adapters=["bfcl"], scenario_manifest=manifest)
            plan["ready"] = True
            plan["blocking_reasons"] = []
            plan["adapters"][0]["ready"] = True
            plan["adapters"][0]["blocking_reasons"] = []
            write_external_eval_plan(plan, out)

            code = run_cli(["validate", "--external-eval-plan", str(out), "--out", str(validation)])

            self.assertEqual(code, 1)
            errors = "\n".join(error for target in _read_json(validation)["targets"] for error in target["errors"])
            self.assertIn("external_eval_plan.ready_adapter_count expected 1", errors)
            self.assertIn("blocking_reasons must include dependencies_missing", errors)
            self.assertIn("ready cannot be true while blockers remain", errors)


def _scenario_manifest(path: Path) -> Path:
    payload = {"schema_version": "hfr.heldout_scenario_manifest.v1", "scenario_ids": ["email_reply_completion"]}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
