import importlib.util
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from flightrecorder.schema_registry import check_schema_file


ROOT = Path(__file__).resolve().parents[1]
SERVING_SCRIPT = ROOT / "scripts" / "check_openai_serving.py"
DEMO_SCRIPT = ROOT / "scripts" / "build_serving_demo_report.py"


def _load_script(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ServingDemoTests(unittest.TestCase):
    def test_mock_serving_check_writes_ready_profile_and_compatibility_report(self):
        check_openai_serving = _load_script(SERVING_SCRIPT, "check_openai_serving")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            adapter = root / "adapter"
            adapter.mkdir()
            (adapter / "adapter_config.json").write_text('{"r": 8}\n', encoding="utf-8")
            (adapter / "adapter_model.safetensors").write_bytes(b"hfr-test-adapter")

            with redirect_stdout(StringIO()):
                code = check_openai_serving.main(
                    [
                        "--out",
                        str(root / "serving"),
                        "--model",
                        "hfr-mock-model",
                        "--arm",
                        "flightrecorder",
                        "--adapter",
                        str(adapter),
                        "--mock-response",
                        "hfr serving smoke ok",
                        "--require-tool-call",
                        "--require-structured-output",
                    ]
                )

            self.assertEqual(code, 0)
            out = root / "serving"
            profile = _read_json(out / "serving_profile.json")
            compatibility = _read_json(out / "compatibility_report.json")
            report = _read_json(out / "serving_check.json")
            schema_result = check_schema_file(out / "serving_profile.json")
            self.assertTrue(schema_result["passed"], schema_result["errors"])
            self.assertEqual(schema_result["schema"]["name"], "serving_profile")
            self.assertTrue(report["passed"], report)
            self.assertEqual(profile["artifacts"]["serving_profile"], "serving_profile.json")
            self.assertEqual(profile["model_identity"]["served_model_id"], "hfr-mock-model+adapter")
            self.assertTrue(profile["model_identity"]["adapter"]["local"])
            self.assertEqual(profile["capabilities"]["tool_calls"], "supported")
            self.assertEqual(profile["capabilities"]["structured_outputs"], "supported")
            self.assertEqual(compatibility["checks"]["tool_calls"]["tool_call_count"], 1)

    def test_demo_report_links_claims_to_replay_artifacts(self):
        build_serving_demo_report = _load_script(DEMO_SCRIPT, "build_serving_demo_report")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline_eval = _write_eval_arm(root, "baseline", "hfr-base", passed=False, score=50)
            candidate_eval = _write_eval_arm(root, "flightrecorder", "hfr-base+adapter", passed=True, score=100)
            demo_json = root / "demo_run.json"
            report_md = root / "DEMO_REPORT.md"

            with redirect_stdout(StringIO()):
                code = build_serving_demo_report.main(
                    [
                        "--arm",
                        f"baseline={baseline_eval}",
                        "--arm",
                        f"flightrecorder={candidate_eval}",
                        "--out",
                        str(demo_json),
                        "--report",
                        str(report_md),
                    ]
                )

            self.assertEqual(code, 0)
            demo = _read_json(demo_json)
            schema_result = check_schema_file(demo_json)
            self.assertTrue(schema_result["passed"], schema_result["errors"])
            self.assertEqual(schema_result["schema"]["name"], "serving_demo_run")
            report = report_md.read_text(encoding="utf-8")
            claim_ids = {claim["id"] for claim in demo["claims"]}
            self.assertTrue(demo["same_scenario_ids"])
            self.assertIn("flightrecorder_repairs_demo_scenario", claim_ids)
            self.assertFalse(Path(demo["arms"][0]["source"]).is_absolute())
            self.assertEqual(demo["arms"][0]["serving_profile"], "baseline/serving_profile.json")
            self.assertFalse(Path(demo["scenarios"][0]["arms"]["baseline"]["trace_path"]).is_absolute())
            self.assertIn("scorecard", report)
            self.assertIn("run_digest", report)
            self.assertIn("live_observer.jsonl", report)


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_eval_arm(root: Path, arm: str, model: str, *, passed: bool, score: int) -> Path:
    arm_dir = root / arm
    run_dir = arm_dir / "demo_scenario"
    run_dir.mkdir(parents=True)
    for name in ("live_observer.jsonl", "scorecard.json", "report.html", "artifact_lineage.json"):
        (run_dir / name).write_text("{}\n", encoding="utf-8")
    (arm_dir / "serving_profile.json").write_text('{"schema_version": "hfr.serving_profile.v1"}\n', encoding="utf-8")
    (run_dir / "run_digest.json").write_text(
        json.dumps({"schema_version": "hfr.run_digest.v1", "outcome": {"passed": passed, "score": score, "summary": "PASS" if passed else "FAIL"}, "trace_signal": {"model": model, "event_count": 4}}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    suite = {
        "total": 1,
        "passed": 1 if passed else 0,
        "failed": 0 if passed else 1,
        "metadata": {"arm": arm, "model": model, "base_url": "http://127.0.0.1:8000/v1", "serving_profile": "serving_profile.json"},
        "metrics": {"pass_rate": 1.0 if passed else 0.0, "average_score": score, "critical_failure_counts": [] if passed else [{"id": "final_answer", "count": 1}]},
        "runs": [
            {
                "scenario_id": "demo_scenario",
                "scenario_title": "Demo Scenario",
                "task_family": "demo",
                "passed": passed,
                "score": score,
                "critical_failures": [] if passed else ["final_answer"],
                "trace_path": str(run_dir / "live_observer.jsonl"),
                "scorecard": str(run_dir / "scorecard.json"),
                "run_digest": str(run_dir / "run_digest.json"),
                "report": str(run_dir / "report.html"),
            }
        ],
    }
    suite_path = arm_dir / "suite_summary.json"
    suite_path.write_text(json.dumps(suite, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    eval_summary = {
        "schema_version": "hfr.hermes_heldout_eval_summary.v1",
        "arm": arm,
        "model": model,
        "base_url": "http://127.0.0.1:8000/v1",
        "serving_profile": "serving_profile.json",
        "suite_summary": str(suite_path),
        "total": 1,
        "passed": suite["passed"],
        "failed": suite["failed"],
        "pass_rate": suite["metrics"]["pass_rate"],
        "average_score": suite["metrics"]["average_score"],
        "critical_failure_total": 0 if passed else 1,
    }
    eval_path = arm_dir / "evaluation_summary.json"
    eval_path.write_text(json.dumps(eval_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return eval_path
