import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from flightrecorder.cli import main
from flightrecorder.schema_registry import check_schema_file
from flightrecorder.validation import validate_artifacts


ROOT = Path(__file__).resolve().parents[1]


def run_cli(args):
    with redirect_stdout(StringIO()):
        return main(args)


class HeldoutManifestTests(unittest.TestCase):
    def test_heldout_manifest_blocks_schema_invalid_suite_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            suite = _suite_summary(root / "baseline_suite.json", ["email_reply_completion"])
            payload = _read_json(suite)
            payload.pop("scenarios_dir")
            suite.write_text(json.dumps(payload), encoding="utf-8")
            out = root / "heldout_manifest.json"

            code = run_cli(["heldout-manifest", "--suite-summary", f"baseline={suite}", "--out", str(out)])

            self.assertEqual(code, 1)
            manifest = _read_json(out)
            self.assertFalse(manifest["ready"])
            self.assertIn("invalid_suite_summary_schema", manifest["blocking_reasons"])

    def test_committed_agentic_training_heldout_manifest_replays_suite_summaries(self):
        eval_root = ROOT / "examples" / "agentic_training" / "heldout_eval"
        suite_manifest_path = eval_root / "heldout_suite_manifest.json"
        manifest_path = eval_root / "heldout_manifest.json"
        manifest = _read_json(manifest_path)

        self.assertEqual(
            manifest["scenario_ids"],
            ["prompt_injection_bad", "prompt_injection_good", "subagent_claim_bad"],
        )
        self.assertTrue(manifest["ready"])
        self.assertEqual(manifest["status"], "identical")
        self.assertTrue(manifest["cross_arm_claims_allowed"])
        self.assertEqual({source["label"] for source in manifest["sources"]}, {"baseline", "candidate"})
        self.assertEqual(run_cli(["schemas", "--check", str(suite_manifest_path)]), 0)
        validation = validate_artifacts(
            eval_suite_manifest_paths=[suite_manifest_path],
            heldout_manifest_paths=[manifest_path],
            strict=True,
        )
        self.assertTrue(validation["passed"], validation)

    def test_heldout_manifest_allows_single_source_external_adapter_seed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            suite = _suite_summary(root / "baseline_suite.json", ["email_reply_completion"])
            out = root / "heldout_manifest.json"

            code = run_cli(["heldout-manifest", "--suite-summary", f"baseline={suite}", "--out", str(out)])
            validate_code = run_cli(["validate", "--heldout-manifest", str(out), "--strict"])
            schema_result = check_schema_file(out)

            self.assertEqual(code, 0)
            self.assertEqual(validate_code, 0)
            self.assertTrue(schema_result["passed"], schema_result["errors"])
            manifest = _read_json(out)
            self.assertTrue(manifest["ready"])
            self.assertEqual(manifest["status"], "single_source")
            self.assertFalse(manifest["cross_arm_claims_allowed"])
            self.assertEqual(manifest["scenario_ids"], ["email_reply_completion"])
            self.assertEqual(manifest["sources"][0]["path"], "baseline_suite.json")

    def test_heldout_manifest_proves_identical_cross_arm_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = _suite_summary(root / "baseline_suite.json", ["email_reply_completion", "prompt_injection"])
            candidate = _suite_summary(root / "candidate_suite.json", ["prompt_injection", "email_reply_completion"])
            out = root / "heldout_manifest.json"

            code = run_cli(
                [
                    "heldout-manifest",
                    "--suite-summary",
                    f"baseline={baseline}",
                    "--suite-summary",
                    f"candidate={candidate}",
                    "--out",
                    str(out),
                ]
            )
            validate_code = run_cli(["validate", "--heldout-manifest", str(out), "--strict"])
            schema_result = check_schema_file(out)

            self.assertEqual(code, 0)
            self.assertEqual(validate_code, 0)
            self.assertTrue(schema_result["passed"], schema_result["errors"])
            manifest = _read_json(out)
            self.assertTrue(manifest["ready"])
            self.assertEqual(manifest["status"], "identical")
            self.assertTrue(manifest["identical"])
            self.assertTrue(manifest["cross_arm_claims_allowed"])
            self.assertEqual(manifest["scenario_count"], 2)

    def test_heldout_manifest_blocks_mismatched_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = _suite_summary(root / "baseline_suite.json", ["email_reply_completion", "prompt_injection"])
            candidate = _suite_summary(root / "candidate_suite.json", ["email_reply_completion"])
            out = root / "heldout_manifest.json"

            code = run_cli(
                [
                    "heldout-manifest",
                    "--suite-summary",
                    f"baseline={baseline}",
                    "--suite-summary",
                    f"candidate={candidate}",
                    "--out",
                    str(out),
                ]
            )
            validate_code = run_cli(["validate", "--heldout-manifest", str(out), "--strict"])

            self.assertEqual(code, 1)
            self.assertEqual(validate_code, 0)
            manifest = _read_json(out)
            self.assertFalse(manifest["ready"])
            self.assertEqual(manifest["status"], "mismatched")
            self.assertIn("heldout_scenario_set_mismatch", manifest["blocking_reasons"])
            self.assertEqual(manifest["mismatches"][0]["missing_from_source"], ["prompt_injection"])

    def test_external_eval_plan_blocks_not_ready_heldout_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = _suite_summary(root / "baseline_suite.json", ["email_reply_completion", "prompt_injection"])
            candidate = _suite_summary(root / "candidate_suite.json", ["email_reply_completion"])
            manifest = root / "heldout_manifest.json"
            plan = root / "external_eval_plan.json"
            run_cli(
                [
                    "heldout-manifest",
                    "--suite-summary",
                    f"baseline={baseline}",
                    "--suite-summary",
                    f"candidate={candidate}",
                    "--out",
                    str(manifest),
                ]
            )

            code = run_cli(
                [
                    "external-eval-plan",
                    "--adapter",
                    "lm_eval_harness",
                    "--scenario-manifest",
                    str(manifest),
                    "--model-endpoint",
                    "http://127.0.0.1:8000/v1",
                    "--lm-eval-task",
                    "mmlu",
                    "--allow-installed",
                    "--out",
                    str(plan),
                ]
            )
            validate_code = run_cli(["validate", "--external-eval-plan", str(plan), "--strict"])
            schema_result = check_schema_file(plan)

            self.assertEqual(code, 1)
            self.assertEqual(validate_code, 0)
            self.assertTrue(schema_result["passed"], schema_result["errors"])
            adapter = _read_json(plan)["adapters"][0]
            self.assertIn("scenario_manifest_not_ready", adapter["blocking_reasons"])

    def test_validate_rejects_forged_ready_mismatched_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = _suite_summary(root / "baseline_suite.json", ["email_reply_completion", "prompt_injection"])
            candidate = _suite_summary(root / "candidate_suite.json", ["email_reply_completion"])
            out = root / "heldout_manifest.json"
            validation = root / "validation.json"
            run_cli(
                [
                    "heldout-manifest",
                    "--suite-summary",
                    f"baseline={baseline}",
                    "--suite-summary",
                    f"candidate={candidate}",
                    "--out",
                    str(out),
                ]
            )
            manifest = _read_json(out)
            manifest["ready"] = True
            manifest["blocking_reasons"] = []
            out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            code = run_cli(["validate", "--heldout-manifest", str(out), "--out", str(validation)])

            self.assertEqual(code, 1)
            errors = "\n".join(error for target in _read_json(validation)["targets"] for error in target["errors"])
            self.assertIn("heldout_manifest.ready expected False", errors)
            self.assertIn("blocking_reasons must include heldout_scenario_set_mismatch", errors)

    def test_strict_validate_warns_on_absolute_source_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            suite = _suite_summary(root / "baseline_suite.json", ["email_reply_completion"])
            out = root / "heldout_manifest.json"
            run_cli(["heldout-manifest", "--suite-summary", f"baseline={suite}", "--out", str(out)])
            manifest = _read_json(out)
            manifest["sources"][0]["path"] = str(suite.resolve())
            out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            permissive_validation = validate_artifacts(heldout_manifest_paths=[out])
            validation = validate_artifacts(heldout_manifest_paths=[out], strict=True)

            self.assertTrue(permissive_validation["passed"], permissive_validation)
            self.assertGreater(permissive_validation["warning_count"], 0, permissive_validation)
            self.assertFalse(validation["passed"], validation)
            self.assertEqual(validation["error_count"], 0, validation)
            warnings = "\n".join(warning for target in validation["targets"] for warning in target["warnings"])
            self.assertIn("heldout_manifest.sources[0].path is absolute", warnings)

    def test_validate_rejects_stale_source_suite_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            suite = _suite_summary(root / "baseline_suite.json", ["email_reply_completion"])
            out = root / "heldout_manifest.json"
            run_cli(["heldout-manifest", "--suite-summary", f"baseline={suite}", "--out", str(out)])
            _suite_summary(suite, ["email_reply_completion", "prompt_injection"])

            validation = validate_artifacts(heldout_manifest_paths=[out], strict=True)

            self.assertFalse(validation["passed"], validation)
            errors = "\n".join(error for target in validation["targets"] for error in target["errors"])
            self.assertIn("heldout_manifest.sources[0].scenario_count must match the current suite summary", errors)
            self.assertIn("heldout_manifest.sources[0].scenario_ids must match the current suite summary", errors)
            self.assertIn("heldout_manifest.sources[0].scenario_fingerprints must match the current suite summary", errors)

    def test_validate_rejects_heldout_manifest_with_unknown_control_plane_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = _suite_summary(root / "baseline_suite.json", ["email_reply_completion", "prompt_injection"])
            candidate = _suite_summary(root / "candidate_suite.json", ["email_reply_completion"])
            out = root / "heldout_manifest.json"
            run_cli(
                [
                    "heldout-manifest",
                    "--suite-summary",
                    f"baseline={baseline}",
                    "--suite-summary",
                    f"candidate={candidate}",
                    "--out",
                    str(out),
                ]
            )
            manifest = _read_json(out)
            manifest["provider_console_url"] = "https://example.invalid/heldout"
            manifest["governance_handoff"]["approval_thread_ref"] = "redacted-thread"
            manifest["sources"][0]["provider_job_id"] = "job-redacted"
            manifest["mismatches"][0]["benchmark_url"] = "https://example.invalid/mismatch"
            out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            schema_result = check_schema_file(out)
            validation = validate_artifacts(heldout_manifest_paths=[out], strict=True)

            self.assertFalse(schema_result["passed"], schema_result)
            self.assertFalse(validation["passed"], validation)
            errors = "\n".join(error for target in validation["targets"] for error in target["errors"])
            self.assertIn("heldout_manifest contains unknown field(s): ['provider_console_url']", errors)
            self.assertIn(
                "heldout_manifest.governance_handoff contains unknown field(s): ['approval_thread_ref']",
                errors,
            )
            self.assertIn("heldout_manifest.sources[0] contains unknown field(s): ['provider_job_id']", errors)
            self.assertIn("heldout_manifest.mismatches[0] contains unknown field(s): ['benchmark_url']", errors)


def _suite_summary(path: Path, scenario_ids: list[str]) -> Path:
    runs = [
        {
            "scenario_id": scenario_id,
            "scenario_title": scenario_id,
            "scenario_sha256": "a" * 64,
            "task_family": scenario_id,
            "scenario_path": f"scenarios/{scenario_id}.json",
            "trace_path": f"traces/{scenario_id}.jsonl",
            "run_dir": f"runs/{scenario_id}",
            "report": f"runs/{scenario_id}/report.html",
            "report_sha256": "b" * 64,
            "report_size_bytes": 1,
            "scorecard": f"runs/{scenario_id}/scorecard.json",
            "scorecard_sha256": "c" * 64,
            "scorecard_size_bytes": 1,
            "run_digest": f"runs/{scenario_id}/run_digest.json",
            "run_digest_sha256": "d" * 64,
            "run_digest_size_bytes": 1,
            "lineage": f"runs/{scenario_id}/artifact_lineage.json",
            "lineage_sha256": "e" * 64,
            "lineage_size_bytes": 1,
            "passed": True,
            "score": 100,
            "failed_rules": [],
            "critical_failures": [],
        }
        for scenario_id in scenario_ids
    ]
    payload = {
        "schema_version": "hfr.run_suite.v1",
        "scenarios_dir": "scenarios",
        "out_dir": "runs",
        "total": len(runs),
        "passed": len(runs),
        "failed": 0,
        "error_count": 0,
        "errors": [],
        "metrics": {
            "pass_rate": 1.0 if runs else 0.0,
            "average_score": 100.0 if runs else 0.0,
            "min_score": 100 if runs else None,
            "max_score": 100 if runs else None,
            "failed_rule_counts": [],
            "critical_failure_counts": [],
            "task_families": [],
            "failed": 0,
            "passed": len(runs),
        },
        "runs": runs,
        "artifacts": {"suite_result": "runs/harness_suite_result.json"},
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
