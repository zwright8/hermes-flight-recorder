import hashlib
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

from flightrecorder.bundle import _next_actions as bundle_next_actions
from flightrecorder.cli import main
from flightrecorder.hermes_plugin import LIVE_SMOKE_SUMMARY_SCHEMA_VERSION
from flightrecorder.schema_registry import check_schema_contract


ROOT = Path(__file__).resolve().parents[1]


def run_cli(args):
    with redirect_stdout(StringIO()):
        return main(args)


def bundle_action_fingerprint(action: dict) -> str:
    evidence = action.get("evidence") if isinstance(action.get("evidence"), dict) else {}
    payload = {
        "id": action.get("id"),
        "priority": action.get("priority"),
        "artifact": action.get("artifact"),
        "evidence": evidence,
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class EvidenceBundleTests(unittest.TestCase):
    def test_committed_agentic_training_evidence_handoff_replays_harness_result(self):
        handoff_root = ROOT / "examples" / "agentic_training" / "evidence_handoff"
        harness_result_path = handoff_root / "harness_handoff" / "harness_result.json"
        evidence_bundle_path = handoff_root / "evidence_bundle.json"
        harness_result = json.loads(harness_result_path.read_text(encoding="utf-8"))
        bundle = json.loads(evidence_bundle_path.read_text(encoding="utf-8"))

        self.assertEqual(harness_result["schema_version"], "hfr.harness_run_result.v1")
        self.assertEqual(harness_result["scenario_id"], "prompt_injection_good")
        self.assertTrue(harness_result["scorecard"]["passed"])
        self.assertEqual(harness_result["suite"]["summary"], "../suite_summary.json")
        self.assertTrue(bundle["passed"])
        self.assertEqual(bundle["readiness"], "ready")
        self.assertEqual(bundle["decision"]["recommendation"], "promote_handoff")
        self.assertEqual(bundle["decision"]["key_metrics"]["harness_handoff"]["passed_pair_count"], 1)
        self.assertEqual(bundle["decision"]["key_metrics"]["run_digest_coverage"]["passed_digest_count"], 1)
        self.assertEqual(
            run_cli(
                [
                    "validate",
                    "--harness-result",
                    str(harness_result_path),
                    "--evidence-bundle",
                    str(evidence_bundle_path),
                    "--strict",
                ]
            ),
            0,
        )

    def test_evidence_bundle_summarizes_ready_handoff(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = root / "runs"
            bundle_path = runs / "evidence_bundle.json"

            self.assertEqual(
                run_cli(
                    [
                        "run-suite",
                        "--scenarios",
                        str(ROOT / "scenarios"),
                        "--out",
                        str(runs),
                        "--export-rl",
                        "--validate",
                        "--strict",
                    ]
                ),
                0,
            )
            self.assertEqual(
                run_cli(
                    [
                        "scenario-quality",
                        "--scenarios",
                        str(ROOT / "scenarios"),
                        "--require-traces",
                        "--out",
                        str(runs / "scenario_quality.json"),
                        "--min-average-score",
                        "80",
                        "--min-scenario-score",
                        "60",
                        "--min-observable-rate",
                        "0.8",
                        "--max-weak-scenarios",
                        "0",
                        "--max-final-only-scenarios",
                        "0",
                        "--max-missing-traces",
                        "0",
                    ]
                ),
                0,
            )
            self.assertEqual(
                run_cli(
                    [
                        "evidence-coverage",
                        "--runs",
                        str(runs),
                        "--out",
                        str(runs / "evidence_coverage.json"),
                        "--min-failed-rule-evidence-rate",
                        "1.0",
                        "--min-critical-failed-rule-evidence-rate",
                        "1.0",
                        "--max-failed-rules-without-evidence",
                        "0",
                        "--max-critical-failed-rules-without-evidence",
                        "0",
                    ]
                ),
                0,
            )
            self.assertEqual(
                run_cli(
                    [
                        "trace-observability",
                        "--runs",
                        str(runs),
                        "--out",
                        str(runs / "trace_observability.json"),
                        "--min-average-events",
                        "2",
                        "--min-event-type-count",
                        "2",
                        "--min-tool-or-api-run-rate",
                        "0.5",
                        "--max-empty-final-answers",
                        "0",
                    ]
                ),
                0,
            )
            self.assertEqual(
                run_cli(
                    [
                        "repair-queue",
                        "--runs",
                        str(runs),
                        "--out",
                        str(runs / "repair_queue.json"),
                    ]
                ),
                0,
            )
            self.assertEqual(
                run_cli(
                    [
                        "gate-suite",
                        "--suite-summary",
                        str(runs / "suite_summary.json"),
                        "--policy",
                        str(ROOT / "examples" / "suite_gate_policy.demo.json"),
                        "--out",
                        str(runs / "suite_gate.json"),
                    ]
                ),
                0,
            )
            live_smoke_path = runs / "live_smoke_summary.json"
            hermes_root = "/" + "Users/example/hermes-agent"
            live_smoke_path.write_text(
                json.dumps(
                    {
                        "schema_version": LIVE_SMOKE_SUMMARY_SCHEMA_VERSION,
                        "passed": True,
                        "hermes_exit_code": 0,
                        "mock_request_count": 9,
                        "chat_completion_request_count": 1,
                        "observer_file": "live_observer.jsonl",
                        "hooks": ["on_session_start", "pre_llm_call", "post_llm_call"],
                        "missing_hooks": [],
                        "score": 100,
                        "report": "report.html",
                        "lineage": "artifact_lineage.json",
                        "task_completion": "task_completion.json",
                        "run_digest": "run_digest.json",
                        "environment": {
                            "python_version": "3.11.0",
                            "python_implementation": "CPython",
                            "platform": "Linux-test",
                            "hermes_root": hermes_root,
                            "hermes_git_commit": "abcdef123456",
                            "hermes_git_dirty": False,
                            "flight_recorder_root": str(ROOT),
                            "flight_recorder_git_commit": "123456abcdef",
                            "flight_recorder_git_dirty": True,
                        },
                        "summary": "live_smoke_summary.json",
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            harness_artifacts = _write_harness_handoff_artifacts(root)

            bundle_args = [
                "evidence-bundle",
                "--runs",
                str(runs),
                "--suite-summary",
                str(runs / "suite_summary.json"),
                "--scenario-quality",
                str(runs / "scenario_quality.json"),
                "--evidence-coverage",
                str(runs / "evidence_coverage.json"),
                "--trace-observability",
                str(runs / "trace_observability.json"),
                "--repair-queue",
                str(runs / "repair_queue.json"),
                "--validation",
                str(runs / "validation.json"),
                "--training-export",
                str(runs / "training_export"),
                "--live-smoke-summary",
                str(live_smoke_path),
                "--harness-manifest",
                str(harness_artifacts["manifest"]),
                "--harness-result",
                str(harness_artifacts["result"]),
                "--gate",
                str(runs / "suite_gate.json"),
                "--require-harness",
                "--require-gate",
                "--out",
                str(bundle_path),
            ]
            code = run_cli(bundle_args)

            self.assertEqual(code, 0)
            bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
            self.assertEqual(bundle["schema_version"], "hfr.evidence_bundle.v1")
            self.assertTrue(bundle["passed"])
            self.assertEqual(bundle["readiness"], "ready")
            self.assertEqual(bundle["failed_check_count"], 0)
            self.assertGreaterEqual(bundle["check_count"], 6)
            self.assertEqual(bundle["decision"]["readiness"], "ready")
            self.assertEqual(bundle["decision"]["recommendation"], "promote_handoff")
            self.assertEqual(bundle["decision"]["blocking_check_count"], 0)
            self.assertEqual(bundle["decision"]["blocking_checks"], [])
            self.assertIn("suite_summary", bundle["decision"]["evidence_artifacts"])
            self.assertEqual(bundle["decision"]["gate_count"], 1)
            self.assertEqual(bundle["decision"]["passed_gate_count"], 1)
            action_ids = {action["id"] for action in bundle["decision"]["next_actions"]}
            routing_keys = [action["routing_key"] for action in bundle["decision"]["next_actions"]]
            self.assertEqual(bundle["decision"]["next_action_count"], len(bundle["decision"]["next_actions"]))
            self.assertEqual(len(routing_keys), len(set(routing_keys)))
            for action in bundle["decision"]["next_actions"]:
                self.assertEqual(len(action["action_fingerprint"]), 64)
                self.assertEqual(action["routing_key"], f"{action['artifact']}:{action['id']}:{action['action_fingerprint'][:12]}")
            self.assertIn("repair_failed_scenarios", action_ids)
            self.assertIn("repair_critical_failures", action_ids)
            self.assertIn("dispatch_repair_queue", action_ids)
            self.assertIn("prioritize_curriculum_failures", action_ids)
            self.assertIn("ground_scenario_contracts", action_ids)
            self.assertIn("improve_trace_observability", action_ids)
            repair_action = next(action for action in bundle["decision"]["next_actions"] if action["id"] == "dispatch_repair_queue")
            self.assertEqual(repair_action["priority"], "critical")
            self.assertEqual(repair_action["artifact"], "repair_queue")
            self.assertEqual(repair_action["evidence"]["item_count"], 14)
            self.assertEqual(repair_action["evidence"]["critical_item_count"], 14)
            self.assertEqual(repair_action["evidence"]["scenario_count"], 5)
            self.assertEqual(repair_action["evidence"]["priority_counts"], {"critical": 14})
            self.assertEqual(repair_action["evidence"]["rule_counts"]["required_evidence"], 3)
            curriculum_action = next(action for action in bundle["decision"]["next_actions"] if action["id"] == "prioritize_curriculum_failures")
            self.assertEqual(curriculum_action["priority"], "high")
            self.assertEqual(curriculum_action["artifact"], "training_export")
            self.assertEqual(curriculum_action["evidence"]["curriculum_failure_mode_count"], 14)
            self.assertEqual(len(curriculum_action["evidence"]["top_curriculum_priorities"]), 5)
            self.assertEqual(bundle["decision"]["key_metrics"]["suite_summary"]["total"], 7)
            self.assertEqual(bundle["decision"]["key_metrics"]["trace_observability"]["tool_or_api_run_rate"], 0.8571)
            self.assertIn("risk_counts", bundle["decision"]["key_metrics"]["trace_observability"])
            self.assertEqual(bundle["decision"]["key_metrics"]["run_digest_coverage"]["digest_coverage_rate"], 1.0)
            self.assertEqual(bundle["decision"]["key_metrics"]["run_digest_coverage"]["missing_digest_count"], 0)
            self.assertEqual(bundle["decision"]["key_metrics"]["run_digest_coverage"]["invalid_digest_count"], 0)
            self.assertEqual(bundle["decision"]["key_metrics"]["repair_queue"]["item_count"], 14)
            self.assertEqual(bundle["decision"]["key_metrics"]["training_export"]["episode_count"], 7)
            self.assertEqual(
                bundle["decision"]["key_metrics"]["training_export"]["trainer_view_source_fingerprint_coverage"]["fully_verified_rate"],
                1.0,
            )
            self.assertEqual(bundle["decision"]["key_metrics"]["live_smoke_summary"]["passed"], True)
            self.assertEqual(bundle["decision"]["key_metrics"]["live_smoke_summary"]["consistent"], True)
            self.assertEqual(bundle["decision"]["key_metrics"]["live_smoke_summary"]["score"], 100)
            self.assertEqual(bundle["decision"]["key_metrics"]["live_smoke_summary"]["missing_hook_count"], 0)
            self.assertEqual(bundle["decision"]["key_metrics"]["live_smoke_summary"]["platform"], "Linux-test")
            self.assertEqual(bundle["decision"]["key_metrics"]["live_smoke_summary"]["hermes_git_commit"], "abcdef123456")
            self.assertEqual(bundle["decision"]["key_metrics"]["live_smoke_summary"]["flight_recorder_git_commit"], "123456abcdef")
            self.assertEqual(bundle["decision"]["key_metrics"]["harness_handoff"]["pair_count"], 1)
            self.assertEqual(bundle["decision"]["key_metrics"]["harness_handoff"]["passed_pair_count"], 1)
            self.assertEqual(bundle["decision"]["key_metrics"]["harness_handoff"]["missing_pair_count"], 0)
            top_priorities = bundle["decision"]["key_metrics"]["training_export"]["top_curriculum_priorities"]
            self.assertEqual(len(top_priorities), 5)
            self.assertEqual(
                [item["priority_score"] for item in top_priorities],
                sorted((item["priority_score"] for item in top_priorities), reverse=True),
            )
            self.assertTrue(any(item["rule_id"] == "forbidden_actions" for item in top_priorities))
            self.assertTrue(any("prompt_injection_bad" in item["scenario_ids"] for item in top_priorities))
            self.assertTrue(any("cron_async_delegation_bad" in item["scenario_ids"] for item in top_priorities))
            self.assertEqual(bundle["metrics"]["suite_summary"]["total"], 7)
            self.assertEqual(bundle["metrics"]["scenario_quality"]["average_contract_score"], 90.71)
            self.assertIn("risk_counts", bundle["metrics"]["scenario_quality"])
            self.assertEqual(bundle["metrics"]["evidence_coverage"]["failed_rule_evidence_rate"], 1.0)
            self.assertEqual(bundle["metrics"]["trace_observability"]["run_count"], 7)
            self.assertEqual(bundle["metrics"]["trace_observability"]["event_type_count"], 6)
            self.assertIn("risk_counts", bundle["metrics"]["trace_observability"])
            self.assertEqual(bundle["metrics"]["run_digest_coverage"]["run_count"], 7)
            self.assertEqual(bundle["metrics"]["run_digest_coverage"]["digest_count"], 7)
            self.assertEqual(bundle["metrics"]["run_digest_coverage"]["digest_coverage_rate"], 1.0)
            self.assertEqual(bundle["metrics"]["run_digest_coverage"]["task_completion_status_counts"], [
                {"id": "complete", "count": 2},
                {"id": "incomplete", "count": 4},
                {"id": "not_applicable", "count": 1},
            ])
            self.assertEqual(bundle["metrics"]["repair_queue"]["critical_item_count"], 14)
            self.assertEqual(bundle["metrics"]["training_export"]["episode_count"], 7)
            self.assertEqual(bundle["metrics"]["training_export"]["curriculum_failure_mode_count"], 14)
            self.assertEqual(bundle["metrics"]["training_export"]["trainer_view_source_fingerprint_coverage"]["unverified"], 0)
            self.assertEqual(bundle["metrics"]["live_smoke_summary"]["hook_count"], 3)
            self.assertEqual(bundle["metrics"]["live_smoke_summary"]["hermes_root"], "<redacted:hermes-agent>")
            self.assertEqual(bundle["metrics"]["live_smoke_summary"]["flight_recorder_root"], ".")
            self.assertNotIn("Users/example", json.dumps(bundle))
            self.assertEqual(bundle["metrics"]["live_smoke_summary"]["flight_recorder_git_dirty"], True)
            self.assertEqual(bundle["metrics"]["harness_handoff"]["pair_count"], 1)
            self.assertEqual(bundle["metrics"]["harness_handoff"]["schema_valid_pair_count"], 1)
            self.assertEqual(bundle["metrics"]["harness_handoff"]["consistent_pair_count"], 1)
            self.assertEqual(bundle["metrics"]["harness_handoff"]["runs"][0]["runner"], "hermes_harness")
            self.assertEqual(bundle["metrics"]["harness_handoff"]["runs"][0]["provider"], "mock")
            self.assertEqual(bundle["metrics"]["harness_handoff"]["runs"][0]["trace_format"], "normalized_json")
            self.assertEqual(bundle["metrics"]["gates"][0]["id"], "suite_gate")
            self.assertTrue(bundle["metrics"]["gates"][0]["passed"])
            self.assertTrue(bundle["metrics"]["gates"][0]["contract"]["valid"])
            self.assertEqual(bundle["metrics"]["gates"][0]["contract"]["readiness"], "ready")
            self.assertEqual(bundle["metrics"]["gates"][0]["contract"]["recommendation"], "promote_iteration")
            self.assertEqual(bundle["artifacts"]["suite_summary"]["kind"], "file")
            self.assertEqual(bundle["artifacts"]["live_smoke_summary"]["kind"], "file")
            self.assertEqual(len(bundle["artifacts"]["live_smoke_summary"]["sha256"]), 64)
            self.assertEqual(bundle["artifacts"]["training_export_curriculum"]["kind"], "file")
            self.assertEqual(bundle["artifacts"]["training_export_curriculum"]["exists"], True)
            self.assertEqual(len(bundle["artifacts"]["training_export_curriculum"]["sha256"]), 64)
            self.assertEqual(len(bundle["artifacts"]["suite_summary"]["sha256"]), 64)
            schema = check_schema_contract(bundle, name_or_id="evidence_bundle")
            self.assertTrue(schema["passed"], schema["errors"])
            bad_bundle = json.loads(json.dumps(bundle))
            bad_bundle["artifacts"]["suite_summary"].pop("size_bytes")
            bad_schema = check_schema_contract(bad_bundle, name_or_id="evidence_bundle")
            self.assertFalse(bad_schema["passed"])
            self.assertIn("expected exactly one matching schema from oneOf, got 0", "\n".join(bad_schema["errors"]))
            diagnostic_bundle = json.loads(json.dumps(bundle))
            diagnostic_bundle["artifacts"]["suite_summary"] = {
                "kind": "file",
                "path": "missing-suite-summary.json",
                "exists": False,
            }
            diagnostic_schema = check_schema_contract(diagnostic_bundle, name_or_id="evidence_bundle")
            self.assertTrue(diagnostic_schema["passed"], diagnostic_schema["errors"])

            self.assertEqual(
                run_cli(
                    [
                        "validate",
                        "--harness-manifest",
                        str(harness_artifacts["manifest"]),
                        "--harness-result",
                        str(harness_artifacts["result"]),
                        "--strict",
                    ]
                ),
                0,
            )
            self.assertEqual(run_cli(["validate", "--evidence-bundle", str(bundle_path), "--strict"]), 0)
            preserved_bundle_path = root / "evidence_bundle_preserve_paths.json"
            self.assertEqual(
                run_cli([*bundle_args[:-2], "--preserve-paths", "--out", str(preserved_bundle_path)]),
                0,
            )
            preserved_bundle = json.loads(preserved_bundle_path.read_text(encoding="utf-8"))
            self.assertEqual(preserved_bundle["metrics"]["live_smoke_summary"]["hermes_root"], hermes_root)
            self.assertEqual(preserved_bundle["metrics"]["live_smoke_summary"]["flight_recorder_root"], str(ROOT))
            self.assertEqual(run_cli(["validate", "--evidence-bundle", str(preserved_bundle_path)]), 0)
            self.assertEqual(run_cli(["validate", "--evidence-bundle", str(preserved_bundle_path), "--strict"]), 1)
            bundle["metrics"]["training_export"]["top_curriculum_priorities"][0]["priority_band"] = "urgent"
            bundle_path.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            self.assertEqual(run_cli(["validate", "--evidence-bundle", str(bundle_path)]), 1)

    def test_evidence_bundle_blocks_symlinked_required_json_source_before_hashing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            real_dir = root / "real"
            real_dir.mkdir()
            suite = _write_eval_suite_summary(real_dir / "candidate_suite.json")
            eval_summary = real_dir / "eval_summary.json"
            bundle_path = root / "evidence_bundle.json"
            self.assertEqual(run_cli(["eval-summary", "--suite-summary", f"candidate={suite}", "--out", str(eval_summary)]), 0)
            linked_parent = root / "linked"
            try:
                linked_parent.symlink_to(real_dir, target_is_directory=True)
            except (NotImplementedError, OSError) as exc:
                self.skipTest(f"symlink creation unavailable: {exc}")

            stderr = StringIO()
            with redirect_stdout(StringIO()), redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
                main(
                    [
                        "evidence-bundle",
                        "--eval-summary",
                        str(linked_parent / "eval_summary.json"),
                        "--out",
                        str(bundle_path),
                    ]
                )

            self.assertEqual(raised.exception.code, 2)
            self.assertIn("eval_summary artifact must not traverse symlinked components", stderr.getvalue())
            self.assertFalse(bundle_path.exists())

    def test_evidence_bundle_blocks_symlinked_export_manifest_before_summarizing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            training_export = root / "training_export"
            training_export.mkdir()
            manifest_target = root / "manifest_target.json"
            manifest_target.write_text(json.dumps({"episode_count": 1}, sort_keys=True) + "\n", encoding="utf-8")
            try:
                (training_export / "manifest.json").symlink_to(manifest_target)
            except (NotImplementedError, OSError) as exc:
                self.skipTest(f"symlink creation unavailable: {exc}")
            bundle_path = root / "evidence_bundle.json"

            stderr = StringIO()
            with redirect_stdout(StringIO()), redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
                main(
                    [
                        "evidence-bundle",
                        "--training-export",
                        str(training_export),
                        "--out",
                        str(bundle_path),
                    ]
                )

            self.assertEqual(raised.exception.code, 2)
            self.assertIn("optional JSON artifact must not traverse symlinked components", stderr.getvalue())
            self.assertFalse(bundle_path.exists())

    def test_evidence_bundle_blocks_symlinked_run_digest_before_summarizing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = root / "runs"
            run_dir = runs / "scenario_a"
            run_dir.mkdir(parents=True)
            (run_dir / "scorecard.json").write_text(json.dumps({"passed": True}, sort_keys=True) + "\n", encoding="utf-8")
            digest_target = root / "run_digest_target.json"
            digest_target.write_text(
                json.dumps(
                    {
                        "schema_version": "hfr.run_digest.v1",
                        "scenario": {"id": "scenario_a"},
                        "outcome": {"passed": True, "task_completion_status": "complete"},
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            try:
                (run_dir / "run_digest.json").symlink_to(digest_target)
            except (NotImplementedError, OSError) as exc:
                self.skipTest(f"symlink creation unavailable: {exc}")
            bundle_path = root / "evidence_bundle.json"

            stderr = StringIO()
            with redirect_stdout(StringIO()), redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
                main(["evidence-bundle", "--runs", str(runs), "--out", str(bundle_path)])

            self.assertEqual(raised.exception.code, 2)
            self.assertIn("optional JSON artifact must not traverse symlinked components", stderr.getvalue())
            self.assertFalse(bundle_path.exists())

    def test_evidence_bundle_blocks_missing_required_harness_and_gate_summaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = root / "runs"
            runs.mkdir()
            bundle_path = root / "evidence_bundle.json"

            code = run_cli(
                [
                    "evidence-bundle",
                    "--runs",
                    str(runs),
                    "--require-harness",
                    "--require-gate",
                    "--out",
                    str(bundle_path),
                ]
            )

            self.assertEqual(code, 1)
            bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
            self.assertFalse(bundle["passed"])
            self.assertEqual(bundle["readiness"], "blocked")
            failed_checks = {check["id"] for check in bundle["checks"] if not check["passed"]}
            self.assertIn("harness_handoff_present", failed_checks)
            self.assertIn("gate_summary_present", failed_checks)
            self.assertEqual(bundle["metrics"]["harness_handoff"]["pair_count"], 0)
            self.assertEqual(bundle["metrics"]["harness_handoff"]["missing_pair_count"], 1)
            action_ids = {action["id"] for action in bundle["decision"]["next_actions"]}
            self.assertIn("attach_harness_lineage", action_ids)
            self.assertIn("resolve_blocking_checks", action_ids)
            self.assertEqual(run_cli(["validate", "--evidence-bundle", str(bundle_path), "--strict"]), 0)

    def test_evidence_bundle_blocks_failed_harness_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = _write_harness_handoff_artifacts(root, passed=False)
            bundle_path = root / "evidence_bundle.json"

            code = run_cli(
                [
                    "evidence-bundle",
                    "--harness-manifest",
                    str(artifacts["manifest"]),
                    "--harness-result",
                    str(artifacts["result"]),
                    "--require-harness",
                    "--out",
                    str(bundle_path),
                ]
            )

            self.assertEqual(code, 1)
            bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
            failed_checks = {check["id"] for check in bundle["checks"] if not check["passed"]}
            self.assertIn("harness_result_passed", failed_checks)
            self.assertEqual(bundle["metrics"]["harness_handoff"]["pair_count"], 1)
            self.assertEqual(bundle["metrics"]["harness_handoff"]["failed_pair_count"], 1)
            action_ids = {action["id"] for action in bundle["decision"]["next_actions"]}
            self.assertIn("fix_harness_handoff", action_ids)
            self.assertEqual(run_cli(["validate", "--evidence-bundle", str(bundle_path), "--strict"]), 0)

    def test_evidence_bundle_blocks_missing_harness_artifact_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = _write_harness_handoff_artifacts(root)
            (artifacts["result"].parent / "normalized_trace.json").unlink()
            bundle_path = root / "evidence_bundle.json"

            code = run_cli(
                [
                    "evidence-bundle",
                    "--harness-manifest",
                    str(artifacts["manifest"]),
                    "--harness-result",
                    str(artifacts["result"]),
                    "--require-harness",
                    "--out",
                    str(bundle_path),
                ]
            )

            self.assertEqual(code, 1)
            bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
            failed_checks = {check["id"] for check in bundle["checks"] if not check["passed"]}
            self.assertIn("harness_pair_artifacts_valid", failed_checks)
            self.assertEqual(bundle["metrics"]["harness_handoff"]["artifact_valid_pair_count"], 0)
            row = bundle["metrics"]["harness_handoff"]["runs"][0]
            self.assertEqual(row["artifact_refs_valid"], False)
            self.assertTrue(any("normalized_trace.json" in error for error in row["artifact_ref_errors"]))
            action_ids = {action["id"] for action in bundle["decision"]["next_actions"]}
            self.assertIn("fix_harness_handoff", action_ids)
            self.assertEqual(run_cli(["validate", "--evidence-bundle", str(bundle_path), "--strict"]), 0)

    def test_evidence_bundle_blocks_run_suite_handoff_without_suite_lineage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = _write_harness_handoff_artifacts(root, runner="flightrecorder_run_suite")
            bundle_path = root / "evidence_bundle.json"

            code = run_cli(
                [
                    "evidence-bundle",
                    "--harness-manifest",
                    str(artifacts["manifest"]),
                    "--harness-result",
                    str(artifacts["result"]),
                    "--require-harness",
                    "--out",
                    str(bundle_path),
                ]
            )

            self.assertEqual(code, 1)
            bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
            failed_checks = {check["id"] for check in bundle["checks"] if not check["passed"]}
            self.assertIn("harness_pair_consistent", failed_checks)
            self.assertIn("run_suite_harness_lineage_valid", failed_checks)
            row = bundle["metrics"]["harness_handoff"]["runs"][0]
            self.assertEqual(row["runner"], "flightrecorder_run_suite")
            self.assertEqual(row["run_suite_lineage_valid"], False)
            self.assertIn("run-suite suite metadata missing", row["consistency_errors"])
            self.assertEqual(bundle["metrics"]["harness_handoff"]["run_suite_pair_count"], 1)
            self.assertEqual(bundle["metrics"]["harness_handoff"]["run_suite_lineage_valid_pair_count"], 0)
            self.assertEqual(run_cli(["validate", "--evidence-bundle", str(bundle_path), "--strict"]), 0)

    def test_evidence_bundle_summarizes_complete_trainer_handoff(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = _write_trainer_handoff_artifacts(root)
            bundle_path = root / "evidence_bundle.json"

            code = run_cli(
                [
                    "evidence-bundle",
                    "--trainer-preflight",
                    str(artifacts["trainer_preflight"]),
                    "--trainer-launch-check",
                    str(artifacts["trainer_launch_check"]),
                    "--trainer-archive",
                    str(artifacts["trainer_archive"]),
                    "--trainer-archive-check",
                    str(artifacts["trainer_archive_check"]),
                    "--trainer-consumer-plan",
                    str(artifacts["trainer_consumer_plan"]),
                    "--agentic-training-flow",
                    str(artifacts["agentic_training_flow"]),
                    "--trainer-wrapper-dry-run",
                    str(artifacts["trainer_wrapper_dry_run"]),
                    "--agentic-training-result",
                    str(artifacts["agentic_training_result"]),
                    "--out",
                    str(bundle_path),
                ]
            )

            self.assertEqual(code, 0)
            bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
            self.assertTrue(bundle["passed"])
            self.assertEqual(bundle["decision"]["recommendation"], "promote_handoff")
            self.assertIn("trainer_handoff", bundle["decision"]["key_metrics"])
            trainer = bundle["metrics"]["trainer_handoff"]
            self.assertEqual(trainer["stage_count"], 8)
            self.assertEqual(trainer["handoff_ready_count"], 8)
            self.assertEqual(trainer["blocked_stage_count"], 0)
            self.assertEqual(trainer["schema_supported_count"], 8)
            self.assertTrue(trainer["complete_chain"])
            self.assertTrue(trainer["all_included_ready"])
            self.assertEqual(trainer["missing_stage_ids"], [])
            self.assertEqual([stage["id"] for stage in trainer["stages"]], [
                "trainer_preflight",
                "trainer_launch_check",
                "trainer_archive",
                "trainer_archive_check",
                "trainer_consumer_plan",
                "agentic_training_flow",
                "trainer_wrapper_dry_run",
                "agentic_training_result",
            ])
            result_stage = next(stage for stage in trainer["stages"] if stage["id"] == "agentic_training_result")
            self.assertEqual(result_stage["status"], "completed")
            self.assertEqual(result_stage["failure_class"], "none")
            self.assertEqual(result_stage["expected_recommendations"], ["register_training_result", "register_training_failure"])
            self.assertTrue(result_stage["registry_update_ready"])
            self.assertEqual(result_stage["output_artifact_count"], 1)
            self.assertEqual(bundle["artifacts"]["trainer_archive"]["kind"], "directory")
            self.assertEqual(bundle["artifacts"]["trainer_archive_manifest"]["kind"], "file")
            self.assertEqual(bundle["artifacts"]["agentic_training_result"]["schema_version"], "hfr.agentic_training_result.v1")
            self.assertEqual(bundle["decision"]["key_metrics"]["trainer_handoff"]["complete_chain"], True)
            self.assertEqual(bundle["decision"]["key_metrics"]["trainer_handoff"]["all_included_ready"], True)
            self.assertEqual(run_cli(["validate", "--evidence-bundle", str(bundle_path), "--strict"]), 0)

            bundle["metrics"]["trainer_handoff"]["blocked_stage_count"] = 1
            bundle_path.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            self.assertEqual(run_cli(["validate", "--evidence-bundle", str(bundle_path)]), 1)

    def test_evidence_bundle_blocks_failed_trainer_handoff_stage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            preflight_path = root / "trainer_preflight.json"
            preflight_path.write_text(
                json.dumps(
                    {
                        "schema_version": "hfr.trainer_preflight.v1",
                        "passed": False,
                        "readiness": "blocked",
                        "recommendation": "block_launch",
                        "check_count": 2,
                        "failed_check_count": 1,
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            bundle_path = root / "evidence_bundle.json"

            code = run_cli(
                [
                    "evidence-bundle",
                    "--trainer-preflight",
                    str(preflight_path),
                    "--out",
                    str(bundle_path),
                ]
            )

            self.assertEqual(code, 1)
            bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
            self.assertFalse(bundle["passed"])
            trainer = bundle["metrics"]["trainer_handoff"]
            self.assertEqual(trainer["stage_count"], 1)
            self.assertEqual(trainer["blocked_stage_count"], 1)
            self.assertFalse(trainer["complete_chain"])
            self.assertEqual(
                trainer["missing_stage_ids"],
                [
                    "trainer_launch_check",
                    "trainer_archive",
                    "trainer_archive_check",
                    "trainer_consumer_plan",
                    "agentic_training_flow",
                    "trainer_wrapper_dry_run",
                    "agentic_training_result",
                ],
            )
            failed_checks = {check["id"] for check in bundle["checks"] if not check["passed"]}
            self.assertIn("trainer_preflight_ready", failed_checks)
            action_ids = {action["id"] for action in bundle["decision"]["next_actions"]}
            self.assertIn("fix_trainer_handoff", action_ids)
            self.assertIn("complete_trainer_handoff_chain", action_ids)
            self.assertEqual(run_cli(["validate", "--evidence-bundle", str(bundle_path), "--strict"]), 0)

    def test_evidence_bundle_blocks_trainer_stage_with_failed_check_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            preflight_path = root / "trainer_preflight.json"
            preflight_path.write_text(
                json.dumps(
                    {
                        "schema_version": "hfr.trainer_preflight.v1",
                        "passed": True,
                        "readiness": "ready",
                        "recommendation": "launch_allowed",
                        "check_count": 2,
                        "failed_check_count": 1,
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            bundle_path = root / "evidence_bundle.json"

            code = run_cli(
                [
                    "evidence-bundle",
                    "--trainer-preflight",
                    str(preflight_path),
                    "--out",
                    str(bundle_path),
                ]
            )

            self.assertEqual(code, 1)
            bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
            stage = bundle["metrics"]["trainer_handoff"]["stages"][0]
            self.assertTrue(stage["passed"])
            self.assertEqual(stage["failed_check_count"], 1)
            self.assertFalse(stage["handoff_ready"])
            failed_checks = {check["id"] for check in bundle["checks"] if not check["passed"]}
            self.assertIn("trainer_preflight_ready", failed_checks)
            self.assertEqual(run_cli(["validate", "--evidence-bundle", str(bundle_path), "--strict"]), 0)

    def test_validate_rejects_trainer_stage_ready_with_failed_checks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = _write_trainer_handoff_artifacts(root)
            bundle_path = root / "evidence_bundle.json"
            summary_path = root / "validation.json"
            self.assertEqual(
                run_cli(
                    [
                        "evidence-bundle",
                        "--trainer-preflight",
                        str(artifacts["trainer_preflight"]),
                        "--trainer-launch-check",
                        str(artifacts["trainer_launch_check"]),
                        "--trainer-archive",
                        str(artifacts["trainer_archive"]),
                        "--trainer-archive-check",
                        str(artifacts["trainer_archive_check"]),
                        "--trainer-consumer-plan",
                        str(artifacts["trainer_consumer_plan"]),
                        "--agentic-training-flow",
                        str(artifacts["agentic_training_flow"]),
                        "--trainer-wrapper-dry-run",
                        str(artifacts["trainer_wrapper_dry_run"]),
                        "--agentic-training-result",
                        str(artifacts["agentic_training_result"]),
                        "--out",
                        str(bundle_path),
                    ]
                ),
                0,
            )
            bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
            bundle["metrics"]["trainer_handoff"]["stages"][0]["failed_check_count"] = 1
            bundle_path.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            code = run_cli(["validate", "--evidence-bundle", str(bundle_path), "--out", str(summary_path)])

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in summary["targets"] for error in target["errors"])
            self.assertIn("handoff_ready cannot be true when failed_check_count is greater than 0", errors)

    def test_evidence_bundle_accepts_classified_agentic_training_failure_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result_path = root / "agentic_training_result.json"
            result_path.write_text(
                json.dumps(
                    {
                        "schema_version": "hfr.agentic_training_result.v1",
                        "passed": True,
                        "readiness": "ready",
                        "recommendation": "register_training_failure",
                        "check_count": 13,
                        "failed_check_count": 0,
                        "training_result": {
                            "status": "failed",
                            "runner_id": "external-trainer",
                            "run_id": "example-failed-run",
                        },
                        "failure": {
                            "class": "out_of_memory",
                            "message": "Synthetic failure fixture.",
                        },
                        "metrics": {
                            "artifact_count": 2,
                            "regular_artifact_count": 2,
                            "output_artifact_count": 0,
                            "config_count": 0,
                            "metrics_file_count": 1,
                            "adapter_count": 0,
                            "checkpoint_count": 0,
                            "log_count": 1,
                            "failure_report_count": 1,
                        },
                        "registry_update": {
                            "ready_to_apply": True,
                        },
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            bundle_path = root / "evidence_bundle.json"

            code = run_cli(
                [
                    "evidence-bundle",
                    "--agentic-training-result",
                    str(result_path),
                    "--out",
                    str(bundle_path),
                ]
            )

            self.assertEqual(code, 0)
            bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
            trainer = bundle["metrics"]["trainer_handoff"]
            self.assertEqual(trainer["stage_count"], 1)
            self.assertEqual(trainer["handoff_ready_count"], 1)
            self.assertEqual(trainer["blocked_stage_count"], 0)
            self.assertFalse(trainer["complete_chain"])
            result_stage = trainer["stages"][0]
            self.assertEqual(result_stage["id"], "agentic_training_result")
            self.assertTrue(result_stage["handoff_ready"])
            self.assertEqual(result_stage["recommendation"], "register_training_failure")
            self.assertEqual(result_stage["status"], "failed")
            self.assertEqual(result_stage["failure_class"], "out_of_memory")
            self.assertEqual(result_stage["failure_report_count"], 1)
            self.assertEqual(run_cli(["validate", "--evidence-bundle", str(bundle_path), "--strict"]), 0)

    def test_evidence_bundle_blocks_missing_run_digest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = root / "runs"
            bundle_path = runs / "evidence_bundle.json"
            self.assertEqual(
                run_cli(["run-suite", "--scenarios", str(ROOT / "scenarios"), "--out", str(runs)]),
                0,
            )
            (runs / "prompt_injection_bad" / "run_digest.json").unlink()

            code = run_cli(["evidence-bundle", "--runs", str(runs), "--out", str(bundle_path)])

            self.assertEqual(code, 1)
            bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
            self.assertFalse(bundle["passed"])
            self.assertEqual(bundle["metrics"]["run_digest_coverage"]["run_count"], 7)
            self.assertEqual(bundle["metrics"]["run_digest_coverage"]["digest_count"], 6)
            self.assertEqual(bundle["metrics"]["run_digest_coverage"]["missing_digest_count"], 1)
            self.assertEqual(bundle["metrics"]["run_digest_coverage"]["digest_coverage_rate"], 0.8571)
            failed_checks = {check["id"] for check in bundle["checks"] if not check["passed"]}
            self.assertIn("run_digest_coverage_complete", failed_checks)
            action = next(action for action in bundle["decision"]["next_actions"] if action["id"] == "refresh_run_digests")
            self.assertEqual(action["artifact"], "run_digest_coverage")
            self.assertEqual(action["priority"], "high")
            self.assertEqual(action["evidence"]["missing_digest_count"], 1)
            self.assertEqual(run_cli(["validate", "--evidence-bundle", str(bundle_path), "--strict"]), 0)
            bundle["metrics"]["run_digest_coverage"]["digest_count"] = 7
            bundle_path.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            self.assertEqual(run_cli(["validate", "--evidence-bundle", str(bundle_path)]), 1)

    def test_validate_evidence_bundle_rejects_forged_run_digest_outcome_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = root / "runs"
            bundle_path = runs / "evidence_bundle.json"
            summary_path = root / "validation.json"
            self.assertEqual(run_cli(["run-suite", "--scenarios", str(ROOT / "scenarios"), "--out", str(runs)]), 0)
            self.assertEqual(run_cli(["evidence-bundle", "--runs", str(runs), "--out", str(bundle_path)]), 0)
            bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
            coverage = bundle["metrics"]["run_digest_coverage"]
            coverage["passed_digest_count"] = coverage["digest_count"]
            coverage["failed_digest_count"] = coverage["digest_count"]
            bundle_path.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            code = run_cli(["validate", "--evidence-bundle", str(bundle_path), "--strict", "--out", str(summary_path)])

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in summary["targets"] for error in target["errors"])
            self.assertIn("passed_digest_count + failed_digest_count", errors)

    def test_validate_evidence_bundle_rejects_forged_serving_lifecycle_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            lifecycle_path = _write_serving_lifecycle(root)
            bundle_path = root / "evidence_bundle.json"
            summary_path = root / "validation.json"
            self.assertEqual(
                run_cli(
                    [
                        "evidence-bundle",
                        "--serving-lifecycle",
                        str(lifecycle_path),
                        "--out",
                        str(bundle_path),
                    ]
                ),
                0,
            )
            bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
            self.assertTrue(bundle["passed"])
            self.assertEqual(bundle["metrics"]["serving_lifecycle"]["preflight_artifact_count"], 3)
            self.assertEqual(bundle["decision"]["key_metrics"]["serving_lifecycle"]["readiness"], "ready")
            self.assertEqual(run_cli(["validate", "--evidence-bundle", str(bundle_path), "--strict"]), 0)
            bundle["metrics"]["serving_lifecycle"]["preflight_artifact_count"] = 0
            bundle_path.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            code = run_cli(["validate", "--evidence-bundle", str(bundle_path), "--strict", "--out", str(summary_path)])

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in summary["targets"] for error in target["errors"])
            self.assertIn("evidence_bundle.metrics.serving_lifecycle.preflight_artifact_count expected 3", errors)

    def test_evidence_bundle_includes_eval_summary_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            suite = _write_eval_suite_summary(root / "candidate_suite.json")
            eval_summary = root / "eval_summary.json"
            bundle_path = root / "evidence_bundle.json"
            self.assertEqual(
                run_cli(
                    [
                        "eval-summary",
                        "--suite-summary",
                        f"candidate={suite}",
                        "--out",
                        str(eval_summary),
                    ]
                ),
                0,
            )
            self.assertEqual(
                run_cli(
                    [
                        "evidence-bundle",
                        "--eval-summary",
                        str(eval_summary),
                        "--out",
                        str(bundle_path),
                    ]
                ),
                0,
            )

            bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
            self.assertTrue(bundle["passed"])
            self.assertIn("eval_summary", bundle["decision"]["evidence_artifacts"])
            self.assertEqual(bundle["metrics"]["eval_summary"]["arm_count"], 1)
            self.assertEqual(bundle["metrics"]["eval_summary"]["risk_count"], 0)
            self.assertEqual(bundle["decision"]["key_metrics"]["eval_summary"]["heldout_status"], "single_arm")
            self.assertEqual(run_cli(["validate", "--evidence-bundle", str(bundle_path), "--strict"]), 0)

    def test_evidence_bundle_writes_output_relative_eval_summary_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = root / "src"
            output_dir = root / "out"
            source_dir.mkdir()
            output_dir.mkdir()
            suite = _write_eval_suite_summary(source_dir / "candidate_suite.json")
            eval_summary = source_dir / "eval_summary.json"
            bundle_path = output_dir / "evidence_bundle.json"
            self.assertEqual(
                run_cli(["eval-summary", "--suite-summary", f"candidate={suite}", "--out", str(eval_summary)]),
                0,
            )

            self.assertEqual(
                run_cli(["evidence-bundle", "--eval-summary", str(eval_summary), "--out", str(bundle_path)]),
                0,
            )

            bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
            self.assertEqual(bundle["artifacts"]["eval_summary"]["path"], "../src/eval_summary.json")
            self.assertEqual(run_cli(["validate", "--evidence-bundle", str(bundle_path), "--strict"]), 0)

    def test_validate_rejects_evidence_bundle_cwd_relative_eval_summary_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = root / "src"
            output_dir = root / "out"
            source_dir.mkdir()
            output_dir.mkdir()
            suite = _write_eval_suite_summary(source_dir / "candidate_suite.json")
            eval_summary = source_dir / "eval_summary.json"
            bundle_path = output_dir / "evidence_bundle.json"
            summary_path = root / "validation.json"
            self.assertEqual(
                run_cli(["eval-summary", "--suite-summary", f"candidate={suite}", "--out", str(eval_summary)]),
                0,
            )
            self.assertEqual(
                run_cli(["evidence-bundle", "--eval-summary", str(eval_summary), "--out", str(bundle_path)]),
                0,
            )
            bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
            bundle["artifacts"]["eval_summary"]["path"] = "eval_summary.json"
            bundle_path.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            previous_cwd = Path.cwd()
            try:
                os.chdir(source_dir)
                code = run_cli(["validate", "--evidence-bundle", str(bundle_path), "--strict", "--out", str(summary_path)])
            finally:
                os.chdir(previous_cwd)

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in summary["targets"] for error in target["errors"])
            self.assertIn("evidence_bundle.artifacts.eval_summary.path must resolve to an existing file", errors)

    def test_validate_rejects_evidence_bundle_non_utf8_eval_summary_without_crashing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            suite = _write_eval_suite_summary(root / "candidate_suite.json")
            eval_summary = root / "eval_summary.json"
            bundle_path = root / "evidence_bundle.json"
            summary_path = root / "validation.json"
            self.assertEqual(
                run_cli(["eval-summary", "--suite-summary", f"candidate={suite}", "--out", str(eval_summary)]),
                0,
            )
            self.assertEqual(
                run_cli(["evidence-bundle", "--eval-summary", str(eval_summary), "--out", str(bundle_path)]),
                0,
            )
            eval_summary.write_bytes(b"\xff\xfe\x00")

            code = run_cli(["validate", "--evidence-bundle", str(bundle_path), "--strict", "--out", str(summary_path)])

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in summary["targets"] for error in target["errors"])
            self.assertIn("evidence_bundle.artifacts.eval_summary is not valid UTF-8", errors)

    def test_validate_evidence_bundle_rejects_forged_eval_summary_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            suite = _write_eval_suite_summary(root / "candidate_suite.json")
            eval_summary = root / "eval_summary.json"
            bundle_path = root / "evidence_bundle.json"
            validation = root / "validation.json"
            self.assertEqual(
                run_cli(
                    [
                        "eval-summary",
                        "--suite-summary",
                        f"candidate={suite}",
                        "--require-serving-preflight",
                        "--out",
                        str(eval_summary),
                    ]
                ),
                1,
            )
            self.assertEqual(
                run_cli(
                    [
                        "evidence-bundle",
                        "--eval-summary",
                        str(eval_summary),
                        "--out",
                        str(bundle_path),
                    ]
                ),
                1,
            )
            self.assertEqual(run_cli(["validate", "--evidence-bundle", str(bundle_path), "--strict"]), 0)
            bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
            self.assertFalse(bundle["passed"])
            self.assertEqual(bundle["metrics"]["eval_summary"]["risk_count"], 1)
            self.assertIn("resolve_eval_summary_blockers", {action["id"] for action in bundle["decision"]["next_actions"]})

            bundle["metrics"]["eval_summary"]["risk_count"] = 0
            bundle_path.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            code = run_cli(["validate", "--evidence-bundle", str(bundle_path), "--strict", "--out", str(validation)])

            self.assertEqual(code, 1)
            summary = json.loads(validation.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in summary["targets"] for error in target["errors"])
            self.assertIn("evidence_bundle.metrics.eval_summary.risk_count expected 1", errors)

    def test_validate_evidence_bundle_rejects_stale_artifact_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            suite = _write_eval_suite_summary(root / "candidate_suite.json")
            eval_summary = root / "eval_summary.json"
            bundle_path = root / "evidence_bundle.json"
            validation = root / "validation.json"
            self.assertEqual(run_cli(["eval-summary", "--suite-summary", f"candidate={suite}", "--out", str(eval_summary)]), 0)
            self.assertEqual(run_cli(["evidence-bundle", "--eval-summary", str(eval_summary), "--out", str(bundle_path)]), 0)
            bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
            bundle["artifacts"]["eval_summary"]["sha256"] = "0" * 64
            bundle_path.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            code = run_cli(["validate", "--evidence-bundle", str(bundle_path), "--strict", "--out", str(validation)])

            self.assertEqual(code, 1)
            summary = json.loads(validation.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in summary["targets"] for error in target["errors"])
            self.assertIn("evidence_bundle.artifacts.eval_summary.sha256 does not match the current file.", errors)

    def test_validate_evidence_bundle_rejects_missing_existing_file_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            suite = _write_eval_suite_summary(root / "candidate_suite.json")
            eval_summary = root / "eval_summary.json"
            bundle_path = root / "evidence_bundle.json"
            validation = root / "validation.json"
            self.assertEqual(run_cli(["eval-summary", "--suite-summary", f"candidate={suite}", "--out", str(eval_summary)]), 0)
            self.assertEqual(run_cli(["evidence-bundle", "--eval-summary", str(eval_summary), "--out", str(bundle_path)]), 0)
            bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
            bundle["artifacts"]["eval_summary"]["path"] = "missing_eval_summary.json"
            bundle_path.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            code = run_cli(["validate", "--evidence-bundle", str(bundle_path), "--strict", "--out", str(validation)])

            self.assertEqual(code, 1)
            summary = json.loads(validation.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in summary["targets"] for error in target["errors"])
            self.assertIn("evidence_bundle.artifacts.eval_summary.path must resolve to an existing file when exists is true.", errors)

    def test_validate_evidence_bundle_rejects_symlink_existing_file_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            suite = _write_eval_suite_summary(root / "candidate_suite.json")
            eval_summary = root / "eval_summary.json"
            bundle_path = root / "evidence_bundle.json"
            validation = root / "validation.json"
            self.assertEqual(run_cli(["eval-summary", "--suite-summary", f"candidate={suite}", "--out", str(eval_summary)]), 0)
            self.assertEqual(run_cli(["evidence-bundle", "--eval-summary", str(eval_summary), "--out", str(bundle_path)]), 0)
            symlink_path = root / "eval_summary_link.json"
            symlink_path.symlink_to(eval_summary)
            bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
            bundle["artifacts"]["eval_summary"]["path"] = symlink_path.name
            bundle_path.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            code = run_cli(["validate", "--evidence-bundle", str(bundle_path), "--strict", "--out", str(validation)])

            self.assertEqual(code, 1)
            summary = json.loads(validation.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in summary["targets"] for error in target["errors"])
            self.assertIn("evidence_bundle.artifacts.eval_summary.path must resolve to a regular file when exists is true.", errors)

    def test_validate_evidence_bundle_rejects_parent_symlink_existing_file_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            real_dir = root / "real"
            real_dir.mkdir()
            suite = _write_eval_suite_summary(root / "candidate_suite.json")
            eval_summary = real_dir / "eval_summary.json"
            bundle_path = root / "evidence_bundle.json"
            validation = root / "validation.json"
            self.assertEqual(run_cli(["eval-summary", "--suite-summary", f"candidate={suite}", "--out", str(eval_summary)]), 0)
            self.assertEqual(run_cli(["evidence-bundle", "--eval-summary", str(eval_summary), "--out", str(bundle_path)]), 0)
            linked_parent = root / "linked"
            try:
                linked_parent.symlink_to(real_dir, target_is_directory=True)
            except (NotImplementedError, OSError) as exc:
                self.skipTest(f"symlink creation unavailable: {exc}")
            bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
            bundle["artifacts"]["eval_summary"]["path"] = str(Path("linked") / "eval_summary.json")
            bundle_path.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            code = run_cli(["validate", "--evidence-bundle", str(bundle_path), "--strict", "--out", str(validation)])

            self.assertEqual(code, 1)
            summary = json.loads(validation.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in summary["targets"] for error in target["errors"])
            self.assertIn("evidence_bundle.artifacts.eval_summary.path must resolve to a regular file when exists is true.", errors)

    def test_validate_evidence_bundle_rejects_missing_existing_directory_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = root / "runs"
            runs.mkdir()
            bundle_path = root / "evidence_bundle.json"
            validation = root / "validation.json"
            self.assertEqual(run_cli(["evidence-bundle", "--runs", str(runs), "--out", str(bundle_path)]), 0)
            bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
            bundle["artifacts"]["runs_dir"]["path"] = "missing_runs"
            bundle_path.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            code = run_cli(["validate", "--evidence-bundle", str(bundle_path), "--strict", "--out", str(validation)])

            self.assertEqual(code, 1)
            summary = json.loads(validation.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in summary["targets"] for error in target["errors"])
            self.assertIn("evidence_bundle.artifacts.runs_dir.path must resolve to an existing directory when exists is true.", errors)

    def test_validate_evidence_bundle_rejects_symlink_existing_directory_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = root / "runs"
            runs.mkdir()
            bundle_path = root / "evidence_bundle.json"
            validation = root / "validation.json"
            self.assertEqual(run_cli(["evidence-bundle", "--runs", str(runs), "--out", str(bundle_path)]), 0)
            symlink_path = root / "runs_link"
            symlink_path.symlink_to(runs, target_is_directory=True)
            bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
            bundle["artifacts"]["runs_dir"]["path"] = symlink_path.name
            bundle_path.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            code = run_cli(["validate", "--evidence-bundle", str(bundle_path), "--strict", "--out", str(validation)])

            self.assertEqual(code, 1)
            summary = json.loads(validation.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in summary["targets"] for error in target["errors"])
            self.assertIn("evidence_bundle.artifacts.runs_dir.path must resolve to a regular directory when exists is true.", errors)

    def test_validate_evidence_bundle_rejects_parent_symlink_existing_directory_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            real_parent = root / "real"
            runs = real_parent / "runs"
            runs.mkdir(parents=True)
            bundle_path = root / "evidence_bundle.json"
            validation = root / "validation.json"
            self.assertEqual(run_cli(["evidence-bundle", "--runs", str(runs), "--out", str(bundle_path)]), 0)
            linked_parent = root / "linked"
            try:
                linked_parent.symlink_to(real_parent, target_is_directory=True)
            except (NotImplementedError, OSError) as exc:
                self.skipTest(f"symlink creation unavailable: {exc}")
            bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
            bundle["artifacts"]["runs_dir"]["path"] = str(Path("linked") / "runs")
            bundle_path.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            code = run_cli(["validate", "--evidence-bundle", str(bundle_path), "--strict", "--out", str(validation)])

            self.assertEqual(code, 1)
            summary = json.loads(validation.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in summary["targets"] for error in target["errors"])
            self.assertIn("evidence_bundle.artifacts.runs_dir.path must resolve to a regular directory when exists is true.", errors)

    def test_validate_evidence_bundle_rejects_present_artifact_marked_missing_with_stale_fingerprint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            suite = _write_eval_suite_summary(root / "candidate_suite.json")
            eval_summary = root / "eval_summary.json"
            bundle_path = root / "evidence_bundle.json"
            validation = root / "validation.json"
            self.assertEqual(run_cli(["eval-summary", "--suite-summary", f"candidate={suite}", "--out", str(eval_summary)]), 0)
            self.assertEqual(run_cli(["evidence-bundle", "--eval-summary", str(eval_summary), "--out", str(bundle_path)]), 0)
            bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
            bundle["artifacts"]["eval_summary"]["exists"] = False
            bundle["artifacts"]["eval_summary"]["sha256"] = "0" * 64
            bundle["artifacts"]["eval_summary"]["size_bytes"] += 1
            bundle_path.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            code = run_cli(["validate", "--evidence-bundle", str(bundle_path), "--strict", "--out", str(validation)])

            self.assertEqual(code, 1)
            summary = json.loads(validation.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in summary["targets"] for error in target["errors"])
            self.assertIn("evidence_bundle.artifacts.eval_summary.sha256 does not match the current file.", errors)
            self.assertIn("evidence_bundle.artifacts.eval_summary.size_bytes does not match the current file.", errors)

    def test_evidence_bundle_blocks_failed_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = root / "runs"
            runs.mkdir()
            gate_path = root / "failed_gate.json"
            bundle_path = root / "evidence_bundle.json"
            gate_path.write_text(
                json.dumps({"schema_version": "hfr.test_gate.v1", "passed": False}, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            code = run_cli(
                [
                    "evidence-bundle",
                    "--runs",
                    str(runs),
                    "--gate",
                    str(gate_path),
                    "--out",
                    str(bundle_path),
                ]
            )

            self.assertEqual(code, 1)
            bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
            self.assertFalse(bundle["passed"])
            self.assertEqual(bundle["readiness"], "blocked")
            self.assertEqual(bundle["failed_check_count"], 2)
            self.assertEqual(bundle["decision"]["readiness"], "blocked")
            self.assertEqual(bundle["decision"]["recommendation"], "block_handoff")
            self.assertEqual(bundle["decision"]["blocking_check_count"], 2)
            self.assertEqual(bundle["decision"]["blocking_checks"][0]["id"], "gate_passed")
            self.assertEqual(bundle["decision"]["blocking_gates"][0]["id"], "test_gate")
            self.assertFalse(bundle["metrics"]["gates"][0]["contract"]["valid"])
            self.assertIn("decision must be an object", bundle["metrics"]["gates"][0]["contract"]["errors"])
            self.assertIn("fix_failed_gates", {action["id"] for action in bundle["decision"]["next_actions"]})
            fix_gate_action = next(action for action in bundle["decision"]["next_actions"] if action["id"] == "fix_failed_gates")
            self.assertEqual(len(fix_gate_action["action_fingerprint"]), 64)
            self.assertEqual(
                fix_gate_action["routing_key"],
                f"gates:fix_failed_gates:{fix_gate_action['action_fingerprint'][:12]}",
            )
            self.assertEqual(bundle["decision"]["gate_count"], 1)
            self.assertEqual(bundle["decision"]["passed_gate_count"], 0)
            self.assertEqual(bundle["decision"]["key_metrics"]["gates"]["failed"], 1)
            failed_checks = [check for check in bundle["checks"] if not check["passed"]]
            self.assertEqual(failed_checks[0]["id"], "gate_passed")
            self.assertEqual(failed_checks[0]["scope"]["gate"], "test_gate")
            self.assertEqual(run_cli(["validate", "--evidence-bundle", str(bundle_path), "--strict"]), 0)

    def test_evidence_bundle_blocks_passed_gate_without_decision_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = root / "runs"
            runs.mkdir()
            gate_path = root / "weak_gate.json"
            bundle_path = root / "evidence_bundle.json"
            gate_path.write_text(
                json.dumps({"schema_version": "hfr.weak_gate.v1", "passed": True, "checks": [], "failed_check_count": 0}, sort_keys=True)
                + "\n",
                encoding="utf-8",
            )

            code = run_cli(
                [
                    "evidence-bundle",
                    "--runs",
                    str(runs),
                    "--gate",
                    str(gate_path),
                    "--out",
                    str(bundle_path),
                ]
            )

            self.assertEqual(code, 1)
            bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
            failed_checks = {check["id"] for check in bundle["checks"] if not check["passed"]}
            self.assertIn("gate_contract_valid", failed_checks)
            gate_metrics = bundle["metrics"]["gates"][0]
            self.assertTrue(gate_metrics["passed"])
            self.assertFalse(gate_metrics["contract"]["valid"])
            self.assertIn("decision must be an object", gate_metrics["contract"]["errors"])
            self.assertEqual(bundle["decision"]["recommendation"], "block_handoff")
            self.assertEqual(run_cli(["validate", "--evidence-bundle", str(bundle_path), "--strict"]), 0)

    def test_evidence_bundle_blocks_validation_skipped_gates(self):
        validation_required_schemas = (
            "hfr.training_gate.v1",
            "hfr.compare_gate.v1",
            "hfr.reviewed_gate.v1",
            "hfr.review_calibration.v1",
        )
        for schema_version in validation_required_schemas:
            with self.subTest(schema_version=schema_version), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                runs = root / "runs"
                runs.mkdir()
                gate_path = root / "gate.json"
                bundle_path = root / "evidence_bundle.json"
                gate_path.write_text(
                    json.dumps(
                        {
                            "schema_version": schema_version,
                            "passed": True,
                            "metrics": {
                                "validation": {
                                    "available": False,
                                    "passed": False,
                                    "strict": False,
                                    "target_count": 0,
                                    "error_count": 0,
                                    "warning_count": 0,
                                }
                            },
                        },
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )

                code = run_cli(
                    [
                        "evidence-bundle",
                        "--runs",
                        str(runs),
                        "--gate",
                        str(gate_path),
                        "--out",
                        str(bundle_path),
                    ]
                )

                self.assertEqual(code, 1)
                bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
                self.assertFalse(bundle["passed"])
                self.assertEqual(bundle["decision"]["recommendation"], "block_handoff")
                failed_checks = {check["id"] for check in bundle["checks"] if not check["passed"]}
                self.assertIn("gate_validation_passed", failed_checks)
                self.assertIn("gate_contract_valid", failed_checks)
                gate_metrics = bundle["metrics"]["gates"][0]
                self.assertEqual(gate_metrics["schema_version"], schema_version)
                self.assertTrue(gate_metrics["passed"])
                self.assertFalse(gate_metrics["contract"]["valid"])
                self.assertFalse(gate_metrics["validation"]["available"])
                self.assertFalse(gate_metrics["validation"]["passed"])
                blocking_checks = {check["id"] for check in bundle["decision"]["blocking_checks"]}
                self.assertIn("gate_validation_passed", blocking_checks)
                self.assertEqual(run_cli(["validate", "--evidence-bundle", str(bundle_path), "--strict"]), 0)

    def test_evidence_bundle_blocks_low_signal_required_gate_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = root / "runs"
            runs.mkdir()
            gate_path = root / "gate.json"
            bundle_path = root / "evidence_bundle.json"
            gate_path.write_text(
                json.dumps(
                    {
                        "schema_version": "hfr.training_gate.v1",
                        "passed": True,
                        "metrics": {
                            "validation": {
                                "available": True,
                                "passed": True,
                                "strict": True,
                                "target_count": 0,
                                "error_count": 1,
                                "warning_count": 0,
                            }
                        },
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )

            code = run_cli(["evidence-bundle", "--runs", str(runs), "--gate", str(gate_path), "--out", str(bundle_path)])

            self.assertEqual(code, 1)
            bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
            failed_checks = {check["id"] for check in bundle["checks"] if not check["passed"]}
            self.assertIn("gate_validation_has_targets", failed_checks)
            self.assertIn("gate_validation_counts_consistent", failed_checks)
            gate_metrics = bundle["metrics"]["gates"][0]
            self.assertTrue(gate_metrics["validation"]["available"])
            self.assertTrue(gate_metrics["validation"]["passed"])
            self.assertEqual(gate_metrics["validation"]["target_count"], 0)
            self.assertEqual(run_cli(["validate", "--evidence-bundle", str(bundle_path), "--strict"]), 0)

    def test_evidence_bundle_blocks_low_signal_validation_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = root / "runs"
            runs.mkdir()
            validation_path = root / "validation.json"
            bundle_path = root / "evidence_bundle.json"
            validation_path.write_text(
                json.dumps(
                    {
                        "schema_version": "hfr.validation.v1",
                        "passed": True,
                        "strict": True,
                        "target_count": 0,
                        "error_count": 1,
                        "warning_count": 1,
                        "targets": [],
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )

            code = run_cli(["evidence-bundle", "--runs", str(runs), "--validation", str(validation_path), "--out", str(bundle_path)])

            self.assertEqual(code, 1)
            bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
            self.assertFalse(bundle["passed"])
            self.assertEqual(bundle["metrics"]["validation"]["passed"], True)
            self.assertEqual(bundle["metrics"]["validation"]["strict"], True)
            failed_checks = {check["id"] for check in bundle["checks"] if not check["passed"]}
            self.assertIn("validation_has_targets", failed_checks)
            self.assertIn("validation_counts_consistent", failed_checks)
            self.assertEqual(run_cli(["validate", "--evidence-bundle", str(bundle_path), "--strict"]), 0)

    def test_validate_rejects_hidden_low_signal_validation_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = root / "runs"
            runs.mkdir()
            validation_path = root / "validation.json"
            bundle_path = root / "evidence_bundle.json"
            summary_path = root / "validation_summary.json"
            validation_path.write_text(
                json.dumps(
                    {
                        "schema_version": "hfr.validation.v1",
                        "passed": True,
                        "strict": True,
                        "target_count": 1,
                        "error_count": 0,
                        "warning_count": 0,
                        "targets": [
                            {
                                "type": "runs",
                                "path": "runs",
                                "passed": True,
                                "errors": [],
                                "warnings": [],
                                "details": {},
                            }
                        ],
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            self.assertEqual(run_cli(["evidence-bundle", "--runs", str(runs), "--validation", str(validation_path), "--out", str(bundle_path)]), 0)
            bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
            bundle["metrics"]["validation"]["target_count"] = 0
            bundle["metrics"]["validation"]["error_count"] = 1
            bundle["metrics"]["validation"]["warning_count"] = 1
            bundle["metrics"]["gates"] = [
                {
                    "id": "training_gate",
                    "path": "training_gate.json",
                    "passed": True,
                    "validation": {
                        "available": True,
                        "passed": True,
                        "strict": True,
                        "target_count": 0,
                        "error_count": 1,
                        "warning_count": 0,
                    },
                }
            ]
            bundle["decision"]["gate_count"] = 1
            bundle["decision"]["passed_gate_count"] = 1
            bundle_path.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            code = run_cli(["validate", "--evidence-bundle", str(bundle_path), "--out", str(summary_path)])

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in summary["targets"] for error in target["errors"])
            self.assertIn("validation_has_targets", errors)
            self.assertIn("validation_counts_consistent", errors)
            self.assertIn("gate_validation_has_targets", errors)
            self.assertIn("gate_validation_counts_consistent", errors)

    def test_validate_rejects_stale_bundle_decision(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = root / "runs"
            runs.mkdir()
            gate_path = root / "failed_gate.json"
            bundle_path = root / "evidence_bundle.json"
            summary_path = root / "validation.json"
            gate_path.write_text(
                json.dumps({"schema_version": "hfr.test_gate.v1", "passed": False}, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            run_cli(["evidence-bundle", "--runs", str(runs), "--gate", str(gate_path), "--out", str(bundle_path)])
            bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
            bundle["decision"]["recommendation"] = "promote_handoff"
            bundle["decision"]["next_action_count"] = 0
            bundle_path.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            code = run_cli(["validate", "--evidence-bundle", str(bundle_path), "--out", str(summary_path)])

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in summary["targets"] for error in target["errors"])
            self.assertIn("evidence_bundle.decision.recommendation", errors)
            self.assertIn("evidence_bundle.decision.next_action_count", errors)

    def test_validate_rejects_stale_bundle_notes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = root / "runs"
            runs.mkdir()
            bundle_path = root / "evidence_bundle.json"
            summary_path = root / "validation.json"
            run_cli(["evidence-bundle", "--runs", str(runs), "--out", str(bundle_path)])
            bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
            bundle["notes"][0] = "Evidence bundles can include rescored traces from manual review."
            bundle_path.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            code = run_cli(["validate", "--evidence-bundle", str(bundle_path), "--out", str(summary_path)])

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in summary["targets"] for error in target["errors"])
            self.assertIn("evidence_bundle.notes must match the producer notes", errors)

    def test_validate_rejects_stale_bundle_action_fingerprint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = root / "runs"
            runs.mkdir()
            gate_path = root / "failed_gate.json"
            bundle_path = root / "evidence_bundle.json"
            summary_path = root / "validation.json"
            gate_path.write_text(
                json.dumps({"schema_version": "hfr.test_gate.v1", "passed": False}, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            run_cli(["evidence-bundle", "--runs", str(runs), "--gate", str(gate_path), "--out", str(bundle_path)])
            bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
            bundle["decision"]["next_actions"][0]["evidence"]["tampered"] = True
            bundle_path.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            code = run_cli(["validate", "--evidence-bundle", str(bundle_path), "--out", str(summary_path)])

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in summary["targets"] for error in target["errors"])
            self.assertIn("action_fingerprint does not match", errors)

    def test_validate_rejects_bundle_action_unknown_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = root / "runs"
            runs.mkdir()
            gate_path = root / "failed_gate.json"
            bundle_path = root / "evidence_bundle.json"
            summary_path = root / "validation.json"
            gate_path.write_text(
                json.dumps({"schema_version": "hfr.test_gate.v1", "passed": False}, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            run_cli(["evidence-bundle", "--runs", str(runs), "--gate", str(gate_path), "--out", str(bundle_path)])
            bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
            action = bundle["decision"]["next_actions"][0]
            action["artifact"] = "missing_artifact"
            action["action_fingerprint"] = bundle_action_fingerprint(action)
            action["routing_key"] = f"{action['artifact']}:{action['id']}:{action['action_fingerprint'][:12]}"
            bundle_path.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            code = run_cli(["validate", "--evidence-bundle", str(bundle_path), "--out", str(summary_path)])

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in summary["targets"] for error in target["errors"])
            self.assertIn("artifact must reference an evidence_bundle.artifacts key, metrics key, or evidence_bundle", errors)

    def test_validate_rejects_forged_bundle_next_actions_with_fresh_fingerprint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = root / "runs"
            runs.mkdir()
            gate_path = root / "failed_gate.json"
            bundle_path = root / "evidence_bundle.json"
            summary_path = root / "validation.json"
            gate_path.write_text(
                json.dumps({"schema_version": "hfr.test_gate.v1", "passed": False}, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            run_cli(["evidence-bundle", "--runs", str(runs), "--gate", str(gate_path), "--out", str(bundle_path)])
            bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
            action = bundle["decision"]["next_actions"][0]
            action["id"] = "defer_blocking_checks"
            action["summary"] = "Defer blocking checks until a later handoff."
            action["action_fingerprint"] = bundle_action_fingerprint(action)
            action["routing_key"] = f"{action['artifact']}:{action['id']}:{action['action_fingerprint'][:12]}"
            bundle_path.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            code = run_cli(["validate", "--evidence-bundle", str(bundle_path), "--out", str(summary_path)])

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in summary["targets"] for error in target["errors"])
            self.assertIn("decision.next_actions must match bundle blockers and metrics", errors)

    def test_validate_rejects_stale_bundle_decision_blocking_checks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = root / "runs"
            runs.mkdir()
            gate_path = root / "failed_gate.json"
            bundle_path = root / "evidence_bundle.json"
            summary_path = root / "validation.json"
            gate_path.write_text(
                json.dumps({"schema_version": "hfr.test_gate.v1", "passed": False}, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            run_cli(["evidence-bundle", "--runs", str(runs), "--gate", str(gate_path), "--out", str(bundle_path)])
            bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
            bundle["decision"]["blocking_checks"][0]["id"] = "stale_check_id"
            bundle_path.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            code = run_cli(["validate", "--evidence-bundle", str(bundle_path), "--out", str(summary_path)])

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in summary["targets"] for error in target["errors"])
            self.assertIn("decision.blocking_checks must match failed evidence_bundle.checks", errors)

    def test_validate_rejects_stale_bundle_decision_blocking_gates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = root / "runs"
            runs.mkdir()
            gate_path = root / "failed_gate.json"
            bundle_path = root / "evidence_bundle.json"
            summary_path = root / "validation.json"
            gate_path.write_text(
                json.dumps({"schema_version": "hfr.test_gate.v1", "passed": False}, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            run_cli(["evidence-bundle", "--runs", str(runs), "--gate", str(gate_path), "--out", str(bundle_path)])
            bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
            bundle["decision"]["blocking_gates"][0]["id"] = "stale_gate_id"
            bundle_path.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            code = run_cli(["validate", "--evidence-bundle", str(bundle_path), "--out", str(summary_path)])

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in summary["targets"] for error in target["errors"])
            self.assertIn("decision.blocking_gates must match failed evidence_bundle.metrics.gates", errors)

    def test_validate_rejects_stale_bundle_decision_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = root / "runs"
            runs.mkdir()
            gate_path = root / "failed_gate.json"
            bundle_path = root / "evidence_bundle.json"
            summary_path = root / "validation.json"
            gate_path.write_text(
                json.dumps({"schema_version": "hfr.test_gate.v1", "passed": False}, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            run_cli(["evidence-bundle", "--runs", str(runs), "--gate", str(gate_path), "--out", str(bundle_path)])
            bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
            bundle["decision"]["summary"] = "Evidence handoff is ready enough for a manual override."
            bundle_path.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            code = run_cli(["validate", "--evidence-bundle", str(bundle_path), "--out", str(summary_path)])

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in summary["targets"] for error in target["errors"])
            self.assertIn("decision.summary must match bundle readiness and failed checks", errors)

    def test_validate_rejects_stale_bundle_decision_key_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = root / "runs"
            runs.mkdir()
            gate_path = root / "failed_gate.json"
            bundle_path = root / "evidence_bundle.json"
            summary_path = root / "validation.json"
            gate_path.write_text(
                json.dumps({"schema_version": "hfr.test_gate.v1", "passed": False}, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            run_cli(["evidence-bundle", "--runs", str(runs), "--gate", str(gate_path), "--out", str(bundle_path)])
            bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
            bundle["decision"]["key_metrics"]["gates"]["failed"] = 0
            bundle_path.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            code = run_cli(["validate", "--evidence-bundle", str(bundle_path), "--out", str(summary_path)])

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in summary["targets"] for error in target["errors"])
            self.assertIn("decision.key_metrics must match evidence_bundle.metrics", errors)

    def test_strict_validate_rejects_absolute_bundle_artifact_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = root / "runs"
            runs.mkdir()
            bundle_path = root / "evidence_bundle.json"
            summary_path = root / "validation.json"
            strict_summary_path = root / "strict_validation.json"
            run_cli(["evidence-bundle", "--runs", str(runs), "--out", str(bundle_path)])
            bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
            bundle["artifacts"]["runs_dir"]["path"] = "/" + "Users/example/private-runs"
            bundle_path.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            code = run_cli(["validate", "--evidence-bundle", str(bundle_path), "--out", str(summary_path)])
            strict_code = run_cli(["validate", "--evidence-bundle", str(bundle_path), "--strict", "--out", str(strict_summary_path)])

            self.assertEqual(code, 0)
            self.assertEqual(strict_code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            warnings = "\n".join(warning for target in summary["targets"] for warning in target["warnings"])
            self.assertIn("evidence_bundle.artifacts.runs_dir.path is absolute", warnings)
            strict_summary = json.loads(strict_summary_path.read_text(encoding="utf-8"))
            strict_warnings = "\n".join(warning for target in strict_summary["targets"] for warning in target["warnings"])
            self.assertIn("evidence_bundle.artifacts.runs_dir.path is absolute", strict_warnings)

    def test_strict_validate_rejects_absolute_bundle_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = root / "runs"
            runs.mkdir()
            bundle_path = root / "evidence_bundle.json"
            summary_path = root / "validation.json"
            strict_summary_path = root / "strict_validation.json"
            run_cli(["evidence-bundle", "--runs", str(runs), "--out", str(bundle_path)])
            bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
            bundle["bundle_path"] = "/" + "Users/example/evidence_bundle.json"
            bundle_path.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            code = run_cli(["validate", "--evidence-bundle", str(bundle_path), "--out", str(summary_path)])
            strict_code = run_cli(["validate", "--evidence-bundle", str(bundle_path), "--strict", "--out", str(strict_summary_path)])

            self.assertEqual(code, 0)
            self.assertEqual(strict_code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            warnings = "\n".join(warning for target in summary["targets"] for warning in target["warnings"])
            self.assertIn("evidence_bundle.bundle_path is absolute", warnings)
            strict_summary = json.loads(strict_summary_path.read_text(encoding="utf-8"))
            strict_warnings = "\n".join(warning for target in strict_summary["targets"] for warning in target["warnings"])
            self.assertIn("evidence_bundle.bundle_path is absolute", strict_warnings)

    def test_strict_validate_rejects_nested_absolute_bundle_metric_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = root / "runs"
            runs.mkdir()
            harness = _write_harness_handoff_artifacts(root)
            trainer = _write_trainer_handoff_artifacts(root)
            bundle_path = root / "evidence_bundle.json"
            summary_path = root / "validation.json"
            strict_summary_path = root / "strict_validation.json"
            self.assertEqual(
                run_cli(
                    [
                        "evidence-bundle",
                        "--runs",
                        str(runs),
                        "--harness-manifest",
                        str(harness["manifest"]),
                        "--harness-result",
                        str(harness["result"]),
                        "--trainer-preflight",
                        str(trainer["trainer_preflight"]),
                        "--trainer-launch-check",
                        str(trainer["trainer_launch_check"]),
                        "--trainer-archive",
                        str(trainer["trainer_archive"]),
                        "--trainer-archive-check",
                        str(trainer["trainer_archive_check"]),
                        "--trainer-consumer-plan",
                        str(trainer["trainer_consumer_plan"]),
                        "--trainer-wrapper-dry-run",
                        str(trainer["trainer_wrapper_dry_run"]),
                        "--agentic-training-result",
                        str(trainer["agentic_training_result"]),
                        "--out",
                        str(bundle_path),
                    ]
                ),
                0,
            )
            absolute_root = "/" + "Users/example"
            bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
            bundle["metrics"]["run_digest_coverage"]["runs_dir"] = f"{absolute_root}/runs"
            bundle["metrics"]["harness_handoff"]["runs"][0]["manifest_path"] = f"{absolute_root}/harness_manifest.json"
            bundle["metrics"]["harness_handoff"]["runs"][0]["result_path"] = f"{absolute_root}/harness_result.json"
            bundle["metrics"]["harness_handoff"]["runs"][0]["suite_summary_path"] = f"{absolute_root}/suite_summary.json"
            bundle["metrics"]["harness_handoff"]["runs"][0]["replay_lineage"] = f"{absolute_root}/artifact_lineage.json"
            bundle["metrics"]["trainer_handoff"]["stages"][0]["path"] = f"{absolute_root}/trainer_preflight.json"
            bundle["metrics"]["live_smoke_summary"] = {
                "hermes_root": f"{absolute_root}/hermes-agent",
                "flight_recorder_root": f"{absolute_root}/hermes-flight-recorder",
            }
            gate_path_value = f"{absolute_root}/gate.json"
            blocking_gates = [{"id": "forged_gate", "path": gate_path_value}]
            bundle["metrics"]["gates"] = [{"id": "forged_gate", "path": gate_path_value, "passed": False}]
            bundle["decision"]["gate_count"] = 1
            bundle["decision"]["passed_gate_count"] = 0
            bundle["decision"]["blocking_gates"] = blocking_gates
            bundle["decision"]["key_metrics"]["gates"] = {"total": 1, "passed": 0, "failed": 1}
            blocking_checks = [
                {
                    "id": str(check.get("id") or "unknown"),
                    "summary": str(check.get("summary") or ""),
                    "scope": check.get("scope") if isinstance(check.get("scope"), dict) else {},
                }
                for check in bundle["checks"]
                if isinstance(check, dict) and check.get("passed") is False
            ]
            bundle["decision"]["next_actions"] = bundle_next_actions(blocking_checks, blocking_gates, bundle["metrics"])
            bundle["decision"]["next_action_count"] = len(bundle["decision"]["next_actions"])
            bundle_path.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            code = run_cli(["validate", "--evidence-bundle", str(bundle_path), "--out", str(summary_path)])
            strict_code = run_cli(["validate", "--evidence-bundle", str(bundle_path), "--strict", "--out", str(strict_summary_path)])

            self.assertEqual(code, 0)
            self.assertEqual(strict_code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            warnings = "\n".join(warning for target in summary["targets"] for warning in target["warnings"])
            self.assertIn("evidence_bundle.metrics.run_digest_coverage.runs_dir is absolute", warnings)
            self.assertIn("evidence_bundle.metrics.harness_handoff.runs[0].manifest_path is absolute", warnings)
            self.assertIn("evidence_bundle.metrics.harness_handoff.runs[0].result_path is absolute", warnings)
            self.assertIn("evidence_bundle.metrics.harness_handoff.runs[0].suite_summary_path is absolute", warnings)
            self.assertIn("evidence_bundle.metrics.harness_handoff.runs[0].replay_lineage is absolute", warnings)
            self.assertIn("evidence_bundle.metrics.trainer_handoff.stages[0].path is absolute", warnings)
            self.assertIn("evidence_bundle.metrics.live_smoke_summary.hermes_root is absolute", warnings)
            self.assertIn("evidence_bundle.metrics.live_smoke_summary.flight_recorder_root is absolute", warnings)
            self.assertIn("evidence_bundle.metrics.gates[0].path is absolute", warnings)
            self.assertIn("evidence_bundle.decision.blocking_gates[0].path is absolute", warnings)
            strict_summary = json.loads(strict_summary_path.read_text(encoding="utf-8"))
            strict_warnings = "\n".join(warning for target in strict_summary["targets"] for warning in target["warnings"])
            self.assertIn("evidence_bundle.metrics.harness_handoff.runs[0].manifest_path is absolute", strict_warnings)


def _write_harness_handoff_artifacts(
    root: Path,
    *,
    passed: bool = True,
    runner: str = "hermes_harness",
) -> dict[str, Path]:
    harness_dir = root / "harness_prompt_injection_good"
    harness_dir.mkdir()
    artifacts = {
        "manifest": harness_dir / "harness_manifest.json",
        "result": harness_dir / "harness_result.json",
    }
    manifest = {
        "schema_version": "hfr.harness_run_manifest.v1",
        "runner": runner,
        "provider": "mock",
        "model": {"id": "hfr-mock"},
        "scenario": {
            "id": "prompt_injection_good",
            "path": "scenarios/prompt_injection_good.json",
        },
        "outputs": {
            "run_dir": ".",
            "manifest": "harness_manifest.json",
            "result": "harness_result.json",
        },
        "sandbox": {
            "root": "sandbox",
            "home": "sandbox/home",
            "workspace": "sandbox/workspace",
            "events": "sandbox/events.jsonl",
            "fake_secret_canaries": [
                {
                    "name": "HFR_FAKE_SECRET_CANARY",
                    "sha256": "0" * 64,
                }
            ],
        },
        "tool_policy": {
            "source": "test_fixture",
            "scenario_policy": {"allowed_tools": ["read_file"]},
            "runtime_policy": {
                "mode": "mock",
                "allowed_tools": ["read_file"],
                "denied_tools": ["shell"],
                "network": {"mode": "disabled", "allowed_hosts": []},
            },
            "blocked_action_canaries": [
                {
                    "type": "secret",
                    "pattern": "HFR_FAKE_SECRET_CANARY",
                    "expected": "absent",
                }
            ],
        },
    }
    result = {
        "schema_version": "hfr.harness_run_result.v1",
        "runner": runner,
        "provider": "mock",
        "model": {"id": "hfr-mock"},
        "scenario_id": "prompt_injection_good",
        "sandbox": manifest["sandbox"],
        "tool_policy": manifest["tool_policy"],
        "trace": {
            "path": "normalized_trace.json",
            "sha256": None,
            "size_bytes": None,
            "format": "normalized_json",
        },
        "scorecard": {
            "path": "scorecard.json",
            "sha256": None,
            "size_bytes": None,
            "passed": passed,
            "score": 100 if passed else 35,
        },
        "artifacts": {
            "normalized_trace": "normalized_trace.json",
            "scorecard": "scorecard.json",
            "run_digest": "run_digest.json",
            "report": "report.html",
            "lineage": "artifact_lineage.json",
        },
        "replay": {
            "lineage": "artifact_lineage.json",
            "lineage_sha256": None,
            "lineage_size_bytes": None,
            "self_contained": True,
        },
    }
    for relative_path, content in {
        "normalized_trace.json": {"schema_version": "hfr.normalized_trace.v1", "events": []},
        "scorecard.json": {"schema_version": "hfr.scorecard.v1", "passed": passed, "score": result["scorecard"]["score"]},
        "run_digest.json": {"schema_version": "hfr.run_digest.v1", "scenario_id": "prompt_injection_good"},
        "artifact_lineage.json": {"schema_version": "hfr.artifact_lineage.v1", "inputs": []},
    }.items():
        (harness_dir / relative_path).write_text(json.dumps(content, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (harness_dir / "report.html").write_text("<!doctype html><title>report</title>\n", encoding="utf-8")
    for name in ("normalized_trace", "scorecard", "run_digest", "report", "lineage"):
        path = harness_dir / result["artifacts"][name]
        result["artifacts"][f"{name}_sha256"] = _sha256_file(path)
        result["artifacts"][f"{name}_size_bytes"] = path.stat().st_size
    result["trace"]["sha256"] = result["artifacts"]["normalized_trace_sha256"]
    result["trace"]["size_bytes"] = result["artifacts"]["normalized_trace_size_bytes"]
    result["scorecard"]["sha256"] = result["artifacts"]["scorecard_sha256"]
    result["scorecard"]["size_bytes"] = result["artifacts"]["scorecard_size_bytes"]
    result["replay"]["lineage_sha256"] = result["artifacts"]["lineage_sha256"]
    result["replay"]["lineage_size_bytes"] = result["artifacts"]["lineage_size_bytes"]
    artifacts["manifest"].write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    artifacts["result"].write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifacts


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_trainer_handoff_artifacts(root: Path) -> dict[str, Path]:
    artifacts = {
        "trainer_preflight": root / "trainer_preflight.json",
        "trainer_launch_check": root / "trainer_launch_check.json",
        "trainer_archive": root / "trainer_archive",
        "trainer_archive_check": root / "trainer_archive_check.json",
        "trainer_consumer_plan": root / "trainer_consumer_plan.json",
        "agentic_training_flow": root / "agentic_training_flow.json",
        "trainer_wrapper_dry_run": root / "trainer_wrapper_dry_run.json",
        "agentic_training_result": root / "agentic_training_result.json",
    }
    artifacts["trainer_archive"].mkdir()
    payloads = {
        artifacts["trainer_preflight"]: {
            "schema_version": "hfr.trainer_preflight.v1",
            "passed": True,
            "readiness": "ready",
            "recommendation": "launch_allowed",
            "check_count": 2,
            "failed_check_count": 0,
            "gate_count": 1,
            "passed_gate_count": 1,
        },
        artifacts["trainer_launch_check"]: {
            "schema_version": "hfr.trainer_launch_check.v1",
            "passed": True,
            "readiness": "ready",
            "recommendation": "launch_allowed",
            "check_count": 3,
            "failed_check_count": 0,
            "gate_count": 1,
            "passed_gate_count": 1,
        },
        artifacts["trainer_archive"] / "trainer_archive.json": {
            "schema_version": "hfr.trainer_archive.v1",
            "passed": True,
            "readiness": "ready",
            "recommendation": "handoff_ready",
            "metrics": {
                "artifact_count": 7,
                "missing_count": 0,
                "path_rewrite_count": 1,
                "trainer_input_count": 4,
            },
        },
        artifacts["trainer_archive_check"]: {
            "schema_version": "hfr.trainer_archive_check.v1",
            "passed": True,
            "readiness": "ready",
            "recommendation": "consumer_ready",
            "check_count": 4,
            "failed_check_count": 0,
            "metrics": {
                "external_code_file_count": 1,
                "missing_external_code_count": 0,
                "missing_trainer_input_count": 0,
                "trainer_input_available_count": 4,
                "trainer_input_count": 4,
            },
        },
        artifacts["trainer_consumer_plan"]: {
            "schema_version": "hfr.trainer_consumer_plan.v1",
            "passed": True,
            "readiness": "ready",
            "recommendation": "ready_for_external_trainer",
            "check_count": 5,
            "failed_check_count": 0,
            "metrics": {
                "command_arg_count": 3,
                "external_code_file_count": 1,
                "external_code_ready_count": 1,
                "trainer_input_count": 4,
                "trainer_input_ready_count": 4,
            },
        },
        artifacts["agentic_training_flow"]: {
            "schema_version": "hfr.agentic_training_flow.v1",
            "passed": True,
            "readiness": "ready",
            "recommendation": "ready_for_delegated_trainer_execution",
            "check_count": 7,
            "failed_check_count": 0,
            "metrics": {
                "command_arg_count": 3,
                "external_code_file_count": 1,
                "stage_count": 2,
                "trainer_input_count": 4,
            },
        },
        artifacts["trainer_wrapper_dry_run"]: {
            "schema_version": "hfr.example_trainer_wrapper_dry_run.v1",
            "passed": True,
            "readiness": "ready",
            "recommendation": "dry_run_ready",
            "check_count": 6,
            "failed_check_count": 0,
            "metrics": {
                "command_arg_count": 3,
                "external_code_file_count": 1,
                "external_code_ready_count": 1,
                "trainer_input_count": 4,
                "trainer_input_ready_count": 4,
            },
        },
        artifacts["agentic_training_result"]: {
            "schema_version": "hfr.agentic_training_result.v1",
            "passed": True,
            "readiness": "ready",
            "recommendation": "register_training_result",
            "check_count": 13,
            "failed_check_count": 0,
            "training_result": {
                "status": "completed",
                "runner_id": "external-trainer",
                "run_id": "example-completed-run",
            },
            "failure": {
                "class": "none",
                "message": "",
            },
            "metrics": {
                "artifact_count": 3,
                "regular_artifact_count": 3,
                "output_artifact_count": 1,
                "config_count": 1,
                "metrics_file_count": 1,
                "adapter_count": 1,
                "checkpoint_count": 0,
                "log_count": 0,
                "failure_report_count": 0,
            },
            "registry_update": {
                "ready_to_apply": True,
            },
        },
    }
    for path, payload in payloads.items():
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifacts


def _write_serving_lifecycle(root: Path) -> Path:
    preflight = root / "preflight"
    preflight.mkdir()
    for filename in ("serving_profile.json", "compatibility_report.json", "serving_check.json"):
        (preflight / filename).write_text("{}\n", encoding="utf-8")
    lifecycle = {
        "schema_version": "hfr.serving_lifecycle.v1",
        "generated_at": "2026-01-01T00:00:00Z",
        "finished_at": "2026-01-01T00:00:01Z",
        "duration_ms": 1000,
        "profile": "mock",
        "engine": "mock",
        "arm": "candidate",
        "provider": "custom",
        "model": "hfr-managed-mock",
        "served_model_name": "hfr-managed-mock",
        "adapter": "",
        "adapter_strategy": {"present": False, "resolved_strategy": "none"},
        "endpoint": {"base_url": "http://127.0.0.1:18080/v1", "host": "127.0.0.1", "port": 18080},
        "launch": {
            "command": ["python3", "scripts/mock_openai_server.py"],
            "command_display": "python3 scripts/mock_openai_server.py",
            "cwd": ".",
            "env_keys": [],
            "startup_timeout_s": 1.0,
            "poll_interval_s": 0.1,
            "grace_period_s": 1.0,
        },
        "process": {"started": True, "pid": 12345, "exit_code": 0},
        "environment": {"python_version": "3.11.0", "platform": "test"},
        "artifacts_root": ".",
        "passed": True,
        "ready": True,
        "readiness": "ready",
        "readiness_probe": {
            "ready": True,
            "summary": "readiness_endpoint_passed",
            "url": "http://127.0.0.1:18080/v1/models",
            "attempts": [{"ok": True, "status_code": 200}],
        },
        "smoke_check": {
            "attempted": True,
            "passed": True,
            "readiness": "ready",
            "failed_checks": [],
            "artifacts": {
                "serving_profile": "preflight/serving_profile.json",
                "compatibility_report": "preflight/compatibility_report.json",
                "serving_check": "preflight/serving_check.json",
            },
        },
        "teardown": {
            "attempted": True,
            "already_exited": False,
            "terminated": True,
            "killed": False,
            "clean": True,
            "running_after_teardown": False,
        },
        "errors": [],
        "artifacts": {
            "serving_lifecycle": "serving_lifecycle.json",
            "stdout_log": "server.stdout.log",
            "stderr_log": "server.stderr.log",
            "serving_profile": "preflight/serving_profile.json",
            "compatibility_report": "preflight/compatibility_report.json",
            "serving_check": "preflight/serving_check.json",
        },
        "logs": {"stdout_tail": "", "stderr_tail": ""},
    }
    lifecycle_path = root / "serving_lifecycle.json"
    lifecycle_path.write_text(json.dumps(lifecycle, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return lifecycle_path


def _write_eval_suite_summary(path: Path) -> Path:
    payload = {
        "schema_version": "hfr.run_suite.v1",
        "total": 1,
        "passed": 1,
        "failed": 0,
        "error_count": 0,
        "errors": [],
        "metrics": {
            "pass_rate": 1.0,
            "average_score": 100.0,
            "failed_rule_counts": [],
            "critical_failure_counts": [],
        },
        "runs": [
            {
                "scenario_id": "email_reply_completion",
                "task_family": "email_reply_completion",
                "passed": True,
                "score": 100,
                "failed_rules": [],
                "critical_failures": [],
            }
        ],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


if __name__ == "__main__":
    unittest.main()
