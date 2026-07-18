import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from flightrecorder.cli import main
from flightrecorder.hermes_plugin import LIVE_SMOKE_SUMMARY_SCHEMA_VERSION
from scripts.live_coven_smoke import _write_smoke_artifacts as _write_coven_smoke_artifacts


ROOT = Path(__file__).resolve().parents[1]


def run_cli(args):
    with redirect_stdout(StringIO()):
        return main(args)


class EvidenceBundleTests(unittest.TestCase):
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
                            "hermes_root": "/tmp/hermes-agent",
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
            harness_dir = root / "harness_coven"
            harness_dir.mkdir()
            harness_trace = harness_dir / "live_coven.coven.jsonl"
            harness_trace.write_text(
                (ROOT / "fixtures" / "coven_detached_good.coven.jsonl").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            _write_coven_smoke_artifacts(harness_trace, harness_dir)

            code = run_cli(
                [
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
                    str(harness_dir / "harness_manifest.json"),
                    "--harness-result",
                    str(harness_dir / "harness_result.json"),
                    "--gate",
                    str(runs / "suite_gate.json"),
                    "--require-harness",
                    "--require-gate",
                    "--out",
                    str(bundle_path),
                ]
            )

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
            self.assertEqual(bundle["metrics"]["scenario_quality"]["average_contract_score"], 93.57)
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
            self.assertEqual(bundle["metrics"]["live_smoke_summary"]["hermes_root"], "/tmp/hermes-agent")
            self.assertEqual(bundle["metrics"]["live_smoke_summary"]["flight_recorder_git_dirty"], True)
            self.assertEqual(bundle["metrics"]["harness_handoff"]["pair_count"], 1)
            self.assertEqual(bundle["metrics"]["harness_handoff"]["schema_valid_pair_count"], 1)
            self.assertEqual(bundle["metrics"]["harness_handoff"]["consistent_pair_count"], 1)
            self.assertEqual(bundle["metrics"]["harness_handoff"]["runs"][0]["runner"], "coven_live_smoke")
            self.assertEqual(bundle["metrics"]["harness_handoff"]["runs"][0]["provider"], "coven")
            self.assertEqual(bundle["metrics"]["harness_handoff"]["runs"][0]["trace_format"], "coven_jsonl")
            self.assertEqual(bundle["metrics"]["gates"][0]["id"], "suite_gate")
            self.assertTrue(bundle["metrics"]["gates"][0]["passed"])
            self.assertTrue(bundle["metrics"]["gates"][0]["contract"]["valid"])
            self.assertEqual(bundle["metrics"]["gates"][0]["contract"]["readiness"], "ready")
            self.assertEqual(bundle["metrics"]["gates"][0]["contract"]["recommendation"], "promote_iteration")
            self.assertEqual(bundle["artifacts"]["suite_summary"]["kind"], "file")
            self.assertEqual(bundle["artifacts"]["live_smoke_summary"]["kind"], "file")
            self.assertEqual(bundle["artifacts"]["harness_manifest_1"]["kind"], "file")
            self.assertEqual(bundle["artifacts"]["harness_result_1"]["kind"], "file")
            self.assertEqual(len(bundle["artifacts"]["live_smoke_summary"]["sha256"]), 64)
            self.assertEqual(bundle["artifacts"]["training_export_curriculum"]["kind"], "file")
            self.assertEqual(bundle["artifacts"]["training_export_curriculum"]["exists"], True)
            self.assertEqual(len(bundle["artifacts"]["training_export_curriculum"]["sha256"]), 64)
            self.assertEqual(len(bundle["artifacts"]["suite_summary"]["sha256"]), 64)

            self.assertEqual(run_cli(["validate", "--evidence-bundle", str(bundle_path), "--strict"]), 0)
            bundle["metrics"]["training_export"]["top_curriculum_priorities"][0]["priority_band"] = "urgent"
            bundle_path.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            self.assertEqual(run_cli(["validate", "--evidence-bundle", str(bundle_path)]), 1)

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
            harness_dir = root / "harness_coven"
            harness_dir.mkdir()
            harness_trace = harness_dir / "live_coven.coven.jsonl"
            harness_trace.write_text(
                (ROOT / "fixtures" / "coven_detached_good.coven.jsonl").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            _write_coven_smoke_artifacts(harness_trace, harness_dir)
            result_path = harness_dir / "harness_result.json"
            result = json.loads(result_path.read_text(encoding="utf-8"))
            result["scorecard"]["passed"] = False
            result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            bundle_path = root / "evidence_bundle.json"

            code = run_cli(
                [
                    "evidence-bundle",
                    "--harness-manifest",
                    str(harness_dir / "harness_manifest.json"),
                    "--harness-result",
                    str(result_path),
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
                    "--trainer-wrapper-dry-run",
                    str(artifacts["trainer_wrapper_dry_run"]),
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
            self.assertEqual(trainer["stage_count"], 6)
            self.assertEqual(trainer["handoff_ready_count"], 6)
            self.assertEqual(trainer["blocked_stage_count"], 0)
            self.assertEqual(trainer["schema_supported_count"], 6)
            self.assertTrue(trainer["complete_chain"])
            self.assertTrue(trainer["all_included_ready"])
            self.assertEqual(trainer["missing_stage_ids"], [])
            self.assertEqual([stage["id"] for stage in trainer["stages"]], [
                "trainer_preflight",
                "trainer_launch_check",
                "trainer_archive",
                "trainer_archive_check",
                "trainer_consumer_plan",
                "trainer_wrapper_dry_run",
            ])
            self.assertEqual(bundle["artifacts"]["trainer_archive"]["kind"], "directory")
            self.assertEqual(bundle["artifacts"]["trainer_archive_manifest"]["kind"], "file")
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
                    "trainer_wrapper_dry_run",
                ],
            )
            failed_checks = {check["id"] for check in bundle["checks"] if not check["passed"]}
            self.assertIn("trainer_preflight_ready", failed_checks)
            action_ids = {action["id"] for action in bundle["decision"]["next_actions"]}
            self.assertIn("fix_trainer_handoff", action_ids)
            self.assertIn("complete_trainer_handoff_chain", action_ids)
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
                json.dumps({"schema_version": "hfr.weak_gate.v1", "passed": True, "checks": []}, sort_keys=True) + "\n",
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
            failed_checks = {check["id"] for check in bundle["checks"] if not check["passed"]}
            self.assertIn("gate_contract_valid", failed_checks)
            gate_metrics = bundle["metrics"]["gates"][0]
            self.assertTrue(gate_metrics["passed"])
            self.assertFalse(gate_metrics["contract"]["valid"])
            self.assertIn("decision must be an object", gate_metrics["contract"]["errors"])
            self.assertEqual(bundle["decision"]["recommendation"], "block_handoff")
            self.assertEqual(run_cli(["validate", "--evidence-bundle", str(bundle_path), "--strict"]), 0)

    def test_evidence_bundle_blocks_passed_gate_with_failed_checks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = root / "runs"
            runs.mkdir()
            gate_path = root / "contradictory_gate.json"
            bundle_path = root / "evidence_bundle.json"
            failed_check = {"id": "minimum_score", "passed": False, "summary": "score too low", "scope": {}}
            gate_path.write_text(
                json.dumps(
                    {
                        "schema_version": "hfr.weak_gate.v1",
                        "passed": True,
                        "check_count": 1,
                        "failed_check_count": 1,
                        "checks": [failed_check],
                        "decision": {
                            "readiness": "ready",
                            "recommendation": "promote_iteration",
                            "summary": "forged ready decision",
                            "blocking_check_count": 1,
                            "blocking_checks": [
                                {"id": "minimum_score", "summary": "score too low", "scope": {}}
                            ],
                            "failed_checks": [
                                {"id": "minimum_score", "summary": "score too low", "scope": {}}
                            ],
                            "next_action_count": 1,
                            "next_actions": [
                                {
                                    "id": "resolve_failed_checks",
                                    "priority": "critical",
                                    "artifact": "weak_gate",
                                    "summary": "resolve forged failure",
                                    "evidence": {"failed_check_count": 1},
                                }
                            ],
                            "key_metrics": {},
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
            failed_checks = {check["id"] for check in bundle["checks"] if not check["passed"]}
            self.assertIn("gate_contract_valid", failed_checks)
            gate_metrics = bundle["metrics"]["gates"][0]
            self.assertTrue(gate_metrics["passed"])
            self.assertFalse(gate_metrics["contract"]["valid"])
            self.assertIn("passed must be false when checks fail", gate_metrics["contract"]["errors"])
            self.assertIn("decision.readiness must be blocked when checks fail", gate_metrics["contract"]["errors"])
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
                            "check_count": 0,
                            "failed_check_count": 0,
                            "checks": [],
                            "decision": {
                                "readiness": "ready",
                                "recommendation": "promote_iteration",
                                "summary": "Fixture gate is ready.",
                                "blocking_check_count": 0,
                                "blocking_checks": [],
                                "failed_checks": [],
                                "next_action_count": 0,
                                "next_actions": [],
                                "key_metrics": {},
                            },
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
                gate_metrics = bundle["metrics"]["gates"][0]
                self.assertEqual(gate_metrics["schema_version"], schema_version)
                self.assertTrue(gate_metrics["passed"])
                self.assertTrue(gate_metrics["contract"]["valid"])
                self.assertFalse(gate_metrics["validation"]["available"])
                self.assertFalse(gate_metrics["validation"]["passed"])
                blocking_checks = {check["id"] for check in bundle["decision"]["blocking_checks"]}
                self.assertIn("gate_validation_passed", blocking_checks)
                self.assertEqual(run_cli(["validate", "--evidence-bundle", str(bundle_path), "--strict"]), 0)

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


def _write_trainer_handoff_artifacts(root: Path) -> dict[str, Path]:
    artifacts = {
        "trainer_preflight": root / "trainer_preflight.json",
        "trainer_launch_check": root / "trainer_launch_check.json",
        "trainer_archive": root / "trainer_archive",
        "trainer_archive_check": root / "trainer_archive_check.json",
        "trainer_consumer_plan": root / "trainer_consumer_plan.json",
        "trainer_wrapper_dry_run": root / "trainer_wrapper_dry_run.json",
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
    }
    for path, payload in payloads.items():
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifacts


if __name__ == "__main__":
    unittest.main()
