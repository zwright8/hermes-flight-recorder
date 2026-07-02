import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

from flightrecorder.cli import main
from flightrecorder.schema_registry import (
    check_schema_contract,
    check_schema_file,
    check_schema_jsonl_file,
    list_schema_records,
    load_schema,
    write_schema_bundle,
)


ROOT = Path(__file__).resolve().parents[1]


class SchemaRegistryTests(unittest.TestCase):
    def test_catalog_loads_public_artifact_contracts(self):
        records = list_schema_records()
        names = {record["name"] for record in records}

        self.assertIn("scenario", names)
        self.assertIn("trace", names)
        self.assertIn("scorecard", names)
        self.assertIn("task_completion", names)
        self.assertIn("state_diff", names)
        self.assertIn("state_snapshot", names)
        self.assertIn("verifier_config", names)
        self.assertIn("state_validator_config", names)
        self.assertIn("state_validator_assertions", names)
        self.assertIn("supervisor_state", names)
        self.assertIn("validation", names)
        self.assertIn("run_suite", names)
        self.assertIn("suite_gate", names)
        self.assertIn("suite_compare", names)
        self.assertIn("suite_trend", names)
        self.assertIn("evidence_coverage", names)
        self.assertIn("scenario_quality", names)
        self.assertIn("trace_observability", names)
        self.assertIn("repair_queue", names)
        self.assertIn("run_digest", names)
        self.assertIn("harness_run_manifest", names)
        self.assertIn("harness_run_result", names)
        self.assertIn("harness_replay_result", names)
        self.assertIn("harness_suite_result", names)
        self.assertIn("harness_model_probe", names)
        self.assertIn("live_smoke_summary", names)
        self.assertIn("live_verifier_smoke_summary", names)
        self.assertIn("openclaw_event", names)
        self.assertIn("live_openclaw_smoke_summary", names)
        self.assertIn("harness_run_manifest", names)
        self.assertIn("harness_run_result", names)
        self.assertIn("serving_profile", names)
        self.assertIn("serving_compatibility_report", names)
        self.assertIn("serving_endpoint_check", names)
        self.assertIn("serving_lifecycle", names)
        self.assertIn("serving_demo_run", names)
        self.assertIn("evidence_bundle", names)
        self.assertIn("improvement_plan", names)
        self.assertIn("improvement_ledger", names)
        self.assertIn("improvement_ledger_gate", names)
        self.assertIn("training_manifest", names)
        self.assertIn("rl_episode", names)
        self.assertIn("rl_reward", names)
        self.assertIn("rl_step_reward", names)
        self.assertIn("rl_preference", names)
        self.assertIn("rl_failure_mode", names)
        self.assertIn("rl_curriculum", names)
        self.assertIn("rl_sft", names)
        self.assertIn("rl_dpo", names)
        self.assertIn("rl_reward_model", names)
        self.assertIn("rl_dataset_metrics", names)
        self.assertIn("dataset_splits", names)
        self.assertIn("dataset_registry", names)
        self.assertIn("compare_rl_manifest", names)
        self.assertIn("compare_rl_pair", names)
        self.assertIn("compare_rl_dpo", names)
        self.assertIn("compare_gate", names)
        self.assertIn("training_gate", names)
        self.assertIn("heldout_manifest", names)
        self.assertIn("eval_suite_manifest", names)
        self.assertIn("external_eval_plan", names)
        self.assertIn("eval_summary", names)
        self.assertIn("review_manifest", names)
        self.assertIn("reviewed_manifest", names)
        self.assertIn("trainer_preflight", names)
        self.assertIn("trainer_launch_check", names)
        self.assertIn("trainer_archive", names)
        self.assertIn("agentic_training_plan", names)
        self.assertIn("agentic_training_runtime_preflight", names)
        self.assertIn("agentic_training_result", names)
        self.assertIn("model_candidate", names)
        self.assertIn("model_scout_manifest", names)
        self.assertIn("model_compatibility_report", names)
        self.assertIn("model_serving_probe_receipt", names)
        self.assertIn("model_adapter_manifest", names)
        self.assertIn("model_registry_entry", names)
        self.assertIn("model_registry", names)
        self.assertIn("promotion_alias_apply", names)
        self.assertIn("promotion_cards", names)
        self.assertIn("promotion_decision", names)
        self.assertIn("promotion_ledger", names)
        self.assertIn("promotion_ledger_gate", names)
        self.assertIn("promotion_policy", names)
        self.assertIn("promotion_release_record", names)
        self.assertIn("promotion_rollback_receipt", names)
        self.assertIn("training_plan", names)
        for record in records:
            schema = load_schema(record["name"])
            self.assertEqual(schema["$id"], record["id"])
            self.assertEqual(schema["$schema"], "https://json-schema.org/draft/2020-12/schema")
            self.assertEqual(schema["type"], "object")

    def test_load_schema_accepts_version_and_filename(self):
        by_name = load_schema("trace")
        by_version = load_schema("hfr.trace.v1")
        by_filename = load_schema("trace.v1.schema.json")

        self.assertEqual(by_name, by_version)
        self.assertEqual(by_name, by_filename)
        self.assertEqual(by_name["properties"]["schema_version"]["const"], "hfr.trace.v1")

    def test_promotion_policy_schema_accepts_demo_policy(self):
        result = check_schema_file(ROOT / "examples" / "promotion_policy.demo.json", "promotion_policy")

        self.assertTrue(result["passed"], result["errors"])
        self.assertEqual(result["schema"]["name"], "promotion_policy")

    def test_schema_check_passes_real_scenario_and_minimal_trace(self):
        scenario_result = check_schema_file(ROOT / "scenarios" / "prompt_injection_good.json", "scenario")
        self.assertTrue(scenario_result["passed"], scenario_result["errors"])

        trace = {
            "schema_version": "hfr.trace.v1",
            "session": {"id": "session-1", "source_format": "observer_jsonl", "model": "unknown"},
            "events": [{"type": "assistant_message", "session_id": "session-1", "text": "ok"}],
            "final_answer": "ok",
        }
        trace_result = check_schema_contract(trace)
        self.assertTrue(trace_result["passed"], trace_result["errors"])
        self.assertEqual(trace_result["schema"]["name"], "trace")

    def test_schema_check_reports_contract_errors(self):
        result = check_schema_contract(
            {
                "schema_version": "hfr.trace.v1",
                "session": {"id": "session-1", "source_format": "observer_jsonl"},
                "events": [{"session_id": "session-1"}],
            }
        )

        self.assertFalse(result["passed"])
        self.assertGreaterEqual(result["error_count"], 3)
        errors = "\n".join(result["errors"])
        self.assertIn("missing required property 'final_answer'", errors)
        self.assertIn("$.session: missing required property 'model'", errors)
        self.assertIn("$.events[0]: missing required property 'type'", errors)

    def test_schema_check_enforces_any_of_contracts(self):
        valid = check_schema_contract(
            {
                "kind": "input",
                "session_id": "coven-session-1",
                "payload_json": "{\"data\":\"hello\"}",
            },
            name_or_id="coven_event",
        )
        self.assertTrue(valid["passed"], valid["errors"])

        invalid = check_schema_contract(
            {
                "kind": "input",
                "session_id": "coven-session-1",
            },
            name_or_id="coven_event",
        )
        self.assertFalse(invalid["passed"])
        self.assertIn("expected exactly one matching schema from oneOf, got 0", "\n".join(invalid["errors"]))

    def test_task_completion_schema_accepts_not_applicable_status(self):
        result = check_schema_contract(
            {
                "schema_version": "hfr.task_completion.v1",
                "status": "not_applicable",
                "passed": True,
                "task_evidence_configured": False,
                "required_check_count": 0,
                "passed_check_count": 0,
                "failed_check_count": 0,
                "blocking_rule_ids": [],
                "checks": [],
                "summary": "No task-completion evidence assertions were configured.",
            }
        )

        self.assertTrue(result["passed"], result["errors"])
        self.assertEqual(result["schema"]["name"], "task_completion")

    def test_model_candidate_schema_enforces_license_and_compatibility_shape(self):
        candidate_path = ROOT / "experiments" / "registry" / "model_candidates" / "local_mock_tiny_chat.json"
        valid = check_schema_file(candidate_path, "model_candidate")
        self.assertTrue(valid["passed"], valid["errors"])

        candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
        del candidate["source"]["url"]
        del candidate["license"]["training_allowed"]
        candidate["compatibility"]["context_length"] = 0
        candidate["compatibility"]["tool_calls"]["verified"] = "false"

        invalid = check_schema_contract(candidate)
        self.assertFalse(invalid["passed"])
        errors = "\n".join(invalid["errors"])
        self.assertIn("$.source: missing required property 'url'", errors)
        self.assertIn("$.license: missing required property 'training_allowed'", errors)
        self.assertIn("$.compatibility.context_length: expected value >= 1", errors)
        self.assertIn("$.compatibility.tool_calls.verified: expected type boolean", errors)

    def test_supervisor_state_schema_accepts_minimal_checkpoint(self):
        result = check_schema_contract(
            {
                "schema_version": "hfr.autonomy.supervisor_state.v1",
                "updated_at": "2026-07-02T00:00:00Z",
                "active_layer": "evidence",
                "current_packet": "publish supervisor-state schema",
                "completed_packets": [
                    {
                        "id": "goal-0-supervisor-state-schema",
                        "layer": "evidence",
                        "title": "Publish supervisor-state schema",
                        "status": "complete",
                    }
                ],
                "blocked_packets": [],
                "next_packets": [
                    {
                        "id": "goal-1-evidence-handoff",
                        "layer": "evidence",
                        "title": "Refresh evidence handoff fixture",
                        "status": "pending",
                    }
                ],
                "latest_artifacts": [
                    {
                        "path": "flightrecorder/schemas/supervisor_state.v1.schema.json",
                        "type": "schema",
                        "sha256": "0" * 64,
                    }
                ],
                "latest_verification": [
                    {
                        "command": "python -m unittest tests.test_schema_registry.SchemaRegistryTests.test_supervisor_state_schema_accepts_minimal_checkpoint",
                        "passed": True,
                        "exit_code": 0,
                    }
                ],
                "privacy_guardrails": {
                    "public_repo": True,
                    "local_state_untracked": True,
                    "daily_email_suppressed": True,
                    "private_fields_forbidden": ["home_directory", "daily_report_recipient"],
                },
                "promotion_readiness": {
                    "evidence": "in_progress",
                    "harness": "not_started",
                    "data": "not_started",
                    "model": "not_started",
                    "training": "not_started",
                    "serving_demo": "not_started",
                    "eval": "not_started",
                    "governance": "not_started",
                },
            }
        )

        self.assertTrue(result["passed"], result["errors"])
        self.assertEqual(result["schema"]["name"], "supervisor_state")

    def test_supervisor_state_schema_requires_all_layer_readiness(self):
        result = check_schema_contract(
            {
                "schema_version": "hfr.autonomy.supervisor_state.v1",
                "updated_at": "2026-07-02T00:00:00Z",
                "active_layer": "evidence",
                "current_packet": "publish supervisor-state schema",
                "completed_packets": [],
                "blocked_packets": [],
                "next_packets": [],
                "latest_artifacts": [],
                "latest_verification": [],
                "promotion_readiness": {"evidence": "in_progress"},
            }
        )

        self.assertFalse(result["passed"])
        errors = "\n".join(result["errors"])
        self.assertIn("$.promotion_readiness: missing required property 'harness'", errors)
        self.assertIn("$.promotion_readiness: missing required property 'governance'", errors)

    def test_run_suite_schema_accepts_minimal_summary(self):
        result = check_schema_contract(
            {
                "schema_version": "hfr.run_suite.v1",
                "scenarios_dir": "scenarios",
                "out_dir": "runs",
                "total": 1,
                "passed": 1,
                "failed": 0,
                "error_count": 0,
                "errors": [],
                "metrics": {
                    "pass_rate": 1.0,
                    "average_score": 100.0,
                    "min_score": 100,
                    "max_score": 100,
                    "failed_rule_counts": [],
                    "critical_failure_counts": [],
                    "task_families": [
                        {
                            "task_family": "email_reply_completion",
                            "total": 1,
                            "passed": 1,
                            "failed": 0,
                            "pass_rate": 1.0,
                            "average_score": 100.0,
                            "failed_rule_counts": [],
                            "critical_failure_counts": [],
                        }
                    ],
                    "failed": 0,
                    "passed": 1,
                },
                "runs": [
                    {
                        "scenario_id": "email_reply_completion_good",
                        "scenario_title": "Email Reply Completion",
                        "task_family": "email_reply_completion",
                        "scenario_path": "scenarios/email_reply_completion_good.json",
                        "scenario_sha256": "0" * 64,
                        "trace_path": "fixtures/email_reply_completion_good.observer.jsonl",
                        "trace_sha256": "1" * 64,
                        "before_state_path": "fixtures/email_before.json",
                        "before_state_sha256": "2" * 64,
                        "state_path": "fixtures/email_after.json",
                        "state_sha256": "3" * 64,
                        "run_dir": "runs/email_reply_completion_good",
                        "report": "runs/email_reply_completion_good/report.html",
                        "scorecard": "runs/email_reply_completion_good/scorecard.json",
                        "run_digest": "runs/email_reply_completion_good/run_digest.json",
                        "lineage": "runs/email_reply_completion_good/artifact_lineage.json",
                        "passed": True,
                        "score": 100,
                        "failed_rules": [],
                        "critical_failures": [],
                    }
                ],
                "artifacts": {"index": "runs/index.html"},
                "metadata": {"candidate": "offline-demo"},
                "training_export": {"episode_count": 1, "reward_count": 1},
                "validation": {"passed": True, "target_count": 1, "error_count": 0, "warning_count": 0},
            }
        )

        self.assertTrue(result["passed"], result["errors"])
        self.assertEqual(result["schema"]["name"], "run_suite")

    def test_suite_trend_schema_accepts_minimal_trend(self):
        result = check_schema_contract(
            {
                "schema_version": "hfr.suite_trend.v1",
                "point_count": 2,
                "points": [
                    {
                        "index": 0,
                        "label": "baseline",
                        "path": "runs/baseline/suite_summary.json",
                        "metadata": {"candidate": "baseline"},
                        "total": 2,
                        "passed": 1,
                        "failed": 1,
                        "error_count": 0,
                        "pass_rate": 0.5,
                        "average_score": 50.0,
                        "failed_rule_counts": {"secret_exposure": 1},
                        "critical_failure_counts": {"secret_exposure": 1},
                        "failed_rule_count": 1,
                        "critical_failure_count": 1,
                        "delta_from_previous": None,
                    },
                    {
                        "index": 1,
                        "label": "candidate",
                        "path": "runs/candidate/suite_summary.json",
                        "metadata": {"candidate": "candidate"},
                        "total": 2,
                        "passed": 2,
                        "failed": 0,
                        "error_count": 0,
                        "pass_rate": 1.0,
                        "average_score": 85.0,
                        "failed_rule_counts": {"secret_exposure": 0, "required_evidence": 2},
                        "critical_failure_counts": {"required_evidence": 1},
                        "failed_rule_count": 2,
                        "critical_failure_count": 1,
                        "delta_from_previous": {
                            "pass_rate_delta": 0.5,
                            "average_score_delta": 35.0,
                            "failed_rule_count_delta": 1,
                            "critical_failure_count_delta": 0,
                        },
                    },
                ],
                "failed_rule_trends": [
                    {
                        "id": "secret_exposure",
                        "first_count": 1,
                        "last_count": 0,
                        "delta": -1,
                        "counts": [
                            {"index": 0, "label": "baseline", "count": 1},
                            {"index": 1, "label": "candidate", "count": 0},
                        ],
                    }
                ],
                "critical_failure_trends": [
                    {
                        "id": "required_evidence",
                        "first_count": 0,
                        "last_count": 1,
                        "delta": 1,
                        "counts": [
                            {"index": 0, "label": "baseline", "count": 0},
                            {"index": 1, "label": "candidate", "count": 1},
                        ],
                    }
                ],
                "summary": "TREND: 2 points; pass_rate_delta=0.5; average_score_delta=35.0; failed_rule_delta=1; critical_failure_delta=0.",
            }
        )

        self.assertTrue(result["passed"], result["errors"])
        self.assertEqual(result["schema"]["name"], "suite_trend")

    def test_suite_compare_schema_accepts_minimal_comparison(self):
        contract_fingerprints = {
            "baseline": {
                "scenario": {"path": "scenarios/prompt_injection_good.json", "sha256": "0" * 64},
                "source_trace": {"path": "fixtures/prompt_injection_good.jsonl", "sha256": "1" * 64},
                "source_state_snapshot": {"path": None, "sha256": None},
            },
            "candidate": {
                "scenario": {"path": "scenarios/prompt_injection_good.json", "sha256": "0" * 64},
                "source_trace": {"path": "fixtures/prompt_injection_good.jsonl", "sha256": "1" * 64},
                "source_state_snapshot": {"path": None, "sha256": None},
            },
        }
        result = check_schema_contract(
            {
                "schema_version": "hfr.suite_compare.v1",
                "baseline": {
                    "label": "baseline",
                    "path": "runs/baseline",
                    "scenario_count": 1,
                    "metadata": {"candidate": "baseline"},
                },
                "candidate": {
                    "label": "candidate",
                    "path": "runs/candidate",
                    "scenario_count": 1,
                    "metadata": {"candidate": "candidate"},
                },
                "aggregate": {
                    "baseline_count": 1,
                    "candidate_count": 1,
                    "paired_count": 1,
                    "baseline_avg_score": 100.0,
                    "candidate_avg_score": 100.0,
                    "avg_score_delta": 0.0,
                    "total_score_delta": 0,
                    "baseline_pass_rate": 1.0,
                    "candidate_pass_rate": 1.0,
                    "failed_rule_deltas": [],
                    "critical_failure_deltas": [],
                    "contract_drift_count": 0,
                    "unverified_contract_count": 0,
                },
                "contract_scope": "scenario",
                "scenario_changes": [
                    {
                        "scenario_id": "prompt_injection_good",
                        "status": "unchanged",
                        "baseline_run": "prompt_injection_good",
                        "candidate_run": "prompt_injection_good",
                        "baseline_score": 100,
                        "candidate_score": 100,
                        "score_delta": 0,
                        "baseline_passed": True,
                        "candidate_passed": True,
                        "rule_regressions": [],
                        "rule_fixes": [],
                        "new_critical_failures": [],
                        "regressed": False,
                        "contract_fingerprint_status": "matched",
                        "contract_fingerprint_scope": "scenario",
                        "contract_fingerprint_reasons": [],
                        "contract_fingerprints": contract_fingerprints,
                        "summary": "NO REGRESSION: score delta 0; rule outcomes unchanged.",
                    }
                ],
                "regressions": [],
                "improvements": [],
                "missing_in_candidate": [],
                "new_in_candidate": [],
                "contract_drifts": [],
                "unverified_contracts": [],
                "regressed": False,
                "summary": "NO REGRESSION: avg score delta 0.0; paired scenarios unchanged.",
            }
        )

        self.assertTrue(result["passed"], result["errors"])
        self.assertEqual(result["schema"]["name"], "suite_compare")

    def test_compare_gate_schema_accepts_minimal_gate(self):
        metrics = {
            "pair_count": 1,
            "dpo_count": 1,
            "candidate_win_count": 1,
            "candidate_win_scenarios": ["email_reply_completion"],
            "baseline_win_count": 0,
            "baseline_win_scenarios": [],
            "skipped_pair_count": 0,
            "contract_drift_count": 0,
            "unverified_contract_count": 0,
            "task_completion_improvement_count": 1,
            "task_completion_improvement_scenarios": ["email_reply_completion"],
            "task_completion_regression_count": 0,
            "task_completion_regression_scenarios": [],
            "task_families": [],
            "validation": {
                "available": True,
                "passed": True,
                "strict": True,
                "target_count": 1,
                "error_count": 0,
                "warning_count": 0,
            },
        }
        result = check_schema_contract(
            {
                "schema_version": "hfr.compare_gate.v1",
                "compare_export": "runs/compare_rl_export",
                "passed": True,
                "check_count": 1,
                "failed_check_count": 0,
                "checks": [
                    {
                        "id": "min_pairs",
                        "passed": True,
                        "actual": 1,
                        "expected": {"min": 1},
                        "summary": "min_pairs: actual=1, min=1",
                    }
                ],
                "metrics": metrics,
                "decision": {
                    "readiness": "ready",
                    "recommendation": "promote_iteration",
                    "summary": "Gate is ready: all checks passed.",
                    "blocking_check_count": 0,
                    "blocking_checks": [],
                    "key_metrics": metrics,
                },
            }
        )

        self.assertTrue(result["passed"], result["errors"])
        self.assertEqual(result["schema"]["name"], "compare_gate")

    def test_suite_gate_schema_accepts_minimal_gate(self):
        metrics = {
            "pass_rate": 1.0,
            "average_score": 100.0,
            "failed": 0,
            "error_count": 0,
            "critical_failure_total": 0,
        }
        result = check_schema_contract(
            {
                "schema_version": "hfr.suite_gate.v1",
                "suite_summary": "runs/suite_summary.json",
                "passed": True,
                "check_count": 1,
                "failed_check_count": 0,
                "checks": [
                    {
                        "id": "min_pass_rate",
                        "passed": True,
                        "actual": 1.0,
                        "expected": {"min": 1.0},
                        "summary": "min_pass_rate: actual=1.0, min=1.0",
                    }
                ],
                "metrics": metrics,
                "decision": {
                    "readiness": "ready",
                    "recommendation": "promote_iteration",
                    "summary": "Gate is ready: all checks passed.",
                    "blocking_check_count": 0,
                    "blocking_checks": [],
                    "key_metrics": metrics,
                },
            }
        )

        self.assertTrue(result["passed"], result["errors"])
        self.assertEqual(result["schema"]["name"], "suite_gate")

    def test_training_gate_schema_accepts_minimal_gate(self):
        metrics = {
            "episode_count": 1,
            "pass_rate": 1.0,
            "average_score": 100.0,
            "quality_flag_count": 0,
            "source_fingerprint_coverage": {"rate": 1.0, "unverified": 0},
            "trainer_view_source_fingerprint_coverage": {"fully_verified_rate": 1.0, "unverified": 0},
            "task_completion": {"complete_count": 1, "incomplete_count": 0},
            "trace_signal": {"average_event_count": 4.0, "event_type_count": 3},
            "dataset_splits": {"episode_count": 1, "family_exclusive": True},
            "artifact_counts": {"episodes": 1, "rewards": 1},
            "validation": {
                "available": True,
                "passed": True,
                "strict": True,
                "target_count": 1,
                "error_count": 0,
                "warning_count": 0,
            },
        }
        result = check_schema_contract(
            {
                "schema_version": "hfr.training_gate.v1",
                "training_export": "runs/training_export",
                "passed": True,
                "check_count": 1,
                "failed_check_count": 0,
                "checks": [
                    {
                        "id": "min_episodes",
                        "passed": True,
                        "actual": 1,
                        "expected": {"min": 1},
                        "summary": "min_episodes: actual=1, min=1",
                    }
                ],
                "metrics": metrics,
                "decision": {
                    "readiness": "ready",
                    "recommendation": "promote_iteration",
                    "summary": "Gate is ready: all checks passed.",
                    "blocking_check_count": 0,
                    "blocking_checks": [],
                    "key_metrics": metrics,
                },
            }
        )

        self.assertTrue(result["passed"], result["errors"])
        self.assertEqual(result["schema"]["name"], "training_gate")

    def test_evidence_coverage_schema_accepts_minimal_summary(self):
        result = check_schema_contract(
            {
                "schema_version": "hfr.evidence_coverage.v1",
                "runs_dir": "runs",
                "passed": True,
                "check_count": 0,
                "failed_check_count": 0,
                "checks": [],
                "metrics": {
                    "run_count": 1,
                    "rule_count": 1,
                    "failed_rule_count": 1,
                    "critical_failed_rule_count": 1,
                    "evidence_ref_count": 1,
                    "failed_rule_evidence_ref_count": 1,
                    "critical_failed_rule_evidence_ref_count": 1,
                    "failed_rules_with_evidence": 1,
                    "failed_rules_without_evidence": 0,
                    "critical_failed_rules_with_evidence": 1,
                    "critical_failed_rules_without_evidence": 0,
                    "task_evidence_ref_count": 1,
                    "failed_rule_evidence_rate": 1.0,
                    "critical_failed_rule_evidence_rate": 1.0,
                    "event_evidence_ref_count": 1,
                    "final_answer_evidence_ref_count": 0,
                    "episode_evidence_ref_count": 0,
                    "evidence_target_counts": [{"id": "event", "count": 1}],
                    "failed_rule_evidence_target_counts": [{"id": "event", "count": 1}],
                    "rule_coverage": [
                        {
                            "rule_id": "required_actions",
                            "rule_name": "Required Actions",
                            "rule_count": 1,
                            "passed": 0,
                            "failed": 1,
                            "critical_failed": 1,
                            "evidence_ref_count": 1,
                            "negative_evidence_ref_count": 1,
                            "failed_with_evidence": 1,
                            "failed_without_evidence": 0,
                            "evidence_target_counts": [{"id": "event", "count": 1}],
                        }
                    ],
                },
                "warnings": [],
                "runs": [
                    {
                        "scenario_id": "email_reply_completion_bad",
                        "scenario_title": "Email Reply Completion",
                        "run_dir": "runs/email_reply_completion_bad",
                        "score": 0,
                        "passed": False,
                        "event_count": 5,
                        "rule_count": 1,
                        "failed_rule_count": 1,
                        "critical_failed_rule_count": 1,
                        "evidence_ref_count": 1,
                        "failed_rule_evidence_ref_count": 1,
                        "critical_failed_rule_evidence_ref_count": 1,
                        "failed_rules_with_evidence": 1,
                        "failed_rules_without_evidence": [],
                        "critical_failed_rules_with_evidence": 1,
                        "critical_failed_rules_without_evidence": [],
                        "task_evidence_ref_count": 1,
                        "evidence_target_counts": [{"id": "event", "count": 1}],
                        "failed_rule_evidence_target_counts": [{"id": "event", "count": 1}],
                        "rules": [
                            {
                                "rule_id": "required_actions",
                                "rule_name": "Required Actions",
                                "passed": False,
                                "failed": True,
                                "critical": True,
                                "evidence_ref_count": 1,
                                "negative_evidence_ref_count": 1,
                                "evidence_target_counts": [{"id": "event", "count": 1}],
                            }
                        ],
                    }
                ],
            }
        )

        self.assertTrue(result["passed"], result["errors"])
        self.assertEqual(result["schema"]["name"], "evidence_coverage")

    def test_scenario_quality_schema_accepts_minimal_summary(self):
        result = check_schema_contract(
            {
                "schema_version": "hfr.scenario_quality.v1",
                "scenarios_dir": "scenarios",
                "pattern": "*.json",
                "recursive": False,
                "require_traces": True,
                "passed": True,
                "check_count": 0,
                "failed_check_count": 0,
                "checks": [],
                "metrics": {
                    "scenario_count": 1,
                    "valid_scenario_count": 1,
                    "invalid_scenario_count": 0,
                    "task_family_count": 1,
                    "average_contract_score": 100.0,
                    "min_contract_score": 100,
                    "max_contract_score": 100,
                    "observable_scenario_count": 1,
                    "observable_scenario_rate": 1.0,
                    "weak_scenario_count": 0,
                    "final_only_scenario_count": 0,
                    "missing_trace_count": 0,
                    "missing_state_count": 0,
                    "task_families": ["email_reply_completion"],
                    "risk_counts": [],
                },
                "scenarios": [
                    {
                        "path": "scenarios/email_reply_completion_good.json",
                        "id": "email_reply_completion_good",
                        "title": "Email Reply Completion",
                        "task_family": "email_reply_completion",
                        "contract_score": 100,
                        "quality": "strong",
                        "errors": [],
                        "risks": [],
                        "signals": {
                            "has_trace_path": True,
                            "trace_exists": True,
                            "has_before_state_path": True,
                            "before_state_exists": True,
                            "has_state_path": True,
                            "state_exists": True,
                            "regex_constraint_count": 1,
                            "budget_constraint_count": 1,
                            "has_secret_policy": True,
                            "has_budget_limits": True,
                            "required_evidence_count": 1,
                            "required_action_count": 1,
                            "required_action_sequence_count": 1,
                            "required_event_count_count": 1,
                            "required_state_count": 1,
                            "required_state_transition_count": 1,
                            "observable_assertion_count": 6,
                            "task_completion_assertion_count": 5,
                            "final_assertion_count": 1,
                            "pass_threshold": 90,
                        },
                        "trace": {
                            "has_trace_path": True,
                            "trace_exists": True,
                            "trace_required": True,
                            "trace_path": "fixtures/email_reply_completion_good.observer.jsonl",
                        },
                        "state": {
                            "has_before_state_path": True,
                            "before_state_exists": True,
                            "has_state_path": True,
                            "state_exists": True,
                            "before_state_path": "fixtures/email_before.json",
                            "state_path": "fixtures/email_after.json",
                        },
                    }
                ],
            }
        )

        self.assertTrue(result["passed"], result["errors"])
        self.assertEqual(result["schema"]["name"], "scenario_quality")

    def test_trace_observability_schema_accepts_minimal_summary(self):
        result = check_schema_contract(
            {
                "schema_version": "hfr.trace_observability.v1",
                "runs_dir": "runs",
                "passed": True,
                "check_count": 0,
                "failed_check_count": 0,
                "checks": [],
                "metrics": {
                    "run_count": 1,
                    "total_event_count": 2,
                    "average_event_count": 2.0,
                    "min_event_count": 2,
                    "max_event_count": 2,
                    "event_type_count": 2,
                    "event_type_counts": [
                        {"id": "assistant_message", "count": 1},
                        {"id": "tool_call", "count": 1},
                    ],
                    "source_format_counts": [{"id": "observer_jsonl", "count": 1}],
                    "model_counts": [{"id": "fixture-model", "count": 1}],
                    "runs_with_final_answer": 1,
                    "empty_final_answer_count": 0,
                    "final_answer_rate": 1.0,
                    "runs_with_tool_or_api_events": 1,
                    "tool_or_api_run_rate": 1.0,
                    "tool_call_count": 1,
                    "tool_result_count": 0,
                    "api_call_count": 0,
                    "subagent_event_count": 0,
                    "approval_event_count": 0,
                    "risk_counts": [{"id": "tool_call_result_imbalance", "count": 1}],
                },
                "warnings": ["prompt_injection_good trace risks: tool_call_result_imbalance"],
                "runs": [
                    {
                        "run_dir": "runs/prompt_injection_good",
                        "scenario_id": "prompt_injection_good",
                        "passed": True,
                        "score": 100,
                        "source_format": "observer_jsonl",
                        "model": "fixture-model",
                        "event_count": 2,
                        "event_type_count": 2,
                        "event_types": [
                            {"id": "assistant_message", "count": 1},
                            {"id": "tool_call", "count": 1},
                        ],
                        "has_final_answer": True,
                        "final_answer_chars": 2,
                        "tool_call_count": 1,
                        "tool_result_count": 0,
                        "api_call_count": 0,
                        "subagent_event_count": 0,
                        "approval_event_count": 0,
                        "has_tool_or_api_events": True,
                        "risks": ["tool_call_result_imbalance"],
                    }
                ],
            }
        )

        self.assertTrue(result["passed"], result["errors"])
        self.assertEqual(result["schema"]["name"], "trace_observability")

    def test_validation_schema_accepts_minimal_summary(self):
        result = check_schema_contract(
            {
                "schema_version": "hfr.validation.v1",
                "passed": True,
                "strict": True,
                "target_count": 1,
                "error_count": 0,
                "warning_count": 0,
                "targets": [
                    {
                        "type": "run",
                        "path": "runs/prompt_injection_good",
                        "passed": True,
                        "errors": [],
                        "warnings": [],
                        "details": {
                            "scenario_id": "prompt_injection_good",
                            "score": 100,
                            "passed": True,
                            "event_count": 4,
                        },
                    }
                ],
            }
        )

        self.assertTrue(result["passed"], result["errors"])
        self.assertEqual(result["schema"]["name"], "validation")

    def test_repair_queue_schema_accepts_minimal_queue(self):
        result = check_schema_contract(
            {
                "schema_version": "hfr.repair_queue.v1",
                "runs_dir": "runs",
                "only_critical": False,
                "passed": True,
                "item_count": 1,
                "metrics": {
                    "item_count": 1,
                    "critical_item_count": 1,
                    "scenario_count": 1,
                    "task_family_count": 1,
                    "scenarios": ["prompt_injection_bad"],
                    "task_families": ["prompt_injection"],
                    "priority_counts": [{"id": "critical", "count": 1}],
                    "rule_counts": [{"id": "forbidden_actions", "count": 1}],
                    "critical_rule_counts": [{"id": "forbidden_actions", "count": 1}],
                    "task_completion_status_counts": [{"id": "incomplete", "count": 1}],
                },
                "items": [
                    {
                        "schema_version": "hfr.repair_item.v1",
                        "repair_item_id": "prompt_injection_bad:forbidden_actions",
                        "run_id": "prompt_injection_bad",
                        "scenario_id": "prompt_injection_bad",
                        "scenario_title": "Prompt Injection Bad",
                        "task_family": "prompt_injection",
                        "priority": "critical",
                        "rule_id": "forbidden_actions",
                        "rule_name": "Forbidden Actions",
                        "critical": True,
                        "penalty": 100,
                        "score": 0,
                        "pass_threshold": 90,
                        "task_completion_status": "incomplete",
                        "task_completion_passed": False,
                        "summary": "forbidden action observed",
                        "suggested_action": "Remove forbidden tool behavior.",
                        "evidence": ["forbidden action observed"],
                        "evidence_refs": [{"target": "event", "event_index": 3, "reason": "forbidden_action"}],
                        "evidence_snippets": [
                            {
                                "target": "event",
                                "event_index": 3,
                                "event_type": "tool_result",
                                "tool_name": "terminal",
                                "status": "ok",
                                "reason": "forbidden_action",
                                "text": "terminal output",
                            }
                        ],
                        "source_artifacts": {
                            "run_dir": "runs/prompt_injection_bad",
                            "normalized_trace": "normalized_trace.json",
                            "scorecard": "scorecard.json",
                            "report": "report.html",
                        },
                        "replay": {
                            "available": True,
                            "self_contained": True,
                            "command": "python -m flightrecorder run --scenario scenarios/prompt_injection_bad.json",
                            "argv": ["python", "-m", "flightrecorder", "run"],
                        },
                    }
                ],
                "notes": [],
            }
        )

        self.assertTrue(result["passed"], result["errors"])
        self.assertEqual(result["schema"]["name"], "repair_queue")

    def test_state_diff_schema_accepts_minimal_diff(self):
        result = check_schema_contract(
            {
                "schema_version": "hfr.state_diff.v1",
                "changed": True,
                "change_count": 1,
                "truncated": False,
                "max_changes": 200,
                "changes": [
                    {
                        "path": "gmail.threads.email-123.sent_replies.0",
                        "kind": "added",
                        "before": None,
                        "after": {"status": "sent"},
                    }
                ],
                "summary": "1 state change(s) detected.",
            }
        )

        self.assertTrue(result["passed"], result["errors"])
        self.assertEqual(result["schema"]["name"], "state_diff")

    def test_state_snapshot_schema_accepts_minimal_snapshot(self):
        result = check_schema_contract(
            {
                "schema_version": "hfr.state_snapshot.v1",
                "filesystem": {
                    "files": {
                        "artifact": {
                            "path": "runs/artifact.json",
                            "exists": True,
                            "kind": "file",
                            "size_bytes": 12,
                            "sha256": "0" * 64,
                        }
                    },
                    "directories": {},
                },
                "json_sources": {
                    "artifact": {
                        "path": "runs/artifact.json",
                        "exists": True,
                        "kind": "file",
                        "size_bytes": 12,
                        "sha256": "0" * 64,
                    }
                },
                "json": {"artifact": {"status": "complete"}},
                "observations": {"task": {"status": "complete"}},
            }
        )

        self.assertTrue(result["passed"], result["errors"])
        self.assertEqual(result["schema"]["name"], "state_snapshot")

    def test_run_digest_schema_accepts_minimal_digest(self):
        result = check_schema_contract(
            {
                "schema_version": "hfr.run_digest.v1",
                "scenario": {"id": "email_reply_completion_good", "title": "Email Reply Completion", "task_family": "email_reply_completion"},
                "outcome": {
                    "passed": True,
                    "score": 100,
                    "pass_threshold": 90,
                    "critical_failures": [],
                    "summary": "passed",
                    "task_completion_status": "complete",
                    "task_completion_passed": True,
                },
                "trace_signal": {
                    "event_count": 1,
                    "event_types": ["tool_call"],
                    "tool_call_count": 1,
                    "tool_result_count": 0,
                    "api_call_count": 0,
                    "subagent_start_count": 0,
                    "max_subagent_depth": 0,
                    "has_final_answer": True,
                    "has_tool_or_api_events": True,
                    "source_format": "observer_jsonl",
                    "model": "unknown",
                },
                "state_changes": {
                    "available": True,
                    "changed": True,
                    "change_count": 1,
                    "truncated": False,
                    "summary": "1 state change(s) detected.",
                    "top_changes": [{"path": "gmail.threads.email-123.sent_replies.0", "kind": "added"}],
                },
                "rules": {"total_count": 10, "failed_count": 0, "critical_failed_count": 0, "failed": []},
                "evidence": {
                    "rule_evidence_ref_count": 1,
                    "failed_rule_evidence_ref_count": 0,
                    "critical_failed_rule_evidence_ref_count": 0,
                    "task_completion_evidence_ref_count": 1,
                    "missing_evidence_ref_count": 0,
                    "total_evidence_ref_count": 2,
                },
                "training_signals": {
                    "score_reward": 1.0,
                    "binary_reward": 1,
                    "task_completion_reward": 1,
                    "task_completion_status": "complete",
                    "task_completion_passed": True,
                    "state_changed": True,
                    "state_change_count": 1,
                    "failure_modes": [],
                },
                "recommended_actions": [{"id": "positive_training_candidate", "priority": "low", "reason": "passed"}],
            }
        )

        self.assertTrue(result["passed"], result["errors"])
        self.assertEqual(result["schema"]["name"], "run_digest")

    def test_harness_run_schemas_accept_minimal_handoff(self):
        manifest_result = check_schema_contract(
            {
                "schema_version": "hfr.harness_run_manifest.v1",
                "runner": "hermes_harness",
                "provider": "mock",
                "model": {"id": "hfr-mock"},
                "scenario": {"id": "prompt_injection_good", "path": "scenarios/prompt_injection_good.json"},
                "outputs": {"run_dir": ".", "manifest": "harness_manifest.json", "result": "harness_result.json"},
                "sandbox": {
                    "root": "sandbox",
                    "home": "sandbox/home",
                    "workspace": "sandbox/workspace",
                    "events": "sandbox/events.jsonl",
                    "fake_secret_canaries": [{"name": "HFR_FAKE_SECRET_CANARY", "sha256": "0" * 64}],
                },
                "tool_policy": {
                    "source": "test_fixture",
                    "scenario_policy": {},
                    "runtime_policy": {"mode": "mock", "network": {"mode": "disabled", "allowed_hosts": []}},
                    "blocked_action_canaries": [
                        {"type": "secret", "pattern": "HFR_FAKE_SECRET_CANARY", "expected": "absent"}
                    ],
                },
            }
        )
        self.assertTrue(manifest_result["passed"], manifest_result["errors"])
        self.assertEqual(manifest_result["schema"]["name"], "harness_run_manifest")

        result_result = check_schema_contract(
            {
                "schema_version": "hfr.harness_run_result.v1",
                "runner": "hermes_harness",
                "provider": "mock",
                "model": {"id": "hfr-mock"},
                "scenario_id": "prompt_injection_good",
                "sandbox": {"root": "sandbox"},
                "tool_policy": {
                    "source": "test_fixture",
                    "scenario_policy": {},
                    "runtime_policy": {"mode": "mock", "network": {"mode": "disabled", "allowed_hosts": []}},
                    "blocked_action_canaries": [
                        {"type": "secret", "pattern": "HFR_FAKE_SECRET_CANARY", "expected": "absent"}
                    ],
                },
                "trace": {"path": "normalized_trace.json", "format": "normalized_json"},
                "scorecard": {"path": "scorecard.json", "passed": True, "score": 100},
                "artifacts": {
                    "normalized_trace": "normalized_trace.json",
                    "scorecard": "scorecard.json",
                    "run_digest": "run_digest.json",
                    "report": "report.html",
                    "lineage": "artifact_lineage.json",
                },
                "replay": {"lineage": "artifact_lineage.json", "self_contained": True},
            }
        )
        self.assertTrue(result_result["passed"], result_result["errors"])
        self.assertEqual(result_result["schema"]["name"], "harness_run_result")

    def test_live_smoke_summary_schema_accepts_current_summary(self):
        result = check_schema_contract(
            {
                "schema_version": "hfr.live_smoke.summary.v2",
                "passed": True,
                "hermes_exit_code": 0,
                "mock_request_count": 9,
                "chat_completion_request_count": 1,
                "score": 100,
                "hooks": ["on_session_start", "pre_llm_call", "post_llm_call"],
                "missing_hooks": [],
                "observer_file": "live_observer.jsonl",
                "report": "report.html",
                "lineage": "artifact_lineage.json",
                "task_completion": "task_completion.json",
                "run_digest": "run_digest.json",
                "summary": "live_smoke_summary.json",
                "environment": {
                    "python_version": "3.11.14",
                    "python_implementation": "CPython",
                    "platform": "Linux-test",
                    "hermes_root": "<hermes-root>",
                    "hermes_git_commit": "abcdef123456",
                    "hermes_git_dirty": False,
                    "flight_recorder_root": "<flight-recorder-root>",
                    "flight_recorder_git_commit": "123456abcdef",
                    "flight_recorder_git_dirty": False,
                },
            }
        )

        self.assertTrue(result["passed"], result["errors"])
        self.assertEqual(result["schema"]["name"], "live_smoke_summary")

    def test_live_openclaw_smoke_summary_schema_accepts_current_summary(self):
        result = check_schema_contract(
            {
                "schema_version": "hfr.openclaw.live_smoke.summary.v1",
                "passed": True,
                "agent_mode": "gateway",
                "openclaw_exit_code": 0,
                "openclaw_setup_exit_code": 1,
                "mock_request_count": 1,
                "chat_completion_request_count": 1,
                "plugin_loaded": True,
                "plugin_runtime_hook_count": 17,
                "score": 100,
                "hooks": ["agent_end", "llm_output"],
                "missing_hooks": [],
                "openclaw_event_file": "live_openclaw.openclaw.jsonl",
                "report": "report.html",
                "lineage": "artifact_lineage.json",
                "task_completion": "task_completion.json",
                "run_digest": "run_digest.json",
                "summary": "live_openclaw_smoke_summary.json",
                "environment": {"openclaw_version": "OpenClaw 2026.6.8"},
            }
        )

        self.assertTrue(result["passed"], result["errors"])
        self.assertEqual(result["schema"]["name"], "live_openclaw_smoke_summary")

    def test_live_verifier_smoke_summary_schema_accepts_current_summary(self):
        result = check_schema_contract(
            {
                "schema_version": "hfr.live_verifier_smoke.summary.v1",
                "passed": True,
                "allow_network": True,
                "configured_only": False,
                "strict_live": True,
                "require_live_provider": True,
                "selected_provider_count": 1,
                "live_attempted_provider_count": 1,
                "passed_provider_count": 1,
                "failed_provider_count": 0,
                "skipped_provider_count": 0,
                "providers": [
                    {
                        "provider": "slack",
                        "source_type": "slack_history",
                        "description": "Read Slack channel history.",
                        "required_env": [{"name": "SLACK_BOT_TOKEN", "present": True}],
                        "optional_env": [],
                        "status": "passed",
                        "reason": "ok",
                        "source_status": "ok",
                        "source_count": 1,
                        "validation_passed": True,
                        "artifacts": {
                            "verifier_config": "slack/verifier_config.json",
                            "state_snapshot": "slack/state_snapshot.json",
                            "validation": "slack/validation.json",
                        },
                    }
                ],
                "artifacts": {
                    "summary": "live_verifier_smoke_summary.json",
                    "schema_check": "live_verifier_smoke_summary.schema_check.json",
                },
                "environment": {"platform": "Linux-test"},
            }
        )

        self.assertTrue(result["passed"], result["errors"])
        self.assertEqual(result["schema"]["name"], "live_verifier_smoke_summary")

    def test_improvement_ledger_gate_schema_accepts_minimal_gate(self):
        result = check_schema_contract(
            {
                "schema_version": "hfr.improvement_ledger_gate.v1",
                "improvement_ledger": "runs/improvement_ledger.json",
                "passed": True,
                "decision": {
                    "readiness": "ready",
                    "recommendation": "promote_iteration",
                    "summary": "Improvement-ledger gate is ready.",
                    "blocking_check_count": 0,
                    "blocking_checks": [],
                    "key_metrics": {
                        "plan_count": 2,
                        "unique_work_item_count": 1,
                        "open_work_item_count": 1,
                        "new_work_item_count": 0,
                        "recurring_work_item_count": 1,
                        "resolved_work_item_count": 0,
                        "critical_open_work_item_count": 0,
                        "high_open_work_item_count": 1,
                        "open_priority_counts": [{"id": "high", "count": 1}],
                        "open_category_counts": [{"id": "repair", "count": 1}],
                    },
                },
                "check_count": 1,
                "failed_check_count": 0,
                "checks": [
                    {
                        "id": "max_recurring_work_items",
                        "passed": True,
                        "actual": 1,
                        "expected": {"max": 1},
                        "summary": "max_recurring_work_items: actual=1, max=1",
                    }
                ],
                "metrics": {
                    "plan_count": 2,
                    "unique_work_item_count": 1,
                    "open_work_item_count": 1,
                    "new_work_item_count": 0,
                    "recurring_work_item_count": 1,
                    "resolved_work_item_count": 0,
                    "critical_open_work_item_count": 0,
                    "high_open_work_item_count": 1,
                    "open_priority_counts": [{"id": "high", "count": 1}],
                    "open_category_counts": [{"id": "repair", "count": 1}],
                },
                "policy": {
                    "schema_version": "hfr.improvement_ledger_gate.policy.v1",
                    "path": "examples/improvement_ledger_gate_policy.demo.json",
                    "effective": {"max_recurring_work_items": 1},
                },
            }
        )

        self.assertTrue(result["passed"], result["errors"])
        self.assertEqual(result["schema"]["name"], "improvement_ledger_gate")

    def test_write_schema_bundle_writes_catalog_and_selected_schemas(self):
        with tempfile.TemporaryDirectory() as tmp:
            written = write_schema_bundle(tmp, ["trace", "scorecard"])
            names = {path.name for path in written}

            self.assertEqual(names, {"manifest.json", "trace.v1.schema.json", "scorecard.v1.schema.json"})
            catalog = json.loads((Path(tmp) / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual({record["name"] for record in catalog["schemas"]}, {"trace", "scorecard"})

    def test_cli_lists_and_exports_schemas(self):
        with tempfile.TemporaryDirectory() as tmp:
            stdout = StringIO()
            with redirect_stdout(stdout):
                list_code = main(["schemas"])
            self.assertEqual(list_code, 0)
            self.assertIn("trace\thfr.trace.v1\ttrace.v1.schema.json", stdout.getvalue())

            schema_out = Path(tmp) / "trace.schema.json"
            with redirect_stdout(StringIO()):
                export_code = main(["schemas", "--name", "trace", "--out", str(schema_out)])
            self.assertEqual(export_code, 0)
            exported = json.loads(schema_out.read_text(encoding="utf-8"))
            self.assertEqual(exported["properties"]["schema_version"]["const"], "hfr.trace.v1")

            bundle_dir = Path(tmp) / "bundle"
            with redirect_stdout(StringIO()):
                bundle_code = main(["schemas", "--name", "task_completion", "--write-dir", str(bundle_dir)])
            self.assertEqual(bundle_code, 0)
            self.assertTrue((bundle_dir / "manifest.json").exists())
            self.assertTrue((bundle_dir / "task_completion.v1.schema.json").exists())

            with self.assertRaises(SystemExit) as raised:
                with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                    main(["schemas", "--name", "task_completion", "--write-dir", str(bundle_dir)])
            self.assertEqual(raised.exception.code, 2)

    def test_cli_checks_artifact_schema_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "trace.json"
            trace_path.write_text(
                json.dumps(
                    {
                        "schema_version": "hfr.trace.v1",
                        "session": {"id": "session-1", "source_format": "observer_jsonl", "model": "unknown"},
                        "events": [],
                        "final_answer": "",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            stdout = StringIO()
            with redirect_stdout(stdout):
                code = main(["schemas", "--check", str(trace_path)])
            self.assertEqual(code, 0)
            result = json.loads(stdout.getvalue())
            self.assertTrue(result["passed"])
            self.assertEqual(result["schema"]["artifact_schema_version"], "hfr.trace.v1")

            bad_path = Path(tmp) / "bad.json"
            bad_path.write_text(json.dumps({"schema_version": "hfr.trace.v1"}) + "\n", encoding="utf-8")
            with redirect_stdout(StringIO()):
                bad_code = main(["schemas", "--check", str(bad_path)])
            self.assertEqual(bad_code, 1)

    def test_cli_checks_training_jsonl_row_schema_contracts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = root / "runs"
            with redirect_stdout(StringIO()):
                self.assertEqual(main(["run-suite", "--scenarios", str(ROOT / "scenarios"), "--out", str(runs), "--export-rl"]), 0)
            # Use the run-suite export so the check exercises real trainer rows.
            out = runs / "training_export"

            episodes_result = check_schema_jsonl_file(out / "episodes.jsonl")
            self.assertTrue(episodes_result["passed"], episodes_result["errors"])
            self.assertEqual(episodes_result["row_schema_counts"], [{"name": "rl_episode", "count": 7}])

            stdout = StringIO()
            with redirect_stdout(stdout):
                code = main(["schemas", "--check-jsonl", str(out / "sft.jsonl"), "--name", "rl_sft"])
            self.assertEqual(code, 0)
            result = json.loads(stdout.getvalue())
            self.assertTrue(result["passed"], result["errors"])
            self.assertEqual(result["schema"]["name"], "rl_sft")
            self.assertGreater(result["row_count"], 0)

            bad_rows = root / "bad_sft.jsonl"
            bad_rows.write_text(json.dumps({"schema_version": "hfr.rl.sft.v1", "sample_id": "missing-fields"}) + "\n", encoding="utf-8")
            with redirect_stdout(StringIO()):
                bad_code = main(["schemas", "--check-jsonl", str(bad_rows), "--name", "rl_sft"])
            self.assertEqual(bad_code, 1)

            dataset_metrics_result = check_schema_file(out / "dataset_metrics.json")
            dataset_registry_result = check_schema_file(out / "dataset_registry.json")
            curriculum_result = check_schema_file(out / "curriculum.json")
            self.assertTrue(dataset_metrics_result["passed"], dataset_metrics_result["errors"])
            self.assertEqual(dataset_metrics_result["schema"]["name"], "rl_dataset_metrics")
            self.assertTrue(dataset_registry_result["passed"], dataset_registry_result["errors"])
            self.assertEqual(dataset_registry_result["schema"]["name"], "dataset_registry")
            self.assertTrue(curriculum_result["passed"], curriculum_result["errors"])
            self.assertEqual(curriculum_result["schema"]["name"], "rl_curriculum")
            manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
            dataset_metrics = json.loads((out / "dataset_metrics.json").read_text(encoding="utf-8"))
            dataset_registry = json.loads((out / "dataset_registry.json").read_text(encoding="utf-8"))
            for payload, field_name in (
                (manifest, "trainer_views"),
                (dataset_metrics, "trainer_views"),
                (dataset_registry, "trainer_views"),
            ):
                bad_payload = json.loads(json.dumps(payload))
                bad_payload.pop(field_name)
                bad_result = check_schema_contract(bad_payload)
                self.assertFalse(bad_result["passed"])
                self.assertIn(field_name, "\n".join(bad_result["errors"]))
            bad_registry = json.loads(json.dumps(dataset_registry))
            bad_registry["selection"].pop("mode_to_view")
            bad_registry_result = check_schema_contract(bad_registry)
            self.assertFalse(bad_registry_result["passed"])
            self.assertIn("mode_to_view", "\n".join(bad_registry_result["errors"]))

    def test_schema_check_passes_trainer_handoff_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = root / "runs"
            gate = root / "training_gate.json"
            preflight = root / "trainer_preflight.json"
            launch_check = root / "trainer_launch_check.json"

            with redirect_stdout(StringIO()):
                self.assertEqual(
                    main(
                        [
                            "run-suite",
                            "--scenarios",
                            str(ROOT / "scenarios"),
                            "--out",
                            str(runs),
                            "--export-rl",
                        ]
                    ),
                    0,
                )
                self.assertEqual(
                    main(
                        [
                            "gate-export",
                            "--training-export",
                            str(runs / "training_export"),
                            "--policy",
                            str(ROOT / "examples" / "training_gate_policy.demo.json"),
                            "--out",
                            str(gate),
                        ]
                    ),
                    0,
                )
                self.assertEqual(
                    main(
                        [
                            "trainer-preflight",
                            "--gate",
                            str(gate),
                            "--training-export",
                            str(runs / "training_export"),
                            "--require-gate",
                            "training_gate",
                            "--trainer-command",
                            "python train.py --dataset runs/training_export",
                            "--out",
                            str(preflight),
                        ]
                    ),
                    0,
                )
                self.assertEqual(
                    main(
                        [
                            "trainer-launch-check",
                            "--preflight",
                            str(preflight),
                            "--require-gate",
                            "training_gate",
                            "--out",
                            str(launch_check),
                        ]
                    ),
                    0,
                )

            preflight_result = check_schema_file(preflight)
            launch_result = check_schema_file(launch_check)

            self.assertTrue(preflight_result["passed"], preflight_result["errors"])
            self.assertEqual(preflight_result["schema"]["name"], "trainer_preflight")
            self.assertTrue(launch_result["passed"], launch_result["errors"])
            self.assertEqual(launch_result["schema"]["name"], "trainer_launch_check")


if __name__ == "__main__":
    unittest.main()
