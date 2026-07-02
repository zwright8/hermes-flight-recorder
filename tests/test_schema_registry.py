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
        self.assertIn("verifier_config", names)
        self.assertIn("state_validator_config", names)
        self.assertIn("state_validator_assertions", names)
        self.assertIn("run_digest", names)
        self.assertIn("harness_run_manifest", names)
        self.assertIn("harness_run_result", names)
        self.assertIn("harness_replay_result", names)
        self.assertIn("live_smoke_summary", names)
        self.assertIn("live_verifier_smoke_summary", names)
        self.assertIn("openclaw_event", names)
        self.assertIn("live_openclaw_smoke_summary", names)
        self.assertIn("harness_run_manifest", names)
        self.assertIn("harness_run_result", names)
        self.assertIn("serving_profile", names)
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
