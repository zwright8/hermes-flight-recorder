import json
import tempfile
import unittest
from pathlib import Path

from flightrecorder.adapters import normalize_trace


ROOT = Path(__file__).resolve().parents[1]


class AdapterTests(unittest.TestCase):
    def test_trajectory_adapter_uses_declared_session_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trajectory.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "session_id": "declared-session",
                        "conversations": [
                            {"from": "human", "value": "hello"},
                            {"from": "gpt", "value": "done"},
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            trace = normalize_trace(path, "trajectory_jsonl")

        self.assertEqual(trace["session"]["id"], "declared-session")
        self.assertEqual({event["session_id"] for event in trace["events"]}, {"declared-session"})

    def test_trajectory_adapter_extracts_tool_calls_results_and_final_answer(self):
        trace = normalize_trace(ROOT / "fixtures" / "prompt_injection_good.trajectory.jsonl")

        self.assertEqual(trace["schema_version"], "hfr.trace.v1")
        self.assertEqual(trace["session"]["source_format"], "trajectory_jsonl")
        self.assertEqual(trace["session"]["model"], "fixture-model")
        self.assertIn("quality gates", trace["final_answer"])
        self.assertTrue(trace["metadata"]["completed"])
        self.assertTrue(any(event["type"] == "tool_call" and event["tool_name"] == "read_file" for event in trace["events"]))
        self.assertTrue(any(event["type"] == "tool_result" and "Malicious note" in event["text"] for event in trace["events"]))

    def test_observer_adapter_maps_subagent_events(self):
        trace = normalize_trace(ROOT / "fixtures" / "subagent_claim_bad.observer.jsonl")

        event_types = [event["type"] for event in trace["events"]]
        self.assertIn("subagent_start", event_types)
        self.assertIn("subagent_stop", event_types)
        self.assertEqual(trace["session"]["source_format"], "observer_jsonl")
        self.assertIn("uploaded report.pdf", trace["final_answer"])

    def test_observer_adapter_maps_async_delegate_cron_trace(self):
        trace = normalize_trace(ROOT / "fixtures" / "cron_async_delegation_bad.observer.jsonl")

        event_types = [event["type"] for event in trace["events"]]
        self.assertEqual(trace["session"]["id"], "cron-session-53027")
        self.assertEqual(trace["session"]["source_format"], "observer_jsonl")
        self.assertEqual(event_types.count("subagent_stop"), 2)
        delegate_result = next(event for event in trace["events"] if event["type"] == "tool_result" and event["tool_name"] == "delegate_task")
        self.assertEqual(delegate_result["result"]["mode"], "background")
        self.assertEqual(delegate_result["result"]["delegation_id"], "deleg_cron_53027")
        self.assertNotIn("ASYNC DELEGATION BATCH COMPLETE", trace["final_answer"])

    def test_observer_finalize_hook_auto_detects_as_observer_jsonl(self):
        with self.subTest("on_session_finalize first row"):
            import tempfile
            import json

            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "finalize.jsonl"
                path.write_text(json.dumps({"hook": "on_session_finalize", "session_id": "s1"}) + "\n", encoding="utf-8")
                trace = normalize_trace(path)

            self.assertEqual(trace["session"]["source_format"], "observer_jsonl")

    def test_openclaw_adapter_maps_plugin_hook_jsonl(self):
        trace = normalize_trace(ROOT / "fixtures" / "openclaw_support_ticket_good.openclaw.jsonl")

        self.assertEqual(trace["schema_version"], "hfr.trace.v1")
        self.assertEqual(trace["session"]["source_format"], "openclaw_jsonl")
        self.assertEqual(trace["session"]["model"], "hfr-openclaw-fixture")
        self.assertIn("TICK-1842", trace["final_answer"])
        self.assertEqual(trace["metadata"]["openclaw_hook_count"], 6)
        self.assertTrue(
            any(
                event["type"] == "tool_call"
                and event["tool_name"] == "create_support_ticket"
                and event["args"]["customer_id"] == "cust_123"
                for event in trace["events"]
            )
        )
        self.assertTrue(
            any(
                event["type"] == "tool_result"
                and event["result"]["ticket_id"] == "TICK-1842"
                and event["source_hook"] == "after_tool_call"
                for event in trace["events"]
            )
        )

    def test_coven_adapter_maps_stream_json_frames(self):
        trace = normalize_trace(ROOT / "fixtures" / "coven_detached_good.coven.jsonl")

        self.assertEqual(trace["schema_version"], "hfr.trace.v1")
        self.assertEqual(trace["session"]["source_format"], "coven_jsonl")
        self.assertEqual(trace["session"]["id"], "coven-session-1")
        self.assertEqual(trace["session"]["model"], "openai/gpt-5.5")
        self.assertTrue(trace["metadata"]["completed"])
        self.assertEqual(trace["metadata"]["coven_stream_event_types"], ["system", "user", "result"])
        self.assertEqual(trace["final_answer"], "")
        self.assertTrue(any(event["type"] == "session_start" and event["source_event_type"] == "system" for event in trace["events"]))
        self.assertTrue(any(event["type"] == "user_message" and "detached Coven smoke" in event["text"] for event in trace["events"]))
        self.assertTrue(any(event["type"] == "session_end" and event["status"] == "success" for event in trace["events"]))

    def test_coven_adapter_maps_daemon_event_rows(self):
        trace = normalize_trace(ROOT / "fixtures" / "coven_daemon_events_good.coven.jsonl")

        self.assertEqual(trace["session"]["source_format"], "coven_jsonl")
        self.assertEqual(trace["session"]["id"], "coven-session-2")
        self.assertTrue(trace["metadata"]["completed"])
        self.assertEqual(trace["metadata"]["coven_daemon_event_kinds"], ["input", "output", "exit"])
        self.assertIn("README inspected successfully", trace["final_answer"])
        self.assertTrue(any(event["type"] == "assistant_message" and event["source_event_kind"] == "output" for event in trace["events"]))


if __name__ == "__main__":
    unittest.main()
