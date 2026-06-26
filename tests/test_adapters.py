import unittest
from pathlib import Path

from flightrecorder.adapters import normalize_trace


ROOT = Path(__file__).resolve().parents[1]


class AdapterTests(unittest.TestCase):
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

    def test_observer_finalize_hook_auto_detects_as_observer_jsonl(self):
        with self.subTest("on_session_finalize first row"):
            import tempfile
            import json

            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "finalize.jsonl"
                path.write_text(json.dumps({"hook": "on_session_finalize", "session_id": "s1"}) + "\n", encoding="utf-8")
                trace = normalize_trace(path)

            self.assertEqual(trace["session"]["source_format"], "observer_jsonl")


if __name__ == "__main__":
    unittest.main()
