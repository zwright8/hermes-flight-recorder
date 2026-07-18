import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from flightrecorder.adapters import normalize_trajectory_v2
from flightrecorder.schema_registry import check_schema_contract
from flightrecorder.trajectory_v2 import (
    canonical_trajectory_v2_bytes,
    check_trajectory_v2,
    load_trajectory_v2,
    write_trajectory_v2,
)


class TrajectoryV2Tests(unittest.TestCase):
    def test_hermes_openclaw_and_coven_round_trip_lossless_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixtures = {
                "trajectory_jsonl": self.write_hermes(root / "hermes.jsonl"),
                "openclaw_jsonl": self.write_openclaw(root / "openclaw.jsonl"),
                "coven_jsonl": self.write_coven(root / "coven.jsonl"),
            }
            for source_format, path in fixtures.items():
                with self.subTest(source_format=source_format):
                    trajectory = normalize_trajectory_v2(path, source_format)
                    validation = check_trajectory_v2(trajectory)

                    self.assertTrue(validation["passed"], validation)
                    self.assertTrue(trajectory["action_training"]["eligible"])
                    self.assertEqual(trajectory["source"]["format"], source_format)
                    self.assertEqual(
                        trajectory["source"]["payload_sha256"],
                        hashlib.sha256(path.read_bytes()).hexdigest(),
                    )
                    self.assertEqual(trajectory["model"]["revision"], "model-revision-42")
                    self.assertEqual(trajectory["tokenizer"]["revision"], "tokenizer-revision-7")
                    self.assertEqual(trajectory["chat_template"]["sha256"], "c" * 64)
                    self.assertEqual(trajectory["policy"]["sha256"], "d" * 64)
                    self.assertEqual(trajectory["environment"]["sha256"], "e" * 64)
                    self.assertEqual(trajectory["tools"][0]["version"], "2.1.0")
                    self.assertEqual(trajectory["tools"][0]["definition"], self.tool_definition())
                    self.assertEqual(trajectory["approvals"][0]["decision"], "approved")
                    self.assertEqual(trajectory["metrics"]["latency_ms"], 42.5)
                    self.assertEqual(trajectory["metrics"]["token_usage"]["total_tokens"], 12)
                    self.assertEqual(
                        [(row["session_id"], row["parent_session_id"]) for row in trajectory["sessions"]],
                        [("root-session", None), ("child-session", "root-session")],
                    )
                    roles = [event["role"] for event in trajectory["events"]]
                    self.assertIn("system", roles)
                    self.assertIn("developer", roles)
                    calls = [event for event in trajectory["events"] if event["type"] == "tool_call"]
                    results = [event for event in trajectory["events"] if event["type"] == "tool_result"]
                    self.assertEqual([event["tool_call_id"] for event in calls], ["call-1"])
                    self.assertEqual([event["tool_call_id"] for event in results], ["call-1"])
                    self.assertEqual(calls[0]["event_id"], "tool-call-event")
                    self.assertEqual(calls[0]["duration_ms"], 12.5)
                    self.assertEqual(calls[0]["cost_usd"], 0.002)
                    self.assertNotIn("private chain", json.dumps(trajectory))
                    self.assertNotIn("hidden_reasoning", json.dumps(trajectory))

                    out = root / f"{source_format}.trajectory_v2.json"
                    write_trajectory_v2(out, trajectory)
                    loaded = load_trajectory_v2(out)
                    self.assertEqual(loaded, trajectory)
                    self.assertEqual(out.read_bytes(), canonical_trajectory_v2_bytes(loaded))
                    self.assertEqual(normalize_trajectory_v2(out), trajectory)

    def test_schema_and_semantic_mutations_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            trajectory = normalize_trajectory_v2(
                self.write_hermes(Path(tmp) / "hermes.jsonl"),
                "trajectory_jsonl",
            )

        missing_policy = copy.deepcopy(trajectory)
        del missing_policy["policy"]
        self.assertFalse(check_schema_contract(missing_policy, name_or_id="trajectory_v2")["passed"])

        required_fields = {
            "schema_version",
            "trajectory_id",
            "root_session_id",
            "source",
            "model",
            "tokenizer",
            "chat_template",
            "policy",
            "environment",
            "governance",
            "tools",
            "sessions",
            "events",
            "approvals",
            "metrics",
            "final_answer",
            "metadata",
            "action_training",
        }
        for field in required_fields:
            with self.subTest(missing_required_field=field):
                missing = copy.deepcopy(trajectory)
                del missing[field]
                self.assertFalse(check_schema_contract(missing, name_or_id="trajectory_v2")["passed"])

        mutations = {}
        duplicate_call = copy.deepcopy(trajectory)
        second_call = copy.deepcopy(next(event for event in duplicate_call["events"] if event["type"] == "tool_call"))
        second_call["event_id"] = "second-call-event"
        second_call["span_id"] = "second-call-span"
        duplicate_call["events"].append(second_call)
        mutations["duplicate_tool_call_id"] = duplicate_call

        unmatched_result = copy.deepcopy(trajectory)
        next(event for event in unmatched_result["events"] if event["type"] == "tool_result")["tool_call_id"] = "missing-call"
        mutations["unmatched_tool_result"] = unmatched_result

        multiple_results = copy.deepcopy(trajectory)
        second_result = copy.deepcopy(next(event for event in multiple_results["events"] if event["type"] == "tool_result"))
        second_result["event_id"] = "second-result-event"
        second_result["span_id"] = "second-result-span"
        multiple_results["events"].append(second_result)
        mutations["tool_call_with_multiple_results"] = multiple_results

        missing_tools = copy.deepcopy(trajectory)
        missing_tools["tools"] = []
        mutations["missing_tool_schema"] = missing_tools

        inferred_tools = copy.deepcopy(trajectory)
        inferred_tools["tools"][0]["schema_provenance"] = "inferred_or_incomplete"
        mutations["inferred_or_incomplete_tool_schema"] = inferred_tools

        tampered_tool = copy.deepcopy(trajectory)
        tampered_tool["tools"][0]["definition"]["version"] = "2.1.1"
        mutations["tool_definition_fingerprint_mismatch"] = tampered_tool

        unknown_parent_span = copy.deepcopy(trajectory)
        unknown_parent_span["events"][0]["parent_span_id"] = "absent-span"
        mutations["unknown_parent_span"] = unknown_parent_span

        unknown_parent_session = copy.deepcopy(trajectory)
        unknown_parent_session["sessions"][1]["parent_session_id"] = "absent-session"
        mutations["unknown_parent_session"] = unknown_parent_session

        missing_environment = copy.deepcopy(trajectory)
        missing_environment["environment"]["sha256"] = "unknown"
        mutations["malformed_environment_identity"] = missing_environment

        malformed_chat_template = copy.deepcopy(trajectory)
        malformed_chat_template["chat_template"]["sha256"] = "not-a-sha256"
        mutations["malformed_chat_template_identity"] = malformed_chat_template

        mutable_model = copy.deepcopy(trajectory)
        mutable_model["model"]["revision"] = "latest"
        mutations["missing_or_mutable_model_identity"] = mutable_model

        event_unknown_parent_session = copy.deepcopy(trajectory)
        event_unknown_parent_session["events"][0]["parent_session_id"] = "absent-session"
        mutations["event_unknown_parent_session"] = event_unknown_parent_session

        cyclic_span = copy.deepcopy(trajectory)
        cyclic_span["events"][0]["parent_span_id"] = cyclic_span["events"][1]["span_id"]
        cyclic_span["events"][1]["parent_span_id"] = cyclic_span["events"][0]["span_id"]
        mutations["cyclic_span_graph"] = cyclic_span

        for expected_reason, mutated in mutations.items():
            with self.subTest(expected_reason=expected_reason):
                mutated["action_training"] = {
                    "eligible": trajectory["action_training"]["eligible"],
                    "quarantine_reasons": trajectory["action_training"]["quarantine_reasons"],
                }
                validation = check_trajectory_v2(mutated)
                self.assertFalse(validation["passed"], validation)
                self.assertIn(expected_reason, validation["quarantine_reasons"] + validation["errors"])

    def context(self) -> dict:
        return {
            "model": {"provider": "fixture-provider", "name": "fixture-model", "revision": "model-revision-42"},
            "tokenizer": {"name": "fixture-tokenizer", "revision": "tokenizer-revision-7"},
            "chat_template": {"name": "fixture-chat", "revision": "chat-revision-3", "sha256": "c" * 64},
            "policy": {"id": "policy-agent-safe", "version": "4", "sha256": "d" * 64},
            "environment": {"id": "fixture-sandbox", "version": "9", "sha256": "e" * 64},
            "governance": {
                "owner": "fixture-owner",
                "tenant": "fixture-tenant",
                "legal_basis": "contract",
                "allowed_purposes": ["agent_training"],
                "sensitivity": "synthetic",
                "jurisdiction": "US",
                "retention_expires_at": "2030-01-01T00:00:00+00:00",
                "license": "Apache-2.0-synthetic-fixture",
                "provenance": {"source": "unit-test", "source_revision": "v1"},
                "deletion_subject_ids": ["fixture-subject"],
            },
            "tools": [self.tool_definition()],
            "sessions": [
                {"session_id": "root-session", "parent_session_id": None, "agent_id": "root", "agent_role": "orchestrator"},
                {
                    "session_id": "child-session",
                    "parent_session_id": "root-session",
                    "agent_id": "child-1",
                    "agent_role": "researcher",
                },
            ],
            "messages": [
                {"role": "system", "content": "Follow the recorded policy.", "session_id": "root-session"},
                {"role": "developer", "content": "Use tools with exact schemas.", "session_id": "root-session"},
            ],
            "approvals": [
                {
                    "approval_id": "approval-1",
                    "session_id": "root-session",
                    "event_id": None,
                    "request": "write output artifact",
                    "decision": "approved",
                    "decided_by": "fixture-reviewer",
                    "timestamp": "2026-07-18T12:00:00Z",
                }
            ],
            "event_metadata": [
                {
                    "match": {"type": "tool_call", "tool_call_id": "call-1"},
                    "event_id": "tool-call-event",
                    "span_id": "tool-call-span",
                    "timestamp": "2026-07-18T12:00:01Z",
                    "duration_ms": 12.5,
                    "token_usage": {"input_tokens": 4, "output_tokens": 2, "total_tokens": 6},
                    "cost_usd": 0.002,
                    "side_effect_status": "requested",
                },
                {
                    "match": {"type": "tool_result", "tool_call_id": "call-1"},
                    "event_id": "tool-result-event",
                    "span_id": "tool-result-span",
                    "timestamp": "2026-07-18T12:00:02Z",
                    "duration_ms": 30.0,
                    "side_effect_status": "completed",
                },
            ],
            "metrics": {
                "started_at": "2026-07-18T12:00:00Z",
                "ended_at": "2026-07-18T12:00:03Z",
                "latency_ms": 42.5,
                "token_usage": {"input_tokens": 8, "output_tokens": 4, "total_tokens": 12},
                "cost_usd": 0.004,
            },
            "metadata": {"fixture": True, "hidden_reasoning": "private chain"},
        }

    @staticmethod
    def tool_definition() -> dict:
        return {
            "type": "function",
            "version": "2.1.0",
            "function": {
                "name": "write_record",
                "description": "Write one deterministic record.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "value": {"type": "string"},
                        "thinking": {
                            "type": "string",
                            "description": "An observable domain field, not model reasoning.",
                        },
                    },
                    "required": ["value"],
                    "additionalProperties": False,
                },
            },
        }

    def write_hermes(self, path: Path) -> Path:
        payload = {
            "session_id": "root-session",
            "model": "fixture-model",
            "completed": True,
            "trajectory_context": self.context(),
            "conversations": [
                {"from": "human", "value": "Write a record."},
                {
                    "from": "gpt",
                    "value": '<think>private chain</think><tool_call>{"id":"call-1","name":"write_record","arguments":{"value":"ok"}}</tool_call>',
                },
                {
                    "from": "tool",
                    "value": '<tool_response>{"tool_call_id":"call-1","name":"write_record","content":{"written":true}}</tool_response>',
                },
                {"from": "gpt", "value": "Record written."},
            ],
        }
        path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
        return path

    def write_openclaw(self, path: Path) -> Path:
        rows = [
            {
                "schema_version": "hfr.openclaw.event.v1",
                "hook": "session_start",
                "captured_at": "2026-07-18T12:00:00Z",
                "trajectory_context": self.context(),
                "payload": {"sessionId": "root-session", "model": "fixture-model"},
            },
            {
                "schema_version": "hfr.openclaw.event.v1",
                "hook": "before_prompt_build",
                "captured_at": "2026-07-18T12:00:00Z",
                "payload": {"sessionId": "root-session", "prompt": "Write a record."},
            },
            {
                "schema_version": "hfr.openclaw.event.v1",
                "hook": "before_tool_call",
                "captured_at": "2026-07-18T12:00:01Z",
                "payload": {
                    "sessionId": "root-session",
                    "toolName": "write_record",
                    "toolCallId": "call-1",
                    "params": {"value": "ok"},
                },
            },
            {
                "schema_version": "hfr.openclaw.event.v1",
                "hook": "after_tool_call",
                "captured_at": "2026-07-18T12:00:02Z",
                "payload": {
                    "sessionId": "root-session",
                    "toolName": "write_record",
                    "toolCallId": "call-1",
                    "success": True,
                    "result": {"written": True},
                },
            },
            {
                "schema_version": "hfr.openclaw.event.v1",
                "hook": "agent_end",
                "captured_at": "2026-07-18T12:00:03Z",
                "payload": {"sessionId": "root-session", "success": True, "finalAnswer": "Record written."},
            },
        ]
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        return path

    def write_coven(self, path: Path) -> Path:
        rows = [
            {
                "type": "system",
                "subtype": "init",
                "session_id": "root-session",
                "model": "fixture-model",
                "cwd": "/tmp/fixture",
                "tools": [self.tool_definition()],
                "trajectory_context": self.context(),
            },
            {
                "type": "user",
                "session_id": "root-session",
                "message": {"role": "user", "content": [{"type": "text", "text": "Write a record."}]},
            },
            {
                "type": "assistant",
                "session_id": "root-session",
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "id": "call-1", "name": "write_record", "input": {"value": "ok"}}
                    ],
                },
            },
            {
                "type": "tool_result",
                "session_id": "root-session",
                "tool_use_id": "call-1",
                "is_error": False,
                "content": [{"type": "text", "text": '{"written":true}'}],
            },
            {
                "type": "assistant",
                "session_id": "root-session",
                "message": {"role": "assistant", "content": [{"type": "text", "text": "Record written."}]},
            },
            {
                "type": "result",
                "subtype": "success",
                "session_id": "root-session",
                "duration_ms": 42.5,
                "num_turns": 2,
                "is_error": False,
            },
        ]
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        return path


if __name__ == "__main__":
    unittest.main()
