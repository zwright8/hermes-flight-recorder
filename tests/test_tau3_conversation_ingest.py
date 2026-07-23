import json
import os
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path

from flightrecorder.schema_registry import check_schema_file
from flightrecorder.tau3_capture import canonical_sha256
from flightrecorder.tau3_conversation_ingest import (
    Tau3ConversationIngestError,
    export_tau3_tool_schemas,
    import_tau3_conversations,
)


REVISION = "1d244f5dca42944b67a379b44bfeb9f5748f189d"


class Tau3ConversationIngestTests(unittest.TestCase):
    def test_exports_tool_schemas_via_tau_subprocess(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "tau"
            repo.mkdir()
            subprocess.run(["git", "init"], cwd=repo, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
            subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
            (repo / "README.md").write_text("fixture\n", encoding="utf-8")
            subprocess.run(["git", "add", "README.md"], cwd=repo, check=True)
            subprocess.run(["git", "commit", "-m", "fixture"], cwd=repo, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            revision = subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=repo, check=True, text=True, stdout=subprocess.PIPE
            ).stdout.strip()
            fake_python = root / "python"
            fake_python.write_text(
                "#!/usr/bin/env python3\n"
                "import json\n"
                "print(json.dumps({'domains': {d: {'tool_count': 1, 'tools': [{'type':'function','function':{'name':'lookup'}}], 'tools_sha256': 'a'*64, 'policy_sha256': 'b'*64} for d in ['airline','retail','telecom']}}))\n",
                encoding="utf-8",
            )
            fake_python.chmod(0o755)

            payload = export_tau3_tool_schemas(
                tau_repo=repo,
                tau_venv_python=fake_python,
                out_path=root / "tools.json",
                expected_tau_revision=revision,
                created_at="2026-07-22T00:00:00Z",
            )

            self.assertEqual(payload["tau_revision"], revision)
            self.assertEqual(payload["domains"]["airline"]["tool_count"], 1)

    def test_imports_successful_train_and_valid_rows_with_private_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = self._write_source(root)
            tools = self._write_tools(root)
            train_results = self._write_results(root, "train", task_id="1", sim_id="sim-train")
            valid_results = self._write_results(root, "valid", task_id="2", sim_id="sim-valid")

            manifest = import_tau3_conversations(
                [train_results, valid_results],
                root / "out",
                source_dir=source_dir,
                tool_schema_path=tools,
                teacher_id="teacher",
                license_id="synthetic-ok",
            )

            self.assertEqual(manifest["counts"], {"train": 1, "valid": 1})
            self.assertEqual(manifest["schema_version"], "hfr.tau3_conversation_import.v1")
            self.assertFalse(manifest["governance"]["sealed_payloads_read"])
            self.assertTrue(check_schema_file(root / "out" / "manifest.json", "tau3_conversation_import")["passed"])
            train_row = _read_jsonl(root / "out" / "train.jsonl")[0]
            valid_row = _read_jsonl(root / "out" / "valid.jsonl")[0]
            self.assertEqual(train_row["metadata"]["split"], "train")
            self.assertEqual(valid_row["metadata"]["split"], "development")
            self.assertEqual(train_row["messages"][0]["role"], "system")
            self.assertIn("<policy>\nAirline policy.\n</policy>", train_row["messages"][0]["content"])
            self.assertNotIn("resolution_steps", train_row["messages"][0]["content"])
            self.assertNotIn("raw_data", json.dumps(train_row["messages"]))
            self.assertNotIn("usage", json.dumps(train_row["messages"]))
            for name in ("train.jsonl", "valid.jsonl", "manifest.json"):
                self.assertEqual(oct(stat.S_IMODE(os.stat(root / "out" / name).st_mode)), "0o600")

    def test_rejects_hidden_evaluator_leakage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = self._write_source(root)
            tools = self._write_tools(root)
            results = self._write_results(
                root,
                "train",
                task_id="1",
                assistant_text="Agent should not approve the cancellation.",
            )

            with self.assertRaisesRegex(Tau3ConversationIngestError, "hidden/evaluator|source hidden"):
                import_tau3_conversations([results], root / "out", source_dir=source_dir, tool_schema_path=tools)

    def test_rejects_sealed_task_hash_without_reading_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = self._write_source(root)
            tools = self._write_tools(root)
            results = self._write_results(root, "train", task_id="1")
            sealed = root / "sealed_manifest.json"
            sealed.write_text(json.dumps({"task_hashes": [self.train_task_sha]}), encoding="utf-8")

            with self.assertRaisesRegex(Tau3ConversationIngestError, "sealed hash"):
                import_tau3_conversations(
                    [results],
                    root / "out",
                    source_dir=source_dir,
                    tool_schema_path=tools,
                    sealed_manifest_path=sealed,
                )

    def test_rejects_real_hash_only_sealed_entry_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = self._write_source(root)
            tools = self._write_tools(root)
            results = self._write_results(root, "train", task_id="1")
            sealed = root / "sealed_manifest.json"
            sealed.write_text(
                json.dumps(
                    {
                        "entries": [
                            {
                                "task_id_sha256": canonical_sha256("unrelated"),
                                "task_sha256": self.train_task_sha,
                                "prompt_sha256": canonical_sha256("unrelated prompt"),
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(Tau3ConversationIngestError, "sealed hash"):
                import_tau3_conversations(
                    [results],
                    root / "out",
                    source_dir=source_dir,
                    tool_schema_path=tools,
                    sealed_manifest_path=sealed,
                )

    def test_rejects_tool_error_in_otherwise_successful_simulation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = self._write_source(root)
            tools = self._write_tools(root)
            results = self._write_results(root, "train", task_id="1")
            payload = json.loads(results.read_text(encoding="utf-8"))
            payload["simulations"][0]["messages"][2]["error"] = True
            results.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(Tau3ConversationIngestError, "errored tool result"):
                import_tau3_conversations(
                    [results], root / "out", source_dir=source_dir, tool_schema_path=tools
                )

    def test_rejects_reward_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = self._write_source(root)
            tools = self._write_tools(root)
            results = self._write_results(root, "train", task_id="1", reward=0)

            with self.assertRaisesRegex(Tau3ConversationIngestError, "reward must be 1"):
                import_tau3_conversations([results], root / "out", source_dir=source_dir, tool_schema_path=tools)

    def test_rejects_wrong_revision(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = self._write_source(root)
            tools = self._write_tools(root)
            results = self._write_results(root, "train", task_id="1", revision="wrong")

            with self.assertRaisesRegex(Tau3ConversationIngestError, "source revision mismatch"):
                import_tau3_conversations([results], root / "out", source_dir=source_dir, tool_schema_path=tools)

    def test_mixed_content_tool_call_requires_explicit_normalization(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = self._write_source(root)
            tools = self._write_tools(root)
            train = self._write_results(root, "train", task_id="1", mixed_tool_content="Let me check.")
            valid = self._write_results(root, "valid", task_id="2", sim_id="sim-valid")

            with self.assertRaisesRegex(Tau3ConversationIngestError, "requires --allow-teacher-protocol-normalization"):
                import_tau3_conversations([train, valid], root / "blocked", source_dir=source_dir, tool_schema_path=tools)

            manifest = import_tau3_conversations(
                [train, valid],
                root / "allowed",
                source_dir=source_dir,
                tool_schema_path=tools,
                allow_teacher_protocol_normalization=True,
            )

            self.assertEqual(manifest["normalization"]["normalized_mixed_content_tool_call_rows"], 1)
            row = _read_jsonl(root / "allowed" / "train.jsonl")[0]
            assistant_tool_turn = next(
                message for message in row["messages"] if message["role"] == "assistant" and "tool_calls" in message
            )
            self.assertNotIn("content", assistant_tool_turn)

    def test_user_tool_calls_are_not_assistant_targets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = self._write_source(root)
            tools = self._write_tools(root)
            train = self._write_results(root, "train", task_id="1", include_user_tool=True)
            valid = self._write_results(root, "valid", task_id="2", sim_id="sim-valid")

            import_tau3_conversations([train, valid], root / "out", source_dir=source_dir, tool_schema_path=tools)

            row = _read_jsonl(root / "out" / "train.jsonl")[0]
            text = json.dumps(row)
            self.assertNotIn("user_lookup", text)
            self.assertNotIn('"requestor": "user"', text)

    def test_rejects_missing_tool_schema_and_undefined_tools(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = self._write_source(root)
            results = self._write_results(root, "train", task_id="1")
            missing_tools = root / "missing_tools.json"
            missing_tools.write_text(
                json.dumps({"domains": {"retail": [{"type": "function", "function": {"name": "x"}}]}}),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(Tau3ConversationIngestError, "missing tool schema"):
                import_tau3_conversations([results], root / "out1", source_dir=source_dir, tool_schema_path=missing_tools)

            bad_tools = root / "bad_tools.json"
            bad_tools.write_text(
                json.dumps({"domains": {"airline": [{"type": "function", "function": {"name": "other_tool"}}]}}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(Tau3ConversationIngestError, "undefined tool"):
                import_tau3_conversations([results], root / "out2", source_dir=source_dir, tool_schema_path=bad_tools)

    def test_rejects_duplicate_episode_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = self._write_source(root)
            tools = self._write_tools(root)
            first = self._write_results(root, "first", task_id="1", sim_id="same")
            second = self._write_results(root, "second", task_id="1", sim_id="same")

            with self.assertRaisesRegex(Tau3ConversationIngestError, "duplicate Tau simulation identity"):
                import_tau3_conversations([first, second], root / "out", source_dir=source_dir, tool_schema_path=tools)

    def _write_source(self, root: Path) -> Path:
        source = root / "source"
        source.mkdir()
        train_task = {
            "id": "1",
            "description": {"purpose": "Cancellation policy case."},
            "evaluation_criteria": {"nl_assertions": ["Agent should not approve the cancellation."]},
            "user_scenario": {
                "instructions": {
                    "known_info": "You are Raj Sanchez.",
                    "task_instructions": "Mention that the customer support representative approved it.",
                }
            },
        }
        valid_task = {
            "id": "2",
            "description": {"purpose": "Membership lookup case."},
            "evaluation_criteria": {"communicate_info": ["4"]},
            "user_scenario": {"instructions": {"known_info": "You are Anya Garcia."}},
        }
        self.train_task_sha = canonical_sha256(train_task)
        train_row = self._source_row("train", train_task, "family-train", self.train_task_sha)
        valid_row = self._source_row("development", valid_task, "family-valid", canonical_sha256(valid_task))
        (source / "train_tasks.jsonl").write_text(json.dumps(train_row) + "\n", encoding="utf-8")
        (source / "development_tasks.jsonl").write_text(json.dumps(valid_row) + "\n", encoding="utf-8")
        return source

    def _source_row(self, split: str, task: dict, family: str, task_sha: str) -> dict:
        return {
            "schema_version": "hfr.tau3_training_source.v1",
            "domain": "airline",
            "split": split,
            "source_revision": REVISION,
            "prompt_sha256": canonical_sha256({"prompt": task["id"]}),
            "task": task,
            "task_family": family,
            "task_sha256": task_sha,
        }

    def _write_tools(self, root: Path) -> Path:
        path = root / "tools.json"
        payload = {
            "domains": {
                "airline": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_reservation_details",
                            "parameters": {"type": "object", "properties": {"reservation_id": {"type": "string"}}},
                        },
                    }
                ]
            }
        }
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def _write_results(
        self,
        root: Path,
        stem: str,
        *,
        task_id: str,
        sim_id: str = "sim",
        revision: str = REVISION,
        reward: float = 1.0,
        assistant_text: str = "I checked the reservation and can help under the policy.",
        mixed_tool_content: str = "",
        include_user_tool: bool = False,
    ) -> Path:
        messages = [{"role": "user", "content": "Please help with my reservation."}]
        if include_user_tool:
            messages.append(
                {
                    "role": "user",
                    "content": "",
                    "tool_calls": [
                        {"id": "user-call", "name": "user_lookup", "arguments": {"x": 1}, "requestor": "user"}
                    ],
                }
            )
        messages.extend(
            [
                {
                    "role": "assistant",
                    "content": mixed_tool_content,
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "name": "get_reservation_details",
                            "arguments": {"reservation_id": "ABC123"},
                            "requestor": "assistant",
                        }
                    ],
                    "raw_data": {"provider": "secret"},
                    "usage": {"prompt_tokens": 1},
                },
                {"role": "tool", "tool_call_id": "call-1", "content": "{\"status\":\"ok\"}"},
                {"role": "assistant", "content": assistant_text},
            ]
        )
        payload = {
            "info": {
                "git_commit": revision,
                "num_trials": 1,
                "max_steps": 30,
                "max_errors": 10,
                "user_info": {"implementation": "user_simulator"},
                "agent_info": {"implementation": "llm_agent_gt"},
                "environment_info": {"domain_name": "airline", "policy": "Airline policy."},
            },
            "tasks": [],
            "simulations": [
                {
                    "id": sim_id,
                    "task_id": task_id,
                    "start_time": "2026-01-01T00:00:00",
                    "end_time": "2026-01-01T00:01:00",
                    "duration": 60.0,
                    "termination_reason": "user_stop",
                    "reward_info": {"reward": reward},
                    "messages": messages,
                    "seed": 42,
                    "mode": "half-duplex",
                }
            ],
        }
        path = root / f"{stem}_results.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


if __name__ == "__main__":
    unittest.main()
