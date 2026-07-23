import hashlib
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
    _resolve_generation_ref_with_cwd_fallback,
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

    def test_imports_successful_rows_derived_from_generation_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = self._write_source(root)
            tools = self._write_tools(root)
            train_results = self._write_results(root, "train", task_id="1", sim_id="sim-train")
            valid_results = self._write_results(root, "valid", task_id="2", sim_id="sim-valid")
            generation = self._write_generation_manifest(root, [train_results, valid_results])

            manifest = import_tau3_conversations(
                [],
                root / "out",
                source_dir=source_dir,
                tool_schema_path=tools,
                generation_manifest_paths=[generation],
                expected_tau_revision=REVISION,
            )

            self.assertEqual(manifest["counts"], {"train": 1, "valid": 1})
            self.assertEqual(manifest["generation_provenance"]["manifest_count"], 1)
            self.assertEqual(manifest["generation_provenance"]["task_receipt_count"], 3)
            self.assertEqual(manifest["generation_provenance"]["success_count"], 2)
            self.assertEqual(manifest["generation_provenance"]["failure_count"], 1)
            self.assertEqual(manifest["generation_provenance"]["admitted_success_count"], 2)
            self.assertEqual(manifest["generation_provenance"]["excluded_success_count"], 0)
            self.assertEqual(manifest["generation_provenance"]["protocol_sha256"], self.protocol_sha)
            admitted = [
                row for row in manifest["source_generation_manifests"][0]["results"]
                if row["training_admitted"]
            ]
            self.assertEqual([Path(row["path"]) for row in admitted], [train_results, valid_results])
            self.assertTrue(all(len(row["task_receipt_sha256"]) == 64 for row in admitted))
            self.assertTrue(all(Path(row["task_receipt_path"]).is_file() for row in admitted))
            self.assertTrue(check_schema_file(root / "out" / "manifest.json", "tau3_conversation_import")["passed"])

    def test_generation_manifest_accepts_repository_relative_prelaunch_receipt(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            root = Path(tmp)
            source_dir = self._write_source(root)
            tools = self._write_tools(root)
            train_results = self._write_results(root, "train", task_id="1", sim_id="sim-train")
            valid_results = self._write_results(root, "valid", task_id="2", sim_id="sim-valid")
            generation = self._write_generation_manifest(
                root,
                [train_results, valid_results],
                repository_relative_prelaunch=True,
            )

            manifest = import_tau3_conversations(
                [],
                root / "out",
                source_dir=source_dir,
                tool_schema_path=tools,
                generation_manifest_paths=[generation],
                expected_tau_revision=REVISION,
            )

            self.assertEqual(manifest["generation_provenance"]["admitted_success_count"], 2)
            self.assertEqual(manifest["counts"], {"train": 1, "valid": 1})

    def test_generation_relative_reference_wins_over_cwd_collision(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            generation = root / "generation"
            generation.mkdir()
            expected = generation / "prelaunch_receipt.json"
            expected.write_text("generation\n", encoding="utf-8")
            unrelated_cwd = root / "cwd"
            unrelated_cwd.mkdir()
            (unrelated_cwd / expected.name).write_text("unrelated\n", encoding="utf-8")
            previous_cwd = Path.cwd()
            try:
                os.chdir(unrelated_cwd)
                resolved = _resolve_generation_ref_with_cwd_fallback(
                    generation,
                    expected.name,
                    "prelaunch_receipt.path",
                )
            finally:
                os.chdir(previous_cwd)
            self.assertEqual(resolved, expected)

    def test_generation_manifest_excludes_errored_tool_success_without_dropping_clean_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = self._write_source(root)
            tools = self._write_tools(root)
            train_results = self._write_results(root, "train", task_id="1", sim_id="sim-train")
            errored_results = self._write_results(root, "errored", task_id="1", sim_id="sim-errored")
            valid_results = self._write_results(root, "valid", task_id="2", sim_id="sim-valid")
            errored_payload = json.loads(errored_results.read_text(encoding="utf-8"))
            errored_payload["simulations"][0]["messages"][2]["error"] = True
            errored_payload["simulations"][0]["messages"][2]["content"] = "sensitive provider failure"
            errored_results.write_text(json.dumps(errored_payload), encoding="utf-8")
            generation = self._write_generation_manifest(
                root,
                [train_results, errored_results, valid_results],
            )

            manifest = import_tau3_conversations(
                [],
                root / "out",
                source_dir=source_dir,
                tool_schema_path=tools,
                generation_manifest_paths=[generation],
                expected_tau_revision=REVISION,
            )

            provenance = manifest["generation_provenance"]
            self.assertEqual(provenance["success_count"], 3)
            self.assertEqual(provenance["failure_count"], 1)
            self.assertEqual(provenance["admitted_success_count"], 2)
            self.assertEqual(provenance["excluded_success_count"], 1)
            self.assertEqual(manifest["counts"], {"train": 1, "valid": 1})
            excluded = [
                result
                for result in manifest["source_generation_manifests"][0]["results"]
                if result["terminal_status"] == "success" and not result["training_admitted"]
            ]
            self.assertEqual(len(excluded), 1)
            self.assertEqual(Path(excluded[0]["path"]), errored_results)
            self.assertEqual(excluded[0]["sha256"], _file_sha(errored_results))
            self.assertEqual(excluded[0]["training_rejection_code"], "errored_tool_result")
            self.assertEqual(
                excluded[0]["training_rejection_reason"],
                "generated result contains an errored assistant tool result",
            )
            self.assertNotIn("sensitive provider failure", json.dumps(excluded[0]))
            self.assertEqual(excluded[0]["terminal_status"], "success")
            self.assertTrue(Path(excluded[0]["task_receipt_path"]).is_file())
            self.assertTrue(check_schema_file(root / "out" / "manifest.json", "tau3_conversation_import")["passed"])

    def test_generation_manifest_excludes_thinking_tag_success_that_manual_import_rejects(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = self._write_source(root)
            tools = self._write_tools(root)
            train_results = self._write_results(root, "train", task_id="1", sim_id="sim-train")
            thinking_results = self._write_results(
                root,
                "thinking",
                task_id="1",
                sim_id="sim-thinking",
                mixed_tool_content="Unsafe hidden reasoning.</think>",
            )
            valid_results = self._write_results(root, "valid", task_id="2", sim_id="sim-valid")

            with self.assertRaisesRegex(Tau3ConversationIngestError, "thinking tags"):
                import_tau3_conversations(
                    [thinking_results],
                    root / "manual-out",
                    source_dir=source_dir,
                    tool_schema_path=tools,
                    allow_teacher_protocol_normalization=True,
                )

            generation = self._write_generation_manifest(
                root,
                [train_results, thinking_results, valid_results],
            )
            manifest = import_tau3_conversations(
                [],
                root / "bound-out",
                source_dir=source_dir,
                tool_schema_path=tools,
                generation_manifest_paths=[generation],
                expected_tau_revision=REVISION,
                allow_teacher_protocol_normalization=True,
            )

            self.assertEqual(manifest["generation_provenance"]["admitted_success_count"], 2)
            self.assertEqual(manifest["generation_provenance"]["excluded_success_count"], 1)
            excluded = next(
                result
                for result in manifest["source_generation_manifests"][0]["results"]
                if result["terminal_status"] == "success" and not result["training_admitted"]
            )
            self.assertEqual(excluded["training_rejection_code"], "thinking_tag")
            self.assertEqual(
                excluded["training_rejection_reason"],
                "generated result contains a forbidden thinking tag",
            )
            self.assertNotIn("Unsafe hidden reasoning", json.dumps(excluded))
            self.assertEqual(manifest["counts"], {"train": 1, "valid": 1})

    def test_rejects_generation_manifest_with_forged_task_receipt_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = self._write_source(root)
            tools = self._write_tools(root)
            train_results = self._write_results(root, "train", task_id="1", sim_id="sim-train")
            valid_results = self._write_results(root, "valid", task_id="2", sim_id="sim-valid")
            generation = self._write_generation_manifest(root, [train_results, valid_results])
            task_receipt = root / "generation" / "task-0000.json"
            receipt = json.loads(task_receipt.read_text(encoding="utf-8"))
            receipt["result_sha256"] = "0" * 64
            task_receipt.write_text(json.dumps(receipt), encoding="utf-8")

            with self.assertRaisesRegex(Tau3ConversationIngestError, "result hash mismatch"):
                import_tau3_conversations(
                    [],
                    root / "out",
                    source_dir=source_dir,
                    tool_schema_path=tools,
                    generation_manifest_paths=[generation],
                    expected_tau_revision=REVISION,
                )

    def test_rejects_generation_manifest_with_forged_result_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = self._write_source(root)
            tools = self._write_tools(root)
            train_results = self._write_results(root, "train", task_id="1", sim_id="sim-train")
            valid_results = self._write_results(root, "valid", task_id="2", sim_id="sim-valid")
            generation = self._write_generation_manifest(root, [train_results, valid_results])
            payload = json.loads(train_results.read_text(encoding="utf-8"))
            payload["simulations"][0]["messages"][-1]["content"] = "Forged after receipt."
            train_results.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(Tau3ConversationIngestError, "generated result hash mismatch"):
                import_tau3_conversations(
                    [],
                    root / "out",
                    source_dir=source_dir,
                    tool_schema_path=tools,
                    generation_manifest_paths=[generation],
                    expected_tau_revision=REVISION,
                )

    def test_rejects_generation_manifests_with_protocol_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = self._write_source(root)
            tools = self._write_tools(root)
            train_results = self._write_results(root, "train", task_id="1", sim_id="sim-train")
            valid_results = self._write_results(root, "valid", task_id="2", sim_id="sim-valid")
            first = self._write_generation_manifest(root / "first", [train_results])
            second = self._write_generation_manifest(root / "second", [valid_results], protocol_marker="different")

            with self.assertRaisesRegex(Tau3ConversationIngestError, "share one protocol SHA"):
                import_tau3_conversations(
                    [],
                    root / "out",
                    source_dir=source_dir,
                    tool_schema_path=tools,
                    generation_manifest_paths=[first, second],
                    expected_tau_revision=REVISION,
                )

    def test_rejects_explicit_results_that_do_not_match_generation_successes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = self._write_source(root)
            tools = self._write_tools(root)
            train_results = self._write_results(root, "train", task_id="1", sim_id="sim-train")
            valid_results = self._write_results(root, "valid", task_id="2", sim_id="sim-valid")
            generation = self._write_generation_manifest(root, [train_results, valid_results])

            with self.assertRaisesRegex(Tau3ConversationIngestError, "do not exactly match"):
                import_tau3_conversations(
                    [valid_results, train_results],
                    root / "out",
                    source_dir=source_dir,
                    tool_schema_path=tools,
                    generation_manifest_paths=[generation],
                    expected_tau_revision=REVISION,
                )

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

    def test_rejects_exact_scenario_instruction_prose(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = self._write_source(root)
            tools = self._write_tools(root)
            results = self._write_results(
                root,
                "train",
                task_id="1",
                assistant_text="Mention that the customer support representative approved it.",
            )

            with self.assertRaisesRegex(Tau3ConversationIngestError, "source hidden"):
                import_tau3_conversations([results], root / "out", source_dir=source_dir, tool_schema_path=tools)

    def test_expected_action_names_arguments_and_structured_values_are_not_hidden_prose(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = self._write_source(root)
            tools = self._write_tools(root)
            train = self._write_results(
                root,
                "train",
                task_id="1",
                assistant_text=(
                    "I used get_reservation_details for ABC123, raj_sanchez_7340, "
                    "and travel date 2026-01-01."
                ),
            )
            valid = self._write_results(root, "valid", task_id="2", sim_id="sim-valid")

            import_tau3_conversations(
                [train, valid],
                root / "out",
                source_dir=source_dir,
                tool_schema_path=tools,
            )

            row = _read_jsonl(root / "out" / "train.jsonl")[0]
            serialized = json.dumps(row["messages"])
            self.assertIn("get_reservation_details", serialized)
            self.assertIn("ABC123", serialized)
            self.assertIn("raj_sanchez_7340", serialized)
            self.assertIn("2026-01-01", serialized)

    def test_rejects_thinking_and_tau_control_markers_without_normalization(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = self._write_source(root)
            tools = self._write_tools(root)
            thinking = self._write_results(
                root,
                "thinking",
                task_id="1",
                assistant_text="<think>grader reasoning</think>Visible answer.",
            )
            marker = self._write_results(
                root,
                "marker",
                task_id="1",
                terminal_user_content="###STOP###",
            )
            embedded_marker = self._write_results(
                root,
                "embedded-marker",
                task_id="1",
                terminal_user_content="Please repeat ###STOP### before continuing.",
            )

            with self.assertRaisesRegex(Tau3ConversationIngestError, "thinking|hidden/evaluator"):
                import_tau3_conversations(
                    [thinking],
                    root / "thinking-out",
                    source_dir=source_dir,
                    tool_schema_path=tools,
                )
            with self.assertRaisesRegex(
                Tau3ConversationIngestError,
                "requires --allow-teacher-protocol-normalization|hidden/evaluator",
            ):
                import_tau3_conversations(
                    [marker],
                    root / "marker-out",
                    source_dir=source_dir,
                    tool_schema_path=tools,
                )
            with self.assertRaisesRegex(Tau3ConversationIngestError, "hidden/evaluator marker"):
                import_tau3_conversations(
                    [embedded_marker],
                    root / "embedded-marker-out",
                    source_dir=source_dir,
                    tool_schema_path=tools,
                    allow_teacher_protocol_normalization=True,
                )

    def test_normalizes_only_trailing_tau_user_control_markers_with_audit_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = self._write_source(root)
            tools = self._write_tools(root)
            train = self._write_results(
                root,
                "train",
                task_id="1",
                terminal_user_content="###STOP###",
            )
            valid = self._write_results(
                root,
                "valid",
                task_id="2",
                sim_id="sim-valid",
                terminal_user_content="Thank you for the help.\n###TRANSFER###",
            )

            manifest = import_tau3_conversations(
                [train, valid],
                root / "out",
                source_dir=source_dir,
                tool_schema_path=tools,
                allow_teacher_protocol_normalization=True,
            )

            self.assertEqual(manifest["normalization"]["normalized_tau_control_marker_rows"], 2)
            self.assertEqual(manifest["normalization"]["stripped_tau_control_markers"], 2)
            self.assertEqual(
                manifest["normalization"]["dropped_tau_control_marker_only_user_messages"],
                1,
            )
            train_row = _read_jsonl(root / "out" / "train.jsonl")[0]
            valid_row = _read_jsonl(root / "out" / "valid.jsonl")[0]
            self.assertNotIn("###STOP###", json.dumps(train_row["messages"]))
            self.assertEqual(valid_row["messages"][-1], {"role": "user", "content": "Thank you for the help."})
            for row, dropped in ((train_row, 1), (valid_row, 0)):
                self.assertEqual(row["metadata"]["normalization"]["tau_control_markers_stripped"], 1)
                self.assertEqual(
                    row["metadata"]["normalization"]["tau_control_marker_only_user_messages_dropped"],
                    dropped,
                )
                assistant_targets = [message for message in row["messages"] if message["role"] == "assistant"]
                self.assertEqual(assistant_targets[0]["tool_calls"][0]["function"]["name"], "get_reservation_details")
                self.assertEqual(
                    assistant_targets[0]["tool_calls"][0]["function"]["arguments"],
                    '{"reservation_id":"ABC123"}',
                )
                self.assertEqual(
                    assistant_targets[1]["content"],
                    "I checked the reservation and can help under the policy.",
                )

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

    def test_preserves_exact_policy_text_when_hashing_and_building_system_prompt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = self._write_source(root)
            exact_policy = "Airline policy with an official trailing newline.\n"
            tools = self._write_tools(
                root,
                policy_sha256=hashlib.sha256(exact_policy.encode("utf-8")).hexdigest(),
            )
            train = self._write_results(root, "train", task_id="1", policy=exact_policy)
            valid = self._write_results(root, "valid", task_id="2", sim_id="sim-valid", policy=exact_policy)

            import_tau3_conversations(
                [train, valid],
                root / "out",
                source_dir=source_dir,
                tool_schema_path=tools,
            )

            row = _read_jsonl(root / "out" / "train.jsonl")[0]
            self.assertIn(
                "<policy>\nAirline policy with an official trailing newline.\n\n</policy>",
                row["messages"][0]["content"],
            )

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
            "evaluation_criteria": {
                "actions": [
                    {
                        "name": "get_reservation_details",
                        "arguments": {
                            "reservation_id": "ABC123",
                            "user_id": "raj_sanchez_7340",
                            "travel_date": "2026-01-01",
                        },
                    }
                ],
                "nl_assertions": ["Agent should not approve the cancellation."],
            },
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

    def _write_tools(self, root: Path, *, policy_sha256: str | None = None) -> Path:
        path = root / "tools.json"
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_reservation_details",
                    "parameters": {"type": "object", "properties": {"reservation_id": {"type": "string"}}},
                },
            }
        ]
        contract: object = tools
        if policy_sha256 is not None:
            contract = {"tools": tools, "policy_sha256": policy_sha256}
        payload = {"domains": {"airline": contract}}
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
        terminal_user_content: str | None = None,
        policy: str = "Airline policy.",
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
        if terminal_user_content is not None:
            messages.append({"role": "user", "content": terminal_user_content})
        payload = {
            "info": {
                "git_commit": revision,
                "num_trials": 1,
                "max_steps": 30,
                "max_errors": 10,
                "user_info": {"implementation": "user_simulator"},
                "agent_info": {"implementation": "llm_agent_gt"},
                "environment_info": {"domain_name": "airline", "policy": policy},
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

    def _write_generation_manifest(
        self,
        root: Path,
        success_results: list[Path],
        *,
        protocol_marker: str = "default",
        repository_relative_prelaunch: bool = False,
    ) -> Path:
        generation_dir = root / "generation"
        generation_dir.mkdir(parents=True)
        protocol_path = generation_dir / "protocol.json"
        protocol_path.write_text(json.dumps(self._protocol_payload(protocol_marker)), encoding="utf-8")
        self.protocol_sha = _file_sha(protocol_path)
        protocol_record = {"path": protocol_path.name, "sha256": self.protocol_sha, "size": protocol_path.stat().st_size}
        prelaunch = {
            "schema_version": "hfr.tau3_teacher_generation_run.v1",
            "phase": "prelaunch",
            "created_at": "2026-01-01T00:00:00Z",
            "tau_revision": REVISION,
            "source": {"path": "source.jsonl", "sha256": "2" * 64},
            "teacher": {"model": "teacher"},
            "user_simulator": {"model": "user"},
            "protocol": protocol_record,
            "config": self._generation_config(len(success_results) + 1),
            "task_count": len(success_results) + 1,
            "sealed_rows": 0,
            "test_rows": 0,
            "loopback_only": True,
            "training_started": False,
            "sealed_payload_accessed": False,
        }
        prelaunch_path = generation_dir / "prelaunch_receipt.json"
        prelaunch_path.write_text(json.dumps(prelaunch), encoding="utf-8")
        task_receipts = []
        for index, result_path in enumerate(success_results):
            receipt = {
                "schema_version": "hfr.tau3_teacher_generation_run.v1",
                "phase": "task",
                "created_at": "2026-01-01T00:00:00Z",
                "task": {"domain": "airline", "task_id": str(index + 1)},
                "command": ["tau2"],
                "result_path": str(result_path),
                "result_sha256": _file_sha(result_path),
                "reward": 1.0,
                "exit_code": 0,
                "timed_out": False,
                "duration_seconds": 1.0,
                "terminal_status": "success",
                "training_started": False,
                "sealed_payload_accessed": False,
            }
            receipt_path = generation_dir / f"task-{index:04d}.json"
            receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
            task_receipts.append(
                {"path": receipt_path.name, "terminal_status": "success", "result_sha256": receipt["result_sha256"]}
            )
        failed_path = generation_dir / f"task-{len(success_results):04d}.json"
        failed_receipt = {
            "schema_version": "hfr.tau3_teacher_generation_run.v1",
            "phase": "task",
            "created_at": "2026-01-01T00:00:00Z",
            "task": {"domain": "airline", "task_id": "failed"},
            "command": ["tau2"],
            "result_path": "",
            "result_sha256": None,
            "reward": 0.0,
            "exit_code": 1,
            "timed_out": False,
            "duration_seconds": 1.0,
            "terminal_status": "failed",
            "training_started": False,
            "sealed_payload_accessed": False,
        }
        failed_path.write_text(json.dumps(failed_receipt), encoding="utf-8")
        task_receipts.append({"path": failed_path.name, "terminal_status": "failed", "result_sha256": None})
        prelaunch_record_path = (
            str(prelaunch_path.relative_to(Path.cwd()))
            if repository_relative_prelaunch
            else prelaunch_path.name
        )
        manifest = {
            "schema_version": "hfr.tau3_teacher_generation_run.v1",
            "phase": "final",
            "created_at": "2026-01-01T00:00:00Z",
            "tau_revision": REVISION,
            "source": {"path": "source.jsonl", "sha256": "2" * 64},
            "teacher": {"model": "teacher"},
            "user_simulator": {"model": "user"},
            "protocol": protocol_record,
            "config": self._generation_config(len(success_results) + 1),
            "task_count": len(success_results) + 1,
            "prelaunch_receipt": {"path": prelaunch_record_path, "sha256": _file_sha(prelaunch_path), "size": prelaunch_path.stat().st_size},
            "success_count": len(success_results),
            "failure_count": 1,
            "task_receipts": task_receipts,
            "sealed_rows": 0,
            "test_rows": 0,
            "loopback_only": True,
            "training_started": False,
            "sealed_payload_accessed": False,
        }
        manifest_path = generation_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        return manifest_path

    def _protocol_payload(self, marker: str) -> dict:
        return {
            "schema_version": "hfr.tau3_protocol_config.v1",
            "protocol_manifest": {"marker": marker},
            "tau_revision": {"revision": REVISION},
            "split_manifest": {},
            "harness_contract": {},
            "model_freeze": {"teachers": ["teacher"]},
            "budget": {},
            "sealed_manifest": {},
            "mlx_qlora_plan": {},
            "recipe_space": {},
            "candidate_selection_contract": {},
            "contamination_attestation": {},
            "redaction_attestation": {},
            "licenses": ["a", "b", "c", "d"],
            "environment_manifest": {},
        }

    def _generation_config(self, max_tasks: int) -> dict:
        return {
            "agent": "auto",
            "communication_protocol_enforced": True,
            "max_errors": 10,
            "max_steps": 30,
            "max_tasks": max_tasks,
            "seed": 101,
            "timeout_seconds": 600,
        }


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _file_sha(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    unittest.main()
