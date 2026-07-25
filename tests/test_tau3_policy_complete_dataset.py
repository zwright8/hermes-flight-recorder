from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

from flightrecorder.schema_registry import check_schema_contract
from flightrecorder.tau3_policy_complete_dataset import (
    BEHAVIORS,
    TAU3_POLICY_COMPLETE_DATASET_SCHEMA_VERSION,
    Tau3PolicyCompleteDatasetError,
    _Target,
    _project_target,
    _v2_family_id,
    build_tau3_policy_complete_dataset,
)


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _tool(name: str) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": f"Fixture {name}",
            "parameters": {
                "type": "object",
                "properties": {"id": {"type": "string"}},
                "required": ["id"],
            },
        },
    }


def _tool_call(call_id: str, name: str, value: str) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": {"id": value}},
            }
        ],
    }


class _Tokenizer:
    chat_template = "policy-complete-fixture-template"

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> list[int]:
        payload = json.dumps(
            {"messages": messages, "tools": kwargs.get("tools")},
            sort_keys=True,
        )
        return list(range(max(1, len(payload))))


def _install_fake_transformers(tokenizer: _Tokenizer) -> mock._patch:
    module = types.ModuleType("transformers")

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(path: Path, **kwargs: Any) -> _Tokenizer:
            return tokenizer

    module.AutoTokenizer = AutoTokenizer
    return mock.patch.dict(sys.modules, {"transformers": module})


class _Fixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.corpus = root / "corpus"
        self.captures = root / "captures.jsonl"
        self.train_split = root / "train.json"
        self.development_split = root / "development.json"
        self.development_tasks = root / "development_tasks.jsonl"
        self.protocol = root / "protocol.json"
        self.tokenizer = root / "tokenizer"
        self.tokenizer.mkdir()
        self._build()

    def _build(self) -> None:
        train_tasks: list[dict[str, Any]] = []
        teacher_rows: list[dict[str, Any]] = []
        family_by_domain: dict[str, list[str]] = {}
        task_ids_by_domain: dict[str, list[str]] = {}
        for domain in ("airline", "retail", "telecom"):
            family_by_domain[domain] = []
            task_ids_by_domain[domain] = []
            for index in (1, 2):
                upstream_family = _canonical_sha256(
                    f"{domain}-upstream-family-{index}"
                )
                if domain == "telecom":
                    upstream_family = _canonical_sha256("telecom-coarse-family")
                    task_id = (
                        f"[mobile_data_issue]condition_{index}"
                        f"[PERSONA:{'Easy' if index == 1 else 'Hard'}]"
                    )
                else:
                    task_id = f"{domain}-task-{index}"
                family_by_domain[domain].append(upstream_family)
                task_ids_by_domain[domain].append(task_id)
                prompt_hash = _canonical_sha256(
                    f"{domain}-agent-visible-prompt-{index}"
                )
                task_hash = _canonical_sha256(f"{domain}-task-hash-{index}")
                train_tasks.append(
                    {
                        "domain": domain,
                        "raw_id": task_id,
                        "raw_id_sha256": _canonical_sha256(task_id),
                        "prompt_sha256": prompt_hash,
                        "task_sha256": task_hash,
                        "family_id": upstream_family,
                    }
                )
                system = f"Exact {domain} policy prompt."
                lookup = f"get_{domain}_record"
                mutation = f"update_{domain}_record"
                messages = [
                    {"role": "system", "content": system},
                    {"role": "assistant", "content": "How can I help?"},
                    {
                        "role": "user",
                        "content": f"Please update my {domain} record {index}.",
                    },
                    _tool_call(f"{domain}-{index}-lookup", lookup, str(index)),
                    {
                        "role": "tool",
                        "name": lookup,
                        "tool_call_id": f"{domain}-{index}-lookup",
                        "content": json.dumps({"id": index, "status": "ready"}),
                    },
                    {
                        "role": "assistant",
                        "content": "I found the record. Please confirm the change.",
                    },
                    {"role": "user", "content": "Yes, confirm the change."},
                    _tool_call(f"{domain}-{index}-update", mutation, str(index)),
                ]
                teacher_rows.append(
                    {
                        "messages": messages,
                        "tools": [_tool(lookup), _tool(mutation)],
                        "metadata": {
                            "schema_version": "hfr.tau3_conversation_import.v1",
                            "episode_id": f"{domain}-episode-{index}",
                            "domain": domain,
                            "split": "train",
                            "task_family": upstream_family,
                            "task_id": task_id,
                            "task_sha256": task_hash,
                            "prompt_sha256": prompt_hash,
                            "system_prompt_sha256": _canonical_sha256(system),
                        },
                    }
                )
        valid_teacher = {
            "messages": [
                {"role": "system", "content": "Excluded development policy."},
                {"role": "user", "content": "Excluded development row."},
                {"role": "assistant", "content": "Excluded."},
            ],
            "metadata": {
                "episode_id": "excluded-development",
                "split": "development",
            },
        }
        _write_jsonl(self.corpus / "train.jsonl", teacher_rows)
        _write_jsonl(self.corpus / "valid.jsonl", [valid_teacher])
        _write_json(
            self.corpus / "manifest.json",
            {
                "schema_version": "hfr.tau3_conversation_import.v1",
                "passed": True,
                "counts": {"train": len(teacher_rows), "valid": 1},
                "files": {
                    "train": {
                        "path": "train.jsonl",
                        "sha256": _sha256(self.corpus / "train.jsonl"),
                    },
                    "valid": {
                        "path": "valid.jsonl",
                        "sha256": _sha256(self.corpus / "valid.jsonl"),
                    },
                },
                "generation_provenance": {"protocol_sha256": "a" * 64},
            },
        )
        train_families = sorted(
            {str(task["family_id"]) for task in train_tasks}
        )
        _write_json(
            self.train_split,
            self._split("train", train_tasks, train_families),
        )
        development_rows = []
        for domain in ("airline", "retail", "telecom"):
            raw_id = (
                "[service_issue]development_only[PERSONA:Easy]"
                if domain == "telecom"
                else f"{domain}-development-only"
            )
            family = _canonical_sha256(f"{domain}-development-family")
            development_rows.append(
                {
                    "domain": domain,
                    "raw_id": raw_id,
                    "raw_id_sha256": _canonical_sha256(raw_id),
                    "prompt_sha256": _canonical_sha256(
                        f"{domain}-development-prompt"
                    ),
                    "task_sha256": _canonical_sha256(
                        f"{domain}-development-task"
                    ),
                    "family_id": family,
                }
            )
        _write_json(
            self.development_split,
            self._split(
                "development",
                development_rows,
                [str(row["family_id"]) for row in development_rows],
            ),
        )
        _write_jsonl(
            self.development_tasks,
            [
                {
                    "task": {
                        "description": f"held-out {row['domain']} development"
                    }
                }
                for row in development_rows
            ],
        )
        capture_rows = []
        for domain in ("airline", "retail", "telecom"):
            tools = [
                _tool(f"get_{domain}_record"),
                _tool(f"update_{domain}_record"),
            ]
            for behavior_index, behavior in enumerate(BEHAVIORS):
                task_index = behavior_index % 2
                capture_rows.append(
                    self._capture(
                        domain=domain,
                        behavior=behavior,
                        split="train",
                        task_id=task_ids_by_domain[domain][task_index],
                        family=family_by_domain[domain][task_index],
                        tools=tools,
                    )
                )
            capture_rows.append(
                self._capture(
                    domain=domain,
                    behavior="success",
                    split="development",
                    task_id=f"{domain}-capture-development",
                    family=_canonical_sha256(
                        f"{domain}-capture-development-family"
                    ),
                    tools=tools,
                )
            )
        _write_jsonl(self.captures, capture_rows)
        _write_json(
            self.protocol,
            {
                "schema_version": "hfr.tau3_protocol_config.v1",
                "split_manifest": {
                    "splits": {
                        "train": {"sha256": _sha256(self.train_split)},
                        "development": {
                            "sha256": _sha256(self.development_split)
                        },
                    }
                },
                "sealed_manifest": {
                    "access_count": 0,
                    "leakage_blocking_hashes": [],
                    "prompt_template_hashes": [],
                },
                "recipe_space": {"sealed_used": False},
                "candidate_selection_contract": {"sealed_used": False},
                "contamination_attestation": {
                    "passed": True,
                    "unresolved_leakage": False,
                    "checks": {"fixture": "passed"},
                },
            },
        )

    @staticmethod
    def _split(
        split: str,
        tasks: list[dict[str, Any]],
        families: list[str],
    ) -> dict[str, Any]:
        return {
            "schema_version": "hfr.tau3_source_split.v1",
            "split": split,
            "source_revision": "b" * 40,
            "task_schema_version": "fixture",
            "algorithm": "fixture-family-split",
            "salt_sha256": "c" * 64,
            "task_count": len(tasks),
            "family_count": len(set(families)),
            "family_ids": sorted(set(families)),
            "tasks": tasks,
        }

    @staticmethod
    def _capture(
        *,
        domain: str,
        behavior: str,
        split: str,
        task_id: str,
        family: str,
        tools: list[dict[str, Any]],
    ) -> dict[str, Any]:
        trajectory_id = f"{split}-{domain}-{behavior}-{_canonical_sha256(task_id)[:8]}"
        prompt = (
            "User scenario:\nKnown info:\nprivate fixture\n"
            "Task instructions:\nprivate simulator direction"
        )
        before = _canonical_sha256([trajectory_id, "before"])
        return {
            "schema_version": "hfr.tau3_capture.v1",
            "trajectory_id": trajectory_id,
            "task_id": task_id,
            "task_family": family,
            "domain": domain,
            "split": split,
            "behavior": behavior,
            "prompt": prompt,
            "prompt_hash": _canonical_sha256(prompt),
            "seed": 7,
            "generator_id": "fixture/teacher",
            "generator_revision": "fixture-revision",
            "policy_revision": f"{domain}-policy-v1",
            "tool_schema_revision": "fixture-tools-v1",
            "starting_state_hash": before,
            "tools": tools,
            "events": [
                {
                    "type": "user_message",
                    "role": "user",
                    "content": prompt,
                },
                {
                    "type": "assistant_message",
                    "role": "assistant",
                    "content": "Reference Tau tool trajectory completed.",
                },
            ],
            "state_transition": {
                "before_hash": before,
                "after_hash": _canonical_sha256([trajectory_id, "after"]),
                "changes": [
                    {
                        "path": "fixture.status",
                        "kind": "changed",
                        "before": "pending",
                        "after": "reviewed",
                    }
                ],
                "executable": True,
            },
            "outcome": {
                "success": behavior == "success",
                "executable_label": "fixture-reviewed",
                "policy_violation": behavior == "policy_failure",
                "harmful_mutation": behavior == "harmful_mutation",
                "evidence_refs": [f"capture:{trajectory_id}"],
            },
            "review": {
                "reviewer": "fixture-reviewer",
                "verifier": "fixture-verifier",
                "disposition": "admit" if behavior == "success" else "reject",
                "reason": "Fixture behavior evidence.",
            },
        }


class Tau3PolicyCompleteDatasetTests(unittest.TestCase):
    def test_builder_excludes_private_capture_content_and_keeps_policy_actions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = _Fixture(Path(tmp))
            output = fixture.root / "dataset"
            with _install_fake_transformers(_Tokenizer()):
                manifest = build_tau3_policy_complete_dataset(
                    teacher_corpus_dir=fixture.corpus,
                    captures_path=fixture.captures,
                    train_split_path=fixture.train_split,
                    development_split_path=fixture.development_split,
                    development_tasks_path=fixture.development_tasks,
                    parent_protocol_path=fixture.protocol,
                    tokenizer_path=fixture.tokenizer,
                    out_dir=output,
                    max_seq_length=100_000,
                    context_window=100_000,
                )

            self.assertTrue(manifest["passed"])
            self.assertTrue(
                check_schema_contract(
                    manifest,
                    name_or_id=TAU3_POLICY_COMPLETE_DATASET_SCHEMA_VERSION,
                )["passed"]
            )
            self.assertEqual(
                manifest["counts"]["capture_training_rows_projected"],
                0,
            )
            self.assertTrue(manifest["balance"]["passed"])
            self.assertEqual(manifest["balance"]["duplicates_added"], 0)
            rows = [
                json.loads(line)
                for split in ("train", "valid")
                for line in (output / f"{split}.jsonl").read_text().splitlines()
            ]
            self.assertTrue(rows)
            for row in rows:
                serialized = json.dumps(row["messages"]).lower()
                self.assertNotIn("user scenario:", serialized)
                self.assertNotIn("known info:", serialized)
                self.assertNotIn("task instructions:", serialized)
                self.assertEqual(row["messages"][0]["role"], "system")
                self.assertEqual(row["messages"][-1]["role"], "assistant")
                self.assertEqual(len(row["tools"]), 2)
                self.assertTrue(row["metadata"]["mask_prompt_required"])
            for split in ("train", "valid"):
                split_rows = [
                    row for row in rows if row["metadata"]["split"] == split
                ]
                for domain in ("airline", "retail", "telecom"):
                    domain_rows = [
                        row
                        for row in split_rows
                        if row["metadata"]["domain"] == domain
                    ]
                    self.assertTrue(
                        any(
                            row["metadata"]["target_tool_name"].startswith(
                                "update_"
                            )
                            and row["metadata"]["target_ordinal"] > 0
                            for row in domain_rows
                        )
                    )
                    recovery = next(
                        row
                        for row in domain_rows
                        if row["metadata"]["behavior"] == "recovery"
                    )
                    self.assertTrue(recovery["metadata"]["after_error_result"])
                    self.assertNotEqual(
                        recovery["metadata"]["target_tool_name"],
                        "invented_tau_tool",
                    )
                    empty = next(
                        row
                        for row in domain_rows
                        if row["metadata"]["behavior"]
                        == "empty_result_recovery"
                    )
                    self.assertEqual(
                        sum(
                            message.get("content") == "[]"
                            for message in empty["messages"]
                        ),
                        2,
                    )
                    self.assertTrue(
                        empty["metadata"]["repeated_call_recovery"]
                    )

    def test_output_is_create_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = _Fixture(Path(tmp))
            output = fixture.root / "dataset"
            output.mkdir()
            (output / "existing").write_text("preserve", encoding="utf-8")
            with self.assertRaisesRegex(
                Tau3PolicyCompleteDatasetError,
                "output",
            ):
                build_tau3_policy_complete_dataset(
                    teacher_corpus_dir=fixture.corpus,
                    captures_path=fixture.captures,
                    train_split_path=fixture.train_split,
                    development_split_path=fixture.development_split,
                    development_tasks_path=fixture.development_tasks,
                    parent_protocol_path=fixture.protocol,
                    tokenizer_path=fixture.tokenizer,
                    out_dir=output,
                )

    def test_telecom_scenario_family_excludes_persona(self) -> None:
        upstream = _canonical_sha256("coarse")
        easy = _v2_family_id(
            "telecom",
            upstream,
            "[mobile_data_issue]airplane_mode_on|roaming_off[PERSONA:Easy]",
        )
        hard = _v2_family_id(
            "telecom",
            upstream,
            "[mobile_data_issue]roaming_off|airplane_mode_on[PERSONA:Hard]",
        )
        different = _v2_family_id(
            "telecom",
            upstream,
            "[mobile_data_issue]roaming_off[PERSONA:Easy]",
        )
        self.assertEqual(easy, hard)
        self.assertNotEqual(easy, different)

    def test_projection_trims_paired_tool_evidence_without_content_truncation(self) -> None:
        tools = [_tool("get_fixture")]
        messages = [
            {"role": "system", "content": "policy"},
            {"role": "user", "content": "old request " + "x" * 80},
            _tool_call("old-call", "get_fixture", "old"),
            {
                "role": "tool",
                "name": "get_fixture",
                "tool_call_id": "old-call",
                "content": "old result " + "y" * 80,
            },
            {"role": "user", "content": "latest request"},
            {"role": "assistant", "content": "safe final target"},
        ]
        target = _Target(
            source_kind="teacher_success",
            source_id="trim-fixture",
            source_sha256="a" * 64,
            family_id="b" * 64,
            domain="airline",
            behavior="success",
            messages=messages,
            tools=tools,
            target_index=5,
            target_kind="assistant_message",
            target_tool_name="",
            target_ordinal=1,
            after_empty_result=False,
            after_error_result=False,
            repeated_call_recovery=False,
            negative_prefix=False,
            pinned_message_indices=frozenset(),
        )
        tokenizer = _Tokenizer()
        minimum = len(
            tokenizer.apply_chat_template(
                [messages[0], messages[4], messages[5]],
                tools=tools,
            )
        )
        row, _ = _project_target(
            target,
            split="train",
            tokenizer=tokenizer,
            max_seq_length=minimum + 20,
            context_window=minimum + 20,
        )
        retained = row["metadata"]["context_projection"]
        self.assertFalse(retained["content_truncated"])
        self.assertGreater(retained["removed_message_count"], 0)
        call_ids = {
            call["id"]
            for message in row["messages"]
            for call in message.get("tool_calls") or []
        }
        result_ids = {
            message["tool_call_id"]
            for message in row["messages"]
            if message.get("role") == "tool"
        }
        self.assertEqual(call_ids, result_ids)
        self.assertEqual(row["messages"][0], messages[0])
        self.assertEqual(row["messages"][-2:], messages[-2:])


if __name__ == "__main__":
    unittest.main()
