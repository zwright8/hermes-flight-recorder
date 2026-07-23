from __future__ import annotations

import json
import hashlib
import sys
import tempfile
import types
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

from flightrecorder.schema_registry import check_schema_contract
from flightrecorder.tau3_training_mixture import (
    TAU3_TRAINING_MIXTURE_SCHEMA_VERSION,
    Tau3TrainingMixtureError,
    build_tau3_training_mixtures,
)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_source_manifest(source: Path, *, protocol_sha256: str = "c" * 64) -> None:
    (source / "manifest.json").write_text(json.dumps({
        "schema_version": "hfr.tau3_conversation_import.v1",
        "passed": True,
        "files": {
            "train": {"path": "train.jsonl", "sha256": _sha256(source / "train.jsonl")},
            "valid": {"path": "valid.jsonl", "sha256": _sha256(source / "valid.jsonl")},
        },
        "generation_provenance": {"protocol_sha256": protocol_sha256},
    }, sort_keys=True), encoding="utf-8")


def _tool(name: str = "lookup") -> dict[str, Any]:
    return {
        "type": "function",
        "name": name,
        "function": {
            "name": name,
            "description": "lookup fixture",
            "parameters": {
                "type": "object",
                "properties": {"id": {"type": "string"}},
                "required": ["id"],
            },
            "version": "tau3-recorded-tool-schema-v1",
        },
    }


def _row(episode_id: str, family: str, *, assistant_text: str = "I checked the record and can proceed.", tool_name: str = "lookup") -> dict[str, Any]:
    return {
        "messages": [
            {"role": "user", "content": "Please check my account."},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": tool_name, "arguments": {"id": "abc"}},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call-1", "name": tool_name, "content": "{\"ok\":true}"},
            {"role": "assistant", "content": assistant_text},
        ],
        "metadata": {
            "episode_id": episode_id,
            "task_family": family,
            "source_view": "action_sft",
            "source_schema_version": "hfr.rl.action_sft.v1",
            "source_fingerprint_status": "verified",
            "source_fingerprints": {
                "scenario": {"sha256": "a" * 64, "path": "source_scenario.json", "exists": True},
                "source_trace": {"sha256": "b" * 64, "path": "source_trace.json", "exists": True},
            },
        },
        "tools": [_tool(tool_name)],
    }


class _Tokenizer:
    chat_template = "fixture-chat-template"

    def __init__(self, *, multiplier: int = 1) -> None:
        self.multiplier = multiplier

    def apply_chat_template(self, messages: list[dict[str, Any]], **kwargs: Any) -> list[int]:
        self.kwargs = kwargs
        size = sum(len(str(message.get("content") or "")) + 1 for message in messages)
        return list(range(max(1, size * self.multiplier)))


class _MappingArgumentsTokenizer(_Tokenizer):
    def apply_chat_template(self, messages: list[dict[str, Any]], **kwargs: Any) -> list[int]:
        for message in messages:
            for tool_call in message.get("tool_calls") or []:
                arguments = tool_call["function"]["arguments"]
                if not isinstance(arguments, dict):
                    raise TypeError("tool-call arguments must be an object")
        return super().apply_chat_template(messages, **kwargs)


class _UserQueryTokenizer(_MappingArgumentsTokenizer):
    def apply_chat_template(self, messages: list[dict[str, Any]], **kwargs: Any) -> list[int]:
        if not any(message.get("role") == "user" for message in messages):
            raise TypeError("no user query")
        return super().apply_chat_template(messages, **kwargs)


def _install_fake_transformers(tokenizer: _Tokenizer) -> mock._patch:
    module = types.ModuleType("transformers")

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(path: Path, **kwargs: Any) -> _Tokenizer:
            AutoTokenizer.path = path
            AutoTokenizer.kwargs = kwargs
            return tokenizer

    module.AutoTokenizer = AutoTokenizer
    return mock.patch.dict(sys.modules, {"transformers": module})


class Tau3TrainingMixtureTests(unittest.TestCase):
    def test_builds_selected_variant_with_a_hash_audited_token_band(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "input_export"
            _write_jsonl(source / "train.jsonl", [
                _row("tau3-train-retail-success-001", "family-train-1", assistant_text="x" * 80),
                _row("tau3-train-retail-success-002", "family-train-2", assistant_text="x" * 600),
            ])
            _write_jsonl(
                source / "valid.jsonl",
                [_row("tau3-development-retail-success-003", "family-valid", assistant_text="x" * 80)],
            )
            _write_source_manifest(source)

            with _install_fake_transformers(_MappingArgumentsTokenizer()):
                manifest = build_tau3_training_mixtures(
                    source,
                    root / "mixtures",
                    tokenizer_path=root / "tokenizer",
                    max_seq_length=160,
                    context_window=160,
                    min_rendered_tokens=100,
                    exclude_over_budget=True,
                    variant_names=("assistant_turn_targets",),
                )

            self.assertEqual([variant["name"] for variant in manifest["variants"]], ["assistant_turn_targets"])
            self.assertFalse((root / "mixtures" / "full_trajectories").exists())
            selected = json.loads(
                (root / "mixtures" / "assistant_turn_targets" / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(selected["budget_filter"]["min_rendered_tokens"], 100)
            self.assertGreater(selected["counts"]["train"], 0)
            self.assertGreater(selected["counts"]["valid"], 0)

    def test_explicit_budget_filter_excludes_and_hashes_only_oversized_derived_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "input_export"
            _write_jsonl(source / "train.jsonl", [
                _row("tau3-train-retail-success-001", "family-train-1", assistant_text="Short answer."),
                _row("tau3-train-retail-success-002", "family-train-2", assistant_text="x" * 600),
            ])
            _write_jsonl(
                source / "valid.jsonl",
                [_row("tau3-development-retail-success-003", "family-valid", assistant_text="Short valid answer.")],
            )
            _write_source_manifest(source)

            with _install_fake_transformers(_MappingArgumentsTokenizer()):
                manifest = build_tau3_training_mixtures(
                    source,
                    root / "mixtures",
                    tokenizer_path=root / "tokenizer",
                    max_seq_length=120,
                    context_window=120,
                    exclude_over_budget=True,
                )

            full_manifest = json.loads(
                (root / "mixtures" / "full_trajectories" / "manifest.json").read_text(encoding="utf-8")
            )
            budget_filter = full_manifest["budget_filter"]
            self.assertTrue(manifest["budget_filter"]["enabled"])
            self.assertEqual(full_manifest["counts"], {"train": 1, "valid": 1})
            self.assertEqual(budget_filter["input_counts"], {"train": 2, "valid": 1})
            self.assertEqual(budget_filter["excluded"]["train"]["count"], 1)
            self.assertEqual(budget_filter["excluded"]["valid"]["count"], 0)
            self.assertRegex(budget_filter["excluded"]["train"]["derived_row_ids_sha256"], r"^[0-9a-f]{64}$")
            self.assertEqual(full_manifest["tokenizer"]["over_max_seq_length_count"], 0)
            self.assertEqual(full_manifest["tokenizer"]["over_context_window_count"], 0)

    def test_excludes_pre_user_assistant_greetings_from_turn_targets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "input_export"
            train = _row("tau3-train-retail-success-001", "family-train")
            valid = _row("tau3-development-retail-success-002", "family-valid")
            for row in (train, valid):
                row["messages"] = [
                    {"role": "system", "content": "Follow policy."},
                    {"role": "assistant", "content": "Hello, how can I help?"},
                    *row["messages"],
                ]
            _write_jsonl(source / "train.jsonl", [train])
            _write_jsonl(source / "valid.jsonl", [valid])
            _write_source_manifest(source)

            with _install_fake_transformers(_UserQueryTokenizer()):
                manifest = build_tau3_training_mixtures(
                    source,
                    root / "mixtures",
                    tokenizer_path=root / "tokenizer",
                )

            assistant_manifest = json.loads(
                (root / "mixtures" / "assistant_turn_targets" / "manifest.json").read_text(encoding="utf-8")
            )
            assistant_rows = [
                json.loads(line)
                for line in (root / "mixtures" / "assistant_turn_targets" / "train.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertTrue(manifest["passed"])
            self.assertEqual(len(assistant_rows), 2)
            self.assertEqual(assistant_manifest["derivation"]["assistant_targets_before_first_user_excluded"], 2)
            self.assertTrue(all(any(message["role"] == "user" for message in row["messages"]) for row in assistant_rows))

    def test_normalizes_canonical_json_tool_arguments_for_qwen_templates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "input_export"
            train = _row("tau3-train-retail-success-001", "family-train")
            valid = _row("tau3-development-retail-success-002", "family-valid")
            train["messages"][1]["tool_calls"][0]["function"]["arguments"] = '{"id":"abc"}'
            valid["messages"][1]["tool_calls"][0]["function"]["arguments"] = '{"id":"def"}'
            _write_jsonl(source / "train.jsonl", [train])
            _write_jsonl(source / "valid.jsonl", [valid])
            _write_source_manifest(source)
            source_train_sha256 = _sha256(source / "train.jsonl")

            with _install_fake_transformers(_MappingArgumentsTokenizer()):
                build_tau3_training_mixtures(source, root / "mixtures", tokenizer_path=root / "tokenizer")

            derived = json.loads(
                (root / "mixtures" / "full_trajectories" / "train.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()[0]
            )
            arguments = derived["messages"][1]["tool_calls"][0]["function"]["arguments"]
            self.assertEqual(arguments, {"id": "abc"})
            self.assertEqual(derived["metadata"]["tool_argument_encoding"], "object")
            self.assertEqual(_sha256(source / "train.jsonl"), source_train_sha256)

    def test_rejects_invalid_or_non_object_json_tool_arguments(self) -> None:
        for invalid_arguments in (
            '{"id":"a","id":"b"}',
            '["abc"]',
            '{not-json}',
            '{"id":NaN}',
            '{"id":Infinity}',
            '{"id":-Infinity}',
        ):
            with self.subTest(arguments=invalid_arguments), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                source = root / "input_export"
                train = _row("tau3-train-retail-success-001", "family-train")
                train["messages"][1]["tool_calls"][0]["function"]["arguments"] = invalid_arguments
                _write_jsonl(source / "train.jsonl", [train])
                _write_jsonl(source / "valid.jsonl", [_row("tau3-development-retail-success-002", "family-valid")])
                _write_source_manifest(source)

                with _install_fake_transformers(_MappingArgumentsTokenizer()):
                    with self.assertRaisesRegex(Tau3TrainingMixtureError, "tool-call arguments"):
                        build_tau3_training_mixtures(source, root / "mixtures", tokenizer_path=root / "tokenizer")

                self.assertFalse((root / "mixtures").exists())

    def test_builds_three_governed_variants_from_safe_fixtures(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "input_export"
            _write_jsonl(source / "train.jsonl", [_row("tau3-train-retail-success-001", "family-train")])
            _write_jsonl(source / "valid.jsonl", [_row("tau3-development-retail-success-002", "family-valid")])
            _write_source_manifest(source)

            with _install_fake_transformers(_Tokenizer()):
                manifest = build_tau3_training_mixtures(source, root / "mixtures", tokenizer_path=root / "tokenizer")

            self.assertTrue(manifest["passed"])
            self.assertEqual(manifest["source_binding"]["protocol_sha256"], "c" * 64)
            self.assertEqual([item["name"] for item in manifest["variants"]], [
                "full_trajectories",
                "assistant_turn_targets",
                "action_upweighted",
            ])
            for variant in ("full_trajectories", "assistant_turn_targets", "action_upweighted"):
                variant_dir = root / "mixtures" / variant
                variant_manifest = json.loads((variant_dir / "manifest.json").read_text(encoding="utf-8"))
                check = check_schema_contract(variant_manifest, name_or_id=TAU3_TRAINING_MIXTURE_SCHEMA_VERSION)
                self.assertTrue(check["passed"], check["errors"])
                rows = [
                    json.loads(line)
                    for line in (variant_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()
                ]
                self.assertIn("provenance_hashes", rows[0]["metadata"])
                self.assertEqual(rows[0]["metadata"]["source_episode_id"], "tau3-train-retail-success-001")
                self.assertFalse((variant_dir / "test.jsonl").exists())

            action_rows = [
                json.loads(line)
                for line in (root / "mixtures" / "action_upweighted" / "train.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            action_count = sum(row["metadata"]["target_kind"] == "tool_call" for row in action_rows)
            non_action_count = len(action_rows) - action_count
            self.assertLessEqual(action_count, 3 * non_action_count)
            self.assertGreater(action_count, 1)

    def test_rejects_evaluator_criteria_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "input_export"
            _write_jsonl(source / "train.jsonl", [
                _row("tau3-train-airline-clarification-001", "family-train", assistant_text="Check that Agent does not offer partial cabin changes.")
            ])
            _write_jsonl(source / "valid.jsonl", [_row("tau3-development-airline-success-002", "family-valid")])
            _write_source_manifest(source)
            with _install_fake_transformers(_Tokenizer()):
                with self.assertRaisesRegex(Tau3TrainingMixtureError, "evaluator-criteria leakage"):
                    build_tau3_training_mixtures(source, root / "mixtures", tokenizer_path=root / "tokenizer")

    def test_rejects_test_or_sealed_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "input_export"
            _write_jsonl(source / "train.jsonl", [_row("tau3-test-retail-success-001", "family-train")])
            _write_jsonl(source / "valid.jsonl", [_row("tau3-development-retail-success-002", "family-valid")])
            _write_source_manifest(source)
            with _install_fake_transformers(_Tokenizer()):
                with self.assertRaisesRegex(Tau3TrainingMixtureError, "split mismatch"):
                    build_tau3_training_mixtures(source, root / "mixtures", tokenizer_path=root / "tokenizer")

    def test_rejects_missing_tool_schema_and_unpaired_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "input_export"
            bad = _row("tau3-train-retail-success-001", "family-train", tool_name="missing_schema")
            bad["tools"] = [_tool("different")]
            _write_jsonl(source / "train.jsonl", [bad])
            _write_jsonl(source / "valid.jsonl", [_row("tau3-development-retail-success-002", "family-valid")])
            _write_source_manifest(source)
            with _install_fake_transformers(_Tokenizer()):
                with self.assertRaisesRegex(Tau3TrainingMixtureError, "missing tool schema"):
                    build_tau3_training_mixtures(source, root / "mixtures", tokenizer_path=root / "tokenizer")

    def test_rejects_source_tamper_duplicate_output_and_overflow(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "input_export"
            tampered = _row("tau3-train-retail-success-001", "family-train")
            tampered["metadata"]["source_fingerprint_status"] = "unverified"
            _write_jsonl(source / "train.jsonl", [tampered])
            _write_jsonl(source / "valid.jsonl", [_row("tau3-development-retail-success-002", "family-valid")])
            _write_source_manifest(source)
            with _install_fake_transformers(_Tokenizer()):
                with self.assertRaisesRegex(Tau3TrainingMixtureError, "unverified source fingerprint"):
                    build_tau3_training_mixtures(source, root / "mixtures", tokenizer_path=root / "tokenizer")

            _write_jsonl(source / "train.jsonl", [_row("tau3-train-retail-success-001", "family-train")])
            _write_source_manifest(source)
            occupied = root / "occupied"
            occupied.mkdir()
            (occupied / "keep").write_text("x", encoding="utf-8")
            with _install_fake_transformers(_Tokenizer()):
                with self.assertRaisesRegex(Tau3TrainingMixtureError, "output directory must be new or empty"):
                    build_tau3_training_mixtures(source, occupied, tokenizer_path=root / "tokenizer")

            with _install_fake_transformers(_Tokenizer(multiplier=100)):
                with self.assertRaisesRegex(Tau3TrainingMixtureError, "exceeds tokenizer"):
                    build_tau3_training_mixtures(source, root / "overflow", tokenizer_path=root / "tokenizer", max_seq_length=8, context_window=8)
            self.assertFalse((root / "overflow").exists())


if __name__ == "__main__":
    unittest.main()
