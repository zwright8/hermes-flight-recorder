from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from flightrecorder.realistic_tool_corpus import (
    DEFAULT_COUNT,
    build_runtime_adapter_rows,
    build_tool_definitions,
    write_runtime_adapter_corpus,
)
from flightrecorder.schema_registry import SchemaRegistryError, check_schema_contract
from flightrecorder.trajectory_v2 import check_trajectory_v2


ROOT = Path(__file__).resolve().parents[1]


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class RuntimeAdapterTrainingCorpusTests(unittest.TestCase):
    def test_builder_is_deterministic_for_fixed_seed_and_count(self) -> None:
        with tempfile.TemporaryDirectory() as left_dir, tempfile.TemporaryDirectory() as right_dir:
            left = write_runtime_adapter_corpus(Path(left_dir), count=36, seed=101)
            right = write_runtime_adapter_corpus(Path(right_dir), count=36, seed=101)

            self.assertEqual(_sha256(left.all_action_sft_path), _sha256(right.all_action_sft_path))
            self.assertEqual(_jsonl(left.all_action_sft_path), _jsonl(right.all_action_sft_path))
            self.assertEqual(left.split_counts, {"development": 3, "sealed_final": 3, "train": 30})
            self.assertEqual(right.task_family_counts, left.task_family_counts)
            self.assertEqual(len(_jsonl(left.all_action_sft_path)), 30)
            self.assertTrue(all(row["split"] == "train" for row in _jsonl(left.all_action_sft_path)))

    def test_default_count_and_downward_test_counts_have_expected_scopes(self) -> None:
        default_rows = build_runtime_adapter_rows()
        self.assertEqual(len(default_rows), DEFAULT_COUNT)
        self.assertGreaterEqual(len(default_rows), 300)
        self.assertLessEqual(len(default_rows), 800)

        rows = build_runtime_adapter_rows(27)
        scopes = {row["task_scope"] for row in rows}
        families = {row["task_family"] for row in rows}
        self.assertEqual(scopes, {"browser", "code_terminal", "database", "generalist", "shared"})
        self.assertTrue(
            {
                "runtime_adapter_router_browser_train",
                "runtime_adapter_router_database_train",
                "runtime_adapter_router_code_terminal_train",
                "runtime_adapter_router_generalist_train",
                "runtime_adapter_router_shared_safety_train",
            }.issubset(families)
        )

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "count must be between"):
                write_runtime_adapter_corpus(Path(tmp), count=801, seed=17)

    def test_rows_include_realistic_native_tool_behaviors(self) -> None:
        rows = build_runtime_adapter_rows(45)
        tools = build_tool_definitions()
        self.assertTrue(all(tool["version"] == "2026-07-01.immutable" for tool in tools))
        for tool in tools:
            replay = dict(tool)
            definition_sha = replay.pop("definition_sha256")
            self.assertRegex(definition_sha, r"^[0-9a-f]{64}$")
            self.assertIn("parameters", tool["function"])

        multi_call_rows = [
            row
            for row in rows
            if sum(len(message.get("tool_calls", [])) for message in row["messages"] if isinstance(message.get("tool_calls"), list)) >= 2
        ]
        self.assertGreaterEqual(len(multi_call_rows), 20)
        self.assertTrue(any("failure_recovery" in row["behavior_tags"] for row in rows))
        self.assertTrue(any("clarification_no_tool_call" in row["behavior_tags"] for row in rows))
        self.assertTrue(any("refusal_no_tool_call" in row["behavior_tags"] for row in rows))
        self.assertTrue(any("write_denial" in row["behavior_tags"] for row in rows))
        self.assertTrue(any(row["task_scope"] == "generalist" and len(row["task_domains"]) > 1 for row in rows))
        self.assertTrue(any(row["approval_evidence"] for row in rows))

        for row in multi_call_rows[:5]:
            calls = [
                call["id"]
                for message in row["messages"]
                for call in (message.get("tool_calls") if isinstance(message.get("tool_calls"), list) else [])
            ]
            results = [message["tool_call_id"] for message in row["messages"] if message.get("role") == "tool"]
            self.assertTrue(set(calls).issubset(set(results)))

    def test_split_keys_are_disjoint_and_manifest_records_no_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = write_runtime_adapter_corpus(Path(tmp), count=36, seed=17)
            manifest = json.loads(result.corpus_manifest.read_text(encoding="utf-8"))
            self.assertTrue(manifest["split_disjointness"]["passed"])
            dataset = json.loads(result.dataset_manifest.read_text(encoding="utf-8"))
            self.assertTrue(dataset["dataset_splits"]["record_key_disjoint"])
            self.assertTrue(dataset["dataset_splits"]["task_id_disjoint"])
            self.assertTrue(dataset["dataset_splits"]["prompt_hash_disjoint"])
            self.assertTrue(dataset["dataset_splits"]["prompt_template_family_disjoint"])

            all_rows = _jsonl(output_all := Path(tmp) / "data" / "all_splits_action_sft.jsonl")
            self.assertTrue(output_all.is_file())
            rows = all_rows
            for key in ("episode_id", "task_id", "prompt_hash", "prompt_template_family"):
                by_split = {
                    split: {row[key] for row in rows if row["split"] == split}
                    for split in ("train", "development", "sealed_final")
                }
                self.assertTrue(by_split["train"].isdisjoint(by_split["development"]))
                self.assertTrue(by_split["train"].isdisjoint(by_split["sealed_final"]))
                self.assertTrue(by_split["development"].isdisjoint(by_split["sealed_final"]))
            families_by_split = {
                split: {row["task_family"] for row in rows if row["split"] == split}
                for split in ("train", "development", "sealed_final")
            }
            self.assertTrue(families_by_split["train"].isdisjoint(families_by_split["development"]))
            self.assertTrue(families_by_split["train"].isdisjoint(families_by_split["sealed_final"]))
            self.assertTrue(families_by_split["development"].isdisjoint(families_by_split["sealed_final"]))

            train_ids = {row["episode_id"] for row in rows if row["split"] == "train"}
            heldout_ids = {row["episode_id"] for row in rows if row["split"] != "train"}
            for candidate_manifest in result.candidate_manifests.values():
                candidate = json.loads(candidate_manifest.read_text(encoding="utf-8"))
                candidate_path = (candidate_manifest.parent / candidate["data_files"]["fr_action_sft"]).resolve()
                candidate_rows = _jsonl(candidate_path)
                candidate_ids = {row["episode_id"] for row in candidate_rows}
                self.assertTrue(candidate_ids.issubset(train_ids))
                self.assertTrue(candidate_ids.isdisjoint(heldout_ids))
                self.assertTrue(all(row["split"] == "train" for row in candidate_rows))

    def test_trajectory_semantics_and_quarantined_non_action_rows_are_explicit(self) -> None:
        rows = build_runtime_adapter_rows(18)
        eligible = 0
        quarantined = 0
        for row in rows:
            result = check_trajectory_v2(row["trajectory_v2"])
            self.assertTrue(result["schema_passed"], result["errors"])
            self.assertFalse(result["errors"], result["errors"])
            reasons = row["trajectory_v2"]["action_training"]["quarantine_reasons"]
            if reasons:
                quarantined += 1
                self.assertTrue({"no_tool_calls", "unmatched_tool_result"}.intersection(reasons))
            else:
                eligible += 1
        self.assertGreater(eligible, 0)
        self.assertGreater(quarantined, 0)

    def test_script_output_is_trainer_manifest_compatible(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as train_tmp:
            output = Path(tmp) / "runtime_adapter_router"
            build = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    "scripts/build_runtime_adapter_training_corpus.py",
                    "--output-dir",
                    str(output),
                    "--count",
                    "36",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(build.returncode, 0, build.stderr)
            summary = json.loads(build.stdout)
            self.assertEqual(summary["total_rows"], 36)
            self.assertTrue(Path(summary["model_manifest"]).is_file())
            self.assertTrue(Path(summary["dataset_manifest"]).is_file())
            model_result = check_schema_contract(
                json.loads(Path(summary["model_manifest"]).read_text(encoding="utf-8")),
                name_or_id="model_candidate",
            )
            self.assertTrue(model_result["passed"], model_result["errors"])
            with self.assertRaises(SchemaRegistryError):
                check_schema_contract(
                    json.loads(Path(summary["dataset_manifest"]).read_text(encoding="utf-8")),
                    name_or_id="hfr.dataset_registry_entry.v1",
                )

            train = subprocess.run(
                [
                    sys.executable,
                    "scripts/train_agentic_lora.py",
                    "--mode",
                    "fr_action_sft",
                    "--dry-run",
                    "--require-registered-inputs",
                    "--experiment-dir",
                    str(output),
                    "--dataset-manifest",
                    summary["dataset_manifest"],
                    "--model-manifest",
                    summary["model_manifest"],
                    "--output-dir",
                    train_tmp,
                    "--limit",
                    "8",
                    "--disable-trackio",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=True,
            )
            plan = json.loads(train.stdout)
            self.assertTrue(plan["passed"], plan["blocked_reasons"])
            self.assertEqual(plan["failed_check_count"], 0)
            self.assertEqual(plan["prepared_counts"]["action_sft"], 8)
            self.assertEqual(plan["raw_counts"]["fr_action_sft"], 30)


if __name__ == "__main__":
    unittest.main()
