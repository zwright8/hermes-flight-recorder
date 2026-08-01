from __future__ import annotations

import hashlib
import json
import copy
import sys
import subprocess
import tempfile
import types
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

from flightrecorder.tau3_competitive_dataset import (
    BEHAVIORS,
    CONTEXT_WINDOW_TOKENS,
    GROUNDED_VALIDATOR_TIMEOUT_SECONDS,
    LINEAGE_ID,
    SOURCE_LINEAGE_ID,
    TAU3_COMPETITIVE_DATASET_SCHEMA_VERSION,
    build_tau3_competitive_dataset,
    validate_tau3_competitive_dataset_bundle,
    _add_dominance_blockers,
    _contamination_report_payload_errors,
    _contamination_summary,
    _load_token_counter,
    _target_export_ordinal,
    _validated_grounded_tool_exemptions,
)
from flightrecorder.tau3_exposure import build_tau3_exposure_ledger, validate_tau3_exposure_ledger
from flightrecorder.tau3_objective_validity import build_tau3_objective_validity_report


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _rewrite_manifest(path: Path, manifest: dict[str, Any]) -> None:
    manifest["manifest_sha256"] = _canonical_sha256(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    )
    _write_json(path, manifest)


def _tool(domain: str) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": f"get_{domain}_record",
            "description": f"Get {domain} record",
            "parameters": {
                "type": "object",
                "properties": {"id": {"type": "string"}},
                "required": ["id"],
            },
        },
    }


def _row(
    split: str,
    domain: str,
    family_index: int,
    row_index: int,
    *,
    completion: bool = False,
    state_evidence: bool = False,
    mutation_result: bool = False,
) -> dict[str, Any]:
    family = _canonical_sha256(f"{split}:{domain}:family:{family_index}")
    source_id = f"{split}-{domain}-{family_index}-{row_index}"
    tool_name = f"update_{domain}_record" if mutation_result else f"get_{domain}_record"
    args = {"id": f"{domain}-{family_index}-{row_index}"}
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": f"Exact {domain} Tau policy prompt."},
        {"role": "user", "content": f"Please help with {domain} {row_index}."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": f"call-{source_id}",
                    "type": "function",
                    "function": {"name": tool_name, "arguments": args},
                }
            ],
        },
    ]
    if completion:
        messages.extend(
            [
                {
                    "role": "tool",
                    "name": tool_name,
                    "tool_call_id": f"call-{source_id}",
                    "content": json.dumps({"status": "ready", "id": args["id"]}),
                },
                {
                    "role": "assistant",
                    "content": f"The {domain} request is completed.",
                },
            ]
        )
    metadata = {
        "schema_version": "hfr.tau3_policy_complete_row.v1",
        "lineage_id": SOURCE_LINEAGE_ID,
        "split": split,
        "domain": domain,
        "source_family_id": family,
        "source_id": source_id,
        "source_kind": "teacher_success",
        "source_sha256": _canonical_sha256(source_id),
        "behavior": "success",
        "target_kind": "assistant_message" if completion else "tool_call",
        "target_tool_name": "" if completion else tool_name,
        "target_ordinal": row_index,
        "after_empty_result": False,
        "after_error_result": False,
        "repeated_call_recovery": False,
        "negative_prefix": False,
    }
    if state_evidence:
        metadata["state_evidence_refs"] = {
            "pre_state": "sha256:" + _canonical_sha256(f"pre:{source_id}"),
            "post_state": "sha256:" + _canonical_sha256(f"post:{source_id}"),
            "replay_validator": "sha256:" + _canonical_sha256(f"validator:{source_id}"),
            "replay_validated": True,
        }
    return {"messages": messages, "tools": [_tool(domain)], "metadata": metadata}


def _write_source_dataset(
    root: Path,
    *,
    omit_domain: str | None = None,
    completion_state_evidence: bool = True,
    completion_mutation_result: bool = False,
) -> Path:
    source = root / "source"
    train: list[dict[str, Any]] = []
    valid: list[dict[str, Any]] = []
    for domain in ("airline", "retail", "telecom"):
        if domain == omit_domain:
            continue
        for family in range(8):
            for index in range(3):
                train.append(_row("train", domain, family, index))
                train.append(
                    _row(
                        "train",
                        domain,
                        family,
                        index + 100,
                        completion=True,
                        state_evidence=completion_state_evidence,
                        mutation_result=completion_mutation_result,
                    )
                )
        for family in range(2):
            for index in range(3):
                valid.append(_row("valid", domain, family, index))
                valid.append(
                    _row(
                        "valid",
                        domain,
                        family,
                        index + 100,
                        completion=True,
                        state_evidence=completion_state_evidence,
                        mutation_result=completion_mutation_result,
                    )
                )
    _write_jsonl(source / "train.jsonl", train)
    _write_jsonl(source / "valid.jsonl", valid)
    _write_json(
        source / "manifest.json",
        {
            "schema_version": "hfr.tau3_policy_complete_dataset.v1",
            "lineage_id": SOURCE_LINEAGE_ID,
            "passed": True,
            "files": {
                "train": {
                    "path": "train.jsonl",
                    "sha256": _sha256(source / "train.jsonl"),
                    "bytes": (source / "train.jsonl").stat().st_size,
                },
                "valid": {
                    "path": "valid.jsonl",
                    "sha256": _sha256(source / "valid.jsonl"),
                    "bytes": (source / "valid.jsonl").stat().st_size,
                },
            },
        },
    )
    return source


def _write_tokenizer_config(
    root: Path,
    *,
    algorithm: str = "pinned_local_apply_chat_template",
    external_chat_template: bool = False,
) -> Path:
    tokenizer_dir = root / "tokenizer"
    tokenizer_dir.mkdir()
    (tokenizer_dir / "tokenizer.json").write_text('{"fixture": true}\n', encoding="utf-8")
    if external_chat_template:
        (tokenizer_dir / "chat_template.jinja").write_text(
            "{{ messages | length }} fixture template\n",
            encoding="utf-8",
        )
    (tokenizer_dir / "tokenizer_config.json").write_text(
        '{}\n' if external_chat_template else '{"chat_template": "fixture"}\n',
        encoding="utf-8",
    )
    path = root / "tokenizer-config.json"
    _write_json(
        path,
        {
            "schema_version": "hfr.tau3_competitive_tokenizer_config.v1",
            "tokenizer_id": "fixture-chat-tokenizer",
            "tokenizer_revision": "fixture-revision",
            "tokenizer_path": str(tokenizer_dir),
            "tokenization_algorithm": algorithm,
            "exact": True,
            "chat_template_aware": True,
        },
    )
    return path


class _FakeTokenizer:
    chat_template = "fixture exact chat template with tools"

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def apply_chat_template(self, messages: list[dict[str, Any]], **kwargs: Any) -> list[int]:
        self.calls.append({"messages": messages, "kwargs": kwargs})
        rendered = json.dumps(
            {
                "messages": messages,
                "tools": kwargs.get("tools"),
                "template": self.chat_template,
                "add_generation_prompt": kwargs.get("add_generation_prompt"),
            },
            sort_keys=True,
        )
        return list(range(1, len(rendered.split()) + 1))


def _install_fake_transformers(tokenizer: _FakeTokenizer) -> mock._patch:
    module = types.ModuleType("transformers")

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(path: Path, **kwargs: Any) -> _FakeTokenizer:
            return tokenizer

    module.AutoTokenizer = AutoTokenizer
    return mock.patch.dict(sys.modules, {"transformers": module})


def _grounded_target(
    behavior: str,
    *,
    decision: int,
    tool_name: str = "",
    text: str | None = None,
    masked: bool = False,
    mask_reason: str | None = None,
    linked_safe_decision_ordinal: int | None = None,
    reviewed: bool = False,
) -> dict[str, Any]:
    canonical = {
        "kind": "tool_call" if tool_name else "assistant_message",
        "text": text or f"Handled {behavior}.",
        "tool_name": tool_name or None,
        "arguments": {"id": f"{behavior}-{decision}"} if tool_name else {},
    }
    return {
        "parent_assistant_decision_ordinal": decision,
        "behavior": behavior,
        "masked": masked,
        "mask_reason": mask_reason,
        "safe_correction_decision_ordinal": linked_safe_decision_ordinal,
        "reviewed": reviewed,
        "canonical_target": canonical,
        "canonical_target_sha256": _canonical_sha256(canonical),
    }


def _grounded_row(
    *,
    split: str,
    domain: str,
    family: int,
    row_index: int,
    behaviors: list[str],
    runtime_family: str = "vendored_tau_tools@" + "a" * 40,
    tool_name: str = "update_record",
    tool_catalog_name: str | None = None,
) -> dict[str, Any]:
    tool_catalog = [{"name": tool_catalog_name or tool_name, "description": "fixture"}]
    targets = [
        _grounded_target(
            behavior,
            decision=index + 1,
            tool_name=tool_name if behavior != "successful_completion" else "",
            text="Completed the account update successfully." if behavior == "successful_completion" else None,
        )
        for index, behavior in enumerate(behaviors)
    ]
    replay = []
    for index in range(len(behaviors) + 1):
        following_behavior = behaviors[index] if index < len(behaviors) else None
        result_class = "success"
        if following_behavior == "empty_result_recovery":
            result_class = "empty"
        elif following_behavior == "error_result_recovery":
            result_class = "exception"
        replay.append(
            {
                "parent_assistant_decision_ordinal": index,
                "tool_name": tool_name,
                "pre_state_sha256": _canonical_sha256(f"pre:{split}:{domain}:{family}:{row_index}:{index}"),
                "post_state_sha256": _canonical_sha256(f"post:{split}:{domain}:{family}:{row_index}:{index}"),
                "state_diff": {"change_count": 1, "changed": True, "changes": []},
                "result_class": result_class,
                "context": {
                    "repeated_call": following_behavior == "repeated_call_recovery",
                },
                "evidence_replayed": True,
            }
        )
    metadata = {
        "schema_version": "hfr.tau3_grounded_generation_row.v1",
        "lineage_id": "tau3-grounded-generation-v1",
        "training_side_only": True,
        "domain": domain,
        "split": "validation" if split == "valid" else split,
        "source_family": "reviewed_synthetic",
        "source_family_id": f"{split}-{domain}-family-{family}",
        "source_id": f"{split}-{domain}-{family}-{row_index}",
        "source_sha256": _canonical_sha256(f"{split}:{domain}:{family}:{row_index}"),
        "parent_trajectory_id": f"traj-{split}-{domain}-{family}-{row_index}",
        "tau_revision": "a" * 40,
        "runtime_family": runtime_family,
        "system_prompt_sha256": _canonical_sha256(f"Tau {domain} policy prompt."),
        "tool_catalog_sha256": _canonical_sha256(tool_catalog),
        "initial_state_sha256": _canonical_sha256({"records": {}}),
        "final_state_sha256": _canonical_sha256({"records": {"changed": True}}),
        "behaviors": sorted(set(behaviors)),
        "tool_exemptions": [
            {
                "tool_name": tool_catalog_name or tool_name,
                "reason": "policy_forbidden",
                "reviewed": True,
                "grounded_validated": True,
            }
        ],
    }
    row = {
        "schema_version": "hfr.tau3_grounded_generation_row.v1",
        "trajectory": {
            "trajectory_id": metadata["parent_trajectory_id"],
            "domain": domain,
            "split": metadata["split"],
            "system_prompt": f"Tau {domain} policy prompt.",
            "turns": [
                {
                    "user": {"content": f"Please handle {domain} decision {index}."},
                    "assistant": {"decision_ordinal": index},
                }
                for index in range(len(behaviors) + 1)
            ],
        },
        "tool_catalog": tool_catalog,
        "initial_state": {"records": {}},
        "tool_replay": replay,
        "training_targets": targets,
        "metadata": metadata,
    }
    metadata["row_sha256"] = _canonical_sha256(
        {key: value for key, value in row.items() if key != "metadata"}
        | {"metadata": {k: v for k, v in metadata.items() if k != "row_sha256"}}
    )
    return row


def _write_grounded_bundle(
    root: Path,
    *,
    runtime_family: str = "vendored_tau_tools@" + "a" * 40,
    tool_catalog_name: str | None = None,
) -> Path:
    bundle = root / "grounded"
    train = []
    valid = []
    for domain in ("airline", "retail", "telecom"):
        for family in range(8):
            for row_index in range(3):
                train.append(
                    _grounded_row(
                        split="train",
                        domain=domain,
                        family=family,
                        row_index=row_index,
                        behaviors=list(BEHAVIORS),
                        runtime_family=runtime_family,
                        tool_name=f"get_{domain}_record",
                        tool_catalog_name=tool_catalog_name,
                    )
                )
        for family in range(5):
            for row_index in range(3):
                valid.append(
                    _grounded_row(
                        split="valid",
                        domain=domain,
                        family=family,
                        row_index=row_index,
                        behaviors=list(BEHAVIORS),
                        runtime_family=runtime_family,
                        tool_name=f"get_{domain}_record",
                        tool_catalog_name=tool_catalog_name,
                    )
                )
    _write_jsonl(bundle / "train.jsonl", train)
    _write_jsonl(bundle / "validation.jsonl", valid)
    manifest = {
        "schema_version": "hfr.tau3_grounded_generation.v1",
        "lineage_id": "tau3-grounded-generation-v1",
        "passed": True,
        "status": "passed",
        "blockers": [],
        "source": {"path_leaf": "fixture.jsonl", "sha256": "a" * 64, "training_side_only": True, "accepted_source_families": ["reviewed_synthetic"]},
        "files": {
            "train": {"path": "train.jsonl", "sha256": _sha256(bundle / "train.jsonl"), "bytes": (bundle / "train.jsonl").stat().st_size},
            "validation": {"path": "validation.jsonl", "sha256": _sha256(bundle / "validation.jsonl"), "bytes": (bundle / "validation.jsonl").stat().st_size},
        },
        "counts": {"train": len(train), "validation": len(valid)},
        "derivation": {"training_side_only": True},
        "coverage": {"passed": True, "blockers": [], "by_split": {}},
        "sealed_access": {"payload_accessed": False, "access_count": 0, "materialized_sealed_fields": []},
        "contamination": {"raw_sealed_payload_read": False, "split_contamination_detected": False, "train_validation_source_hash_disjoint": True},
    }
    manifest["manifest_sha256"] = _canonical_sha256(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    )
    _write_json(bundle / "manifest.json", manifest)
    return bundle


def _add_masked_negative_contexts(bundle: Path, behavior: str, negative_target: dict[str, Any]) -> None:
    for filename in ("train.jsonl", "validation.jsonl"):
        path = bundle / filename
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
        ]
        for row in rows:
            targets = row["training_targets"]
            correction = next(
                target
                for target in targets
                if target["behavior"] == behavior
            )
            negative = {
                **negative_target,
                "parent_assistant_decision_ordinal": 0,
                "safe_correction_decision_ordinal": correction["parent_assistant_decision_ordinal"],
            }
            negative["canonical_target_sha256"] = _canonical_sha256(negative["canonical_target"])
            insert_at = targets.index(correction)
            targets.insert(insert_at, negative)
            row["metadata"]["row_sha256"] = _canonical_sha256(
                {key: value for key, value in row.items() if key != "metadata"}
                | {"metadata": {k: v for k, v in row["metadata"].items() if k != "row_sha256"}}
            )
        _write_jsonl(path, rows)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    for split, filename in (("train", "train.jsonl"), ("validation", "validation.jsonl")):
        manifest["files"][split]["sha256"] = _sha256(bundle / filename)
        manifest["files"][split]["bytes"] = (bundle / filename).stat().st_size
    _rewrite_manifest(bundle / "manifest.json", manifest)


def _masked_negative_target(kind: str, domain: str) -> dict[str, Any]:
    if kind == "hallucinated_tool":
        canonical = {
            "kind": "tool_call",
            "tool_name": f"delete_{domain}_secret",
            "arguments": {"id": "forbidden"},
        }
    elif kind == "harmful_mutation":
        canonical = {
            "kind": "tool_call",
            "tool_name": f"get_{domain}_record",
            "arguments": {"id": "unsafe-mutation"},
        }
    else:
        canonical = {
            "kind": "assistant_message",
            "text": "Done before verifying the required evidence.",
            "tool_name": None,
            "arguments": {},
        }
    return {
        "parent_assistant_decision_ordinal": 0,
        "behavior": kind,
        "masked": True,
        "mask_reason": "unsafe_or_negative_action",
        "reviewed": True,
        "canonical_target": canonical,
        "negative_text": canonical.get("text") or "",
    }


def _write_contamination_report(root: Path, *, sealed_access_count: int = 0) -> Path:
    path = root / "contamination_report.json"
    _write_json(
        path,
        {
            "schema_version": "hfr.tau3_v3_contamination_report.v1",
            "passed": sealed_access_count == 0,
            "blockers": [] if sealed_access_count == 0 else ["sealed access"],
            "train_validation_source_hash_disjoint": True,
            "train_internal_family_disjoint": True,
            "train_internal_source_disjoint": True,
            "train_internal_prompt_disjoint": True,
            "development_checks_passed": True,
            "sealed_checks_passed": True,
            "sealed_access": {
                "access_count": sealed_access_count,
                "payload_accessed": False,
                "materialized_sealed_fields": [],
            },
            "raw_sealed_payload_read": False,
        },
    )
    return path


def _scenario_contamination_report() -> dict[str, Any]:
    zero_overlap = {"overlap_count": 0, "overlaps": []}
    return {
        "schema_version": "hfr.tau3_v3_scenario_contamination_report.v1",
        "passed": True,
        "blockers": [],
        "new_split_disjointness": {
            "source_id_hashes": zero_overlap,
            "family_hashes": zero_overlap,
            "prompt_hashes": zero_overlap,
        },
        "development_comparison": {
            "source_id_hashes": zero_overlap,
            "family_hashes": zero_overlap,
            "prompt_hashes": zero_overlap,
        },
        "development_hash_only_evidence": {
            "row_count": 54,
            "valid_row_count": 54,
            "malformed_row_count": 0,
            "missing_or_unreadable": False,
        },
        "sealed_hash_only_comparison": {
            "sealed_payload_access_count": 0,
            "malformed_identity_hash_count": 0,
            "malformed_prompt_hash_count": 0,
            "identity_overlap_count": 0,
            "prompt_template_overlap_count": 3,
            "prompt_template_overlap_resolved": True,
        },
    }


def _grounded_validation_patch() -> mock._patch:
    return mock.patch(
        "flightrecorder.tau3_competitive_dataset.validate_tau3_grounded_generation_bundle",
        return_value={"passed": True, "errors": []},
    )


class Tau3CompetitiveDatasetTests(unittest.TestCase):
    def test_target_export_ordinal_does_not_collapse_duplicate_targets(self) -> None:
        first = {"canonical_target_sha256": "a" * 64, "masked": False}
        second = {"canonical_target_sha256": "a" * 64, "masked": False}
        row = {"training_targets": [first, second]}

        self.assertEqual(_target_export_ordinal(row, first), 0)
        self.assertEqual(_target_export_ordinal(row, second), 1)

    def test_promotes_strict_grounded_zero_arg_exemption(self) -> None:
        self.assertEqual(
            _validated_grounded_tool_exemptions(
                {
                    "tool_exemptions": [
                        {
                            "tool_name": "list_all_airports",
                            "reason": "zero_arg",
                            "reviewed": True,
                            "reviewer": "grounded-validator",
                        }
                    ]
                }
            ),
            [
                {
                    "tool_name": "list_all_airports",
                    "reason": "zero_arg",
                    "reviewed": True,
                    "grounded_validated": True,
                }
            ],
        )

    def test_accepts_exact_v3_scenario_contamination_report(self) -> None:
        report = _scenario_contamination_report()

        self.assertEqual(_contamination_report_payload_errors(report), [])
        self.assertEqual(
            _contamination_summary(report),
            {
                "passed": True,
                "blocker_count": 0,
                "sealed_access_count": 0,
                "train_validation_source_hash_disjoint": True,
                "development_checks_passed": True,
                "sealed_checks_passed": True,
            },
        )

    def test_rejects_malformed_v3_scenario_contamination_evidence(self) -> None:
        report = _scenario_contamination_report()
        report["development_hash_only_evidence"]["valid_row_count"] = 53
        report["sealed_hash_only_comparison"]["identity_overlap_count"] = 1

        errors = _contamination_report_payload_errors(report)

        self.assertIn("contamination_report.development_checks_passed must be true", errors)
        self.assertIn("contamination_report.sealed_checks_passed must be true", errors)

    def test_projects_grounded_unicode_prompt_with_grounded_hash_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = _write_source_dataset(root)
            grounded = _write_grounded_bundle(root)
            train_path = grounded / "train.jsonl"
            rows = [
                json.loads(line)
                for line in train_path.read_text(encoding="utf-8").splitlines()
            ]
            row = next(
                item for item in rows if item["metadata"]["domain"] == "telecom"
            )
            prompt = row["trajectory"]["system_prompt"] + " Prices include €."
            row["trajectory"]["system_prompt"] = prompt
            row["metadata"]["system_prompt_sha256"] = _canonical_sha256(prompt)
            row["metadata"]["row_sha256"] = _canonical_sha256(
                {key: value for key, value in row.items() if key != "metadata"}
                | {
                    "metadata": {
                        key: value
                        for key, value in row["metadata"].items()
                        if key != "row_sha256"
                    }
                }
            )
            _write_jsonl(train_path, rows)
            manifest_path = grounded / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["files"]["train"]["sha256"] = _sha256(train_path)
            manifest["files"]["train"]["bytes"] = train_path.stat().st_size
            _rewrite_manifest(manifest_path, manifest)
            contamination = _write_contamination_report(root)
            tokenizer_config = _write_tokenizer_config(root)

            with _install_fake_transformers(_FakeTokenizer()), _grounded_validation_patch():
                built = build_tau3_competitive_dataset(
                    source_dataset_dir=source,
                    out_dir=root / "v3",
                    tokenizer_config_path=tokenizer_config,
                    grounded_generation_bundle=grounded,
                    contamination_report_path=contamination,
                )

            self.assertTrue(built["passed"])

    def test_builds_and_validates_threshold_complete_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = _write_source_dataset(root)
            tokenizer_config = _write_tokenizer_config(root)
            out = root / "v3"

            tokenizer = _FakeTokenizer()
            with _install_fake_transformers(tokenizer):
                manifest = build_tau3_competitive_dataset(
                    source_dataset_dir=source,
                    out_dir=out,
                    tokenizer_config_path=tokenizer_config,
                    include_template_supplements=True,
                )
            with _install_fake_transformers(_FakeTokenizer()):
                result = validate_tau3_competitive_dataset_bundle(out)

            self.assertEqual(manifest["schema_version"], TAU3_COMPETITIVE_DATASET_SCHEMA_VERSION)
            self.assertEqual(manifest["lineage_id"], LINEAGE_ID)
            self.assertFalse(manifest["passed"])
            self.assertFalse(result["passed"])
            self.assertIn(
                "candidate eligibility requires all rows to be grounded_generation_target",
                manifest["blockers"],
            )
            self.assertGreater(len(tokenizer.calls), 0)
            self.assertTrue(
                all(call["kwargs"].get("tokenize") is True for call in tokenizer.calls)
            )
            self.assertIn("empty_result_recovery", BEHAVIORS)
            self.assertEqual(
                0,
                result["coverage"]["by_split"]["train"]["airline"]["behavior_counts"][
                    "successful_completion"
                ],
            )
            train_rows = [
                json.loads(line)
                for line in (out / "train.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            first = train_rows[0]
            manual = _FakeTokenizer()
            full = manual.apply_chat_template(
                first["messages"],
                tools=first["tools"],
                tokenize=True,
                add_generation_prompt=False,
            )
            prompt = manual.apply_chat_template(
                first["messages"][:-1],
                tools=first["tools"],
                tokenize=True,
                add_generation_prompt=True,
            )
            self.assertEqual(
                len(prompt),
                first["metadata"]["token_counts"]["prompt_tokens"],
            )
            self.assertEqual(
                len(full) - len(prompt),
                first["metadata"]["token_counts"]["supervised_tokens"],
            )
            derived_success = [
                row
                for row in train_rows
                if row["metadata"]["source_kind"] == "derived_reviewed_safe_decision"
                and row["metadata"]["behavior"] == "successful_completion"
            ]
            self.assertEqual([], derived_success)
            assistant_text = "\n".join(
                str(message.get("content") or "")
                for row in train_rows
                if row["metadata"]["source_kind"] == "derived_reviewed_safe_decision"
                for message in row["messages"]
                if message.get("role") == "assistant"
            )
            self.assertNotIn("train-", assistant_text)
            self.assertNotIn("/", assistant_text)

    def test_validation_reports_missing_domain_blocker(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = _write_source_dataset(root, omit_domain="telecom")
            tokenizer_config = _write_tokenizer_config(root)
            out = root / "v3"

            with _install_fake_transformers(_FakeTokenizer()):
                manifest = build_tau3_competitive_dataset(
                    source_dataset_dir=source,
                    out_dir=out,
                    tokenizer_config_path=tokenizer_config,
                    include_template_supplements=True,
                )
            result = validate_tau3_competitive_dataset_bundle(out)

            self.assertFalse(manifest["passed"])
            self.assertFalse(result["passed"])
            self.assertTrue(
                any("telecom" in error for error in result["errors"]),
                result["errors"][:10],
            )

    def test_validation_rejects_ungrounded_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = _write_source_dataset(root)
            tokenizer_config = _write_tokenizer_config(root)
            out = root / "v3"
            with _install_fake_transformers(_FakeTokenizer()):
                build_tau3_competitive_dataset(
                    source_dataset_dir=source,
                    out_dir=out,
                    tokenizer_config_path=tokenizer_config,
                    include_template_supplements=True,
                )
            rows = [
                json.loads(line)
                for line in (out / "train.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            rows[0]["metadata"]["review"]["grounded"] = False
            rows[0]["metadata"]["derived_row_sha256"] = _canonical_sha256(
                {key: value for key, value in rows[0].items() if key != "metadata"}
                | {
                    "metadata": {
                        key: value
                        for key, value in rows[0]["metadata"].items()
                        if key != "derived_row_sha256"
                    }
                }
            )
            _write_jsonl(out / "train.jsonl", rows)
            manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
            manifest["files"]["train"]["sha256"] = _sha256(out / "train.jsonl")
            manifest["files"]["train"]["bytes"] = (out / "train.jsonl").stat().st_size
            manifest["manifest_sha256"] = _canonical_sha256(
                {
                    key: value
                    for key, value in manifest.items()
                    if key != "manifest_sha256"
                }
            )
            _write_json(out / "manifest.json", manifest)

            result = validate_tau3_competitive_dataset_bundle(out)

            self.assertFalse(result["passed"])
            self.assertTrue(
                any("not grounded" in error for error in result["errors"]),
                result["errors"],
            )

    def test_missing_tokenizer_config_blocks_eligibility(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = _write_source_dataset(root)
            out = root / "v3"

            manifest = build_tau3_competitive_dataset(
                source_dataset_dir=source,
                out_dir=out,
                include_template_supplements=True,
            )
            result = validate_tau3_competitive_dataset_bundle(out)

            self.assertFalse(manifest["passed"])
            self.assertFalse(result["passed"])
            self.assertTrue(
                any("token_counts" in error for error in result["errors"]),
                result["errors"][:10],
            )

    def test_lexical_counter_config_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = _write_source_dataset(root)
            tokenizer_config = _write_tokenizer_config(
                root,
                algorithm="json_chat_template_tokenizer_v1",
            )

            with self.assertRaisesRegex(Exception, "unsupported tokenizer config"):
                with _install_fake_transformers(_FakeTokenizer()):
                    build_tau3_competitive_dataset(
                        source_dataset_dir=source,
                        out_dir=root / "v3",
                        tokenizer_config_path=tokenizer_config,
                        include_template_supplements=True,
                    )

    def test_external_chat_template_asset_is_copied_and_hash_bound(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            tokenizer_config = _write_tokenizer_config(
                root,
                external_chat_template=True,
            )
            out = root / "bundle"

            with _install_fake_transformers(_FakeTokenizer()):
                token_counter = _load_token_counter(
                    tokenizer_config,
                    bundle_out_dir=out,
                    copy_into_bundle=True,
                )

            self.assertIsNotNone(token_counter)
            record = token_counter.config_record
            self.assertTrue((out / "tokenizer" / "chat_template.jinja").is_file())
            self.assertEqual(
                _sha256(out / "tokenizer" / "chat_template.jinja"),
                record["chat_template_file_sha256"],
            )
            self.assertEqual(
                record["chat_template_file_sha256"],
                record["copied_assets"]["chat_template.jinja"],
            )

    def test_real_local_base_tokenizer_external_template_is_replayable_when_available(self) -> None:
        base = Path(__file__).resolve().parents[1] / "local" / "tau3" / "models" / "base"
        if not (base / "tokenizer_config.json").is_file():
            self.skipTest("local Tau-3 base tokenizer is unavailable")
        try:
            import transformers  # noqa: F401
        except ImportError:
            self.skipTest("transformers is unavailable in this test environment")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            tokenizer_config = root / "tokenizer-config.json"
            _write_json(
                tokenizer_config,
                {
                    "schema_version": "hfr.tau3_competitive_tokenizer_config.v1",
                    "tokenizer_id": "local-tau3-base",
                    "tokenizer_revision": "local",
                    "tokenizer_path": str(base),
                    "tokenization_algorithm": "pinned_local_apply_chat_template",
                    "exact": True,
                    "chat_template_aware": True,
                },
            )
            token_counter = _load_token_counter(
                tokenizer_config,
                bundle_out_dir=root / "bundle",
                copy_into_bundle=True,
            )

            record = token_counter.config_record
            copied = root / "bundle" / "tokenizer"
            self.assertTrue((copied / "chat_template.jinja").is_file())
            self.assertEqual(
                _sha256(base / "chat_template.jinja"),
                record["chat_template_file_sha256"],
            )
            self.assertEqual(record["chat_template_file_sha256"], _sha256(copied / "chat_template.jinja"))
            self.assertTrue(record["chat_template_sha256"])

    def test_successful_completion_requires_state_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = _write_source_dataset(root)
            grounded = _write_grounded_bundle(root)
            contamination = _write_contamination_report(root)
            tokenizer_config = _write_tokenizer_config(root)
            out = root / "v3"
            with _install_fake_transformers(_FakeTokenizer()), _grounded_validation_patch():
                build_tau3_competitive_dataset(
                    source_dataset_dir=source,
                    out_dir=out,
                    tokenizer_config_path=tokenizer_config,
                    grounded_generation_bundle=grounded,
                    contamination_report_path=contamination,
                )
            rows = [
                json.loads(line)
                for line in (out / "train.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            completion = next(
                row
                for row in rows
                if row["metadata"]["behavior"] == "successful_completion"
            )
            completion["metadata"]["review"]["completion_claim_has_replayable_state"] = False
            completion["metadata"]["state_evidence_refs"]["post_state"] = None
            completion["metadata"]["derived_row_sha256"] = _canonical_sha256(
                {key: value for key, value in completion.items() if key != "metadata"}
                | {
                    "metadata": {
                        key: value
                        for key, value in completion["metadata"].items()
                        if key != "derived_row_sha256"
                    }
                }
            )
            _write_jsonl(out / "train.jsonl", rows)
            manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
            manifest["files"]["train"]["sha256"] = _sha256(out / "train.jsonl")
            manifest["files"]["train"]["bytes"] = (out / "train.jsonl").stat().st_size
            manifest["manifest_sha256"] = _canonical_sha256(
                {
                    key: value
                    for key, value in manifest.items()
                    if key != "manifest_sha256"
                }
            )
            _write_json(out / "manifest.json", manifest)

            with _install_fake_transformers(_FakeTokenizer()), _grounded_validation_patch():
                result = validate_tau3_competitive_dataset_bundle(out)

            self.assertFalse(result["passed"])
            self.assertTrue(any("successful_completion" in error for error in result["errors"]), result["errors"])

    def test_tool_result_with_completed_text_does_not_create_success_without_state_refs(self) -> None:
        for mutation_result in (False, True):
            with self.subTest(mutation_result=mutation_result):
                with tempfile.TemporaryDirectory() as temp:
                    root = Path(temp)
                    source = _write_source_dataset(
                        root,
                        completion_state_evidence=False,
                        completion_mutation_result=mutation_result,
                    )
                    tokenizer_config = _write_tokenizer_config(root)
                    out = root / "v3"
                    with _install_fake_transformers(_FakeTokenizer()):
                        manifest = build_tau3_competitive_dataset(
                            source_dataset_dir=source,
                            out_dir=out,
                            tokenizer_config_path=tokenizer_config,
                            include_template_supplements=True,
                        )
                    result = validate_tau3_competitive_dataset_bundle(out)

                    self.assertFalse(manifest["passed"])
                    self.assertFalse(result["passed"])
                    for split in ("train", "valid"):
                        for domain in ("airline", "retail", "telecom"):
                            self.assertEqual(
                                0,
                                result["coverage"]["by_split"][split][domain][
                                    "behavior_counts"
                                ]["successful_completion"],
                            )

    def test_grounded_bundle_projects_multi_decision_targets_and_exact_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = _write_source_dataset(root)
            grounded = _write_grounded_bundle(root)
            contamination = _write_contamination_report(root)
            tokenizer_config = _write_tokenizer_config(root)
            out = root / "v3"
            tokenizer = _FakeTokenizer()

            with _install_fake_transformers(tokenizer), _grounded_validation_patch():
                manifest = build_tau3_competitive_dataset(
                    source_dataset_dir=source,
                    out_dir=out,
                    tokenizer_config_path=tokenizer_config,
                    grounded_generation_bundle=grounded,
                    contamination_report_path=contamination,
                )
                result = validate_tau3_competitive_dataset_bundle(out)

            self.assertTrue(manifest["passed"], manifest["blockers"][:5])
            self.assertTrue(result["passed"], result["errors"][:5])
            self.assertEqual(8 * 3 * 3 * len(BEHAVIORS), manifest["counts"]["grounded_train_targets"])
            self.assertIn("parent_trajectories", manifest["files"])
            self.assertIn("objective_training_export", manifest["files"])
            objective_rows = [
                json.loads(line)
                for line in (out / "objective_training_export.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(
                manifest["counts"]["grounded_train_targets"] + manifest["counts"]["grounded_valid_targets"],
                len(objective_rows),
            )
            first_objective = objective_rows[0]
            self.assertEqual(
                len(first_objective["input_token_ids"]) - 1,
                len(first_objective["loss_mask"]),
            )
            offset = first_objective["token_accounting"]["prompt_tokens"]
            self.assertEqual(
                [0] * (offset - 1) + [1] * first_objective["token_accounting"]["target_tokens"],
                first_objective["loss_mask"],
            )
            self.assertEqual("mlx_lm_shifted_targets_v1", first_objective["loss_mask_semantics"])
            rows = [
                json.loads(line)
                for line in (out / "train.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            grounded_rows = [
                row for row in rows if row["metadata"]["source_kind"] == "grounded_generation_target"
            ]
            self.assertGreater(len(grounded_rows), len(BEHAVIORS))
            first = grounded_rows[0]
            manual = _FakeTokenizer()
            full = manual.apply_chat_template(
                first["messages"],
                tools=first["tools"],
                tokenize=True,
                add_generation_prompt=False,
            )
            prompt = manual.apply_chat_template(
                first["messages"][:-1],
                tools=first["tools"],
                tokenize=True,
                add_generation_prompt=True,
            )
            self.assertEqual(len(prompt), first["metadata"]["token_counts"]["prompt_tokens"])
            self.assertEqual(len(full) - len(prompt), first["metadata"]["token_counts"]["supervised_tokens"])
            self.assertFalse(
                any(row["metadata"]["source_kind"] == "derived_reviewed_safe_decision" for row in rows)
            )
            objective_report = build_tau3_objective_validity_report(
                training_export_path=out / "objective_training_export.jsonl",
                parent_trajectories_path=out / "parent_trajectories.jsonl",
                source_root=out,
            )
            self.assertTrue(objective_report["passed"], objective_report["checks"][:5])
            self.assertEqual(CONTEXT_WINDOW_TOKENS, manifest["context_window"]["max_tokens"])
            exposure = build_tau3_exposure_ledger(
                out / "train.jsonl",
                root / "exposure",
                seed=7,
                epochs=2,
                batch_size=2,
                gradient_accumulation_steps=1,
            )
            exposure_validation = validate_tau3_exposure_ledger(
                out / "train.jsonl",
                exposure["receipt_path"],
            )
            self.assertTrue(exposure_validation["passed"])
            self.assertIn("assistant_message", exposure["coverage"]["target_tools"])

    def test_objective_export_preserves_full_long_assistant_target(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = _write_source_dataset(root)
            grounded = _write_grounded_bundle(root)
            contamination = _write_contamination_report(root)
            tokenizer_config = _write_tokenizer_config(root)
            long_target = (
                "I verified the complete tool result and can now report the requested "
                "outcome without truncating the supervised assistant decision. "
            ) * 3
            train_path = grounded / "train.jsonl"
            grounded_rows = [
                json.loads(line)
                for line in train_path.read_text(encoding="utf-8").splitlines()
            ]
            completion = next(
                target
                for target in grounded_rows[0]["training_targets"]
                if target["behavior"] == "successful_completion"
            )
            completion["canonical_target"]["text"] = long_target
            completion["canonical_target_sha256"] = _canonical_sha256(
                completion["canonical_target"]
            )
            metadata = grounded_rows[0]["metadata"]
            metadata["row_sha256"] = _canonical_sha256(
                {key: value for key, value in grounded_rows[0].items() if key != "metadata"}
                | {"metadata": {key: value for key, value in metadata.items() if key != "row_sha256"}}
            )
            _write_jsonl(train_path, grounded_rows)
            grounded_manifest = json.loads(
                (grounded / "manifest.json").read_text(encoding="utf-8")
            )
            grounded_manifest["files"]["train"]["sha256"] = _sha256(train_path)
            grounded_manifest["files"]["train"]["bytes"] = train_path.stat().st_size
            _rewrite_manifest(grounded / "manifest.json", grounded_manifest)
            out = root / "v3"

            with _install_fake_transformers(_FakeTokenizer()), _grounded_validation_patch():
                manifest = build_tau3_competitive_dataset(
                    source_dataset_dir=source,
                    out_dir=out,
                    tokenizer_config_path=tokenizer_config,
                    grounded_generation_bundle=grounded,
                    contamination_report_path=contamination,
                )

            objective_rows = [
                json.loads(line)
                for line in (out / "objective_training_export.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertTrue(manifest["passed"], manifest["blockers"][:5])
            self.assertTrue(manifest["objective_validity"]["passed"])
            self.assertIn(long_target, [row["target_text"] for row in objective_rows])

    def test_objective_validity_failure_blocks_build_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = _write_source_dataset(root)
            grounded = _write_grounded_bundle(root)
            contamination = _write_contamination_report(root)
            tokenizer_config = _write_tokenizer_config(root)

            with (
                _install_fake_transformers(_FakeTokenizer()),
                _grounded_validation_patch(),
                mock.patch(
                    "flightrecorder.tau3_competitive_dataset.build_tau3_objective_validity_report",
                    return_value={
                        "passed": False,
                        "failed_check_count": 3,
                        "eligible_decision_count": 1,
                        "supervised_row_count": 1,
                    },
                ),
            ):
                manifest = build_tau3_competitive_dataset(
                    source_dataset_dir=source,
                    out_dir=root / "v3",
                    tokenizer_config_path=tokenizer_config,
                    grounded_generation_bundle=grounded,
                    contamination_report_path=contamination,
                )

            self.assertFalse(manifest["passed"])
            self.assertFalse(manifest["objective_validity"]["passed"])
            self.assertIn("objective validity failed 3 checks", manifest["blockers"])

    def test_objective_export_tamper_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = _write_source_dataset(root)
            grounded = _write_grounded_bundle(root)
            contamination = _write_contamination_report(root)
            tokenizer_config = _write_tokenizer_config(root)
            out = root / "v3"
            with _install_fake_transformers(_FakeTokenizer()), _grounded_validation_patch():
                build_tau3_competitive_dataset(
                    source_dataset_dir=source,
                    out_dir=out,
                    tokenizer_config_path=tokenizer_config,
                    grounded_generation_bundle=grounded,
                    contamination_report_path=contamination,
                )
            objective_rows = [
                json.loads(line)
                for line in (out / "objective_training_export.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            objective_rows[0]["target_sha256"] = "d" * 64
            _write_jsonl(out / "objective_training_export.jsonl", objective_rows)
            manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
            manifest["files"]["objective_training_export"]["sha256"] = _sha256(out / "objective_training_export.jsonl")
            manifest["files"]["objective_training_export"]["bytes"] = (out / "objective_training_export.jsonl").stat().st_size
            _rewrite_manifest(out / "manifest.json", manifest)

            with _install_fake_transformers(_FakeTokenizer()), _grounded_validation_patch():
                result = validate_tau3_competitive_dataset_bundle(out)

            self.assertFalse(result["passed"])
            self.assertTrue(any("objective export replay failed" in error for error in result["errors"]), result["errors"][:20])

    def test_stale_objective_export_fails_after_train_loss_mask_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = _write_source_dataset(root)
            grounded = _write_grounded_bundle(root)
            contamination = _write_contamination_report(root)
            tokenizer_config = _write_tokenizer_config(root)
            out = root / "v3"
            with _install_fake_transformers(_FakeTokenizer()), _grounded_validation_patch():
                build_tau3_competitive_dataset(
                    source_dataset_dir=source,
                    out_dir=out,
                    tokenizer_config_path=tokenizer_config,
                    grounded_generation_bundle=grounded,
                    contamination_report_path=contamination,
                )
            rows = [
                json.loads(line)
                for line in (out / "train.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            rows[0]["metadata"]["token_counts"]["loss_mask"][0] = 1
            rows[0]["metadata"]["token_counts"]["loss_mask_sha256"] = _canonical_sha256(
                rows[0]["metadata"]["token_counts"]["loss_mask"]
            )
            rows[0]["metadata"]["derived_row_sha256"] = _canonical_sha256(
                {key: value for key, value in rows[0].items() if key != "metadata"}
                | {"metadata": {k: v for k, v in rows[0]["metadata"].items() if k != "derived_row_sha256"}}
            )
            _write_jsonl(out / "train.jsonl", rows)
            manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
            manifest["files"]["train"]["sha256"] = _sha256(out / "train.jsonl")
            manifest["files"]["train"]["bytes"] = (out / "train.jsonl").stat().st_size
            _rewrite_manifest(out / "manifest.json", manifest)

            with _install_fake_transformers(_FakeTokenizer()), _grounded_validation_patch():
                result = validate_tau3_competitive_dataset_bundle(out)

            self.assertFalse(result["passed"])
            self.assertTrue(any("token_counts.loss_mask does not replay" in error for error in result["errors"]), result["errors"][:20])
            self.assertTrue(any("objective_training_export rows do not match" in error for error in result["errors"]), result["errors"][:20])

    def test_stale_parent_export_fails_deterministic_replay(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = _write_source_dataset(root)
            grounded = _write_grounded_bundle(root)
            contamination = _write_contamination_report(root)
            tokenizer_config = _write_tokenizer_config(root)
            out = root / "v3"
            with _install_fake_transformers(_FakeTokenizer()), _grounded_validation_patch():
                build_tau3_competitive_dataset(
                    source_dataset_dir=source,
                    out_dir=out,
                    tokenizer_config_path=tokenizer_config,
                    grounded_generation_bundle=grounded,
                    contamination_report_path=contamination,
                )
            parent_rows = [
                json.loads(line)
                for line in (out / "parent_trajectories.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            parent_rows[0]["assistant_decisions"][0]["target_sha256"] = "e" * 64
            _write_jsonl(out / "parent_trajectories.jsonl", parent_rows)
            manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
            manifest["files"]["parent_trajectories"]["sha256"] = _sha256(out / "parent_trajectories.jsonl")
            manifest["files"]["parent_trajectories"]["bytes"] = (out / "parent_trajectories.jsonl").stat().st_size
            _rewrite_manifest(out / "manifest.json", manifest)

            with _install_fake_transformers(_FakeTokenizer()), _grounded_validation_patch():
                result = validate_tau3_competitive_dataset_bundle(out)

            self.assertFalse(result["passed"])
            self.assertTrue(any("parent_trajectories rows do not match" in error for error in result["errors"]), result["errors"][:20])

    def test_manifest_paths_reject_traversal_and_symlink(self) -> None:
        for mode in ("traversal", "symlink"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                source = _write_source_dataset(root)
                grounded = _write_grounded_bundle(root)
                contamination = _write_contamination_report(root)
                tokenizer_config = _write_tokenizer_config(root)
                out = root / "v3"
                with _install_fake_transformers(_FakeTokenizer()), _grounded_validation_patch():
                    build_tau3_competitive_dataset(
                        source_dataset_dir=source,
                        out_dir=out,
                        tokenizer_config_path=tokenizer_config,
                        grounded_generation_bundle=grounded,
                        contamination_report_path=contamination,
                    )
                manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
                if mode == "traversal":
                    manifest["files"]["train"]["path"] = "../escape.jsonl"
                else:
                    (out / "train-link.jsonl").symlink_to(out / "train.jsonl")
                    manifest["files"]["train"]["path"] = "train-link.jsonl"
                    manifest["files"]["train"]["sha256"] = _sha256(out / "train.jsonl")
                    manifest["files"]["train"]["bytes"] = (out / "train.jsonl").stat().st_size
                _rewrite_manifest(out / "manifest.json", manifest)

                with _install_fake_transformers(_FakeTokenizer()), _grounded_validation_patch():
                    result = validate_tau3_competitive_dataset_bundle(out)

                self.assertFalse(result["passed"])
                needle = "path traversal" if mode == "traversal" else "symlink"
                self.assertTrue(any(needle in error for error in result["errors"]), result["errors"])

    def test_context_gate_drops_only_oldest_complete_prefix_units(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = _write_source_dataset(root)
            grounded = _write_grounded_bundle(root)
            train_path = grounded / "train.jsonl"
            train_rows = [
                json.loads(line)
                for line in train_path.read_text(encoding="utf-8").splitlines()
            ]
            train_rows[0]["trajectory"]["turns"][0]["user"]["content"] = "old context " * 20_000
            _write_jsonl(train_path, train_rows)
            grounded_manifest = json.loads((grounded / "manifest.json").read_text(encoding="utf-8"))
            grounded_manifest["files"]["train"]["sha256"] = _sha256(train_path)
            grounded_manifest["files"]["train"]["bytes"] = train_path.stat().st_size
            _rewrite_manifest(grounded / "manifest.json", grounded_manifest)
            contamination = _write_contamination_report(root)
            tokenizer_config = _write_tokenizer_config(root)
            out = root / "v3"

            with _install_fake_transformers(_FakeTokenizer()), _grounded_validation_patch():
                build_tau3_competitive_dataset(
                    source_dataset_dir=source,
                    out_dir=out,
                    tokenizer_config_path=tokenizer_config,
                    grounded_generation_bundle=grounded,
                    contamination_report_path=contamination,
                )
                result = validate_tau3_competitive_dataset_bundle(out)

            self.assertTrue(result["passed"], result["errors"][:10])
            rows = [
                json.loads(line)
                for line in (out / "train.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            fitted = next(
                row
                for row in rows
                if row["metadata"]["source_provenance"]["grounded_parent_row_sha256"]
                == train_rows[0]["metadata"]["row_sha256"]
                and row["metadata"]["source_target_ordinal"] >= 2
            )
            self.assertGreater(
                fitted["metadata"]["context_window"]["excluded_oldest_complete_interaction_units"],
                0,
            )
            self.assertLessEqual(
                fitted["metadata"]["token_counts"]["total_tokens"],
                CONTEXT_WINDOW_TOKENS,
            )
            self.assertIn(
                "Please handle airline decision 2.",
                [message.get("content") for message in fitted["messages"] if message.get("role") == "user"],
            )

    def test_grounded_prompt_and_replay_refs_exclude_future_results(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = _write_source_dataset(root)
            grounded = _write_grounded_bundle(root)
            contamination = _write_contamination_report(root)
            tokenizer_config = _write_tokenizer_config(root)
            out = root / "v3"
            with _install_fake_transformers(_FakeTokenizer()), _grounded_validation_patch():
                build_tau3_competitive_dataset(
                    source_dataset_dir=source,
                    out_dir=out,
                    tokenizer_config_path=tokenizer_config,
                    grounded_generation_bundle=grounded,
                    contamination_report_path=contamination,
                )
            rows = [
                json.loads(line)
                for line in (out / "train.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            target = next(
                row
                for row in rows
                if row["metadata"]["domain"] == "airline"
                and row["metadata"]["source_target_ordinal"] == 2
                and row["metadata"]["target_tool_name"]
            )
            user_messages = [
                message.get("content")
                for message in target["messages"]
                if message.get("role") == "user"
            ]
            self.assertIn("Please handle airline decision 2.", user_messages)
            self.assertNotIn("Please handle airline decision 3.", user_messages)
            assistant_messages = [
                str(message.get("content") or "")
                for message in target["messages"]
                if message.get("role") == "assistant"
            ]
            self.assertIn("Completed the account update successfully.", assistant_messages)
            self.assertNotIn("Handled authentication.", assistant_messages)
            tool_results = [
                message
                for message in target["messages"]
                if message.get("role") == "tool"
            ]
            self.assertTrue(all("decision 2" not in str(message.get("content")) for message in tool_results))
            grounded_source_rows = [
                json.loads(line)
                for line in (grounded / "train.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            source_parent = next(
                row
                for row in grounded_source_rows
                if row["metadata"]["row_sha256"]
                == target["metadata"]["source_provenance"]["grounded_parent_row_sha256"]
            )
            expected_target_call_hash = _canonical_sha256(
                [
                    call
                    for call in source_parent["tool_replay"]
                    if call["parent_assistant_decision_ordinal"] == 2
                ]
            )
            self.assertEqual(
                expected_target_call_hash,
                target["metadata"]["source_provenance"]["tool_replay_sha256"],
            )

    def test_masked_negative_contexts_are_prompt_only_for_safe_corrections(self) -> None:
        cases = (
            ("hallucinated_tool_correction", "hallucinated_tool", "delete_airline_secret"),
            ("harmful_mutation_correction", "harmful_mutation", "unsafe-mutation"),
            ("premature_completion_correction", "premature_completion", "Done before verifying"),
        )
        for correction_behavior, negative_kind, expected_text in cases:
            with self.subTest(correction_behavior=correction_behavior), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                source = _write_source_dataset(root)
                grounded = _write_grounded_bundle(root)
                _add_masked_negative_contexts(
                    grounded,
                    correction_behavior,
                    _masked_negative_target(negative_kind, "airline"),
                )
                contamination = _write_contamination_report(root)
                tokenizer_config = _write_tokenizer_config(root)
                out = root / "v3"

                with _install_fake_transformers(_FakeTokenizer()), _grounded_validation_patch():
                    build_tau3_competitive_dataset(
                        source_dataset_dir=source,
                        out_dir=out,
                        tokenizer_config_path=tokenizer_config,
                        grounded_generation_bundle=grounded,
                        contamination_report_path=contamination,
                    )
                    result = validate_tau3_competitive_dataset_bundle(out)

                self.assertTrue(result["passed"], result["errors"][:10])
                rows = [
                    json.loads(line)
                    for line in (out / "train.jsonl").read_text(encoding="utf-8").splitlines()
                ]
                correction = next(
                    row
                    for row in rows
                    if row["metadata"]["domain"] == "airline"
                    and row["metadata"]["behavior"] == correction_behavior
                )
                rendered_prompt = json.dumps(correction["messages"][:-1], sort_keys=True)
                rendered_target = json.dumps(correction["messages"][-1], sort_keys=True)
                self.assertIn(expected_text, rendered_prompt)
                self.assertNotIn(expected_text, rendered_target)
                self.assertEqual("safe_correction", next(
                    row
                    for row in [
                        json.loads(line)
                        for line in (out / "objective_training_export.jsonl").read_text(encoding="utf-8").splitlines()
                    ]
                    if row["row_id"] == correction["metadata"]["derived_row_sha256"]
                )["target_kind"])

    def test_masked_negative_context_requires_review_and_later_safe_linkage(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = _write_source_dataset(root)
            grounded = _write_grounded_bundle(root)
            bad_negative = _masked_negative_target("harmful_mutation", "airline")
            bad_negative.pop("reviewed")
            _add_masked_negative_contexts(
                grounded,
                "harmful_mutation_correction",
                bad_negative,
            )
            contamination = _write_contamination_report(root)
            tokenizer_config = _write_tokenizer_config(root)

            with self.assertRaisesRegex(Exception, "masked negative target must be explicitly reviewed"):
                with _install_fake_transformers(_FakeTokenizer()), _grounded_validation_patch():
                    build_tau3_competitive_dataset(
                        source_dataset_dir=source,
                        out_dir=root / "v3",
                        tokenizer_config_path=tokenizer_config,
                        grounded_generation_bundle=grounded,
                        contamination_report_path=contamination,
                    )

    def test_fixed_schema_tool_argument_dominance_uses_payload_identity(self) -> None:
        rows = []
        for index in range(8):
            rows.append(
                {
                    "metadata": {
                        "canonical_target_sha256": f"{index:064x}",
                        "source_family_id": f"family-{index}",
                        "target_action_class": "tool_call",
                        "target_tool_name": "get_airline_record",
                        "canonical_target": {
                            "kind": "tool_call",
                            "tool_name": "get_airline_record",
                            "arguments": {"id": f"acct-{index}"},
                            "arguments_sha256": f"{index + 1:064x}",
                        },
                    }
                }
            )
        blockers: list[str] = []
        _add_dominance_blockers(blockers, "train", "airline", rows)
        self.assertFalse(any("argument_template_share" in blocker for blocker in blockers), blockers)
        repeated = copy.deepcopy(rows)
        for row in repeated:
            row["metadata"]["canonical_target"]["arguments"] = {"id": "acct-0"}
            row["metadata"]["canonical_target"]["arguments_sha256"] = "1" * 64
        blockers = []
        _add_dominance_blockers(blockers, "train", "airline", repeated)
        self.assertTrue(any("argument_template_share" in blocker for blocker in blockers), blockers)

    def test_tampered_token_counts_are_recomputed_and_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = _write_source_dataset(root)
            grounded = _write_grounded_bundle(root)
            contamination = _write_contamination_report(root)
            tokenizer_config = _write_tokenizer_config(root)
            out = root / "v3"
            with _install_fake_transformers(_FakeTokenizer()), _grounded_validation_patch():
                build_tau3_competitive_dataset(
                    source_dataset_dir=source,
                    out_dir=out,
                    tokenizer_config_path=tokenizer_config,
                    grounded_generation_bundle=grounded,
                    contamination_report_path=contamination,
                )
            rows = [
                json.loads(line)
                for line in (out / "train.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            rows[0]["metadata"]["token_counts"]["supervised_tokens"] += 1
            rows[0]["metadata"]["derived_row_sha256"] = _canonical_sha256(
                {key: value for key, value in rows[0].items() if key != "metadata"}
                | {"metadata": {k: v for k, v in rows[0]["metadata"].items() if k != "derived_row_sha256"}}
            )
            _write_jsonl(out / "train.jsonl", rows)
            manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
            manifest["files"]["train"]["sha256"] = _sha256(out / "train.jsonl")
            manifest["files"]["train"]["bytes"] = (out / "train.jsonl").stat().st_size
            manifest["manifest_sha256"] = _canonical_sha256(
                {key: value for key, value in manifest.items() if key != "manifest_sha256"}
            )
            _write_json(out / "manifest.json", manifest)

            with _install_fake_transformers(_FakeTokenizer()), _grounded_validation_patch():
                result = validate_tau3_competitive_dataset_bundle(out)

            self.assertFalse(result["passed"])
            self.assertTrue(
                any("token_counts.supervised_tokens does not replay" in error for error in result["errors"]),
                result["errors"][:20],
            )

    def test_grounded_bundle_tamper_is_rejected_by_competitive_validator(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = _write_source_dataset(root)
            grounded = _write_grounded_bundle(root)
            contamination = _write_contamination_report(root)
            tokenizer_config = _write_tokenizer_config(root)
            out = root / "v3"
            with _install_fake_transformers(_FakeTokenizer()), _grounded_validation_patch():
                build_tau3_competitive_dataset(
                    source_dataset_dir=source,
                    out_dir=out,
                    tokenizer_config_path=tokenizer_config,
                    grounded_generation_bundle=grounded,
                    contamination_report_path=contamination,
                )
            manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
            manifest["grounded_generation"]["manifest_sha256"] = "b" * 64
            manifest["manifest_sha256"] = _canonical_sha256(
                {key: value for key, value in manifest.items() if key != "manifest_sha256"}
            )
            _write_json(out / "manifest.json", manifest)

            with _grounded_validation_patch():
                result = validate_tau3_competitive_dataset_bundle(out)

            self.assertFalse(result["passed"])
            self.assertTrue(any("grounded_generation manifest hash" in error for error in result["errors"]))

    def test_grounded_recovery_condition_tamper_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = _write_source_dataset(root)
            grounded = _write_grounded_bundle(root)
            contamination = _write_contamination_report(root)
            tokenizer_config = _write_tokenizer_config(root)
            out = root / "v3"
            with _install_fake_transformers(_FakeTokenizer()), _grounded_validation_patch():
                build_tau3_competitive_dataset(
                    source_dataset_dir=source,
                    out_dir=out,
                    tokenizer_config_path=tokenizer_config,
                    grounded_generation_bundle=grounded,
                    contamination_report_path=contamination,
                )
            rows = [
                json.loads(line)
                for line in (out / "train.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            mutations = {
                "empty_result_recovery": ("preceding_result_class", "success"),
                "error_result_recovery": ("preceding_result_class", "success"),
                "repeated_call_recovery": ("preceding_repeated_call", False),
            }
            for behavior, (field, value) in mutations.items():
                row = next(
                    candidate
                    for candidate in rows
                    if candidate["metadata"]["behavior"] == behavior
                )
                row["metadata"][field] = value
                row["metadata"]["derived_row_sha256"] = _canonical_sha256(
                    {key: item for key, item in row.items() if key != "metadata"}
                    | {
                        "metadata": {
                            key: item
                            for key, item in row["metadata"].items()
                            if key != "derived_row_sha256"
                        }
                    }
                )
            _write_jsonl(out / "train.jsonl", rows)
            manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
            manifest["files"]["train"]["sha256"] = _sha256(out / "train.jsonl")
            manifest["files"]["train"]["bytes"] = (out / "train.jsonl").stat().st_size
            _rewrite_manifest(out / "manifest.json", manifest)

            with _install_fake_transformers(_FakeTokenizer()), _grounded_validation_patch():
                result = validate_tau3_competitive_dataset_bundle(out)

            self.assertFalse(result["passed"])
            self.assertTrue(
                any("empty_result_recovery must follow an empty" in error for error in result["errors"]),
                result["errors"][:30],
            )
            self.assertTrue(
                any("error_result_recovery must follow an error" in error for error in result["errors"]),
                result["errors"][:30],
            )
            self.assertTrue(
                any("repeated_call_recovery must follow" in error for error in result["errors"]),
                result["errors"][:30],
            )

    def test_external_grounded_validator_is_used_for_build(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = _write_source_dataset(root)
            grounded = _write_grounded_bundle(root)
            contamination = _write_contamination_report(root)
            tokenizer_config = _write_tokenizer_config(root)
            out = root / "v3"
            completed = types.SimpleNamespace(
                returncode=0,
                stdout=json.dumps(
                    {
                        "schema_version": "hfr.validation.v1",
                        "target": str(grounded),
                        "passed": True,
                        "strict": True,
                        "errors": [],
                    }
                ),
                stderr="",
            )
            with _install_fake_transformers(_FakeTokenizer()), mock.patch(
                "flightrecorder.tau3_competitive_dataset.subprocess.run",
                return_value=completed,
            ) as run:
                manifest = build_tau3_competitive_dataset(
                    source_dataset_dir=source,
                    out_dir=out,
                    tokenizer_config_path=tokenizer_config,
                    grounded_generation_bundle=grounded,
                    grounded_validator_python=sys.executable,
                    contamination_report_path=contamination,
                )

            self.assertTrue(manifest["grounded_generation"]["strict_validation_passed"])
            args, kwargs = run.call_args
            self.assertEqual(Path(args[0][0]), Path(sys.executable).absolute())
            self.assertTrue(str(args[0][1]).endswith("scripts/validate_tau3_grounded_generation.py"))
            self.assertEqual(Path(args[0][2]), grounded)
            env = kwargs["env"]
            self.assertEqual(env["PYTHONNOUSERSITE"], "1")
            self.assertIn(str(Path(__file__).resolve().parents[1]), env["PYTHONPATH"].split(":"))
            self.assertEqual(kwargs["timeout"], GROUNDED_VALIDATOR_TIMEOUT_SECONDS)

    def test_external_grounded_validator_timeout_rejects_build(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = _write_source_dataset(root)
            grounded = _write_grounded_bundle(root)
            tokenizer_config = _write_tokenizer_config(root)
            timeout = subprocess.TimeoutExpired(
                cmd=[sys.executable, "validate_tau3_grounded_generation.py"],
                timeout=GROUNDED_VALIDATOR_TIMEOUT_SECONDS,
            )

            with _install_fake_transformers(_FakeTokenizer()), mock.patch(
                "flightrecorder.tau3_competitive_dataset.subprocess.run",
                side_effect=timeout,
            ):
                with self.assertRaisesRegex(Exception, "grounded validator subprocess failed"):
                    build_tau3_competitive_dataset(
                        source_dataset_dir=source,
                        out_dir=root / "v3",
                        tokenizer_config_path=tokenizer_config,
                        grounded_generation_bundle=grounded,
                        grounded_validator_python=sys.executable,
                    )

    def test_external_grounded_validator_failure_rejects_build(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = _write_source_dataset(root)
            grounded = _write_grounded_bundle(root)
            tokenizer_config = _write_tokenizer_config(root)
            completed = types.SimpleNamespace(returncode=9, stdout="", stderr="tau import failed")

            with _install_fake_transformers(_FakeTokenizer()), mock.patch(
                "flightrecorder.tau3_competitive_dataset.subprocess.run",
                return_value=completed,
            ):
                with self.assertRaisesRegex(Exception, "grounded validator subprocess failed"):
                    build_tau3_competitive_dataset(
                        source_dataset_dir=source,
                        out_dir=root / "v3",
                        tokenizer_config_path=tokenizer_config,
                        grounded_generation_bundle=grounded,
                        grounded_validator_python=sys.executable,
                    )

    def test_external_grounded_validator_tamper_failure_rejects_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = _write_source_dataset(root)
            grounded = _write_grounded_bundle(root)
            contamination = _write_contamination_report(root)
            tokenizer_config = _write_tokenizer_config(root)
            out = root / "v3"
            ok = types.SimpleNamespace(
                returncode=0,
                stdout=json.dumps(
                    {
                        "schema_version": "hfr.validation.v1",
                        "target": str(grounded),
                        "passed": True,
                        "strict": True,
                        "errors": [],
                    }
                ),
                stderr="",
            )
            with _install_fake_transformers(_FakeTokenizer()), mock.patch(
                "flightrecorder.tau3_competitive_dataset.subprocess.run",
                return_value=ok,
            ):
                build_tau3_competitive_dataset(
                    source_dataset_dir=source,
                    out_dir=out,
                    tokenizer_config_path=tokenizer_config,
                    grounded_generation_bundle=grounded,
                    grounded_validator_python=sys.executable,
                    contamination_report_path=contamination,
                )

            failed = types.SimpleNamespace(
                returncode=1,
                stdout=json.dumps(
                    {
                        "schema_version": "hfr.validation.v1",
                        "target": str(out / "evidence" / "grounded_generation"),
                        "passed": False,
                        "strict": True,
                        "errors": ["grounded train file hash does not replay"],
                    }
                ),
                stderr="noisy dependency logs",
            )
            with _install_fake_transformers(_FakeTokenizer()), mock.patch(
                "flightrecorder.tau3_competitive_dataset.subprocess.run",
                return_value=failed,
            ):
                result = validate_tau3_competitive_dataset_bundle(
                    out,
                    grounded_validator_python=sys.executable,
                )

            self.assertFalse(result["passed"])
            self.assertTrue(any("grounded_generation strict replay failed" in error for error in result["errors"]))
            self.assertTrue(any("grounded train file hash does not replay" in error for error in result["errors"]))

    def test_grounded_fake_runtime_rejected_without_patch(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = _write_source_dataset(root)
            grounded = _write_grounded_bundle(root, runtime_family="fake_test_tau_tools")
            tokenizer_config = _write_tokenizer_config(root)

            with self.assertRaisesRegex(Exception, "grounded generation bundle failed strict validation"):
                with _install_fake_transformers(_FakeTokenizer()):
                    build_tau3_competitive_dataset(
                        source_dataset_dir=source,
                        out_dir=root / "v3",
                        tokenizer_config_path=tokenizer_config,
                        grounded_generation_bundle=grounded,
                    )

    def test_grounded_row_target_and_catalog_mismatch_rejected(self) -> None:
        for field in ("canonical_target_sha256", "parent_tools_sha256"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                source = _write_source_dataset(root)
                grounded = _write_grounded_bundle(root)
                contamination = _write_contamination_report(root)
                tokenizer_config = _write_tokenizer_config(root)
                out = root / "v3"
                with _install_fake_transformers(_FakeTokenizer()), _grounded_validation_patch():
                    build_tau3_competitive_dataset(
                        source_dataset_dir=source,
                        out_dir=out,
                        tokenizer_config_path=tokenizer_config,
                        grounded_generation_bundle=grounded,
                        contamination_report_path=contamination,
                    )
                rows = [
                    json.loads(line)
                    for line in (out / "train.jsonl").read_text(encoding="utf-8").splitlines()
                ]
                row = next(row for row in rows if row["metadata"]["source_kind"] == "grounded_generation_target")
                row["metadata"][field] = "c" * 64
                row["metadata"]["derived_row_sha256"] = _canonical_sha256(
                    {key: value for key, value in row.items() if key != "metadata"}
                    | {"metadata": {k: v for k, v in row["metadata"].items() if k != "derived_row_sha256"}}
                )
                _write_jsonl(out / "train.jsonl", rows)
                manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
                manifest["files"]["train"]["sha256"] = _sha256(out / "train.jsonl")
                manifest["files"]["train"]["bytes"] = (out / "train.jsonl").stat().st_size
                manifest["manifest_sha256"] = _canonical_sha256(
                    {key: value for key, value in manifest.items() if key != "manifest_sha256"}
                )
                _write_json(out / "manifest.json", manifest)

                with _install_fake_transformers(_FakeTokenizer()), _grounded_validation_patch():
                    result = validate_tau3_competitive_dataset_bundle(out)

                self.assertFalse(result["passed"])
                self.assertTrue(
                    any("grounded row" in error and ("binding mismatch" in error or "tool catalog hash mismatch" in error) for error in result["errors"]),
                    result["errors"][:20],
                )

    def test_schema_file_is_valid_json(self) -> None:
        schema_path = (
            Path(__file__).resolve().parents[1]
            / "flightrecorder"
            / "schemas"
            / "tau3_competitive_dataset.v1.schema.json"
        )
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        self.assertEqual(
            schema["properties"]["schema_version"]["const"],
            TAU3_COMPETITIVE_DATASET_SCHEMA_VERSION,
        )


if __name__ == "__main__":
    unittest.main()
