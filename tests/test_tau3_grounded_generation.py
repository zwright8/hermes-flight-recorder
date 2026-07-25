from __future__ import annotations

import json
import tempfile
import unittest
from datetime import date, datetime, time
from enum import Enum
from pathlib import Path
from typing import Any

from flightrecorder.tau3_grounded_generation import (
    BEHAVIORS,
    LINEAGE_ID,
    TAU3_GROUNDED_DATASET_SCHEMA_VERSION,
    Tau3GroundedGenerationError,
    _model_to_json,
    build_tau3_grounded_generation_dataset,
    canonical_sha256,
    validate_tau3_grounded_generation_bundle,
)


TAU_REVISION = "a" * 40


class _UnitEnum(Enum):
    ACTIVE = "Active"


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _scenario(
    *,
    split: str,
    domain: str,
    family_index: int,
    behavior: str,
    source_id: str | None = None,
    unsafe: bool = False,
    fabricated_success: bool = False,
) -> dict[str, Any]:
    family_id = f"{split}-{domain}-family-{family_index}"
    row_id = source_id or f"{split}-{domain}-{family_index}-{behavior}"
    record_id = f"{domain}-{family_index}"
    tool_calls = [{"tool_name": "get_record", "arguments": {"id": record_id}}]
    target_text = f"Handled {behavior} for {domain}."
    target: dict[str, Any] = {
        "behavior": behavior,
        "kind": "assistant_message",
        "text": target_text,
    }
    if behavior == "successful_completion":
        if fabricated_success:
            target["text"] = "Completed the account update successfully."
        else:
            tool_calls = [
                {
                    "tool_name": "update_record",
                    "arguments": {"id": record_id, "patch": {"status": "updated"}},
                }
            ]
            target["text"] = "Completed the account update successfully."
    elif behavior == "empty_result_recovery":
        tool_calls = [{"tool_name": "empty_search", "arguments": {}}]
    elif behavior == "error_result_recovery":
        tool_calls = [{"tool_name": "raise_tool_exception", "arguments": {"message": "boom"}}]
    elif behavior == "repeated_call_recovery":
        tool_calls = [
            {"tool_name": "get_record", "arguments": {"id": record_id}},
            {"tool_name": "get_record", "arguments": {"id": record_id}},
        ]
    elif behavior in {"harmful_mutation_correction", "premature_completion_correction"}:
        target.update(
            {
                "tool_name": "update_record",
                "arguments": {"id": record_id, "patch": {"status": "unsafe"}},
                "requires_confirmation": True,
            }
        )
        if unsafe:
            target["unsafe"] = True
            target["mask_reason"] = "unsafe_or_negative_action"
    return {
        "trajectory_id": f"traj-{row_id}",
        "domain": domain,
        "split": split,
        "source_family": "reviewed_synthetic",
        "source_family_id": family_id,
        "source_id": row_id,
        "tau_revision": TAU_REVISION,
        "runtime_family": "fake_test_tau_tools",
        "system_prompt": f"Tau {domain} policy prompt.",
        "initial_state": {
            "records": {record_id: {"id": record_id, "status": "open"}},
            "notes": [],
        },
        "turns": [
            {
                "user": {"content": f"Please handle {behavior}."},
                "assistant": {
                    "decision_ordinal": 0,
                    "tool_calls": tool_calls,
                    "safe_corrected_target": target,
                },
            }
        ],
        "recipe": {"id": "unit-recipe", "sha256": canonical_sha256("recipe")},
        "teacher": {"id": "unit-teacher", "sha256": canonical_sha256("teacher")},
        "reviewer": {"id": "unit-reviewer", "sha256": canonical_sha256("reviewer")},
        "redaction": {"passed": True, "method": "synthetic-no-pii"},
        "contamination": {
            "source_split": split,
            "raw_sealed_payload_read": False,
            "sealed_hash_only": True,
        },
    }


def _complete_source(path: Path) -> None:
    rows: list[dict[str, Any]] = []
    for domain in ("airline", "retail", "telecom"):
        for split, family_count in (("train", 8), ("validation", 2)):
            for family_index in range(family_count):
                for behavior in BEHAVIORS:
                    rows.append(
                        _scenario(
                            split=split,
                            domain=domain,
                            family_index=family_index,
                            behavior=behavior,
                            unsafe=behavior in {"harmful_mutation_correction", "premature_completion_correction"},
                        )
                    )
    _write_jsonl(path, rows)


def _rewrite_bundle_manifest(bundle: Path) -> None:
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    for split in ("train", "validation"):
        path = bundle / f"{split}.jsonl"
        manifest["files"][split]["sha256"] = _sha256(path)
        manifest["files"][split]["bytes"] = path.stat().st_size
    manifest["manifest_sha256"] = canonical_sha256(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    )
    _write_json(bundle / "manifest.json", manifest)


class Tau3GroundedGenerationTests(unittest.TestCase):
    def test_model_to_json_normalizes_scalar_dates_times_and_enums(self) -> None:
        payload = {
            "date": date(2026, 7, 25),
            "datetime": datetime(2026, 7, 25, 12, 3, 4),
            "time": time(12, 3, 4),
            "enum": _UnitEnum.ACTIVE,
        }

        normalized = _model_to_json(payload)

        self.assertEqual(
            normalized,
            {
                "date": "2026-07-25",
                "datetime": "2026-07-25T12:03:04",
                "time": "12:03:04",
                "enum": "Active",
            },
        )
        self.assertIsInstance(canonical_sha256(normalized), str)

    def test_complete_fake_fixture_builds_blocked_not_candidate_eligible(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.jsonl"
            _complete_source(source)

            manifest = build_tau3_grounded_generation_dataset(
                source=source,
                out_dir=root / "out",
                strict_coverage=False,
            )
            result = validate_tau3_grounded_generation_bundle(root / "out", strict=False)

            self.assertEqual(manifest["schema_version"], TAU3_GROUNDED_DATASET_SCHEMA_VERSION)
            self.assertEqual(manifest["lineage_id"], LINEAGE_ID)
            self.assertFalse(manifest["passed"])
            self.assertFalse(result["passed"])
            self.assertTrue(
                any("candidate-eligible vendored Tau replay" in error for error in result["errors"]),
                result["errors"][:5],
            )
            self.assertEqual(result["coverage"]["by_split"]["train"]["airline"]["source_family_count"], 8)
            self.assertEqual(result["coverage"]["by_split"]["validation"]["airline"]["source_family_count"], 2)
            row = json.loads((root / "out" / "train.jsonl").read_text(encoding="utf-8").splitlines()[0])
            self.assertNotIn("initial_state", row)
            self.assertIn("initial_state_ref", row)

    def test_shared_initial_states_are_deduplicated_into_one_blob(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.jsonl"
            first = _scenario(
                split="train",
                domain="airline",
                family_index=0,
                behavior="authentication",
            )
            second = _scenario(
                split="train",
                domain="airline",
                family_index=1,
                behavior="clarification_refusal",
            )
            second["initial_state"] = json.loads(json.dumps(first["initial_state"]))
            _write_jsonl(source, [first, second])

            build_tau3_grounded_generation_dataset(
                source=source,
                out_dir=root / "out",
                strict_coverage=False,
            )

            state_blobs = list((root / "out" / "states").glob("*.json"))
            rows = [
                json.loads(line)
                for line in (root / "out" / "train.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(state_blobs), 1)
            self.assertEqual(rows[0]["initial_state_ref"], rows[1]["initial_state_ref"])

    def test_validation_rejects_tampered_and_traversing_state_refs(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.jsonl"
            _write_jsonl(
                source,
                [
                    _scenario(
                        split="train",
                        domain="airline",
                        family_index=0,
                        behavior="authentication",
                    )
                ],
            )
            out = root / "out"
            build_tau3_grounded_generation_dataset(
                source=source,
                out_dir=out,
                strict_coverage=False,
            )
            state_path = next((out / "states").glob("*.json"))
            state = json.loads(state_path.read_text(encoding="utf-8"))
            state["records"]["tampered"] = {"id": "tampered"}
            _write_json(state_path, state)

            result = validate_tau3_grounded_generation_bundle(out, strict=False)

            self.assertFalse(result["passed"])
            self.assertTrue(
                any("initial_state_ref.sha256 does not replay" in error for error in result["errors"]),
                result["errors"],
            )

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.jsonl"
            _write_jsonl(
                source,
                [
                    _scenario(
                        split="train",
                        domain="airline",
                        family_index=0,
                        behavior="authentication",
                    )
                ],
            )
            out = root / "out"
            build_tau3_grounded_generation_dataset(
                source=source,
                out_dir=out,
                strict_coverage=False,
            )
            result = validate_tau3_grounded_generation_bundle(out, strict=False)

            self.assertFalse(result["passed"])
            self.assertFalse(
                any("runtime cannot be instantiated" in error for error in result["errors"]),
                result["errors"],
            )
            self.assertFalse(
                any("tool_replay" in error for error in result["errors"]),
                result["errors"],
            )
            rows = [
                json.loads(line)
                for line in (out / "train.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            rows[0]["initial_state_ref"]["path"] = "../escaped.json"
            rows[0]["metadata"]["row_sha256"] = canonical_sha256(
                {
                    key: value
                    for key, value in rows[0].items()
                    if key != "metadata"
                }
                | {
                    "metadata": {
                        key: value
                        for key, value in rows[0]["metadata"].items()
                        if key != "row_sha256"
                    }
                }
            )
            _write_jsonl(out / "train.jsonl", rows)
            _rewrite_bundle_manifest(out)

            result = validate_tau3_grounded_generation_bundle(out, strict=False)

            self.assertFalse(result["passed"])
            self.assertTrue(
                any("initial_state_ref.path must be a safe relative path" in error for error in result["errors"]),
                result["errors"],
            )

    def test_validation_replays_and_rejects_tampered_state_result_tool_and_args(self) -> None:
        tamper_cases = (
            ("arguments_sha256", lambda call: call.update({"canonical_arguments": {"id": "missing"}})),
            ("canonical_result", lambda call: call.update({"canonical_result": {"id": "tampered"}})),
            ("tool_definition_sha256", lambda call: call.update({"tool_definition_sha256": "b" * 64})),
            ("post_state_sha256", lambda call: call.update({"post_state_sha256": "c" * 64})),
        )
        for label, mutate in tamper_cases:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                source = root / "source.jsonl"
                _complete_source(source)
                out = root / "out"
                build_tau3_grounded_generation_dataset(
                    source=source,
                    out_dir=out,
                    strict_coverage=False,
                )
                rows = [
                    json.loads(line)
                    for line in (out / "train.jsonl").read_text(encoding="utf-8").splitlines()
                ]
                mutate(rows[0]["tool_replay"][0])
                rows[0]["metadata"]["row_sha256"] = canonical_sha256(
                    {
                        key: value
                        for key, value in rows[0].items()
                        if key != "metadata"
                    }
                    | {
                        "metadata": {
                            key: value
                            for key, value in rows[0]["metadata"].items()
                            if key != "row_sha256"
                        }
                    }
                )
                _write_jsonl(out / "train.jsonl", rows)
                _rewrite_bundle_manifest(out)

                result = validate_tau3_grounded_generation_bundle(out)

                self.assertFalse(result["passed"])
                self.assertTrue(
                    any(label in error for error in result["errors"]),
                    result["errors"][:10],
                )

    def test_source_expected_result_hash_and_class_are_verified(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.jsonl"
            row = _scenario(
                split="train",
                domain="airline",
                family_index=0,
                behavior="authentication",
            )
            expected = {
                "id": "airline-0",
                "status": "open",
            }
            row["turns"][0]["assistant"]["tool_calls"][0]["expected_result_sha256"] = canonical_sha256(expected)
            row["turns"][0]["assistant"]["tool_calls"][0]["expected_result_class"] = "success"
            _write_jsonl(source, [row])
            out = root / "out"

            build_tau3_grounded_generation_dataset(
                source=source,
                out_dir=out,
                strict_coverage=False,
            )

            exported = json.loads((out / "train.jsonl").read_text(encoding="utf-8").splitlines()[0])
            evidence = exported["tool_replay"][0]
            self.assertTrue(evidence["source_expected_result_verified"])
            self.assertEqual(evidence["source_expected_result_sha256"], canonical_sha256(expected))
            self.assertEqual(evidence["source_expected_result_class"], "success")

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.jsonl"
            row = _scenario(
                split="train",
                domain="airline",
                family_index=0,
                behavior="authentication",
            )
            row["turns"][0]["assistant"]["tool_calls"][0]["expected_result_sha256"] = "0" * 64
            _write_jsonl(source, [row])

            with self.assertRaises(Tau3GroundedGenerationError):
                build_tau3_grounded_generation_dataset(
                    source=source,
                    out_dir=root / "out",
                    strict_coverage=False,
                )

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.jsonl"
            row = _scenario(
                split="train",
                domain="airline",
                family_index=0,
                behavior="empty_result_recovery",
            )
            row["turns"][0]["assistant"]["tool_calls"][0]["expected_result_class"] = "success"
            _write_jsonl(source, [row])

            with self.assertRaises(Tau3GroundedGenerationError):
                build_tau3_grounded_generation_dataset(
                    source=source,
                    out_dir=root / "out",
                    strict_coverage=False,
                )

    def test_validator_rejects_tampered_source_expected_result_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.jsonl"
            row = _scenario(
                split="train",
                domain="airline",
                family_index=0,
                behavior="authentication",
            )
            expected = {"id": "airline-0", "status": "open"}
            row["turns"][0]["assistant"]["tool_calls"][0]["expected_result_sha256"] = canonical_sha256(expected)
            _write_jsonl(source, [row])
            out = root / "out"
            build_tau3_grounded_generation_dataset(
                source=source,
                out_dir=out,
                strict_coverage=False,
            )
            rows = [
                json.loads(line)
                for line in (out / "train.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            rows[0]["tool_replay"][0]["source_expected_result_sha256"] = "1" * 64
            rows[0]["metadata"]["row_sha256"] = canonical_sha256(
                {
                    key: value
                    for key, value in rows[0].items()
                    if key != "metadata"
                }
                | {
                    "metadata": {
                        key: value
                        for key, value in rows[0]["metadata"].items()
                        if key != "row_sha256"
                    }
                }
            )
            _write_jsonl(out / "train.jsonl", rows)
            _rewrite_bundle_manifest(out)

            result = validate_tau3_grounded_generation_bundle(out, strict=False)

            self.assertFalse(result["passed"])
            self.assertTrue(
                any("expected_result_sha256 does not match replayed result" in error for error in result["errors"]),
                result["errors"],
            )

    def test_validation_rejects_fabricated_success_without_replayed_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.jsonl"
            _write_jsonl(
                source,
                [
                    _scenario(
                        split="train",
                        domain="airline",
                        family_index=0,
                        behavior="successful_completion",
                        fabricated_success=True,
                    )
                ],
            )
            out = root / "out"
            build_tau3_grounded_generation_dataset(
                source=source,
                out_dir=out,
                strict_coverage=False,
            )

            result = validate_tau3_grounded_generation_bundle(out, strict=False)

            self.assertFalse(result["passed"])
            self.assertTrue(
                any("fabricates completion" in error for error in result["errors"]),
                result["errors"],
            )

    def test_validation_reports_split_contamination(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.jsonl"
            shared_source_id = "shared-source"
            _write_jsonl(
                source,
                [
                    _scenario(
                        split="train",
                        domain="airline",
                        family_index=0,
                        behavior="successful_completion",
                        source_id=shared_source_id,
                    ),
                    _scenario(
                        split="validation",
                        domain="airline",
                        family_index=0,
                        behavior="successful_completion",
                        source_id=shared_source_id,
                    ),
                ],
            )
            out = root / "out"
            build_tau3_grounded_generation_dataset(
                source=source,
                out_dir=out,
                strict_coverage=False,
            )

            result = validate_tau3_grounded_generation_bundle(out, strict=False)

            self.assertFalse(result["passed"])
            self.assertTrue(
                any("source_id crosses splits" in error for error in result["errors"]),
                result["errors"],
            )

    def test_builder_requires_explicit_complete_matching_contamination_metadata(self) -> None:
        cases = (
            ("missing", lambda row: row.pop("contamination")),
            ("partial", lambda row: row.update({"contamination": {"raw_sealed_payload_read": False}})),
            (
                "sealed_payload",
                lambda row: row.update(
                    {
                        "contamination": {
                            "source_split": "train",
                            "raw_sealed_payload_read": True,
                            "sealed_hash_only": True,
                        }
                    }
                ),
            ),
            (
                "not_hash_only",
                lambda row: row.update(
                    {
                        "contamination": {
                            "source_split": "train",
                            "raw_sealed_payload_read": False,
                            "sealed_hash_only": False,
                        }
                    }
                ),
            ),
            (
                "split_mismatch",
                lambda row: row.update(
                    {
                        "contamination": {
                            "source_split": "validation",
                            "raw_sealed_payload_read": False,
                            "sealed_hash_only": True,
                        }
                    }
                ),
            ),
        )
        for label, mutate in cases:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                source = root / "source.jsonl"
                row = _scenario(
                    split="train",
                    domain="airline",
                    family_index=0,
                    behavior="authentication",
                )
                mutate(row)
                _write_jsonl(source, [row])

                with self.assertRaises(Tau3GroundedGenerationError):
                    build_tau3_grounded_generation_dataset(
                        source=source,
                        out_dir=root / "out",
                        strict_coverage=False,
                    )

    def test_validator_replays_row_contamination_metadata(self) -> None:
        tamper_cases = (
            (
                "raw_sealed_payload_read must be false",
                lambda contamination: contamination.update({"raw_sealed_payload_read": True}),
            ),
            (
                "sealed_hash_only must be true",
                lambda contamination: contamination.update({"sealed_hash_only": False}),
            ),
            (
                "source_split must match row split",
                lambda contamination: contamination.update({"source_split": "validation"}),
            ),
        )
        for expected, mutate in tamper_cases:
            with self.subTest(expected=expected), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                source = root / "source.jsonl"
                _write_jsonl(
                    source,
                    [
                        _scenario(
                            split="train",
                            domain="airline",
                            family_index=0,
                            behavior="authentication",
                        )
                    ],
                )
                out = root / "out"
                build_tau3_grounded_generation_dataset(
                    source=source,
                    out_dir=out,
                    strict_coverage=False,
                )
                rows = [
                    json.loads(line)
                    for line in (out / "train.jsonl").read_text(encoding="utf-8").splitlines()
                ]
                mutate(rows[0]["metadata"]["contamination"])
                rows[0]["metadata"]["row_sha256"] = canonical_sha256(
                    {
                        key: value
                        for key, value in rows[0].items()
                        if key != "metadata"
                    }
                    | {
                        "metadata": {
                            key: value
                            for key, value in rows[0]["metadata"].items()
                            if key != "row_sha256"
                        }
                    }
                )
                _write_jsonl(out / "train.jsonl", rows)
                _rewrite_bundle_manifest(out)

                result = validate_tau3_grounded_generation_bundle(out, strict=False)

                self.assertFalse(result["passed"])
                self.assertTrue(
                    any(expected in error for error in result["errors"]),
                    result["errors"],
                )

    def test_builder_rejects_unmasked_unsafe_corrected_target(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.jsonl"
            _write_jsonl(
                source,
                [
                    _scenario(
                        split="train",
                        domain="airline",
                        family_index=0,
                        behavior="harmful_mutation_correction",
                        unsafe=False,
                    )
                ],
            )

            with self.assertRaises(Tau3GroundedGenerationError):
                build_tau3_grounded_generation_dataset(
                    source=source,
                    out_dir=root / "out",
                    strict_coverage=False,
                )

    def test_builder_rejects_unbound_unmasked_tool_target(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.jsonl"
            row = _scenario(
                split="train",
                domain="airline",
                family_index=0,
                behavior="authentication",
            )
            row["turns"][0]["assistant"]["safe_corrected_target"].update(
                {
                    "tool_name": "get_record",
                    "arguments": {"id": "different-record"},
                    "safe_precondition": "forged bypass must not admit this target",
                }
            )
            _write_jsonl(source, [row])

            with self.assertRaises(Tau3GroundedGenerationError):
                build_tau3_grounded_generation_dataset(
                    source=source,
                    out_dir=root / "out",
                    strict_coverage=False,
                )

    def test_builder_rejects_malformed_canonical_target_kind_and_tool_fields(self) -> None:
        cases = (
            ("bad_kind", {"kind": "other"}),
            ("message_with_tool", {"kind": "assistant_message", "tool_name": "get_record"}),
            ("tool_call_without_args", {"kind": "tool_call", "tool_name": "get_record", "arguments": {}}),
        )
        for label, update in cases:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                source = root / "source.jsonl"
                row = _scenario(
                    split="train",
                    domain="airline",
                    family_index=0,
                    behavior="authentication",
                )
                row["turns"][0]["assistant"]["safe_corrected_target"].update(update)
                _write_jsonl(source, [row])

                with self.assertRaises(Tau3GroundedGenerationError):
                    build_tau3_grounded_generation_dataset(
                        source=source,
                        out_dir=root / "out",
                        strict_coverage=False,
                    )

    def test_zero_argument_tool_target_is_grounded_by_exact_catalog_and_replay(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.jsonl"
            row = _scenario(
                split="train",
                domain="airline",
                family_index=0,
                behavior="empty_result_recovery",
            )
            row["turns"][0]["assistant"]["safe_corrected_target"].update(
                {
                    "kind": "tool_call",
                    "tool_name": "empty_search",
                    "arguments": {},
                }
            )
            _write_jsonl(source, [row])

            build_tau3_grounded_generation_dataset(
                source=source,
                out_dir=root / "out",
                strict_coverage=False,
            )

            result = validate_tau3_grounded_generation_bundle(
                root / "out",
                strict=False,
            )
            self.assertFalse(
                any("empty target arguments" in error for error in result["errors"]),
                result["errors"],
            )
            exported = json.loads(
                (root / "out" / "train.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()[0]
            )
            self.assertEqual(
                exported["training_targets"][0]["canonical_target"]["arguments"],
                {},
            )

    def test_tool_exemptions_export_and_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.jsonl"
            row = _scenario(
                split="train",
                domain="airline",
                family_index=0,
                behavior="empty_result_recovery",
            )
            row["tool_exemptions"] = [
                {
                    "tool_name": "empty_search",
                    "reason": "zero_arg",
                    "reviewer": "unit-reviewer",
                },
                {
                    "tool_name": "get_record",
                    "reason": "policy_forbidden",
                    "reviewer": "unit-reviewer",
                    "policy_hash": canonical_sha256("policy"),
                    "citation": "policy:do-not-use-get-record-for-this-case",
                },
            ]
            _write_jsonl(source, [row])

            build_tau3_grounded_generation_dataset(
                source=source,
                out_dir=root / "out",
                strict_coverage=False,
            )

            exported = json.loads((root / "out" / "train.jsonl").read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(exported["metadata"]["tool_exemptions"], row["tool_exemptions"])

        bad_cases = (
            (
                "zero_arg_on_required_tool",
                [
                    {
                        "tool_name": "get_record",
                        "reason": "zero_arg",
                        "reviewer": "unit-reviewer",
                    }
                ],
            ),
            (
                "policy_missing_evidence",
                [
                    {
                        "tool_name": "get_record",
                        "reason": "policy_forbidden",
                        "reviewer": "unit-reviewer",
                    }
                ],
            ),
            (
                "not_in_catalog",
                [
                    {
                        "tool_name": "missing_tool",
                        "reason": "zero_arg",
                        "reviewer": "unit-reviewer",
                    }
                ],
            ),
            (
                "bad_reason",
                [
                    {
                        "tool_name": "empty_search",
                        "reason": "reviewed",
                        "reviewer": "unit-reviewer",
                    }
                ],
            ),
        )
        for label, exemptions in bad_cases:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                source = root / "source.jsonl"
                row = _scenario(
                    split="train",
                    domain="airline",
                    family_index=0,
                    behavior="authentication",
                )
                row["tool_exemptions"] = exemptions
                _write_jsonl(source, [row])

                with self.assertRaises(Tau3GroundedGenerationError):
                    build_tau3_grounded_generation_dataset(
                        source=source,
                        out_dir=root / "out",
                        strict_coverage=False,
                    )

    def test_validator_replays_tool_exemptions(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.jsonl"
            _write_jsonl(
                source,
                [
                    _scenario(
                        split="train",
                        domain="airline",
                        family_index=0,
                        behavior="authentication",
                    )
                ],
            )
            out = root / "out"
            build_tau3_grounded_generation_dataset(
                source=source,
                out_dir=out,
                strict_coverage=False,
            )
            rows = [
                json.loads(line)
                for line in (out / "train.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            rows[0]["metadata"]["tool_exemptions"] = [
                {
                    "tool_name": "missing_tool",
                    "reason": "zero_arg",
                    "reviewer": "unit-reviewer",
                }
            ]
            rows[0]["metadata"]["row_sha256"] = canonical_sha256(
                {
                    key: value
                    for key, value in rows[0].items()
                    if key != "metadata"
                }
                | {
                    "metadata": {
                        key: value
                        for key, value in rows[0]["metadata"].items()
                        if key != "row_sha256"
                    }
                }
            )
            _write_jsonl(out / "train.jsonl", rows)
            _rewrite_bundle_manifest(out)

            result = validate_tau3_grounded_generation_bundle(out, strict=False)

            self.assertFalse(result["passed"])
            self.assertTrue(
                any("tool_name is not in exact catalog" in error for error in result["errors"]),
                result["errors"],
            )

    def test_validator_rejects_forged_safe_precondition_on_unexecuted_target(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.jsonl"
            _write_jsonl(
                source,
                [
                    _scenario(
                        split="train",
                        domain="airline",
                        family_index=0,
                        behavior="authentication",
                    )
                ],
            )
            out = root / "out"
            build_tau3_grounded_generation_dataset(
                source=source,
                out_dir=out,
                strict_coverage=False,
            )
            rows = [
                json.loads(line)
                for line in (out / "train.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            rows[0]["training_targets"][0]["canonical_target"]["tool_name"] = "get_record"
            rows[0]["training_targets"][0]["canonical_target"]["arguments"] = {
                "id": "not-executed"
            }
            rows[0]["training_targets"][0]["canonical_target_sha256"] = canonical_sha256(
                rows[0]["training_targets"][0]["canonical_target"]
            )
            rows[0]["training_targets"][0]["safe_precondition"] = "forged bypass"
            rows[0]["metadata"]["row_sha256"] = canonical_sha256(
                {
                    key: value
                    for key, value in rows[0].items()
                    if key != "metadata"
                }
                | {
                    "metadata": {
                        key: value
                        for key, value in rows[0]["metadata"].items()
                        if key != "row_sha256"
                    }
                }
            )
            _write_jsonl(out / "train.jsonl", rows)
            _rewrite_bundle_manifest(out)

            result = validate_tau3_grounded_generation_bundle(out, strict=False)

            self.assertFalse(result["passed"])
            self.assertTrue(
                any("target tool call is not exactly bound" in error for error in result["errors"]),
                result["errors"],
            )

    def test_vendored_tau_adapter_replays_actual_airline_tool_when_available(self) -> None:
        repo = Path(__file__).resolve().parents[1] / "local" / "tau3" / "repository"
        if not repo.is_dir():
            self.skipTest("local/tau3/repository is absent")
        try:
            import sys

            sys.path.insert(0, str(repo / "src"))
            from tau2.domains.airline.data_model import FlightDB
            from tau2.domains.airline.utils import AIRLINE_DB_PATH
        except Exception as exc:
            self.skipTest(f"vendored Tau toolkit dependencies unavailable: {exc}")
        state = FlightDB.load(AIRLINE_DB_PATH).model_dump(mode="json")
        user_id = next(iter(state["users"]))
        import subprocess

        revision = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.jsonl"
            row = _scenario(
                split="train",
                domain="airline",
                family_index=0,
                behavior="authentication",
            )
            row["tau_revision"] = revision
            row["runtime_family"] = f"vendored_tau_tools@{revision}"
            row["tau_repo"] = repo.relative_to(Path(__file__).resolve().parents[1]).as_posix()
            row["initial_state"] = state
            expected_user = state["users"][user_id]
            row["turns"][0]["assistant"]["tool_calls"] = [
                {
                    "tool_name": "get_user_details",
                    "arguments": {"user_id": user_id},
                    "expected_result_sha256": canonical_sha256(expected_user),
                    "expected_result_class": "success",
                }
            ]
            row["turns"][0]["assistant"]["safe_corrected_target"] = {
                "behavior": "authentication",
                "kind": "assistant_message",
                "text": "I verified the user details with the Tau airline tool.",
            }
            _write_jsonl(source, [row])

            out = root / "out"
            build_tau3_grounded_generation_dataset(
                source=source,
                out_dir=out,
                strict_coverage=False,
            )
            clean = validate_tau3_grounded_generation_bundle(out, strict=False)
            self.assertFalse(clean["passed"])
            self.assertFalse(
                any("runtime cannot be instantiated" in error for error in clean["errors"]),
                clean["errors"],
            )
            self.assertFalse(
                any("tool_replay" in error for error in clean["errors"]),
                clean["errors"],
            )
            rows = [
                json.loads(line)
                for line in (out / "train.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertTrue(rows[0]["tool_replay"][0]["source_expected_result_verified"])
            self.assertEqual(
                rows[0]["tool_replay"][0]["source_expected_result_sha256"],
                canonical_sha256(expected_user),
            )
            rows[0]["metadata"]["tau_repo"]["tree_sha256"] = "f" * 64
            rows[0]["metadata"]["row_sha256"] = canonical_sha256(
                {
                    key: value
                    for key, value in rows[0].items()
                    if key != "metadata"
                }
                | {
                    "metadata": {
                        key: value
                        for key, value in rows[0]["metadata"].items()
                        if key != "row_sha256"
                    }
                }
            )
            _write_jsonl(out / "train.jsonl", rows)
            _rewrite_bundle_manifest(out)

            tampered = validate_tau3_grounded_generation_bundle(out, strict=False)

            self.assertFalse(tampered["passed"])
            self.assertTrue(
                any("tau_repo.tree_sha256 does not replay" in error for error in tampered["errors"]),
                tampered["errors"],
            )

    def test_vendored_tau_telecom_date_result_is_canonical_json_when_available(self) -> None:
        repo = Path(__file__).resolve().parents[1] / "local" / "tau3" / "repository"
        if not repo.is_dir():
            self.skipTest("local/tau3/repository is absent")
        try:
            import subprocess
            import sys

            sys.path.insert(0, str(repo / "src"))
            from tau2.domains.telecom.data_model import TelecomDB
            from tau2.domains.telecom.utils import TELECOM_DB_PATH
            from flightrecorder.tau3_grounded_generation import _VendoredTauRuntime
        except Exception as exc:
            self.skipTest(f"vendored Tau toolkit dependencies unavailable: {exc}")
        revision = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        state = TelecomDB.load(TELECOM_DB_PATH).model_dump(mode="json")
        customer = state["customers"][0]
        line_id = customer["line_ids"][0]
        runtime = _VendoredTauRuntime(
            domain="telecom",
            revision=revision,
            state=state,
            repo=repo.relative_to(Path(__file__).resolve().parents[1]).as_posix(),
        )

        result = runtime.call(
            "get_data_usage",
            {"customer_id": customer["customer_id"], "line_id": line_id},
        )

        self.assertIsInstance(result["cycle_end_date"], str)
        self.assertRegex(result["cycle_end_date"], r"^2025-\d{2}-\d{2}$")
        self.assertIsInstance(canonical_sha256(result), str)

    def test_schema_file_is_valid_json(self) -> None:
        schema_path = (
            Path(__file__).resolve().parents[1]
            / "flightrecorder"
            / "schemas"
            / "tau3_grounded_generation.v1.schema.json"
        )
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        self.assertEqual(
            schema["properties"]["schema_version"]["const"],
            TAU3_GROUNDED_DATASET_SCHEMA_VERSION,
        )


if __name__ == "__main__":
    unittest.main()
