from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from typing import Any

from flightrecorder.tau3_v3_scenarios import (
    TAU3_V3_SCENARIO_SUMMARY_SCHEMA_VERSION,
    Tau3V3ScenarioError,
    _coverage_summary,
    _telecom_state_variant_for_tool,
    build_tau3_v3_scenario_sources,
)
from flightrecorder.tau3_grounded_generation import canonical_sha256


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _init_git_repo(path: Path) -> None:
    path.mkdir(parents=True)
    subprocess.run(["git", "init"], cwd=path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "agent@example.com"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "Agent"], cwd=path, check=True)
    (path / "README.md").write_text("fixture\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-m", "fixture"], cwd=path, check=True, capture_output=True)


def _fixture_inputs(root: Path) -> dict[str, Path]:
    repo = root / "tau_repo"
    _init_git_repo(repo)
    mixture = root / "mixture"
    _write_jsonl(
        mixture / "train.jsonl",
        [
            {
                "messages": [{"content": f"exact {domain} system prompt"}],
                "metadata": {"domain": domain},
            }
            for domain in ("airline", "retail", "telecom")
        ],
    )
    catalog = root / "tool-catalog.json"
    _write_json(
        catalog,
        {
            "domains": {
                "airline": {
                    "tools": [
                        {"function": {"name": "get_record"}},
                        {"function": {"name": "list_all_airports"}},
                        {"function": {"name": "update_record"}},
                    ]
                },
                "retail": {
                    "tools": [
                        {"function": {"name": "get_record"}},
                        {"function": {"name": "list_all_product_types"}},
                        {"function": {"name": "update_record"}},
                    ]
                },
                "telecom": {
                    "tools": [
                        {"function": {"name": "get_record"}},
                        {"function": {"name": "update_record"}},
                    ]
                },
            }
        },
    )
    corpus = root / "corpus.jsonl"
    _write_jsonl(
        corpus,
        [
            {"metadata": {"domain": "airline", "reward": 1}},
            {"metadata": {"domain": "retail", "reward": 1}},
            {"metadata": {"domain": "telecom", "reward": 1}},
        ],
    )
    development = root / "development.jsonl"
    _write_jsonl(
        development,
        [
            {
                "task_sha256": "a" * 64,
                "task_family": "b" * 64,
                "prompt_sha256": "c" * 64,
                "task": {
                    "user_scenario": {
                        "instructions": {
                            "known_info": "development only",
                            "reason_for_call": "different request",
                            "task_instructions": "do not overlap",
                        }
                    }
                },
            }
        ],
    )
    protocol = root / "protocol.json"
    _write_json(
        protocol,
        {
            "sealed_manifest": {
                "leakage_blocking_hashes": ["d" * 64],
                "prompt_template_hashes": ["e" * 64],
            }
        },
    )
    return {
        "repo": repo,
        "mixture": mixture,
        "catalog": catalog,
        "corpus": corpus,
        "development": development,
        "protocol": protocol,
    }


class _FakeRuntime:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.domain = payload["domain"]
        self.state = payload["initial_state"]

    def tool_catalog(self) -> list[dict[str, Any]]:
        if self.domain == "airline":
            return [
                {"name": "get_record"},
                {"name": "list_all_airports"},
                {"name": "update_record"},
            ]
        if self.domain == "retail":
            return [
                {"name": "get_record"},
                {"name": "list_all_product_types"},
                {"name": "update_record"},
            ]
        return [{"name": "get_record"}, {"name": "update_record"}]

    def call(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        if tool_name in {"list_all_airports", "list_all_product_types"}:
            return {"ok": True}
        if args.get("id") not in self.state["records"]:
            raise ValueError("missing record")
        if tool_name == "update_record":
            self.state["records"][args["id"]]["status"] = "updated"
        return self.state["records"][args["id"]]


def _fake_runtime(payload: dict[str, Any]) -> _FakeRuntime:
    return _FakeRuntime(payload)


class Tau3V3ScenarioSourceTests(unittest.TestCase):
    def test_dry_run_builds_chronological_rows_without_writing_private_output(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp:
            paths = _fixture_inputs(Path(temp))
            out = Path(temp) / "sources.jsonl"
            result = build_tau3_v3_scenario_sources(
                out=out,
                tau_repo=paths["repo"],
                v2_mixture_dir=paths["mixture"],
                official_tool_catalog=paths["catalog"],
                natural_corpus=paths["corpus"],
                development_tasks=paths["development"],
                protocol=paths["protocol"],
                strict=False,
                dry_run=True,
                runtime_factory=_fake_runtime,
                max_rows_per_domain_split=1,
            )

            self.assertFalse(out.exists())
            self.assertEqual(result.summary["schema_version"], TAU3_V3_SCENARIO_SUMMARY_SCHEMA_VERSION)
            self.assertFalse(result.summary["passed"])
            self.assertEqual(result.summary["sealed_access"]["dev_or_sealed_payload_access_count"], 0)
            first = result.rows[0]
            self.assertEqual(first["turns"][0]["assistant"]["decision_ordinal"], 0)
            self.assertEqual(
                first["turns"][0]["assistant"]["safe_corrected_target"]["tool_name"],
                first["turns"][0]["assistant"]["tool_calls"][0]["tool_name"],
            )
            self.assertEqual(first["turns"][1]["assistant"]["decision_ordinal"], 1)
            self.assertIn("safe_corrected_target", first["turns"][1]["assistant"])
            self.assertEqual(
                first["contamination"],
                {
                    "source_split": first["split"],
                    "raw_sealed_payload_read": False,
                    "sealed_hash_only": True,
                    "dev_payload_read": False,
                },
            )

    def test_negative_corrections_are_reviewed_context_before_safe_targets(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp:
            paths = _fixture_inputs(Path(temp))
            result = build_tau3_v3_scenario_sources(
                out=Path(temp) / "unused.jsonl",
                tau_repo=paths["repo"],
                v2_mixture_dir=paths["mixture"],
                official_tool_catalog=paths["catalog"],
                natural_corpus=paths["corpus"],
                development_tasks=paths["development"],
                protocol=paths["protocol"],
                strict=False,
                dry_run=True,
                runtime_factory=_fake_runtime,
                max_rows_per_domain_split=104,
            )

            expected_negative = {
                "hallucinated_tool_correction": "hallucinated_tool",
                "harmful_mutation_correction": "harmful_mutation",
                "premature_completion_correction": "premature_completion",
            }
            for behavior, negative_behavior in expected_negative.items():
                row = next(
                    item
                    for item in result.rows
                    if item["domain"] == "airline"
                    and item["split"] == "train"
                    and f"-{behavior}-" in item["source_id"]
                )
                negative = row["turns"][0]["assistant"]["safe_corrected_target"]
                correction = row["turns"][1]["assistant"]["safe_corrected_target"]
                self.assertTrue(negative["masked"])
                self.assertTrue(negative["reviewed"])
                self.assertEqual(negative["negative_behavior"], negative_behavior)
                self.assertEqual(negative["safe_correction_decision_ordinal"], 1)
                self.assertEqual(correction["behavior"], behavior)
                self.assertFalse(correction.get("masked", False))
                self.assertEqual(
                    result.summary["coverage"]["behavior_counts"]["train"]["airline"][behavior],
                    24,
                )
                self.assertEqual(
                    result.summary["coverage"]["negative_context_counts"]["train"]["airline"][
                        negative_behavior
                    ],
                    24,
                )
            success = next(
                item
                for item in result.rows
                if item["domain"] == "airline"
                and item["split"] == "train"
                and "-successful_completion-" in item["source_id"]
            )
            action = success["turns"][0]["assistant"]["safe_corrected_target"]
            completion = success["turns"][1]["assistant"]["safe_corrected_target"]
            self.assertEqual(action["behavior"], "later_task_completion_actions")
            self.assertEqual(action["kind"], "tool_call")
            self.assertEqual(completion["behavior"], "successful_completion")
            self.assertEqual(completion["kind"], "assistant_message")
            self.assertEqual(
                result.summary["coverage"]["behavior_counts"]["train"]["airline"][
                    "successful_completion"
                ],
                24,
            )

    def test_writes_source_and_sibling_contamination_report_when_not_dry_run(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp:
            paths = _fixture_inputs(Path(temp))
            out = Path(temp) / "sources.jsonl"
            contamination = Path(temp) / "contamination_report.json"

            result = build_tau3_v3_scenario_sources(
                out=out,
                tau_repo=paths["repo"],
                v2_mixture_dir=paths["mixture"],
                official_tool_catalog=paths["catalog"],
                natural_corpus=paths["corpus"],
                development_tasks=paths["development"],
                protocol=paths["protocol"],
                contamination_report_out=contamination,
                strict=False,
                runtime_factory=_fake_runtime,
                max_rows_per_domain_split=1,
            )

            self.assertTrue(out.is_file())
            self.assertTrue(contamination.is_file())
            report = json.loads(contamination.read_text(encoding="utf-8"))
            self.assertTrue(report["passed"])
            self.assertEqual(report["sealed_hash_only_comparison"]["sealed_payload_access_count"], 0)
            self.assertEqual(
                result.summary["contamination_report"]["path"],
                contamination.relative_to(Path.cwd()).as_posix(),
            )
            rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
            airline = next(row for row in rows if row["domain"] == "airline")
            self.assertEqual(
                airline["tool_exemptions"],
                [
                    {
                        "reason": "zero_arg",
                        "reviewed": True,
                        "reviewer": "flightrecorder-v3-source-coverage-gate",
                        "reviewer_sha256": airline["tool_exemptions"][0]["reviewer_sha256"],
                        "tool_name": "list_all_airports",
                    }
                ],
            )
            closure_targets = [
                turn["assistant"]["safe_corrected_target"]
                for row in rows
                if "tool-closure" in row["source_id"]
                for turn in row["turns"]
            ]
            self.assertTrue(closure_targets)
            self.assertEqual(
                {target["behavior"] for target in closure_targets},
                {"later_task_completion_actions"},
            )

    def test_strict_mode_fails_closed_on_incomplete_coverage(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp:
            paths = _fixture_inputs(Path(temp))
            with self.assertRaises(Tau3V3ScenarioError):
                build_tau3_v3_scenario_sources(
                    out=None,
                    tau_repo=paths["repo"],
                    v2_mixture_dir=paths["mixture"],
                    official_tool_catalog=paths["catalog"],
                    natural_corpus=paths["corpus"],
                    development_tasks=paths["development"],
                    protocol=paths["protocol"],
                    strict=True,
                    dry_run=True,
                    runtime_factory=_fake_runtime,
                    max_rows_per_domain_split=1,
                )

    def test_natural_import_preserves_chronology_and_normalized_hash_overlap(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp:
            paths = _fixture_inputs(Path(temp))
            revision = subprocess.run(
                ["git", "-C", str(paths["repo"]), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            task_sha = "1" * 64
            family_sha = "2" * 64
            prompt_sha = "3" * 64
            system_prompt = "exact airline system prompt"
            tools = [
                {"function": {"name": "get_record"}},
                {"function": {"name": "list_all_airports"}},
                {"function": {"name": "update_record"}},
            ]
            _write_jsonl(
                paths["corpus"],
                [
                    {
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": "Look up airline-1 twice."},
                            {
                                "role": "assistant",
                                "tool_calls": [
                                    {
                                        "id": "call-1",
                                        "type": "function",
                                        "function": {
                                            "name": "get_record",
                                            "arguments": json.dumps({"id": "airline-1"}),
                                        },
                                    }
                                ],
                            },
                            {
                                "role": "tool",
                                "tool_call_id": "call-1",
                                "content": json.dumps({"id": "airline-1"}),
                            },
                            {
                                "role": "assistant",
                                "tool_calls": [
                                    {
                                        "id": "call-2",
                                        "type": "function",
                                        "function": {
                                            "name": "get_record",
                                            "arguments": json.dumps({"id": "airline-1"}),
                                        },
                                    }
                                ],
                            },
                            {
                                "role": "tool",
                                "tool_call_id": "call-2",
                                "content": json.dumps({"id": "airline-1"}),
                            },
                            {"role": "assistant", "content": "The record has been checked twice."},
                        ],
                        "metadata": {
                            "domain": "airline",
                            "episode_id": "natural-airline-fixture",
                            "privacy": {"sealed_payload_read": False},
                            "prompt_sha256": prompt_sha,
                            "reward": 1.0,
                            "source_revision": revision,
                            "split": "train",
                            "system_prompt_sha256": canonical_sha256(system_prompt),
                            "task_family": family_sha,
                            "task_sha256": task_sha,
                        },
                        "tools": tools,
                    }
                ],
            )
            _write_jsonl(
                paths["development"],
                [
                    {
                        "task_sha256": task_sha,
                        "task_family": "b" * 64,
                        "prompt_sha256": "c" * 64,
                    }
                ],
            )

            result = build_tau3_v3_scenario_sources(
                out=None,
                tau_repo=paths["repo"],
                v2_mixture_dir=paths["mixture"],
                official_tool_catalog=paths["catalog"],
                natural_corpus=paths["corpus"],
                development_tasks=paths["development"],
                protocol=paths["protocol"],
                strict=False,
                dry_run=True,
                runtime_factory=_fake_runtime,
                max_rows_per_domain_split=0,
            )

            natural = next(row for row in result.rows if row["source_family"] == "official_train_derived")
            user_turns = [turn for turn in natural["turns"] if "user" in turn]
            self.assertEqual(len(user_turns), 1)
            self.assertEqual(natural["source_generation"]["source_identity_sha256"], task_sha)
            self.assertEqual(natural["source_generation"]["source_family_sha256"], family_sha)
            self.assertEqual(natural["source_generation"]["source_prompt_sha256"], prompt_sha)
            self.assertIn("development source_id_hashes overlap", result.summary["blockers"])

    def test_telecom_state_variants_replay_and_count_as_distinct_targets(self) -> None:
        base_state = {
            "customers": [
                {
                    "customer_id": "C1001",
                    "full_name": "John Smith",
                    "date_of_birth": "1985-06-15",
                    "email": "john.smith@example.com",
                    "phone_number": "555-123-2002",
                    "address": {"street": "123 Main St"},
                    "account_status": "Active",
                    "payment_methods": [],
                    "line_ids": [],
                    "bill_ids": [],
                    "created_at": "2025-01-15T10:30:00",
                    "last_extension_date": None,
                    "goodwill_credit_used_this_year": 0.0,
                }
            ],
            "lines": [
                {
                    "line_id": "L1001",
                    "phone_number": "555-123-2001",
                    "status": "Suspended",
                    "plan_id": "P1001",
                    "device_id": "D1001",
                    "data_used_gb": 3.2,
                    "data_refueling_gb": 0.0,
                    "roaming_enabled": False,
                    "contract_end_date": "2026-12-31",
                    "last_plan_change_date": "2025-01-10",
                    "last_sim_replacement_date": None,
                    "suspension_start_date": "2025-01-01",
                }
            ],
            "bills": [
                {
                    "bill_id": "B1001",
                    "customer_id": "C1001",
                    "period_start": "2025-01-01",
                    "period_end": "2025-01-31",
                    "issue_date": "2025-01-05",
                    "total_due": 160.5,
                    "due_date": "2025-01-19",
                    "line_items": [],
                    "status": "Paid",
                }
            ],
        }
        distinct: set[str] = set()
        rows = []
        for index in range(8):
            variant = _telecom_state_variant_for_tool(
                base_state,
                "get_customer_by_id",
                index,
                distinct,
            )
            self.assertIsNotNone(variant)
            state, args, derivation = variant
            distinct.add(canonical_sha256(args))
            self.assertTrue(any(customer["customer_id"] == args["customer_id"] for customer in state["customers"]))
            self.assertEqual(derivation["base_state_sha256"], canonical_sha256(base_state))
            self.assertEqual(derivation["variant_state_sha256"], canonical_sha256(state))
            rows.append(
                {
                    "split": "train",
                    "domain": "telecom",
                    "source_family_id": f"variant-{index}",
                    "turns": [
                        {
                            "assistant": {
                                "safe_corrected_target": {
                                    "behavior": "successful_completion",
                                    "kind": "tool_call",
                                    "tool_name": "get_customer_by_id",
                                    "arguments": args,
                                }
                            }
                        }
                    ],
                }
            )

        summary = _coverage_summary(
            rows,
            {"telecom": {"get_customer_by_id": [{}]}},
            [],
            {},
            {"blockers": [], "passed": True},
        )
        record = summary["coverage"]["tool_counts"]["train"]["telecom"]["get_customer_by_id"]
        self.assertEqual(record["supervised_target_count"], 8)
        self.assertEqual(record["distinct_argument_count"], 8)

    def test_missing_development_hash_file_blocks_and_strict_raises(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp:
            paths = _fixture_inputs(Path(temp))
            missing = Path(temp) / "missing-development.jsonl"

            result = build_tau3_v3_scenario_sources(
                out=None,
                tau_repo=paths["repo"],
                v2_mixture_dir=paths["mixture"],
                official_tool_catalog=paths["catalog"],
                natural_corpus=paths["corpus"],
                development_tasks=missing,
                protocol=paths["protocol"],
                strict=False,
                dry_run=True,
                runtime_factory=_fake_runtime,
                max_rows_per_domain_split=0,
            )

            self.assertIn(
                "development hash-only evidence is missing or unreadable: "
                + missing.relative_to(Path.cwd()).as_posix(),
                result.summary["blockers"],
            )
            self.assertTrue(
                result.summary["contamination_report"]["development_hash_only_evidence"]["missing_or_unreadable"]
            )
            with self.assertRaises(Tau3V3ScenarioError):
                build_tau3_v3_scenario_sources(
                    out=None,
                    tau_repo=paths["repo"],
                    v2_mixture_dir=paths["mixture"],
                    official_tool_catalog=paths["catalog"],
                    natural_corpus=paths["corpus"],
                    development_tasks=missing,
                    protocol=paths["protocol"],
                    strict=True,
                    dry_run=True,
                    runtime_factory=_fake_runtime,
                    max_rows_per_domain_split=0,
                )

    def test_empty_protocol_blocks_and_strict_raises(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp:
            paths = _fixture_inputs(Path(temp))
            _write_json(paths["protocol"], {})

            result = build_tau3_v3_scenario_sources(
                out=None,
                tau_repo=paths["repo"],
                v2_mixture_dir=paths["mixture"],
                official_tool_catalog=paths["catalog"],
                natural_corpus=paths["corpus"],
                development_tasks=paths["development"],
                protocol=paths["protocol"],
                strict=False,
                dry_run=True,
                runtime_factory=_fake_runtime,
                max_rows_per_domain_split=0,
            )

            self.assertIn("protocol sealed_manifest is missing or empty", result.summary["blockers"])
            self.assertIn(
                "protocol sealed_manifest.leakage_blocking_hashes is empty",
                result.summary["blockers"],
            )
            self.assertIn(
                "protocol sealed_manifest.prompt_template_hashes is empty",
                result.summary["blockers"],
            )
            with self.assertRaises(Tau3V3ScenarioError):
                build_tau3_v3_scenario_sources(
                    out=None,
                    tau_repo=paths["repo"],
                    v2_mixture_dir=paths["mixture"],
                    official_tool_catalog=paths["catalog"],
                    natural_corpus=paths["corpus"],
                    development_tasks=paths["development"],
                    protocol=paths["protocol"],
                    strict=True,
                    dry_run=True,
                    runtime_factory=_fake_runtime,
                    max_rows_per_domain_split=0,
                )

    def test_malformed_and_empty_hash_evidence_blocks(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp:
            paths = _fixture_inputs(Path(temp))
            _write_jsonl(
                paths["development"],
                [
                    {
                        "task_sha256": "A" * 64,
                        "task_family": "",
                        "prompt_sha256": "not-a-sha",
                    }
                ],
            )
            _write_json(
                paths["protocol"],
                {
                    "sealed_manifest": {
                        "leakage_blocking_hashes": [],
                        "prompt_template_hashes": ["F" * 64, "not-a-sha"],
                    }
                },
            )

            result = build_tau3_v3_scenario_sources(
                out=None,
                tau_repo=paths["repo"],
                v2_mixture_dir=paths["mixture"],
                official_tool_catalog=paths["catalog"],
                natural_corpus=paths["corpus"],
                development_tasks=paths["development"],
                protocol=paths["protocol"],
                strict=False,
                dry_run=True,
                runtime_factory=_fake_runtime,
                max_rows_per_domain_split=0,
            )

            blockers = result.summary["blockers"]
            self.assertIn(
                "development hash-only row 1 must carry lowercase sha256 field task_sha256",
                blockers,
            )
            self.assertIn(
                "development hash-only row 1 must carry lowercase sha256 field task_family",
                blockers,
            )
            self.assertIn(
                "development hash-only row 1 must carry lowercase sha256 field prompt_sha256",
                blockers,
            )
            self.assertIn("development hash-only evidence has no valid rows", blockers)
            self.assertIn("protocol sealed_manifest.leakage_blocking_hashes must be nonempty", blockers)
            self.assertIn("protocol sealed_manifest.leakage_blocking_hashes is empty", blockers)
            self.assertIn(
                "protocol sealed_manifest.prompt_template_hashes[0] must be a lowercase sha256 string",
                blockers,
            )
            self.assertIn(
                "protocol sealed_manifest.prompt_template_hashes[1] must be a lowercase sha256 string",
                blockers,
            )
            evidence = result.summary["contamination_report"]["development_hash_only_evidence"]
            self.assertEqual(evidence["row_count"], 1)
            self.assertEqual(evidence["valid_row_count"], 0)
            self.assertEqual(evidence["malformed_row_count"], 1)


if __name__ == "__main__":
    unittest.main()
