from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

from flightrecorder.tau3_exposure import (
    Tau3ExposureError,
    build_tau3_exposure_ledger,
    validate_tau3_exposure_ledger,
)

ROOT = Path(__file__).resolve().parents[1]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _length_bucket(prompt_tokens: int, supervised_tokens: int) -> str:
    total = prompt_tokens + supervised_tokens
    if total <= 512:
        return "short"
    if total <= 1024:
        return "medium"
    if total <= 2048:
        return "long"
    return "extra_long"


def _row(
    index: int,
    domain: str,
    behavior: str,
    *,
    target_tool: str = "lookup_order",
    assistant_message: bool = False,
) -> dict[str, Any]:
    prompt_tokens = 100 + index
    supervised_tokens = 20 + index
    target_action_class = "assistant_message" if assistant_message else (
        "retry_tool_call" if "recovery" in behavior else "tool_call"
    )
    target_tool_name = "assistant_message" if assistant_message else target_tool
    preceding_result_class = "none"
    if "recovery" in behavior:
        preceding_result_class = "empty_result" if behavior == "empty_result_recovery" else "error_result"
    return {
        "messages": [
            {"role": "system", "content": "Tau policy"},
            {"role": "user", "content": f"request {index}"},
            {"role": "assistant", "content": f"answer {index}"} if assistant_message else {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": f"call-{index}",
                        "type": "function",
                        "function": {"name": target_tool, "arguments": "{}"},
                    }
                ],
            },
        ],
        "metadata": {
            "schema_version": "hfr.tau3_competitive_dataset_row.v1",
            "lineage_id": "tau3-competitive-agent-v3",
            "split": "train",
            "domain": domain,
            "behavior": behavior,
            "target_action_class": target_action_class,
            "target_tool_name": target_tool_name,
            "canonical_target": {"kind": target_action_class, "tool_name": target_tool_name},
            "canonical_target_sha256": "0" * 64,
            "preceding_result_class": preceding_result_class,
            "source_family_id": f"{domain}-family-{index % 2}",
            "source_provenance": {
                "method": "direct_parent_projection",
                "grounded_to_parent": True,
                "reviewed": True,
                "training_side_only": True,
            },
            "token_counts": {
                "method": "test_exact_chat_template_counter",
                "exact": True,
                "chat_template_aware": True,
                "prompt_tokens": prompt_tokens,
                "supervised_tokens": supervised_tokens,
            },
            "length_bucket": _length_bucket(prompt_tokens, supervised_tokens),
        },
        "tools": [{"type": "function", "function": {"name": target_tool, "parameters": {"type": "object"}}}],
    }


def _legacy_row(index: int, domain: str, behavior: str, *, target_tool: str = "lookup_order") -> dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": "Tau policy"},
            {"role": "user", "content": f"legacy request {index}"},
            {"role": "assistant", "content": f"legacy answer {index}"},
        ],
        "metadata": {
            "domain": domain,
            "behavior": behavior,
            "target_tool": target_tool,
            "action_class": "tool_call" if "recovery" not in behavior else "retry_tool_call",
            "result_class": "success" if "recovery" not in behavior else "empty_result",
            "length_bucket": "short",
            "source_family": f"{domain}-family-{index % 2}",
            "source_provenance": "legacy-fixture",
            "prompt_tokens": 100 + index,
            "supervised_tokens": 20 + index,
        },
        "tools": [{"type": "function", "function": {"name": target_tool, "parameters": {"type": "object"}}}],
    }


def _eligible_rows() -> list[dict[str, Any]]:
    behaviors = [
        ("airline", "successful_completion", "book_flight"),
        ("retail", "clarification_refusal", "lookup_order"),
        ("telecom", "authentication", "verify_account"),
        ("airline", "confirmation_before_mutation", "cancel_flight"),
        ("retail", "later_task_completion_actions", "return_order"),
        ("telecom", "safe_stopping", "lookup_service"),
        ("airline", "transfer_handoff", "transfer_to_agent"),
        ("retail", "empty_result_recovery", "lookup_order"),
        ("telecom", "error_result_recovery", "lookup_service"),
        ("airline", "repeated_call_recovery", "search_flights"),
        ("retail", "hallucinated_tool_correction", "lookup_order"),
        ("telecom", "harmful_mutation_correction", "change_plan"),
        ("airline", "premature_completion_correction", "book_flight"),
        ("telecom", "successful_completion", "lookup_service"),
    ]
    return [
        _row(
            index,
            domain,
            behavior,
            target_tool=tool,
            assistant_message=behavior in {"clarification_refusal", "safe_stopping", "transfer_handoff"},
        )
        for index, (domain, behavior, tool) in enumerate(behaviors, start=1)
    ]


class Tau3ExposureTests(unittest.TestCase):
    def test_builds_and_validates_replayable_full_epoch_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "train.jsonl"
            _write_jsonl(dataset, _eligible_rows())

            receipt = build_tau3_exposure_ledger(
                dataset,
                root / "exposure",
                seed=101,
                epochs=2,
                batch_size=2,
                gradient_accumulation_steps=2,
            )

            self.assertTrue(receipt["passed"])
            self.assertTrue(receipt["candidate_eligibility"]["passed"])
            self.assertEqual(receipt["coverage"]["effective_epochs"], 2)
            self.assertEqual(receipt["coverage"]["min_row_exposure"], 2)
            self.assertTrue(receipt["coverage"]["complete_optimizer_steps"])
            self.assertEqual(receipt["coverage"]["row_schema_versions"], ["hfr.tau3_competitive_dataset_row.v1"])
            self.assertEqual(receipt["dataset"]["label"], "train.jsonl")
            self.assertNotIn(str(root), json.dumps(receipt["dataset"]))
            self.assertEqual(set(receipt["coverage"]["domains"]), {"airline", "retail", "telecom"})
            self.assertEqual(receipt["coverage"]["behavior_exposure_counts"]["repeated_call_recovery"], 2)
            result = validate_tau3_exposure_ledger(dataset, root / "exposure" / "training_exposure_receipt.json")
            self.assertTrue(result["passed"])
            ledger_rows = (root / "exposure" / "training_exposure_ledger.jsonl").read_text(encoding="utf-8").splitlines()
            first_step = json.loads(ledger_rows[0])
            self.assertEqual(first_step["microbatch_size"], 2)
            self.assertEqual(first_step["gradient_accumulation_steps"], 2)
            self.assertEqual(first_step["effective_batch_size"], 4)
            self.assertEqual(first_step["effective_batch_row_count"], 4)
            self.assertEqual(first_step["microbatch_count"], 2)
            self.assertEqual(first_step["rows"][0]["cumulative_exposure"], 1)
            all_ledger_rows = [
                row
                for step in (json.loads(line) for line in ledger_rows)
                for row in step["rows"]
            ]
            first_safe_stop = next(row for row in all_ledger_rows if row["behavior"] == "safe_stopping")
            self.assertEqual(first_safe_stop["target_tool"], "assistant_message")
            self.assertEqual(first_safe_stop["action_class"], "assistant_message")
            self.assertEqual(first_safe_stop["source_provenance"], "direct_parent_projection")

    def test_optimizer_steps_group_gradient_accumulation_microbatches(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "train.jsonl"
            _write_jsonl(dataset, _eligible_rows())

            build_tau3_exposure_ledger(
                dataset,
                root / "exposure",
                seed=111,
                epochs=2,
                batch_size=2,
                gradient_accumulation_steps=7,
            )

            steps = [
                json.loads(line)
                for line in (root / "exposure" / "training_exposure_ledger.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(steps[0]["effective_batch_size"], 14)
            self.assertEqual(steps[0]["microbatch_count"], 7)
            self.assertEqual([len(item["row_hashes"]) for item in steps[0]["microbatches"]], [2, 2, 2, 2, 2, 2, 2])
            self.assertEqual(len(steps), 2)

    def test_schedule_is_stable_for_same_content_and_seed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "train.jsonl"
            _write_jsonl(dataset, _eligible_rows())
            first = build_tau3_exposure_ledger(dataset, root / "first", seed=202, epochs=2, batch_size=2, gradient_accumulation_steps=2)
            second = build_tau3_exposure_ledger(dataset, root / "second", seed=202, epochs=2, batch_size=2, gradient_accumulation_steps=2)

            self.assertEqual(first["files"]["ledger"]["sha256"], second["files"]["ledger"]["sha256"])
            self.assertEqual(
                (root / "first" / "training_exposure_ledger.jsonl").read_text(encoding="utf-8"),
                (root / "second" / "training_exposure_ledger.jsonl").read_text(encoding="utf-8"),
            )

    def test_validator_fails_when_dataset_row_is_mutated_after_build(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "train.jsonl"
            rows = _eligible_rows()
            _write_jsonl(dataset, rows)
            build_tau3_exposure_ledger(dataset, root / "exposure", seed=303, epochs=2, batch_size=2, gradient_accumulation_steps=2)
            rows[0]["metadata"]["prompt_tokens"] = 999
            _write_jsonl(dataset, rows)

            with self.assertRaisesRegex(Tau3ExposureError, "ledger does not replay"):
                validate_tau3_exposure_ledger(dataset, root / "exposure" / "training_exposure_receipt.json")

    def test_candidate_gate_fails_for_partial_epoch_and_weak_batch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "train.jsonl"
            _write_jsonl(dataset, _eligible_rows())
            receipt = build_tau3_exposure_ledger(dataset, root / "exposure", seed=404, epochs=1, batch_size=2, gradient_accumulation_steps=1)

            self.assertFalse(receipt["passed"])
            failed = {check["id"] for check in receipt["candidate_eligibility"]["checks"] if not check["passed"]}
            self.assertIn("at_least_two_effective_epochs", failed)
            self.assertIn("effective_batch_at_least_four", failed)

    def test_candidate_gate_fails_for_partial_final_optimizer_step(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "train.jsonl"
            _write_jsonl(dataset, _eligible_rows())
            receipt = build_tau3_exposure_ledger(dataset, root / "exposure", seed=407, epochs=2, batch_size=3, gradient_accumulation_steps=2)

            self.assertFalse(receipt["passed"])
            self.assertFalse(receipt["coverage"]["complete_optimizer_steps"])
            failed = {check["id"] for check in receipt["candidate_eligibility"]["checks"] if not check["passed"]}
            self.assertIn("complete_optimizer_steps", failed)

    def test_candidate_gate_requires_all_v3_behaviors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "train.jsonl"
            rows = [row for row in _eligible_rows() if row["metadata"]["behavior"] != "safe_stopping"]
            _write_jsonl(dataset, rows)
            receipt = build_tau3_exposure_ledger(dataset, root / "exposure", seed=405, epochs=2, batch_size=2, gradient_accumulation_steps=2)

            self.assertFalse(receipt["passed"])
            failures = {check["id"]: check for check in receipt["candidate_eligibility"]["checks"] if not check["passed"]}
            self.assertIn("required_behaviors_exact", failures)
            self.assertIn("safe_stopping", failures["required_behaviors_exact"]["actual"]["missing"])
            self.assertIn("stopping_strata_nonzero", failures)

    def test_legacy_fixture_rows_are_supported_but_never_candidate_eligible(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "train.jsonl"
            _write_jsonl(
                dataset,
                [
                    _legacy_row(index, domain, behavior, target_tool=tool)
                    for index, (domain, behavior, tool) in enumerate(
                        [
                            ("airline", "successful_completion", "book_flight"),
                            ("retail", "clarification_refusal", "lookup_order"),
                            ("telecom", "authentication", "verify_account"),
                        ],
                        start=1,
                    )
                ],
            )

            receipt = build_tau3_exposure_ledger(
                dataset,
                root / "exposure",
                seed=1,
                epochs=2,
                batch_size=2,
                gradient_accumulation_steps=2,
            )

            self.assertFalse(receipt["passed"])
            self.assertEqual(receipt["coverage"]["row_schema_versions"], ["hfr.tau3_training_exposure_legacy_fixture.v1"])
            failures = {check["id"] for check in receipt["candidate_eligibility"]["checks"] if not check["passed"]}
            self.assertIn("competitive_dataset_row_schema", failures)

    def test_missing_required_competitive_metadata_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = _eligible_rows()[0]
            del row["metadata"]["target_tool_name"]
            dataset = root / "train.jsonl"
            _write_jsonl(dataset, [row])

            with self.assertRaisesRegex(Tau3ExposureError, "missing metadata.target_tool_name"):
                build_tau3_exposure_ledger(dataset, root / "exposure", seed=1, epochs=2, batch_size=2, gradient_accumulation_steps=2)

    def test_estimated_token_counts_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = _eligible_rows()[0]
            row["metadata"]["token_counts"]["method"] = "deterministic_json_char4_estimate"
            dataset = root / "train.jsonl"
            _write_jsonl(dataset, [row])

            with self.assertRaisesRegex(Tau3ExposureError, "metadata.token_counts.method must not be estimated"):
                build_tau3_exposure_ledger(dataset, root / "exposure", seed=1, epochs=2, batch_size=2, gradient_accumulation_steps=2)

    def test_missing_source_provenance_method_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = _eligible_rows()[0]
            del row["metadata"]["source_provenance"]["method"]
            dataset = root / "train.jsonl"
            _write_jsonl(dataset, [row])

            with self.assertRaisesRegex(Tau3ExposureError, "metadata.source_provenance.method"):
                build_tau3_exposure_ledger(dataset, root / "exposure", seed=1, epochs=2, batch_size=2, gradient_accumulation_steps=2)

    def test_cli_build_and_validate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "train.jsonl"
            _write_jsonl(dataset, _eligible_rows())
            build = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "build_tau3_exposure_ledger.py"),
                    "--dataset",
                    str(dataset),
                    "--out",
                    str(root / "exposure"),
                    "--seed",
                    "505",
                    "--epochs",
                    "2",
                    "--batch-size",
                    "2",
                    "--gradient-accumulation-steps",
                    "2",
                ],
                check=False,
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.assertEqual(build.returncode, 0, build.stderr)
            payload = json.loads(build.stdout)
            self.assertTrue(payload["candidate_eligible"])

            validate = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "validate_tau3_exposure_ledger.py"),
                    "--dataset",
                    str(dataset),
                    "--receipt",
                    str(root / "exposure" / "training_exposure_receipt.json"),
                ],
                check=False,
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.assertEqual(validate.returncode, 0, validate.stderr)
            self.assertTrue(json.loads(validate.stdout)["passed"])


if __name__ == "__main__":
    unittest.main()
