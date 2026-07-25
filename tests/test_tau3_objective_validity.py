from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

from flightrecorder.tau3_objective_validity import (
    build_tau3_objective_validity_report,
    parent_trajectory_sha256,
    validate_tau3_objective_validity_report,
)


ROOT = Path(__file__).resolve().parents[1]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _fixture(root: Path) -> tuple[Path, Path]:
    domain_hashes = {
        "airline": ("a" * 64, "b" * 64, "I can help after confirming the flight change."),
        "retail": ("c" * 64, "d" * 64, "I can look up the order before changing it."),
        "telecom": ("e" * 64, "f" * 64, "I should not do that. I will ask for confirmation first."),
    }
    parents = []
    for domain, (system_hash, tool_hash, target) in domain_hashes.items():
        is_negative = domain == "telecom"
        parents.append(
            {
                "trajectory_id": f"tau-{domain}-001",
                "domain": domain,
                "system_prompt_sha256": system_hash,
                "ordered_tool_catalog_sha256": tool_hash,
                "assistant_decisions": [
                    {
                        "decision_ordinal": 0,
                        "eligible_for_supervision": True,
                        "kind": "unsafe_prefix_correction" if is_negative else "assistant_message",
                        "target_sha256": _canonical_sha256(target),
                        "safe_correction_required": is_negative,
                    },
                    {
                        "decision_ordinal": 1,
                        "eligible_for_supervision": False,
                        "kind": "intermediate_observation",
                        "target_sha256": str(len(domain)) * 64,
                        "safe_correction_required": False,
                    },
                ],
            }
        )
    parent_hashes = {row["trajectory_id"]: parent_trajectory_sha256(row) for row in parents}
    rows = []
    for domain, (system_hash, tool_hash, target) in domain_hashes.items():
        trajectory_id = f"tau-{domain}-001"
        rows.append(
            _training_row(
                row_id=f"{domain}-001-d0",
                trajectory_id=trajectory_id,
                domain=domain,
                ordinal=0,
                parent_hash=parent_hashes[trajectory_id],
                system_hash=system_hash,
                tool_hash=tool_hash,
                target=target,
                negative_prefix=domain == "telecom",
            )
        )
    parent_path = root / "parents.jsonl"
    train_path = root / "train.jsonl"
    _write_jsonl(parent_path, parents)
    _write_jsonl(train_path, rows)
    return train_path, parent_path


def _training_row(
    *,
    row_id: str,
    trajectory_id: str,
    domain: str,
    ordinal: int,
    parent_hash: str,
    system_hash: str,
    tool_hash: str,
    target: str,
    negative_prefix: bool,
) -> dict[str, Any]:
    prompt_tokens = 11 if not negative_prefix else 14
    target_tokens = 7
    negative_action = 0 if not negative_prefix else 3
    input_token_ids = list(range(100, 100 + prompt_tokens + target_tokens))
    loss_mask = [0] * max(0, prompt_tokens - 1) + [1] * target_tokens
    return {
        "row_id": row_id,
        "trajectory_id": trajectory_id,
        "domain": domain,
        "decision_ordinal": ordinal,
        "parent_trajectory_sha256": parent_hash,
        "supervised_decision": True,
        "target_text": target,
        "target_sha256": _canonical_sha256(target),
        "target_kind": "safe_correction" if negative_prefix else "positive_action",
        "negative_prefix": negative_prefix,
        "system_prompt_sha256": system_hash,
        "ordered_tool_catalog_sha256": tool_hash,
        "token_accounting": {
            "prompt_tokens": prompt_tokens,
            "target_tokens": target_tokens,
            "total_tokens": prompt_tokens + target_tokens,
            "masked_tokens": prompt_tokens,
            "supervised_tokens": target_tokens,
        },
        "target_boundaries": {
            "start_token": prompt_tokens,
            "end_token": prompt_tokens + target_tokens,
            "complete_message": True,
            "truncated": False,
        },
        "token_class_counts": {
            "prompt": 2,
            "tool_result": 2,
            "user": 2,
            "private_reference": 1,
            "grader": 1,
            "negative_action": negative_action,
            "assistant_target": target_tokens,
            "other_prompt": prompt_tokens - 8 - negative_action,
        },
        "masked_token_class_counts": {
            "prompt": 2,
            "tool_result": 2,
            "user": 2,
            "private_reference": 1,
            "grader": 1,
            "negative_action": negative_action,
            "other_prompt": prompt_tokens - 8 - negative_action,
        },
        "input_token_ids": input_token_ids,
        "loss_mask": loss_mask,
        "input_token_ids_sha256": _canonical_sha256(input_token_ids),
        "loss_mask_sha256": _canonical_sha256(loss_mask),
        "loss_mask_semantics": "mlx_lm_shifted_targets_v1",
    }


class Tau3ObjectiveValidityTests(unittest.TestCase):
    def test_report_passes_and_replays_for_complete_per_decision_export(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_path, parent_path = _fixture(root)
            report = build_tau3_objective_validity_report(
                training_export_path=train_path,
                parent_trajectories_path=parent_path,
                source_root=root,
            )
            self.assertTrue(report["passed"])
            self.assertEqual(report["eligible_decision_count"], 3)
            self.assertEqual(report["negative_prefix_count"], 1)
            self.assertEqual(report["complete_message_truncation_count"], 0)
            self.assertEqual(set(report["system_prompt_sha256_by_domain"]), {"airline", "retail", "telecom"})
            self.assertNotEqual(
                report["system_prompt_sha256_by_domain"]["airline"],
                report["system_prompt_sha256_by_domain"]["retail"],
            )
            self.assertEqual(report["sources"]["training_export"]["path"], "train.jsonl")
            report_path = root / "objective_validity.json"
            _write_json(report_path, report)
            replay = validate_tau3_objective_validity_report(report_path)
            self.assertTrue(replay["passed"], replay)

    def test_missing_eligible_decision_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_path, parent_path = _fixture(root)
            rows = [json.loads(line) for line in train_path.read_text(encoding="utf-8").splitlines()]
            _write_jsonl(train_path, rows[:2])
            report = build_tau3_objective_validity_report(
                training_export_path=train_path,
                parent_trajectories_path=parent_path,
                source_root=root,
            )
            self.assertFalse(report["passed"])
            failed_ids = {check["id"] for check in report["checks"] if not check["passed"]}
            self.assertIn("every_eligible_assistant_decision_supervised", failed_ids)

    def test_negative_prefix_requires_masked_negative_action_and_safe_correction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_path, parent_path = _fixture(root)
            rows = [json.loads(line) for line in train_path.read_text(encoding="utf-8").splitlines()]
            rows[2]["target_kind"] = "unsafe_tool_call"
            rows[2]["masked_token_class_counts"]["negative_action"] = 0
            _write_jsonl(train_path, rows)
            report = build_tau3_objective_validity_report(
                training_export_path=train_path,
                parent_trajectories_path=parent_path,
                source_root=root,
            )
            self.assertFalse(report["passed"])
            failed_ids = {check["id"] for check in report["checks"] if not check["passed"]}
            self.assertIn("row.telecom-001-d0.negative_prefix_targets_safe_correction", failed_ids)
            self.assertIn("row.telecom-001-d0.negative_action_tokens_present_and_masked", failed_ids)

    def test_parent_hash_and_target_hash_are_replayed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_path, parent_path = _fixture(root)
            rows = [json.loads(line) for line in train_path.read_text(encoding="utf-8").splitlines()]
            rows[0]["parent_trajectory_sha256"] = "0" * 64
            rows[0]["target_sha256"] = "1" * 64
            _write_jsonl(train_path, rows)
            report = build_tau3_objective_validity_report(
                training_export_path=train_path,
                parent_trajectories_path=parent_path,
                source_root=root,
            )
            self.assertFalse(report["passed"])
            failed_ids = {check["id"] for check in report["checks"] if not check["passed"]}
            self.assertIn("row.airline-001-d0.parent_hash_matches", failed_ids)
            self.assertIn("row.airline-001-d0.target_sha256_matches_text", failed_ids)
            self.assertIn("row.airline-001-d0.target_sha256_matches_parent", failed_ids)

    def test_token_accounting_and_complete_message_truncation_fail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_path, parent_path = _fixture(root)
            rows = [json.loads(line) for line in train_path.read_text(encoding="utf-8").splitlines()]
            rows[0]["token_accounting"]["supervised_tokens"] = 3
            rows[0]["target_boundaries"]["truncated"] = True
            _write_jsonl(train_path, rows)
            report = build_tau3_objective_validity_report(
                training_export_path=train_path,
                parent_trajectories_path=parent_path,
                source_root=root,
            )
            self.assertFalse(report["passed"])
            self.assertEqual(report["complete_message_truncation_count"], 1)
            failed_ids = {check["id"] for check in report["checks"] if not check["passed"]}
            self.assertIn("row.airline-001-d0.target_boundaries_complete_message", failed_ids)
            self.assertIn("row.airline-001-d0.loss_tokens_match_target", failed_ids)

    def test_loss_mask_arrays_reject_protected_prompt_unmasking_with_plausible_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_path, parent_path = _fixture(root)
            rows = [json.loads(line) for line in train_path.read_text(encoding="utf-8").splitlines()]
            rows[2]["loss_mask"][0] = 1
            rows[2]["loss_mask"][-1] = 0
            rows[2]["loss_mask_sha256"] = _canonical_sha256(rows[2]["loss_mask"])
            _write_jsonl(train_path, rows)

            report = build_tau3_objective_validity_report(
                training_export_path=train_path,
                parent_trajectories_path=parent_path,
                source_root=root,
            )

            self.assertFalse(report["passed"])
            failed_ids = {check["id"] for check in report["checks"] if not check["passed"]}
            self.assertIn(
                "row.telecom-001-d0.loss_mask_exact_mlx_shifted_prompt_target_boundary",
                failed_ids,
            )

    def test_domain_and_per_domain_hash_mismatches_fail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_path, parent_path = _fixture(root)
            rows = [json.loads(line) for line in train_path.read_text(encoding="utf-8").splitlines()]
            rows[0]["domain"] = "retail"
            rows[1]["system_prompt_sha256"] = rows[0]["system_prompt_sha256"]
            _write_jsonl(train_path, rows)
            report = build_tau3_objective_validity_report(
                training_export_path=train_path,
                parent_trajectories_path=parent_path,
                source_root=root,
            )
            self.assertFalse(report["passed"])
            self.assertIsNone(report["system_prompt_sha256_by_domain"])
            failed_ids = {check["id"] for check in report["checks"] if not check["passed"]}
            self.assertIn("row.airline-001-d0.domain_matches_parent", failed_ids)
            self.assertIn("system_prompt_hash_stable_by_domain", failed_ids)

    def test_builder_rejects_sources_outside_report_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = root / "sources"
            report_dir = root / "bundle"
            train_path, parent_path = _fixture(source_dir)
            with self.assertRaisesRegex(ValueError, "inside objective-validity report directory"):
                build_tau3_objective_validity_report(
                    training_export_path=train_path,
                    parent_trajectories_path=parent_path,
                    source_root=report_dir,
                )

    def test_validator_rejects_parent_directory_source_ref(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_path, parent_path = _fixture(root)
            report = build_tau3_objective_validity_report(
                training_export_path=train_path,
                parent_trajectories_path=parent_path,
                source_root=root,
            )
            report["sources"]["training_export"]["path"] = "../train.jsonl"
            report_path = root / "objective_validity.json"
            _write_json(report_path, report)
            replay = validate_tau3_objective_validity_report(report_path)
            self.assertFalse(replay["passed"])
            self.assertTrue(any("must not contain '..'" in error for error in replay["errors"]))

    def test_saved_report_fails_if_source_changes_after_build(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_path, parent_path = _fixture(root)
            report = build_tau3_objective_validity_report(
                training_export_path=train_path,
                parent_trajectories_path=parent_path,
                source_root=root,
            )
            report_path = root / "objective_validity.json"
            _write_json(report_path, report)
            rows = [json.loads(line) for line in train_path.read_text(encoding="utf-8").splitlines()]
            rows[0]["target_text"] = "changed"
            _write_jsonl(train_path, rows)
            replay = validate_tau3_objective_validity_report(report_path)
            self.assertFalse(replay["passed"])
            self.assertTrue(any("sha256 mismatch" in error for error in replay["errors"]))

    def test_cli_build_and_validate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_path, parent_path = _fixture(root)
            report_path = root / "objective_validity.json"
            build = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "build_tau3_objective_validity.py"),
                    "--training-export",
                    str(train_path),
                    "--parent-trajectories",
                    str(parent_path),
                    "--out",
                    str(report_path),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(build.returncode, 0, build.stderr + build.stdout)
            validate = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "validate_tau3_objective_validity.py"),
                    "--report",
                    str(report_path),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(validate.returncode, 0, validate.stderr + validate.stdout)
