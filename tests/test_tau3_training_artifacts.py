from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

from flightrecorder.schema_registry import check_schema_contract
from flightrecorder.tau3_training_artifacts import (
    REQUIRED_ARTIFACTS,
    build_bundle_manifest,
    validate_tau3_training_bundle,
)


ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _base_bundle(root: Path, *, mode: str = "production", ready: bool = True) -> Path:
    bundle = root / "bundle"
    sealed_hash = "f" * 64
    split_hashes = {
        "train": "a" * 64,
        "development": "b" * 64,
        "sealed": "c" * 64,
    }
    domains = ["airline", "retail", "telecom"]
    behavior_rows = [
        ("tau-air-1", "airline", ["success"], False),
        ("tau-ret-1", "retail", ["correction"], False),
        ("tau-tel-1", "telecom", ["clarification", "refusal"], False),
        ("tau-air-2", "airline", ["recovery"], False),
        ("tau-ret-2", "retail", ["policy_failure"], True),
        ("tau-tel-2", "telecom", ["harmful", "unnecessary_mutation"], True),
        ("tau-air-3", "airline", ["hallucinated_tool"], True),
        ("tau-ret-3", "retail", ["premature_completion"], True),
    ]
    trajectories = [
        {
            "id": row_id,
            "domain": domain,
            "tokens": 100,
            "behavior_tags": tags,
            "unsafe": unsafe,
            "safety_label": "unsafe" if unsafe else "safe",
            "prompt_hash": f"{idx:064x}",
        }
        for idx, (row_id, domain, tags, unsafe) in enumerate(behavior_rows, start=1)
    ]
    admitted = [
        {
            "id": row["id"],
            "accepted": True,
            "evidence": {
                "lineage": "lineage.json",
                "state_transition": "state-delta.json",
                "executable": True,
                "safety": "reviewed",
                "reviewer": "local-reviewer",
            },
        }
        for row in trajectories
    ]
    rejected = [{"id": "tau-rejected-1", "reason": "unsafe positive target"}]
    sft = [{"id": row["id"], "label": "positive", "messages": []} for row in trajectories if not row["unsafe"]]
    action_sft = [{"id": row["id"], "label": "positive", "actions": []} for row in trajectories if not row["unsafe"]]
    dpo = [
        {
            "id": "pref-1",
            "chosen": {"id": "tau-air-1"},
            "rejected": {"id": "tau-rejected-1"},
            "preference_evidence": {"reviewer": "local-reviewer", "basis": "executable outcome"},
        }
    ]

    json_artifacts: dict[str, dict[str, Any]] = {
        "protocol/protocol_manifest.json": {"schema_version": "hfr.tau3.protocol_manifest.v1", "domains": domains},
        "protocol/tau_revision.json": {
            "schema_version": "hfr.tau3.revision.v1",
            "repository": "local/tau3-fixture",
            "local_git": True,
            "revision": "1234567890abcdef",
            "task_schema_version": "tau3-fixture-v1",
            "split_hashes": split_hashes,
        },
        "protocol/split_manifest.json": {
            "schema_version": "hfr.tau3.split_manifest.v1",
            "strategy": "task_family_before_generation",
            "splits": {
                "train": {"sha256": split_hashes["train"]},
                "development": {"sha256": split_hashes["development"]},
                "sealed": {"sha256": split_hashes["sealed"]},
            },
        },
        "protocol/harness_contract.json": {
            "schema_version": "hfr.tau3.harness_contract.v1",
            "fixed": True,
            "no_test_time_search": True,
            "domains": domains,
            "decoding": {"temperature": 0, "top_p": 1},
            "seeds": [17, 23, 42],
            "system_prompt_sha256": "e" * 64,
            "tool_order": "frozen",
            "context_window": 8192,
            "turn_limit": 30,
            "retry_policy": "none",
            "local_only": True,
        },
        "protocol/model_freeze.json": {
            "schema_version": "hfr.tau3.model_freeze.v1",
            "base_model": {
                "name": "local-dense-8b",
                "parameters_billion": 8.0,
                "architecture": "dense decoder",
                "revision": "base-rev",
                "license": "apache-2.0",
                "quantization": "4-bit",
                "tokenizer": "fixture-tokenizer",
                "chat_template": "fixture-chat-v1",
                "model_card_url": "local:fixture-model-card",
            },
            "comparators": [
                {
                    "name": "comparator-a-7b",
                    "parameters_billion": 7.0,
                    "architecture": "dense decoder",
                    "revision": "cmp-a",
                    "license": "apache-2.0",
                    "quantization": "4-bit",
                },
                {
                    "name": "comparator-b-9b",
                    "parameters_billion": 9.0,
                    "architecture": "dense decoder",
                    "revision": "cmp-b",
                    "license": "mit",
                    "quantization": "4-bit",
                },
            ],
        },
        "protocol/budget.json": {
            "schema_version": "hfr.tau3.budget.v1",
            "max_seconds": 604800,
            "reserved_final_eval": True,
            "reserved_final_eval_seconds": 86400,
            "stages": {"generation": 172800, "search": 172800, "training": 172800, "final_eval": 86400},
            "local_only": True,
        },
        "sealed/sealed_manifest.json": {
            "schema_version": "hfr.tau3.sealed_manifest.v1",
            "quarantine_predates_generation": True,
            "prompt_hashes": [sealed_hash],
        },
        "generation/balance_report.json": {
            "schema_version": "hfr.tau3.balance_report.v1",
            "token_share_by_domain": {"airline": 0.375, "retail": 0.375, "telecom": 0.25},
        },
        "generation/contamination_report.json": {
            "schema_version": "hfr.tau3.contamination_report.v1",
            "passed": True,
            "leakage_found": False,
        },
        "generation/redaction_report.json": {
            "schema_version": "hfr.tau3.redaction_report.v1",
            "passed": True,
            "secrets_found": False,
        },
        "generation/license_report.json": {"schema_version": "hfr.tau3.license_report.v1", "passed": True},
        "generation/dataset_identity.json": {
            "schema_version": "hfr.tau3.dataset_identity.v1",
            "dataset_sha256": "d" * 64,
            "deletion_lineage": {"key": "trajectory_id"},
        },
        "training/model_manifest.json": {"schema_version": "hfr.tau3.model_manifest.v1", "local_only": True},
        "training/dataset_manifest.json": {"schema_version": "hfr.tau3.dataset_manifest.v1", "local_only": True},
        "training/mlx_qlora_plan.json": {
            "schema_version": "hfr.tau3.mlx_qlora_plan.v1",
            "passed": True,
            "method": "QLoRA LoRA adapter",
            "quantization": "4-bit",
            "local_only": True,
            "command_argv": [
                "python",
                "-m",
                "mlx_lm.lora",
                "--model",
                "model_input",
                "--train",
                "--data",
                "input_export",
                "--adapter-path",
                "adapter_output",
            ],
            "resume": {"enabled": True},
            "stop_conditions": ["divergence", "budget"],
            "output_contract": {"adapter_only": True},
            "tokenizer_compatibility": {
                "schema_version": "hfr.tau3_tokenizer_compatibility.v1",
                "passed": True,
                "checked": True,
                "row_count": 3,
                "max_rendered_tokens": 1024,
                "max_seq_length": 4096,
                "harness_context_window": 8192,
                "over_max_seq_length_count": 0,
                "over_context_window_count": 0,
                "local_only": True,
                "network": False,
                "training_started": False,
            },
        },
        "training/recipe_space.json": {
            "schema_version": "hfr.tau3.recipe_space.v1",
            "bounded": True,
            "max_trials": 6,
            "development_only": True,
            "sealed_used": False,
        },
        "training/candidate_selection_contract.json": {
            "schema_version": "hfr.tau3.candidate_selection_contract.v1",
            "passed": True,
            "no_test_time_search": True,
            "development_only": True,
            "sealed_used": False,
            "one_untouched_checkpoint": True,
            "primary_metric": "macro_pass_1",
            "safety_non_inferiority_margin": 0.01,
            "per_domain_non_inferiority_margin": 0.03,
            "bootstrap": {"kind": "paired", "confidence": 0.95},
        },
        "training/agentic_training_plan.json": {
            "schema_version": "hfr.tau3.agentic_training_plan.v1",
            "training_started": False,
            "weights_updated": False,
            "sealed_evaluation_started": False,
            "promotion_applied": False,
        },
        "training/runtime_preflight.json": {
            "schema_version": "hfr.tau3.runtime_preflight.v1",
            "passed": True,
            "local_only": True,
            "allow_network": False,
        },
        "training/trainer_preflight.json": {
            "schema_version": "hfr.tau3.trainer_preflight.v1",
            "passed": True,
            "training_started": False,
        },
        "training/trainer_launch_check.json": {
            "schema_version": "hfr.tau3.trainer_launch_check.v1",
            "passed": True,
            "approved_command": "python -m mlx_lm.lora --model model_input --train --data input_export --adapter-path adapter_output",
            "executed": False,
            "training_started": False,
            "weights_updated": False,
            "sealed_evaluation_started": False,
            "promotion_applied": False,
        },
        "training/trainer_archive/trainer_archive.json": {
            "schema_version": "hfr.tau3.trainer_archive.v1",
            "passed": True,
            "training_started": False,
            "weights_updated": False,
        },
        "training/trainer_archive_check.json": {
            "schema_version": "hfr.tau3.trainer_archive_check.v1",
            "passed": True,
            "training_started": False,
        },
        "training/trainer_consumer_plan.json": {"schema_version": "hfr.tau3.trainer_consumer_plan.v1", "passed": True},
        "rehearsal/rehearsal_result.json": {
            "schema_version": "hfr.tau3.rehearsal_result.v1",
            "tiny": True,
            "non_sealed": True,
            "passed": True,
            "sealed_evaluation_started": False,
            "training_started": False,
        },
        "evidence/evidence_bundle.json": {
            "schema_version": "hfr.tau3.evidence_bundle.v1",
            "passed": True,
            "hash_checked": True,
        },
    }
    json_artifacts["protocol/protocol_manifest.json"]["signature"] = _canonical_sha256({
        "protocol_manifest": json_artifacts["protocol/protocol_manifest.json"],
        "tau_revision": json_artifacts["protocol/tau_revision.json"],
        "split_manifest": json_artifacts["protocol/split_manifest.json"],
        "harness_contract": json_artifacts["protocol/harness_contract.json"],
        "model_freeze": json_artifacts["protocol/model_freeze.json"],
        "budget": json_artifacts["protocol/budget.json"],
    })
    for rel, payload in json_artifacts.items():
        _write_json(bundle / rel, payload)
    _write_jsonl(bundle / "generation/trajectories.jsonl", trajectories)
    _write_jsonl(bundle / "generation/admission_ledger.jsonl", admitted)
    _write_jsonl(bundle / "generation/rejection_ledger.jsonl", rejected)
    _write_jsonl(bundle / "exports/sft.jsonl", sft)
    _write_jsonl(bundle / "exports/action_sft.jsonl", action_sft)
    _write_jsonl(bundle / "exports/dpo.jsonl", dpo)
    (bundle / "exports").mkdir(parents=True, exist_ok=True)
    (bundle / "exports/DATASET_CARD.md").write_text("# Tau-3 core dataset\n\nLocal-only synthetic bundle.\n", encoding="utf-8")

    export_hashes = {}
    for rel in ("exports/sft.jsonl", "exports/action_sft.jsonl", "exports/dpo.jsonl"):
        export_hashes[Path(rel).stem] = hashlib.sha256((bundle / rel).read_bytes()).hexdigest()
    _write_json(
        bundle / "exports/manifest.json",
        {
            "schema_version": "hfr.tau3.export_manifest.v1",
            "counts": {"sft": len(sft), "action_sft": len(action_sft), "dpo": len(dpo)},
            "hashes": export_hashes,
        },
    )
    _rewrite_manifest(bundle, mode=mode, ready=ready)
    return bundle


def _rewrite_manifest(bundle: Path, *, mode: str = "production", ready: bool = True) -> None:
    evidence_rows = []
    for role, rel_path in REQUIRED_ARTIFACTS:
        if role == "evidence_bundle":
            continue
        path = bundle / rel_path
        evidence_rows.append({
            "role": role,
            "path": rel_path,
            "size": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })
    _write_json(
        bundle / "evidence/evidence_bundle.json",
        {
            "schema_version": "hfr.tau3.evidence_bundle.v1",
            "passed": True,
            "hash_checked": True,
            "artifacts": evidence_rows,
        },
    )
    manifest = build_bundle_manifest(bundle, bundle_mode=mode, ready_for_training=ready, created_at="2026-07-22T00:00:00Z")
    _write_json(bundle / "manifest.json", manifest)


def _mutate_json(path: Path, mutate) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutate(payload)
    _write_json(path, payload)


class Tau3TrainingArtifactValidatorTests(unittest.TestCase):
    def test_happy_path_passes_strict_and_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bundle = _base_bundle(Path(tmp))
            result = validate_tau3_training_bundle(bundle, strict=True)
            self.assertTrue(result["passed"], [check for check in result["checks"] if not check["passed"]])
            schema = check_schema_contract(
                json.loads((bundle / "manifest.json").read_text(encoding="utf-8")),
                name_or_id="tau3_training_bundle",
            )
            self.assertTrue(schema["passed"], schema["errors"])

    def test_cli_writes_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cwd = Path(tmp) / "cwd"
            cwd.mkdir()
            bundle = _base_bundle(Path(tmp))
            out = Path(tmp) / "validation.json"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "validate_tau3_training_artifacts.py"),
                    "--bundle",
                    str(bundle),
                    "--strict",
                    "--out",
                    str(out),
                ],
                cwd=cwd,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr + proc.stdout)
            receipt = json.loads(out.read_text(encoding="utf-8"))
            self.assertTrue(receipt["passed"])

    def test_rehearsal_requires_explicit_allow_rehearsal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bundle = _base_bundle(Path(tmp), mode="rehearsal", ready=False)
            strict = validate_tau3_training_bundle(bundle, strict=True)
            allowed = validate_tau3_training_bundle(bundle, strict=True, allow_rehearsal=True)
            self.assertFalse(strict["passed"])
            self.assertIn("strict_requires_production_mode", {check["id"] for check in strict["checks"] if not check["passed"]})
            self.assertTrue(allowed["passed"], [check for check in allowed["checks"] if not check["passed"]])
            self.assertFalse(allowed["ready_for_training"])

    def test_rejects_missing_required_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bundle = _base_bundle(Path(tmp))
            (bundle / "generation/redaction_report.json").unlink()
            result = validate_tau3_training_bundle(bundle, strict=True)
            self.assertFalse(result["passed"])
            self.assertIn("artifact_exists:redaction_report", {check["id"] for check in result["checks"] if not check["passed"]})

    def test_rejects_tampered_artifact_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bundle = _base_bundle(Path(tmp))
            _mutate_json(bundle / "protocol/budget.json", lambda payload: payload.__setitem__("note", "tampered"))
            result = validate_tau3_training_bundle(bundle, strict=True)
            self.assertFalse(result["passed"])
            self.assertIn("artifact_sha256_replays:budget", {check["id"] for check in result["checks"] if not check["passed"]})

    def test_accepts_native_runtime_launch_manifest_and_dpo_shapes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bundle = _base_bundle(Path(tmp))
            _write_json(
                bundle / "training/runtime_preflight.json",
                {
                    "schema_version": "hfr.agentic_training_runtime_preflight.v1",
                    "passed": True,
                    "execution_boundary": {
                        "cloud_jobs_started": False,
                        "model_downloads_started": False,
                        "training_started": False,
                        "weights_updated": False,
                    },
                },
            )
            _write_json(
                bundle / "training/trainer_launch_check.json",
                {
                    "schema_version": "hfr.trainer_launch_check.v1",
                    "passed": True,
                    "approved_command": {
                        "argv": [
                            "python",
                            "-m",
                            "mlx_lm.lora",
                            "--model",
                            "model_input",
                            "--train",
                            "--data",
                            "input_export",
                            "--adapter-path",
                            "adapter_output",
                        ]
                    },
                    "executed": False,
                    "training_started": False,
                    "weights_updated": False,
                    "sealed_evaluation_started": False,
                    "promotion_applied": False,
                },
            )
            _write_jsonl(
                bundle / "exports/dpo.jsonl",
                [
                    {
                        "id": "pref-native",
                        "chosen": "tau-air-1",
                        "rejected": "tau-rejected-1",
                        "chosen_episode_id": "tau-air-1",
                        "rejected_episode_id": "tau-rejected-1",
                        "score_gap": 0.25,
                        "reason": "chosen row has executable success and safer state mutation",
                        "label_provenance": {
                            "chosen": {"reviewer": "local-reviewer"},
                            "rejected": {"reviewer": "local-reviewer"},
                        },
                    }
                ],
            )
            import hashlib

            hashes = {
                "sft": hashlib.sha256((bundle / "exports/sft.jsonl").read_bytes()).hexdigest(),
                "action_sft": hashlib.sha256((bundle / "exports/action_sft.jsonl").read_bytes()).hexdigest(),
                "dpo": hashlib.sha256((bundle / "exports/dpo.jsonl").read_bytes()).hexdigest(),
            }
            _write_json(
                bundle / "exports/manifest.json",
                {
                    "schema_version": "hfr.rl.manifest.v1",
                    "sft_count": 4,
                    "action_sft_count": 4,
                    "dpo_count": 1,
                    "artifact_fingerprints": hashes,
                },
            )
            _rewrite_manifest(bundle)
            result = validate_tau3_training_bundle(bundle, strict=True)
            self.assertTrue(result["passed"], [check for check in result["checks"] if not check["passed"]])

    def test_rejects_sealed_prompt_leak(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bundle = _base_bundle(Path(tmp))
            _mutate_json(bundle / "exports/manifest.json", lambda payload: payload.__setitem__("leaked_hash", "f" * 64))
            _rewrite_manifest(bundle)
            result = validate_tau3_training_bundle(bundle, strict=True)
            self.assertFalse(result["passed"])
            self.assertIn("sealed_prompt_hashes_absent_from_training_artifacts", {check["id"] for check in result["checks"] if not check["passed"]})

    def test_rejects_unsafe_positive_sft_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bundle = _base_bundle(Path(tmp))
            with (bundle / "exports/sft.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"id": "tau-ret-2", "label": "positive"}) + "\n")
            _mutate_json(
                bundle / "exports/manifest.json",
                lambda payload: payload["counts"].__setitem__("sft", payload["counts"]["sft"] + 1),
            )
            _rewrite_manifest(bundle)
            result = validate_tau3_training_bundle(bundle, strict=True)
            self.assertFalse(result["passed"])
            self.assertIn("unsafe_rows_not_positive_sft", {check["id"] for check in result["checks"] if not check["passed"]})

    def test_rejects_out_of_band_base_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bundle = _base_bundle(Path(tmp))
            _mutate_json(bundle / "protocol/model_freeze.json", lambda payload: payload["base_model"].__setitem__("parameters_billion", 13.0))
            _rewrite_manifest(bundle)
            result = validate_tau3_training_bundle(bundle, strict=True)
            self.assertFalse(result["passed"])
            self.assertIn("base_model_7_to_9b_dense", {check["id"] for check in result["checks"] if not check["passed"]})

    def test_rejects_missing_second_comparator(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bundle = _base_bundle(Path(tmp))
            _mutate_json(bundle / "protocol/model_freeze.json", lambda payload: payload.__setitem__("comparators", payload["comparators"][:1]))
            _rewrite_manifest(bundle)
            result = validate_tau3_training_bundle(bundle, strict=True)
            self.assertFalse(result["passed"])
            self.assertIn("at_least_two_eligible_comparators", {check["id"] for check in result["checks"] if not check["passed"]})

    def test_rejects_cloud_or_network_flags(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bundle = _base_bundle(Path(tmp))
            _mutate_json(bundle / "training/runtime_preflight.json", lambda payload: payload.__setitem__("allow_network", True))
            _rewrite_manifest(bundle)
            result = validate_tau3_training_bundle(bundle, strict=True)
            failed = {check["id"] for check in result["checks"] if not check["passed"]}
            self.assertFalse(result["passed"])
            self.assertIn("all_local_only_flags_hold", failed)

    def test_rejects_private_paths_and_credentials_in_any_required_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bundle = _base_bundle(Path(tmp))
            _mutate_json(
                bundle / "training/dataset_manifest.json",
                lambda payload: payload.__setitem__("local_path", "/Users/alice/.cache/private-dataset"),
            )
            _mutate_json(
                bundle / "training/model_manifest.json",
                lambda payload: payload.__setitem__("api_key", "hf_abcdefghijklmnop"),
            )
            _rewrite_manifest(bundle)

            result = validate_tau3_training_bundle(bundle, strict=True)

            failed = {check["id"] for check in result["checks"] if not check["passed"]}
            self.assertFalse(result["passed"])
            self.assertIn("private_local_paths_absent", failed)
            self.assertIn("credential_like_values_absent", failed)

    def test_rejects_remote_or_mismatched_production_command_bindings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bundle = _base_bundle(Path(tmp))
            remote_argv = [
                "python",
                "-m",
                "mlx_lm.lora",
                "--model",
                "mlx-community/remote-model",
                "--train",
                "--data",
                "../other-data",
                "--adapter-path",
                "adapter_output",
            ]
            _mutate_json(
                bundle / "training/mlx_qlora_plan.json",
                lambda payload: payload.__setitem__("command_argv", remote_argv),
            )
            _mutate_json(
                bundle / "training/trainer_launch_check.json",
                lambda payload: payload.__setitem__("approved_command", " ".join(remote_argv)),
            )
            _rewrite_manifest(bundle)

            result = validate_tau3_training_bundle(bundle, strict=True)

            failed = {check["id"] for check in result["checks"] if not check["passed"]}
            self.assertFalse(result["passed"])
            self.assertIn("production_launch_command_uses_local_bundle_bindings", failed)

    def test_rejects_budget_over_seven_days(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bundle = _base_bundle(Path(tmp))
            _mutate_json(bundle / "protocol/budget.json", lambda payload: payload.__setitem__("max_seconds", 604801))
            _rewrite_manifest(bundle)
            result = validate_tau3_training_bundle(bundle, strict=True)
            self.assertFalse(result["passed"])
            self.assertIn("budget_max_seven_days", {check["id"] for check in result["checks"] if not check["passed"]})

    def test_rejects_unapproved_qlora_or_candidate_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bundle = _base_bundle(Path(tmp))
            _mutate_json(
                bundle / "training/mlx_qlora_plan.json",
                lambda payload: payload.__setitem__("passed", False),
            )
            _mutate_json(
                bundle / "training/candidate_selection_contract.json",
                lambda payload: payload.__setitem__("passed", False),
            )
            _rewrite_manifest(bundle)

            result = validate_tau3_training_bundle(bundle, strict=True)

            failed = {check["id"] for check in result["checks"] if not check["passed"]}
            self.assertFalse(result["passed"])
            self.assertIn("mlx_qlora_plan_approved", failed)
            self.assertIn("candidate_selection_contract_approved", failed)

    def test_rejects_rows_beyond_tokenizer_sequence_budget(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bundle = _base_bundle(Path(tmp))
            def exceed(payload):
                compatibility = payload["tokenizer_compatibility"]
                compatibility["passed"] = False
                compatibility["max_rendered_tokens"] = 5000
                compatibility["over_max_seq_length_count"] = 1
            _mutate_json(bundle / "training/mlx_qlora_plan.json", exceed)
            _rewrite_manifest(bundle)

            result = validate_tau3_training_bundle(bundle, strict=True)

            failed = {check["id"] for check in result["checks"] if not check["passed"]}
            self.assertFalse(result["passed"])
            self.assertIn("mlx_tokenizer_sequence_budget_passed", failed)

    def test_rejects_full_training_or_launch_started(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bundle = _base_bundle(Path(tmp))
            _mutate_json(bundle / "training/trainer_launch_check.json", lambda payload: payload.__setitem__("training_started", True))
            _rewrite_manifest(bundle)
            result = validate_tau3_training_bundle(bundle, strict=True)
            failed = {check["id"] for check in result["checks"] if not check["passed"]}
            self.assertFalse(result["passed"])
            self.assertIn("trainer_launch_check_did_not_start_training", failed)
            self.assertIn("all_local_only_flags_hold", failed)

    def test_required_artifact_constant_matches_contract(self) -> None:
        roles = [role for role, _ in REQUIRED_ARTIFACTS]
        self.assertEqual(len(roles), len(set(roles)))
        self.assertIn(("evidence_bundle", "evidence/evidence_bundle.json"), REQUIRED_ARTIFACTS)


if __name__ == "__main__":
    unittest.main()
