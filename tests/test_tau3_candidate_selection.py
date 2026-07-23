from __future__ import annotations

import json
import hashlib
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

from flightrecorder.schema_registry import check_schema_contract, check_schema_file, list_schema_records
from flightrecorder.tau3_candidate_selection import (
    TAU3_CANDIDATE_LOCK_SCHEMA_VERSION,
    Tau3CandidateEntry,
    Tau3CandidateSelectionError,
    select_tau3_candidate,
)

ROOT = Path(__file__).resolve().parents[1]
REV = "2" * 40


class Tau3CandidateSelectionTests(unittest.TestCase):
    def test_selects_one_eligible_candidate_and_writes_hash_only_lock(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = _benchmark_manifest(root, "base", reward=0.0, db_match=False)
            good = _candidate_entry(root, "candidate-b", reward=1.0, db_match=True)
            better_tie = _candidate_entry(root, "candidate-a", reward=1.0, db_match=True)

            result = select_tau3_candidate(
                base_manifest_path=base,
                candidates=[good, better_tie],
                report_path=root / "selection.json",
                lock_path=root / "candidate-lock.json",
                bootstrap_samples=200,
                bootstrap_seed=9,
                created_at="2026-07-23T00:00:00+00:00",
            )

            self.assertEqual(result["selected_candidate_id"], "candidate-a")
            report = _read(root / "selection.json")
            lock = _read(root / "candidate-lock.json")
            self.assertEqual(report["selected_candidate_id"], "candidate-a")
            self.assertEqual(lock["schema_version"], TAU3_CANDIDATE_LOCK_SCHEMA_VERSION)
            self.assertEqual(lock["development_selection_report_sha256"], _sha256(root / "selection.json"))
            self.assertEqual(lock["endpoint_model_sha256"], _hash("candidate-a-endpoint-model"))
            self.assertEqual(lock["candidate_identity_sha256"], _sha256(better_tie.candidate_identity_path))
            self.assertNotEqual(lock["candidate_identity_sha256"], report["selection"]["candidate_identity_canonical_sha256"])
            self.assertEqual(report["selection"]["candidate_identity_sha256"], _sha256(better_tie.candidate_identity_path))
            self.assertTrue(lock["hashes_only"])
            self.assertFalse(lock["local_paths_included"])
            self.assertTrue(check_schema_file(root / "candidate-lock.json", "tau3_candidate_lock")["passed"])
            encoded_lock = json.dumps(lock, sort_keys=True)
            for forbidden in ("/Users/", str(root), "result_path", "messages", "raw_data", "policy", '"path"'):
                self.assertNotIn(forbidden, encoded_lock)

    def test_fails_closed_on_sealed_input_without_writing_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = _benchmark_manifest(root, "base", reward=0.0, db_match=False)
            candidate = _candidate_entry(root, "candidate-a", reward=1.0, db_match=True)
            manifest = _read(candidate.development_manifest_path)
            manifest["mode"] = "sealed"
            _write(candidate.development_manifest_path, manifest)

            with self.assertRaisesRegex(Tau3CandidateSelectionError, "sealed-mode"):
                select_tau3_candidate(
                    base_manifest_path=base,
                    candidates=[candidate],
                    report_path=root / "selection.json",
                    lock_path=root / "candidate-lock.json",
                    bootstrap_samples=200,
                )

            self.assertFalse((root / "selection.json").exists())
            self.assertFalse((root / "candidate-lock.json").exists())

    def test_rejects_tampered_raw_result_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = _benchmark_manifest(root, "base", reward=0.0, db_match=False)
            candidate = _candidate_entry(root, "candidate-a", reward=1.0, db_match=True)
            first_raw = root / "candidate-a" / "raw" / "airline-101.json"
            payload = _read(first_raw)
            payload["simulations"][0]["reward_info"]["reward"] = 0.0
            _write(first_raw, payload)

            with self.assertRaisesRegex(Tau3CandidateSelectionError, "hash does not replay"):
                select_tau3_candidate(
                    base_manifest_path=base,
                    candidates=[candidate],
                    report_path=root / "selection.json",
                    lock_path=root / "candidate-lock.json",
                    bootstrap_samples=200,
                )

    def test_rejects_run_ref_domain_seed_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = _benchmark_manifest(root, "base", reward=0.0, db_match=False)
            candidate = _candidate_entry(root, "candidate-a", reward=1.0, db_match=True)
            manifest = _read(candidate.development_manifest_path)
            manifest["run_receipts"][0]["seed"] = 202
            _write(candidate.development_manifest_path, manifest)

            with self.assertRaisesRegex(Tau3CandidateSelectionError, "domain/seed mismatch"):
                select_tau3_candidate(
                    base_manifest_path=base,
                    candidates=[candidate],
                    report_path=root / "selection.json",
                    lock_path=root / "candidate-lock.json",
                    bootstrap_samples=200,
                )

    def test_rejects_rewritten_prelaunch_source_and_receipt_refs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = _benchmark_manifest(root, "base", reward=0.0, db_match=False)
            candidate = _candidate_entry(root, "candidate-a", reward=1.0, db_match=True)
            prelaunch = _read(candidate.development_manifest_path.parent / "prelaunch_receipt.json")
            prelaunch["protocol_sha256"] = "0" * 64
            _write(candidate.development_manifest_path.parent / "prelaunch_receipt.json", prelaunch)
            with self.assertRaisesRegex(Tau3CandidateSelectionError, "prelaunch_receipt sha256"):
                select_tau3_candidate(
                    base_manifest_path=base,
                    candidates=[candidate],
                    report_path=root / "selection.json",
                    lock_path=root / "candidate-lock.json",
                    bootstrap_samples=200,
                )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = _benchmark_manifest(root, "base", reward=0.0, db_match=False)
            candidate = _candidate_entry(root, "candidate-a", reward=1.0, db_match=True)
            source_path = candidate.development_manifest_path.parent / "development.json"
            source = _read(source_path)
            source["split"] = "train"
            _write(source_path, source)
            with self.assertRaisesRegex(Tau3CandidateSelectionError, "development source sha256"):
                select_tau3_candidate(
                    base_manifest_path=base,
                    candidates=[candidate],
                    report_path=root / "selection.json",
                    lock_path=root / "candidate-lock.json",
                    bootstrap_samples=200,
                )

    def test_blocks_ineligible_safety_or_incomplete_training(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = _benchmark_manifest(root, "base", reward=1.0, db_match=True)
            unsafe = _candidate_entry(root, "candidate-unsafe", reward=1.0, db_match=False)

            with self.assertRaisesRegex(Tau3CandidateSelectionError, "no eligible"):
                select_tau3_candidate(
                    base_manifest_path=base,
                    candidates=[unsafe],
                    report_path=root / "selection.json",
                    lock_path=root / "candidate-lock.json",
                    bootstrap_samples=200,
                )

    def test_rejects_candidate_identity_and_protocol_mismatches(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = _benchmark_manifest(root, "base", reward=0.0, db_match=False)
            candidate = _candidate_entry(root, "candidate-a", reward=1.0, db_match=True)
            identity = _read(candidate.candidate_identity_path)
            identity["adapter_tree_sha256"] = "0" * 64
            _write(candidate.candidate_identity_path, identity)
            _rebind_candidate_identity(candidate)

            with self.assertRaisesRegex(Tau3CandidateSelectionError, "adapter_tree_sha256"):
                select_tau3_candidate(
                    base_manifest_path=base,
                    candidates=[candidate],
                    report_path=root / "selection.json",
                    lock_path=root / "candidate-lock.json",
                    bootstrap_samples=200,
                )

            candidate = _candidate_entry(root, "candidate-b", reward=1.0, db_match=True)
            receipt = _read(candidate.training_receipt_path)
            receipt["training_binding"]["protocol"]["sha256"] = "f" * 64
            _write(candidate.training_receipt_path, receipt)
            identity = _read(candidate.candidate_identity_path)
            identity["training_receipt_sha256"] = _sha256(candidate.training_receipt_path)
            identity["final_training_receipt_sha256"] = _sha256(candidate.training_receipt_path)
            _write(candidate.candidate_identity_path, identity)
            _rebind_candidate_identity(candidate)
            with self.assertRaisesRegex(Tau3CandidateSelectionError, "no eligible"):
                select_tau3_candidate(
                    base_manifest_path=base,
                    candidates=[candidate],
                    report_path=root / "selection-2.json",
                    lock_path=root / "candidate-lock-2.json",
                    bootstrap_samples=200,
                )

    def test_rejects_non_schema_real_candidate_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = _benchmark_manifest(root, "base", reward=0.0, db_match=False)
            candidate = _candidate_entry(root, "candidate-a", reward=1.0, db_match=True)
            identity = _read(candidate.candidate_identity_path)
            identity["schema_version"] = "hfr.tau3_candidate_selection.v1"
            _write(candidate.candidate_identity_path, identity)
            _rebind_candidate_identity(candidate)

            with self.assertRaisesRegex(Tau3CandidateSelectionError, "candidate identity schema_version"):
                select_tau3_candidate(
                    base_manifest_path=base,
                    candidates=[candidate],
                    report_path=root / "selection.json",
                    lock_path=root / "candidate-lock.json",
                    bootstrap_samples=200,
                )

    def test_replays_training_adapter_telemetry_and_required_proofs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = _benchmark_manifest(root, "base", reward=0.0, db_match=False)

            adapter_tampered = _candidate_entry(root, "candidate-adapter-tampered", reward=1.0, db_match=True)
            (adapter_tampered.training_receipt_path.parent / "adapter" / "adapters.safetensors").write_bytes(b"tampered")
            with self.assertRaisesRegex(Tau3CandidateSelectionError, "adapter_tree_replay_mismatch"):
                select_tau3_candidate(
                    base_manifest_path=base,
                    candidates=[adapter_tampered],
                    report_path=root / "selection-adapter.json",
                    lock_path=root / "lock-adapter.json",
                    bootstrap_samples=200,
                )

            missing_loss = _candidate_entry(root, "candidate-missing-loss", reward=1.0, db_match=True)
            receipt = _read(missing_loss.training_receipt_path)
            receipt["losses"]["train"] = []
            _write(missing_loss.training_receipt_path, receipt)
            _rebind_training_receipt(missing_loss)
            with self.assertRaisesRegex(Tau3CandidateSelectionError, "training_train_losses_missing"):
                select_tau3_candidate(
                    base_manifest_path=base,
                    candidates=[missing_loss],
                    report_path=root / "selection-loss.json",
                    lock_path=root / "lock-loss.json",
                    bootstrap_samples=200,
                )

            missing_resume_proof = _candidate_entry(root, "candidate-missing-resume", reward=1.0, db_match=True)
            receipt = _read(missing_resume_proof.training_receipt_path)
            receipt["checks"] = [check for check in receipt["checks"] if check["id"] != "resume_hyperparameters_match"]
            _write(missing_resume_proof.training_receipt_path, receipt)
            _rebind_training_receipt(missing_resume_proof)
            with self.assertRaisesRegex(Tau3CandidateSelectionError, "resume_hyperparameters_match_missing"):
                select_tau3_candidate(
                    base_manifest_path=base,
                    candidates=[missing_resume_proof],
                    report_path=root / "selection-resume.json",
                    lock_path=root / "lock-resume.json",
                    bootstrap_samples=200,
                )

            telemetry_tampered = _candidate_entry(root, "candidate-telemetry-tampered", reward=1.0, db_match=True)
            (telemetry_tampered.training_receipt_path.parent / "telemetry.jsonl").write_text(
                json.dumps({"stream": "stdout", "text": "iter 1 train loss 9.99"}) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(Tau3CandidateSelectionError, "telemetry_sha256_replay_mismatch|telemetry_event_count_replay_mismatch|training_train_losses_not_in_telemetry"):
                select_tau3_candidate(
                    base_manifest_path=base,
                    candidates=[telemetry_tampered],
                    report_path=root / "selection-telemetry.json",
                    lock_path=root / "lock-telemetry.json",
                    bootstrap_samples=200,
                )

            symlinked_adapter = _candidate_entry(root, "candidate-symlink-adapter", reward=1.0, db_match=True)
            real_adapter = symlinked_adapter.training_receipt_path.parent / "adapter"
            link_adapter = symlinked_adapter.training_receipt_path.parent / "adapter-link"
            try:
                os.symlink(real_adapter, link_adapter)
            except (OSError, NotImplementedError) as exc:
                self.skipTest(f"symlink creation unavailable: {exc}")
            receipt = _read(symlinked_adapter.training_receipt_path)
            receipt["adapter"]["path"] = "adapter-link"
            _write(symlinked_adapter.training_receipt_path, receipt)
            _rebind_training_receipt(symlinked_adapter)
            with self.assertRaisesRegex(Tau3CandidateSelectionError, "adapter_path_invalid"):
                select_tau3_candidate(
                    base_manifest_path=base,
                    candidates=[symlinked_adapter],
                    report_path=root / "selection-symlink.json",
                    lock_path=root / "lock-symlink.json",
                    bootstrap_samples=200,
                )

    def test_rejects_duplicates_and_cli_writes_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = _benchmark_manifest(root, "base", reward=0.0, db_match=False)
            candidate = _candidate_entry(root, "candidate-a", reward=1.0, db_match=True)
            with self.assertRaisesRegex(Tau3CandidateSelectionError, "duplicate"):
                select_tau3_candidate(
                    base_manifest_path=base,
                    candidates=[candidate, candidate],
                    report_path=root / "selection.json",
                    lock_path=root / "candidate-lock.json",
                    bootstrap_samples=200,
                )

            cmd = [
                sys.executable,
                str(ROOT / "scripts" / "select_tau3_candidate.py"),
                "--base-manifest",
                str(base),
                "--candidate",
                f"candidate-a={candidate.development_manifest_path},{candidate.training_receipt_path},{candidate.candidate_identity_path}",
                "--report-out",
                str(root / "cli-selection.json"),
                "--lock-out",
                str(root / "cli-lock.json"),
                "--bootstrap-samples",
                "200",
            ]
            completed = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=False)
            self.assertEqual(completed.returncode, 0, completed.stderr + completed.stdout)
            self.assertTrue(check_schema_file(root / "cli-lock.json", "tau3_candidate_lock")["passed"])

    def test_schemas_are_registered(self) -> None:
        names = {record["name"] for record in list_schema_records()}
        self.assertIn("tau3_candidate_selection", names)
        self.assertIn("tau3_candidate_lock", names)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = _benchmark_manifest(root, "base", reward=0.0, db_match=False)
            candidate = _candidate_entry(root, "candidate-a", reward=1.0, db_match=True)
            select_tau3_candidate(
                base_manifest_path=base,
                candidates=[candidate],
                report_path=root / "selection.json",
                lock_path=root / "lock.json",
                bootstrap_samples=200,
                created_at="2026-07-23T00:00:00+00:00",
            )
            self.assertTrue(check_schema_contract(_read(root / "selection.json"), name_or_id="tau3_candidate_selection")["passed"])
            self.assertTrue(check_schema_contract(_read(root / "lock.json"), name_or_id="tau3_candidate_lock")["passed"])


def _candidate_entry(root: Path, candidate_id: str, *, reward: float, db_match: bool) -> Tau3CandidateEntry:
    protocol_sha = _ensure_protocol(root)
    receipt = _training_receipt(root, candidate_id, protocol_sha=protocol_sha)
    identity = root / candidate_id / "identity.json"
    receipt_sha = _sha256(receipt)
    receipt_payload = _read(receipt)
    adapter_tree_sha = receipt_payload["adapter"]["tree_sha256"]
    endpoint_model_sha = _hash(candidate_id + "-endpoint-model")
    _write(
        identity,
        {
            "schema_version": "hfr.tau3_candidate_identity.v1",
            "created_at": "2026-07-23T00:00:00+00:00",
            "candidate_id": candidate_id,
            "training_receipt_sha256": receipt_sha,
            "final_training_receipt_sha256": receipt_sha,
            "adapter_tree_sha256": adapter_tree_sha,
            "endpoint_model_sha256": endpoint_model_sha,
            "training_binding": {
                "protocol_sha256": receipt_payload["training_binding"]["protocol"]["sha256"],
                "protocol_signature": receipt_payload["training_binding"]["protocol"]["protocol_signature"],
                "model_freeze_sha256": "b" * 64,
                "recipe_space_sha256": "c" * 64,
                "mlx_qlora_plan_sha256": "d" * 64,
                "base_identity_sha256": receipt_payload["training_binding"]["model"]["identity_sha256"],
                "base_tree_sha256": receipt_payload["training_binding"]["model"]["tree_sha256"],
                "dataset_manifest_sha256": receipt_payload["training_binding"]["dataset"]["manifest_sha256"],
                "dataset_files_sha256": receipt_payload["training_binding"]["dataset"]["files_sha256"],
                "source_binding_sha256": receipt_payload["training_binding"]["dataset"]["source_binding_sha256"],
                "recipe_sha256": receipt_payload["training_binding"]["recipe"]["recipe_sha256"],
            },
            "adapter_identity": {
                "adapter_tree_sha256": adapter_tree_sha,
                "tree_sha256": adapter_tree_sha,
                "file_count": receipt_payload["adapter"]["file_count"],
                "adapter_weight_file_count": receipt_payload["adapter_weight_file_count"],
                "declared_file_set_sha256": adapter_tree_sha,
                "replayed_file_set_sha256": adapter_tree_sha,
            },
            "governance": {
                "training_receipt_schema_checked": True,
                "training_receipt_final": True,
                "training_receipt_success": True,
                "training_weights_updated": True,
                "adapter_files_replayed": True,
                "endpoint_model_hash_only": True,
                "hashes_only": True,
                "local_paths_included": False,
                "absolute_paths_included": False,
                "raw_endpoint_model_included": False,
                "raw_training_receipt_included": False,
                "public_safe": True,
                "private_material_included": False,
                "sealed_access_authorized": False,
            },
            "schema_checked": True,
            "read_only": True,
        },
    )
    manifest = _benchmark_manifest(
        root,
        "adapter",
        reward=reward,
        db_match=db_match,
        directory=candidate_id,
        candidate_identity=_file_ref(identity, "identity.json"),
        endpoint_model_sha256=endpoint_model_sha,
    )
    return Tau3CandidateEntry(candidate_id, manifest, receipt, identity)


def _rebind_candidate_identity(candidate: Tau3CandidateEntry) -> None:
    payload = _read(candidate.development_manifest_path)
    payload["candidate_identity"]["sha256"] = _sha256(candidate.candidate_identity_path)
    payload["candidate_identity"]["size"] = candidate.candidate_identity_path.stat().st_size
    _write(candidate.development_manifest_path, payload)
    prelaunch_path = candidate.development_manifest_path.parent / "prelaunch_receipt.json"
    prelaunch = _read(prelaunch_path)
    prelaunch["candidate_identity"]["sha256"] = _sha256(candidate.candidate_identity_path)
    prelaunch["candidate_identity"]["size"] = candidate.candidate_identity_path.stat().st_size
    _write(prelaunch_path, prelaunch)
    payload = _read(candidate.development_manifest_path)
    payload["prelaunch_receipt"]["sha256"] = _sha256(prelaunch_path)
    payload["prelaunch_receipt"]["size"] = prelaunch_path.stat().st_size
    _write(candidate.development_manifest_path, payload)


def _rebind_training_receipt(candidate: Tau3CandidateEntry) -> None:
    identity = _read(candidate.candidate_identity_path)
    receipt_sha = _sha256(candidate.training_receipt_path)
    identity["training_receipt_sha256"] = receipt_sha
    identity["final_training_receipt_sha256"] = receipt_sha
    _write(candidate.candidate_identity_path, identity)
    _rebind_candidate_identity(candidate)


def _ensure_protocol(root: Path) -> str:
    source = root / "development.json"
    if not source.exists():
        tasks = [
            ("airline", "air-1", "1" * 64, "a" * 64, "b" * 64),
            ("retail", "ret-1", "2" * 64, "c" * 64, "d" * 64),
            ("telecom", "tel-1", "3" * 64, "e" * 64, "f" * 64),
        ]
        _write(
            source,
            {
                "schema_version": "hfr.tau3_source_split.v1",
                "source_revision": REV,
                "split": "development",
                "task_schema_version": "tau2.tasks.v1",
                "algorithm": "test-family-split",
                "salt_sha256": "0" * 64,
                "task_count": len(tasks),
                "family_count": len(tasks),
                "family_ids": [item[2] for item in tasks],
                "tasks": [
                    {
                        "domain": domain,
                        "raw_id": raw_id,
                        "raw_id_sha256": _hash(raw_id),
                        "task_sha256": task_sha,
                        "prompt_sha256": prompt_sha,
                        "family_id": family_id,
                    }
                    for domain, raw_id, family_id, task_sha, prompt_sha in tasks
                ],
            },
        )
    protocol = root / "protocol.json"
    source_sha = _sha256(source)
    payload = {
        "schema_version": "hfr.tau3_protocol_config.v1",
        "protocol_manifest": {"candidate_lock_sha256": "c" * 64},
        "tau_revision": {"revision": REV, "split_hashes": {"train": "1" * 64, "development": source_sha, "sealed": "2" * 64}},
        "split_manifest": {
            "splits": {
                "train": {"sha256": "1" * 64, "sealed": False},
                "development": {"sha256": source_sha, "sealed": False},
                "sealed": {"sha256": "2" * 64, "sealed": True},
            }
        },
        "harness_contract": {},
        "model_freeze": {},
        "budget": {},
        "sealed_manifest": {"access_count": 0, "manifest_sha256": "2" * 64},
        "mlx_qlora_plan": {},
        "recipe_space": {},
        "candidate_selection_contract": {"candidate_lock_sha256": "c" * 64},
        "contamination_attestation": {},
        "redaction_attestation": {},
        "licenses": [{}, {}, {}, {}],
        "environment_manifest": {},
    }
    _write(protocol, payload)
    return _sha256(protocol)


def _benchmark_manifest(
    root: Path,
    arm: str,
    *,
    reward: float,
    db_match: bool,
    directory: str | None = None,
    candidate_identity: dict[str, Any] | None = None,
    endpoint_model_sha256: str | None = None,
) -> Path:
    out = root / (directory or arm)
    raw_dir = out / "raw"
    protocol_sha = _ensure_protocol(root)
    staged_protocol = out / "protocol.json"
    staged_source = out / "development.json"
    _write(staged_protocol, _read(root / "protocol.json"))
    _write(staged_source, _read(root / "development.json"))
    receipts = []
    for domain in ("airline", "retail", "telecom"):
        for seed in (101, 202, 303, 404):
            raw_path = raw_dir / f"{domain}-{seed}.json"
            result_payload = _result(domain, seed, reward=reward, db_match=db_match)
            _write(raw_path, result_payload)
            copied_result_rel = f"results/{domain}/seed-{seed}/results.json"
            copied_result_path = out / copied_result_rel
            _write(copied_result_path, result_payload)
            raw_sha = _sha256(raw_path)
            self_check_sha = _sha256(copied_result_path)
            if self_check_sha != raw_sha:
                raise AssertionError("fixture copied result hash mismatch")
            receipt_path = out / f"run-{domain}-seed{seed}.json"
            receipt = {
                "schema_version": "hfr.tau3_benchmark_run.v1",
                "phase": "domain_seed",
                "created_at": "2026-07-23T00:00:00+00:00",
                "protocol_sha256": protocol_sha,
                "arm_identity": {"arm": arm},
                "mode": "development",
                "arm_id": arm,
                "domain": domain,
                "seed": seed,
                "command": ["tau2", "run", domain],
                "result_path": str(raw_path),
                "result_sha256": raw_sha,
                "result_summary": _summary(reward),
                "exit_code": 0,
                "timed_out": False,
                "duration_seconds": 1.0,
                "terminal_status": "completed",
                "stdout_tail": "",
                "stderr_tail": "",
                "training_started": False,
                "sealed_payload_accessed": False,
                "sealed_task_ids_materialized": False,
            }
            _write(receipt_path, receipt)
            receipts.append(
                {
                    "path": receipt_path.name,
                    "receipt_sha256": _sha256(receipt_path),
                    "result_path": copied_result_rel,
                    "domain": domain,
                    "seed": seed,
                    "terminal_status": "completed",
                    "result_sha256": raw_sha,
                    "result_summary": _summary(reward),
                }
            )
    arm_identity = {"arm": arm}
    if endpoint_model_sha256 is not None:
        arm_identity["endpoint_model_sha256"] = endpoint_model_sha256
    manifest = {
        "schema_version": "hfr.tau3_benchmark_run.v1",
        "phase": "final",
        "created_at": "2026-07-23T00:00:00+00:00",
        "protocol": _file_ref(staged_protocol, "protocol.json"),
        "protocol_sha256": protocol_sha,
        "tau_revision": REV,
        "mode": "development",
        "arm_id": arm,
        "arm_identity": arm_identity,
        "agent": _endpoint("agent"),
        "user_simulator": _endpoint("user"),
        "reviewer": _endpoint("reviewer"),
        "config": {
            "agent": "llm_agent",
            "auto_review": True,
            "communication_protocol_enforced": True,
            "context_window": 16384,
            "domains": ["airline", "retail", "telecom"],
            "hallucination_retries": 0,
            "max_concurrency": 1,
            "max_errors": 10,
            "max_retries": 0,
            "max_steps": 30,
            "num_trials": 1,
            "resume": False,
            "review_mode": "full",
            "seeds": [101, 202, 303, 404],
            "test_time_search": False,
            "timeout_seconds": 600,
            "user": "user_simulator",
        },
        "source": _file_ref(staged_source, "development.json"),
        "sealed_task_count_manifest": None,
        "candidate_lock": None,
        "candidate_identity": candidate_identity,
        "task_selection": {"split": "development"},
        "run_count": 12,
        "success_count": 12,
        "failure_count": 0,
        "run_receipts": receipts,
        "loopback_only": True,
        "training_started": False,
        "sealed_payload_accessed": False,
        "sealed_task_ids_materialized": False,
    }
    prelaunch = {
        **{key: manifest[key] for key in (
            "schema_version",
            "created_at",
            "protocol",
            "protocol_sha256",
            "tau_revision",
            "mode",
            "arm_id",
            "arm_identity",
            "agent",
            "user_simulator",
            "reviewer",
            "config",
            "source",
            "sealed_task_count_manifest",
            "candidate_lock",
            "candidate_identity",
            "task_selection",
            "training_started",
            "sealed_payload_accessed",
            "sealed_task_ids_materialized",
        )},
        "phase": "prelaunch",
    }
    prelaunch_path = out / "prelaunch_receipt.json"
    _write(prelaunch_path, prelaunch)
    manifest["prelaunch_receipt"] = _file_ref(prelaunch_path, prelaunch_path.name)
    manifest_path = out / "manifest.json"
    _write(manifest_path, manifest)
    return manifest_path


def _training_receipt(root: Path, candidate_id: str, *, protocol_sha: str) -> Path:
    path = root / candidate_id / "training_receipt.json"
    adapter_dir = path.parent / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    (adapter_dir / "adapter_config.json").write_text('{"r":16}\n', encoding="utf-8")
    (adapter_dir / "adapters.safetensors").write_bytes(f"{candidate_id}-adapter".encode("utf-8"))
    checkpoint_dir = adapter_dir / "checkpoint-0001"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "weights.npz").write_bytes(f"{candidate_id}-checkpoint".encode("utf-8"))
    adapter_tree = _adapter_tree(adapter_dir)
    telemetry = path.parent / "telemetry.jsonl"
    telemetry.write_text(
        "\n".join(
            [
                json.dumps({"stream": "stdout", "text": "iter 1 train loss 1.25"}),
                json.dumps({"stream": "stdout", "text": "iter 1 validation loss 0.75"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    payload = {
        "schema_version": "hfr.tau3_mlx_training_run.v1",
        "phase": "final",
        "created_at": "2026-07-23T00:00:00+00:00",
        "bundle": {"kind": "mixture"},
        "output_dir": ".",
        "command": ["python", "-m", "mlx_lm", "lora"],
        "config": {"seed": 17},
        "checks": [
            {"id": "protocol_schema_passed", "passed": True, "actual": [], "expected": "registered tau3_protocol_config schema"},
            {"id": "recipe_within_protocol_recipe_space", "passed": True, "actual": {"passed": True}, "expected": "recipe inside frozen recipe_space"},
            {"id": "mixture_manifest_protocol_sha_matches", "passed": True, "actual": protocol_sha, "expected": protocol_sha},
            {"id": "mixture_no_sealed_or_test_rows", "passed": True, "actual": {"sealed": 0, "test": 0}, "expected": {"sealed": 0, "test": 0}},
            {"id": "smoke_update_observed", "passed": True, "actual": True, "expected": True},
            {"id": "smoke_checkpoint_observed", "passed": True, "actual": True, "expected": True},
            {"id": "resume_receipt_schema_passed", "passed": True, "actual": True, "expected": True},
            {"id": "resume_adapter_tree_fingerprint_replays", "passed": True, "actual": True, "expected": True},
            {"id": "resume_adapter_file_bound_to_prior_fingerprint", "passed": True, "actual": True, "expected": True},
            {"id": "resume_protocol_model_dataset_match", "passed": True, "actual": True, "expected": True},
            {"id": "resume_hyperparameters_match", "passed": True, "actual": True, "expected": True},
        ],
        "weights_updated": True,
        "terminal_status": "success",
        "adapter_weight_file_count": 1,
        "adapter": {"path": "adapter", **adapter_tree},
        "telemetry": {"path": "telemetry.jsonl", "sha256": _sha256(telemetry), "event_count": 2},
        "losses": {"train": [1.25], "validation": [0.75], "last_train": 1.25, "last_validation": 0.75},
        "schema_checked": True,
        "training_binding": {
            "protocol": {"sha256": protocol_sha, "protocol_signature": "5" * 64},
            "model": {"identity_sha256": "6" * 64, "tree_sha256": "7" * 64},
            "dataset": {"manifest_sha256": "8" * 64, "files_sha256": "9" * 64, "source_binding_sha256": "a" * 64},
            "recipe": {"recipe_sha256": _hash(candidate_id + "-recipe")},
            "mlx_qlora_plan": {"smoke": {"required": True}, "resume": {"enabled": True}},
        },
    }
    _write(path, payload)
    return path


def _adapter_tree(root: Path) -> dict[str, Any]:
    records = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        rel = path.relative_to(root).as_posix()
        records.append({"path": rel, "size": path.stat().st_size, "sha256": _sha256(path), "kind": _fingerprint_kind(rel)})
    digest = hashlib.sha256()
    for record in records:
        digest.update(json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    return {"file_count": len(records), "files": records, "tree_sha256": digest.hexdigest()}


def _fingerprint_kind(rel: str) -> str:
    name = Path(rel).name
    if name in {"adapter_config.json", "config.json"}:
        return "config"
    if "checkpoint" in rel.lower():
        return "checkpoint"
    if Path(rel).suffix in {".safetensors", ".npz", ".bin"}:
        return "adapter"
    return "artifact"


def _result(domain: str, seed: int, *, reward: float, db_match: bool) -> dict[str, Any]:
    task_id = f"{domain}-{seed}"
    return {
        "timestamp": "2026-07-23T00:00:00",
        "info": {
            "git_commit": REV,
            "num_trials": 1,
            "max_steps": 30,
            "max_errors": 10,
            "max_retries": 0,
            "auto_resume": False,
            "auto_review": True,
            "review_mode": "full",
            "review_model": "fixed-reviewer",
            "hallucination_retries": 0,
            "text_streaming_config": {"chunk_by": "words", "chunk_size": 1},
            "retrieval_config": None,
            "user_info": {"implementation": "user_simulator", "llm": "fixed-user", "llm_args": _llm_args()},
            "agent_info": {"implementation": "llm_agent", "llm": "candidate-model", "llm_args": _llm_args()},
            "environment_info": {"domain_name": domain, "policy": f"{domain} policy", "tool_defs": [{"name": "tool"}]},
        },
        "tasks": [{"id": task_id, "user_scenario": {"instructions": "raw"}, "evaluation_criteria": {"assertions": ["raw"]}}],
        "simulations": [
            {
                "id": f"sim-{task_id}",
                "task_id": task_id,
                "trial": 0,
                "seed": seed,
                "termination_reason": "user_stop",
                "reward_info": {"reward": reward, "db_check": {"db_match": db_match}, "reward_basis": ["DB", "COMMUNICATE"]},
                "messages": [{"role": "user", "content": "raw"}],
                "raw_data": {"provider": "raw"},
                "review": {"errors": []},
            }
        ],
    }


def _endpoint(label: str) -> dict[str, Any]:
    return {
        "context_window": 16384,
        "endpoint_hash": _hash(label + "-endpoint"),
        "loopback": True,
        "max_tokens": 1024,
        "model_sha256": _hash(label + "-model"),
        "temperature": 0.0,
        "top_p": 1.0,
    }


def _llm_args() -> dict[str, Any]:
    return {"temperature": 0.0, "top_p": 1.0, "max_tokens": 1024, "num_retries": 0}


def _summary(reward: float) -> dict[str, Any]:
    return {
        "prompt_token_ceiling": 16384,
        "prompt_token_ceiling_checked": True,
        "prompt_token_ceiling_exceeded": False,
        "prompt_token_observation_count": 1,
        "reward_sum": reward,
        "simulation_count": 1,
        "success_count": 1 if reward >= 1.0 else 0,
    }


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _file_ref(path: Path, rel_path: str) -> dict[str, Any]:
    return {"path": rel_path, "sha256": _sha256(path), "size": path.stat().st_size}


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


if __name__ == "__main__":
    unittest.main()
