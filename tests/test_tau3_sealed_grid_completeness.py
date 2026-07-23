from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

from flightrecorder.schema_registry import check_schema_contract, check_schema_file
from flightrecorder.tau3_sealed_authorization import create_tau3_sealed_authorization
import flightrecorder.tau3_sealed_grid_completeness as sealed_grid
from flightrecorder.tau3_sealed_grid_completeness import (
    REQUIRED_ARMS,
    REQUIRED_DOMAINS,
    REQUIRED_SEEDS,
    Tau3SealedGridCompletenessError,
    build_tau3_sealed_grid_completeness,
)


class Tau3SealedGridCompletenessTests(unittest.TestCase):
    def test_builds_public_safe_completeness_artifact_for_exact_grid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = build_grid_fixture(root)

            artifact = build_tau3_sealed_grid_completeness(
                arm_manifests=fixture["arm_manifests"],
                candidate_lock=fixture["candidate_lock"],
                protocol=fixture["protocol"],
                sealed_source_manifest=fixture["sealed_source"],
                sealed_authorization=fixture["authorization"],
                expected_tau_revision="a" * 40,
                out=root / "completeness.json",
                created_at="2026-07-23T02:00:00Z",
            )

            self.assertTrue(artifact["passed"])
            self.assertEqual(artifact["counts"]["total_episodes"], 1600)
            self.assertEqual(check_schema_contract(artifact, name_or_id="tau3_sealed_grid_completeness")["passed"], True)
            rendered = json.dumps(artifact, sort_keys=True)
            self.assertNotIn("airline-0", rendered)
            self.assertNotIn(str(root), rendered)
            self.assertNotIn("reward", rendered)

    def test_rejects_99_and_101_tasks(self) -> None:
        for count in (99, 101):
            with self.subTest(count=count), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                fixture = build_grid_fixture(root)
                rewrite_result_tasks(root, "adapter", "airline", 101, [f"airline-{index}" for index in range(count)])
                with self.assertRaisesRegex(Tau3SealedGridCompletenessError, "coverage|episode"):
                    build_fixture_artifact(root, fixture)

    def test_rejects_duplicate_task_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = build_grid_fixture(root)
            tasks = [f"airline-{index}" for index in range(34)]
            tasks[-1] = tasks[0]
            rewrite_result_tasks(root, "adapter", "airline", 101, tasks)
            with self.assertRaisesRegex(Tau3SealedGridCompletenessError, "duplicate"):
                build_fixture_artifact(root, fixture)

    def test_rejects_cross_arm_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = build_grid_fixture(root)
            rewrite_result_tasks(root, "comparator_1", "retail", 202, [f"retail-{index}" for index in range(33, 66)] + ["retail-extra"])
            with self.assertRaisesRegex(Tau3SealedGridCompletenessError, "coverage"):
                build_fixture_artifact(root, fixture)

    def test_rejects_wrong_seed_and_arm(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = build_grid_fixture(root)
            mutate_json(root / "sealed" / "adapter" / "run-airline-seed101.json", lambda payload: payload.__setitem__("seed", 202))
            refresh_arm_manifest_ref(root, "adapter", "airline", 101)
            with self.assertRaisesRegex(Tau3SealedGridCompletenessError, "domain/seed"):
                build_fixture_artifact(root, fixture)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = build_grid_fixture(root)
            mutate_json(root / "sealed" / "base" / "manifest.json", lambda payload: payload.__setitem__("arm_id", "adapter"))
            fixture["arm_manifests"][1] = root / "sealed" / "base" / "manifest.json"
            with self.assertRaisesRegex(Tau3SealedGridCompletenessError, "arm|model identity"):
                build_fixture_artifact(root, fixture)

    def test_rejects_symlink_and_mutated_result_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = build_grid_fixture(root)
            result = root / "sealed" / "adapter" / "results" / "airline" / "seed-101" / "results.json"
            real = result.with_name("real-results.json")
            result.rename(real)
            result.symlink_to(real)
            with self.assertRaisesRegex(Tau3SealedGridCompletenessError, "symlink"):
                build_fixture_artifact(root, fixture)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = build_grid_fixture(root)
            mutate_json(root / "sealed" / "adapter" / "results" / "airline" / "seed-101" / "results.json", lambda payload: payload["simulations"].append({"task_id": "airline-0", "seed": 101}))
            with self.assertRaisesRegex(Tau3SealedGridCompletenessError, "hash"):
                build_fixture_artifact(root, fixture)

    def test_artifact_schema_is_strict_and_cli_works(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = build_grid_fixture(root)
            out = root / "completeness.json"

            proc = subprocess.run(
                [
                    sys.executable,
                    "scripts/build_tau3_sealed_grid_completeness.py",
                    "--sealed-source-manifest",
                    str(fixture["sealed_source"]),
                    "--sealed-authorization",
                    str(fixture["authorization"]),
                    "--candidate-lock",
                    str(fixture["candidate_lock"]),
                    "--protocol",
                    str(fixture["protocol"]),
                    "--expected-tau-revision",
                    "a" * 40,
                    "--out",
                    str(out),
                    *sum((["--arm-manifest", str(path)] for path in fixture["arm_manifests"]), []),
                ],
                cwd=Path(__file__).resolve().parents[1],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr + proc.stdout)
            self.assertNotIn(str(root), proc.stdout)
            self.assertNotIn(str(root), proc.stderr)
            self.assertNotIn('"out"', proc.stdout)
            self.assertTrue(check_schema_file(out, "tau3_sealed_grid_completeness")["passed"])

            payload = read_json(out)
            payload["path"] = "/tmp/leak"
            self.assertFalse(check_schema_contract(payload, name_or_id="tau3_sealed_grid_completeness")["passed"])

    def test_cli_failure_stderr_is_path_free(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = build_grid_fixture(root)
            out = root / "existing.json"
            write_json(out, {"already": True})

            proc = subprocess.run(
                [
                    sys.executable,
                    "scripts/build_tau3_sealed_grid_completeness.py",
                    "--sealed-source-manifest",
                    str(fixture["sealed_source"]),
                    "--sealed-authorization",
                    str(fixture["authorization"]),
                    "--candidate-lock",
                    str(fixture["candidate_lock"]),
                    "--protocol",
                    str(fixture["protocol"]),
                    "--expected-tau-revision",
                    "a" * 40,
                    "--out",
                    str(out),
                    *sum((["--arm-manifest", str(path)] for path in fixture["arm_manifests"]), []),
                ],
                cwd=Path(__file__).resolve().parents[1],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            self.assertEqual(proc.returncode, 2)
            self.assertEqual(proc.stdout, "")
            self.assertNotIn(str(root), proc.stderr)
            self.assertIn("error_code", proc.stderr)

    def test_rejects_authorization_harness_model_and_chronology_mutations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = build_grid_fixture(root)
            mutate_json(root / "sealed" / "adapter" / "authorization.json", lambda payload: payload["candidate_lock"].__setitem__("sha256", "0" * 64))
            refresh_arm_auth_ref(root, "adapter")
            with self.assertRaisesRegex(Tau3SealedGridCompletenessError, "authorization"):
                build_fixture_artifact(root, fixture)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = build_grid_fixture(root)
            mutate_json(root / "sealed" / "base" / "manifest.json", lambda payload: payload["config"].__setitem__("seeds", [101]))
            with self.assertRaisesRegex(Tau3SealedGridCompletenessError, "harness|schema"):
                build_fixture_artifact(root, fixture)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = build_grid_fixture(root)
            mutate_json(root / "sealed" / "comparator_1" / "manifest.json", lambda payload: payload["arm_identity"].__setitem__("model_identity_sha256", "0" * 64))
            with self.assertRaisesRegex(Tau3SealedGridCompletenessError, "model identity"):
                build_fixture_artifact(root, fixture)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = build_grid_fixture(root)
            mutate_json(root / "sealed" / "adapter" / "run-airline-seed101.json", lambda payload: payload.__setitem__("created_at", "2026-07-23T00:05:00Z"))
            refresh_arm_manifest_ref(root, "adapter", "airline", 101)
            with self.assertRaisesRegex(Tau3SealedGridCompletenessError, "chronology"):
                build_fixture_artifact(root, fixture)

    def test_rejects_result_mutation_during_stable_read(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = build_grid_fixture(root)
            target = root / "sealed" / "adapter" / "results" / "airline" / "seed-101" / "results.json"
            original_read = sealed_grid.os.read
            target_stat = target.stat()
            mutated = False

            def racing_read(fd: int, size: int) -> bytes:
                nonlocal mutated
                chunk = original_read(fd, size)
                try:
                    current = sealed_grid.os.fstat(fd)
                except OSError:
                    return chunk
                if not mutated and chunk and current.st_ino == target_stat.st_ino and current.st_dev == target_stat.st_dev:
                    mutated = True
                    target.write_text(json.dumps({"simulations": [{"task_id": "airline-0", "seed": 101}, {"task_id": "airline-extra", "seed": 101}]}) + "\n", encoding="utf-8")
                return chunk

            try:
                sealed_grid.os.read = racing_read
                with self.assertRaisesRegex(Tau3SealedGridCompletenessError, "changed while being read"):
                    build_fixture_artifact(root, fixture)
            finally:
                sealed_grid.os.read = original_read


def build_fixture_artifact(root: Path, fixture: dict[str, Any]) -> dict[str, Any]:
    return build_tau3_sealed_grid_completeness(
        arm_manifests=fixture["arm_manifests"],
        candidate_lock=fixture["candidate_lock"],
        protocol=fixture["protocol"],
        sealed_source_manifest=fixture["sealed_source"],
        sealed_authorization=fixture["authorization"],
        expected_tau_revision="a" * 40,
        out=root / "completeness.json",
        created_at="2026-07-23T02:00:00Z",
    )


def build_grid_fixture(root: Path) -> dict[str, Any]:
    tasks_by_domain = {
        "airline": [f"airline-{index}" for index in range(34)],
        "retail": [f"retail-{index}" for index in range(33)],
        "telecom": [f"telecom-{index}" for index in range(33)],
    }
    sealed_source = {
        "schema_version": "hfr.tau3_sealed_source_manifest.v1",
        "source_revision": "a" * 40,
        "hashes_only": True,
        "task_count": 100,
        "entries": [
            {
                "task_id_sha256": task_hash(domain, task_id),
                "prompt_sha256": sha256_text(f"prompt:{domain}:{task_id}"),
                "task_sha256": sha256_text(f"task:{domain}:{task_id}"),
            }
            for domain in REQUIRED_DOMAINS
            for task_id in tasks_by_domain[domain]
        ],
    }
    sealed_source_path = write_json(root / "sealed-source.json", sealed_source)
    protocol_path = write_json(root / "protocol.json", protocol_payload(sealed_source_path))
    lock_path = write_json(root / "candidate-lock.json", candidate_lock_payload(protocol_path))
    authorization_path = root / "authorization.json"
    create_tau3_sealed_authorization(
        candidate_lock=lock_path,
        protocol=protocol_path,
        sealed_source_manifest=sealed_source_path,
        out=authorization_path,
        created_at="2026-07-23T00:20:00Z",
    )
    arm_paths = [
        build_arm(root, arm, tasks_by_domain, protocol_path=protocol_path, lock_path=lock_path, sealed_source_path=sealed_source_path, authorization_path=authorization_path)
        for arm in REQUIRED_ARMS
    ]
    return {"sealed_source": sealed_source_path, "authorization": authorization_path, "candidate_lock": lock_path, "protocol": protocol_path, "arm_manifests": arm_paths}


def protocol_payload(sealed_source_path: Path) -> dict[str, Any]:
    sealed_sha = sha256_file(sealed_source_path)
    return {
        "schema_version": "hfr.tau3_protocol_config.v1",
        "protocol_manifest": {"signature": "fixture"},
        "tau_revision": {"revision": "a" * 40, "split_hashes": {"sealed": sealed_sha}},
        "split_manifest": {"splits": {"sealed": {"sha256": sealed_sha}}},
        "harness_contract": {
            "domains": list(REQUIRED_DOMAINS),
            "context_window": 16384,
            "decoding": {"temperature": 0.0, "top_p": 1.0, "max_output_tokens": 1024, "seeds": list(REQUIRED_SEEDS)},
            "turn_limit": 30,
            "retry_policy": "none",
            "test_time_search": False,
            "no_test_time_search": True,
            "domain_contracts": {},
            "prompt_contract": {},
        },
        "model_freeze": {
            "base_model": {"name": "base", "revision": "base-rev", "local_identity_sha256": "8" * 64},
            "comparators": [
                {"name": "comparator-1", "revision": "cmp1-rev", "local_identity_sha256": "9" * 64},
                {"name": "comparator-2", "revision": "cmp2-rev", "local_identity_sha256": "a" * 64},
            ],
        },
        "budget": {"passed": True},
        "sealed_manifest": {"manifest_sha256": sealed_sha, "access_count": 0},
        "mlx_qlora_plan": {},
        "recipe_space": {},
        "candidate_selection_contract": {"passed": True},
        "contamination_attestation": {"passed": True},
        "redaction_attestation": {"passed": True},
        "licenses": [{"status": "approved", "training_allowed": True} for _ in range(4)],
        "environment_manifest": {},
    }


def candidate_lock_payload(protocol_path: Path) -> dict[str, Any]:
    return {
        "schema_version": "hfr.tau3_candidate_lock.v1",
        "created_at": "2026-07-23T00:10:00Z",
        "selected_candidate_id_hash": "d" * 64,
        "candidate_identity_sha256": "5" * 64,
        "development_selection_report_sha256": "e" * 64,
        "development_benchmark_manifest_sha256": "f" * 64,
        "training_receipt_sha256": "1" * 64,
        "endpoint_model_sha256": "7" * 64,
        "adapter_tree_sha256": "6" * 64,
        "recipe_sha256": "2" * 64,
        "base_identity_sha256": "8" * 64,
        "base_tree_sha256": "3" * 64,
        "dataset_manifest_sha256": "4" * 64,
        "dataset_files_sha256": "b" * 64,
        "source_binding_sha256": "c" * 64,
        "protocol_sha256": sha256_file(protocol_path),
        "protocol_signature": "1" * 64,
        "hashes_only": True,
        "sealed_access_authorized": True,
        "local_paths_included": False,
        "raw_payload_included": False,
    }


def build_arm(
    root: Path,
    arm: str,
    tasks_by_domain: dict[str, list[str]],
    *,
    protocol_path: Path,
    lock_path: Path,
    sealed_source_path: Path,
    authorization_path: Path,
) -> Path:
    arm_dir = root / "sealed" / arm
    arm_dir.mkdir(parents=True)
    copy_json(protocol_path, arm_dir / "protocol.json")
    copy_json(lock_path, arm_dir / "candidate-lock.json")
    copy_json(sealed_source_path, arm_dir / "sealed-source.json")
    copy_json(authorization_path, arm_dir / "authorization.json")
    run_receipts = []
    for domain in REQUIRED_DOMAINS:
        for seed in REQUIRED_SEEDS:
            result_rel = f"results/{domain}/seed-{seed}/results.json"
            result_path = arm_dir / result_rel
            write_json(result_path, {"simulations": [{"task_id": task_id, "seed": seed} for task_id in tasks_by_domain[domain]]})
            receipt = {
                "schema_version": "hfr.tau3_benchmark_run.v1",
                "phase": "domain_seed",
                "created_at": "2026-07-23T01:00:00Z",
                "protocol_sha256": sha256_file(protocol_path),
                "arm_identity": {"arm_id": arm, "identity": sha256_text(arm)},
                "mode": "sealed",
                "arm_id": arm,
                "domain": domain,
                "seed": seed,
                "result_path": result_rel,
                "result_sha256": sha256_file(result_path),
                "result_summary": result_summary(len(tasks_by_domain[domain])),
                "terminal_status": "completed",
                "training_started": False,
                "sealed_payload_accessed": False,
                "sealed_task_ids_materialized": False,
            }
            receipt_path = write_json(arm_dir / f"run-{domain}-seed{seed}.json", receipt)
            run_receipts.append(
                {
                    "path": receipt_path.name,
                    "receipt_sha256": sha256_file(receipt_path),
                    "domain": domain,
                    "seed": seed,
                    "terminal_status": "completed",
                    "result_path": result_rel,
                    "result_sha256": sha256_file(result_path),
                    "result_summary": result_summary(len(tasks_by_domain[domain])),
                }
            )
    prelaunch_path = write_json(arm_dir / "prelaunch_receipt.json", {"schema_version": "hfr.tau3_benchmark_run.v1", "phase": "prelaunch", "created_at": "2026-07-23T00:30:00Z"})
    arm_identity = arm_identity_payload(arm, authorization_path, lock_path)
    manifest = {
        "schema_version": "hfr.tau3_benchmark_run.v1",
        "phase": "final",
        "created_at": "2026-07-23T01:05:00Z",
        "tau_revision": "a" * 40,
        "protocol": ref("protocol.json", arm_dir / "protocol.json"),
        "protocol_sha256": sha256_file(protocol_path),
        "mode": "sealed",
        "arm_id": arm,
        "arm_identity": arm_identity,
        "agent": endpoint(arm),
        "user_simulator": endpoint("user"),
        "reviewer": endpoint("reviewer"),
        "config": config(),
        "source": None,
        "sealed_task_count_manifest": {**ref("sealed-source.json", arm_dir / "sealed-source.json"), "hashes_only": True, "task_count": 100},
        "sealed_authorization": {
            **ref("authorization.json", arm_dir / "authorization.json"),
            "authorized": True,
            "candidate_lock_sha256": sha256_file(lock_path),
            "protocol_sha256": sha256_file(protocol_path),
            "sealed_source_sha256": sha256_file(sealed_source_path),
            "task_count": 100,
            "arms": list(REQUIRED_ARMS),
            "seeds": list(REQUIRED_SEEDS),
        },
        "candidate_lock": ref("candidate-lock.json", arm_dir / "candidate-lock.json"),
        "candidate_identity": None,
        "task_selection": {
            "official_split": "test",
            "task_ids_in_command": False,
            "task_payload_accessed": False,
            "domains": list(REQUIRED_DOMAINS),
            "sealed_task_count": 100,
            "task_count_by_domain": None,
        },
        "prelaunch_receipt": ref("prelaunch_receipt.json", prelaunch_path),
        "run_count": 12,
        "success_count": 12,
        "failure_count": 0,
        "run_receipts": run_receipts,
        "loopback_only": True,
        "training_started": False,
        "sealed_payload_accessed": False,
        "sealed_task_ids_materialized": False,
    }
    return write_json(arm_dir / "manifest.json", manifest)


def rewrite_result_tasks(root: Path, arm: str, domain: str, seed: int, task_ids: list[str]) -> None:
    path = root / "sealed" / arm / "results" / domain / f"seed-{seed}" / "results.json"
    write_json(path, {"simulations": [{"task_id": task_id, "seed": seed} for task_id in task_ids]})
    receipt_path = root / "sealed" / arm / f"run-{domain}-seed{seed}.json"
    receipt = read_json(receipt_path)
    receipt["result_sha256"] = sha256_file(path)
    write_json(receipt_path, receipt)
    refresh_arm_manifest_ref(root, arm, domain, seed)


def refresh_arm_manifest_ref(root: Path, arm: str, domain: str, seed: int) -> None:
    manifest_path = root / "sealed" / arm / "manifest.json"
    manifest = read_json(manifest_path)
    receipt_path = root / "sealed" / arm / f"run-{domain}-seed{seed}.json"
    result_path = root / "sealed" / arm / "results" / domain / f"seed-{seed}" / "results.json"
    for record in manifest["run_receipts"]:
        if record["domain"] == domain and record["seed"] == seed:
            record["receipt_sha256"] = sha256_file(receipt_path)
            record["result_sha256"] = sha256_file(result_path)
    write_json(manifest_path, manifest)


def refresh_arm_auth_ref(root: Path, arm: str) -> None:
    manifest_path = root / "sealed" / arm / "manifest.json"
    manifest = read_json(manifest_path)
    auth_path = root / "sealed" / arm / "authorization.json"
    manifest["sealed_authorization"]["sha256"] = sha256_file(auth_path)
    write_json(manifest_path, manifest)


def arm_identity_payload(arm: str, authorization_path: Path, lock_path: Path) -> dict[str, Any]:
    authorization = read_json(authorization_path)
    refs = authorization["model_identity_refs"]
    if arm == "adapter":
        return {
            "arm_id": "adapter",
            "source": "candidate_lock",
            "candidate_lock_sha256": sha256_file(lock_path),
            "candidate_identity_sha256": refs["candidate_identity_sha256"],
            "endpoint_model_sha256": refs["endpoint_model_sha256"],
            "adapter": {"tree_sha256": refs["adapter_tree_sha256"]},
        }
    key = {
        "base": "base_identity_sha256",
        "comparator_1": "comparator_1_identity_sha256",
        "comparator_2": "comparator_2_identity_sha256",
    }[arm]
    return {"arm_id": arm, "source": "protocol_model_freeze", "model_identity_sha256": refs[key]}


def mutate_json(path: Path, mutate: Any) -> None:
    payload = read_json(path)
    mutate(payload)
    write_json(path, payload)


def endpoint(label: str) -> dict[str, Any]:
    return {
        "context_window": 16384,
        "endpoint_hash": sha256_text(f"endpoint:{label}"),
        "loopback": True,
        "max_tokens": 1024,
        "model_sha256": sha256_text(f"model:{label}"),
        "temperature": 0.0,
        "top_p": 1.0,
    }


def config() -> dict[str, Any]:
    return {
        "agent": "llm_agent",
        "auto_review": True,
        "communication_protocol_enforced": True,
        "context_window": 16384,
        "domains": list(REQUIRED_DOMAINS),
        "hallucination_retries": 0,
        "max_concurrency": 1,
        "max_errors": 10,
        "max_retries": 0,
        "max_steps": 30,
        "num_trials": 1,
        "resume": False,
        "review_mode": "full",
        "seeds": list(REQUIRED_SEEDS),
        "test_time_search": False,
        "timeout_seconds": 600,
        "user": "user_simulator",
    }


def result_summary(count: int) -> dict[str, Any]:
    return {
        "prompt_token_ceiling": 16384,
        "prompt_token_ceiling_checked": True,
        "prompt_token_ceiling_exceeded": False,
        "prompt_token_observation_count": count,
        "reward_sum": float(count),
        "simulation_count": count,
        "success_count": count,
    }


def ref(name: str, path: Path) -> dict[str, Any]:
    return {"path": name, "sha256": sha256_file(path), "size": path.stat().st_size}


def copy_json(src: Path, dst: Path) -> None:
    write_json(dst, read_json(src))


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def task_hash(domain: str, task_id: str) -> str:
    return sha256_text(f"{domain}:{task_id}")


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    unittest.main()
