from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

from flightrecorder.schema_registry import check_schema_contract
from flightrecorder.tau3_generation_retry import (
    Tau3GenerationRetryError,
    build_tau3_generation_retry_source,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_tau3_generation_retry_source.py"
REVISION = "1d244f5dca42944b67a379b44bfeb9f5748f189d"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Tau3GenerationRetryTests(unittest.TestCase):
    def test_selects_only_tasks_without_reward_1_normal_success_and_cli(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self._source_jsonl(root)
            manifest = self._generation_run(root, success_indexes={0})
            out = root / "retry.jsonl"
            receipt = build_tau3_generation_retry_source(
                source_jsonl_paths=[source],
                generation_manifest_paths=[manifest],
                out_jsonl=out,
                manifest_path=root / "retry-manifest.json",
                expected_tau_revision=REVISION,
                created_at="2026-07-23T00:00:00Z",
            )

            self.assertEqual(receipt["schema_version"], "hfr.tau3_generation_retry_source.v1")
            self.assertEqual(receipt["source_task_count"], 3)
            self.assertEqual(receipt["covered_success_task_count"], 1)
            self.assertEqual(receipt["selected_task_count"], 2)
            self.assertTrue(check_schema_contract(receipt, name_or_id="tau3_generation_retry_source")["passed"])
            selected = read_jsonl(out)
            self.assertEqual([row["task"]["id"] for row in selected], ["task-2", "task-3"])
            self.assertEqual(oct(stat.S_IMODE(out.stat().st_mode)), "0o600")
            self.assertEqual(oct(stat.S_IMODE((root / "retry-manifest.json").stat().st_mode)), "0o600")

            cli_out = root / "retry-cli.jsonl"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--source-jsonl",
                    str(source),
                    "--generation-manifest",
                    str(manifest),
                    "--out-jsonl",
                    str(cli_out),
                    "--manifest",
                    str(root / "retry-cli-manifest.json"),
                    "--domain",
                    "telecom",
                    "--expected-tau-revision",
                    REVISION,
                ],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr + proc.stdout)
            self.assertEqual([row["task"]["id"] for row in read_jsonl(cli_out)], ["task-3"])

    def test_rejects_successful_receipt_with_result_hash_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self._source_jsonl(root)
            manifest = self._generation_run(root, success_indexes={0})
            result_path = root / "results" / "result-0.json"
            result_payload = read_json(result_path)
            result_payload["simulations"][0]["reward_info"]["reward"] = 0.0
            result_path.write_text(json.dumps(result_payload) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(Tau3GenerationRetryError, "result hash mismatch"):
                build_tau3_generation_retry_source(
                    source_jsonl_paths=[source],
                    generation_manifest_paths=[manifest],
                    out_jsonl=root / "retry.jsonl",
                    manifest_path=root / "retry-manifest.json",
                )

    def test_rejects_unsafe_receipt_reference(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self._source_jsonl(root)
            manifest = self._generation_run(root, success_indexes={0})
            payload = read_json(manifest)
            payload["task_receipts"][0]["path"] = "../escape.json"
            manifest.write_text(json.dumps(payload) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(Tau3GenerationRetryError, "relative path inside"):
                build_tau3_generation_retry_source(
                    source_jsonl_paths=[source],
                    generation_manifest_paths=[manifest],
                    out_jsonl=root / "retry.jsonl",
                    manifest_path=root / "retry-manifest.json",
                )

    def test_rejects_sealed_source_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self._source_jsonl(root, split="sealed")
            manifest = self._generation_run(root, success_indexes=set())
            with self.assertRaisesRegex(Tau3GenerationRetryError, "sealed/test split rejected"):
                build_tau3_generation_retry_source(
                    source_jsonl_paths=[source],
                    generation_manifest_paths=[manifest],
                    out_jsonl=root / "retry.jsonl",
                    manifest_path=root / "retry-manifest.json",
                )

    def test_training_excluded_reward_1_result_remains_retry_eligible(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self._source_jsonl(root)
            manifest = self._generation_run(root, success_indexes={0})
            result_path = root / "results" / "result-0.json"
            result = read_json(result_path)
            result["simulations"][0]["messages"] = [
                {"role": "assistant", "content": "<think>hidden reasoning</think>"}
            ]
            self._rebind_result(manifest, 0, result_path, result)

            out = root / "retry.jsonl"
            receipt = build_tau3_generation_retry_source(
                source_jsonl_paths=[source],
                generation_manifest_paths=[manifest],
                out_jsonl=out,
                manifest_path=root / "retry-manifest.json",
            )
            self.assertEqual(receipt["covered_success_task_count"], 0)
            self.assertEqual(receipt["generation_manifests"][0]["training_excluded_success_count"], 1)
            self.assertEqual([row["task"]["id"] for row in read_jsonl(out)], ["task-1", "task-2", "task-3"])

    def test_errored_assistant_tool_result_remains_retry_eligible(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self._source_jsonl(root)
            manifest = self._generation_run(root, success_indexes={0})
            result_path = root / "results" / "result-0.json"
            result = read_json(result_path)
            result["simulations"][0]["messages"] = [
                {
                    "role": "tool",
                    "requestor": "assistant",
                    "id": "call-1",
                    "content": "tool failed",
                    "error": True,
                }
            ]
            self._rebind_result(manifest, 0, result_path, result)
            out = root / "retry.jsonl"
            receipt = build_tau3_generation_retry_source(
                source_jsonl_paths=[source],
                generation_manifest_paths=[manifest],
                out_jsonl=out,
            )
            self.assertEqual(receipt["covered_success_task_count"], 0)
            self.assertEqual(receipt["generation_manifests"][0]["training_excluded_success_count"], 1)
            self.assertEqual([row["task"]["id"] for row in read_jsonl(out)], ["task-1", "task-2", "task-3"])

    def test_rejects_result_bound_to_a_different_task(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self._source_jsonl(root)
            manifest = self._generation_run(root, success_indexes={0})
            result_path = root / "results" / "result-0.json"
            result = read_json(result_path)
            result["simulations"][0]["task_id"] = "task-2"
            self._rebind_result(manifest, 0, result_path, result)
            with self.assertRaisesRegex(Tau3GenerationRetryError, "simulation task_id"):
                build_tau3_generation_retry_source(
                    source_jsonl_paths=[source],
                    generation_manifest_paths=[manifest],
                    out_jsonl=root / "retry.jsonl",
                )

    def test_rejects_source_task_hash_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self._source_jsonl(root)
            rows = read_jsonl(source)
            rows[0]["task"]["instruction"] = "tampered"
            source.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
            manifest = self._generation_run(root, success_indexes=set())
            with self.assertRaisesRegex(Tau3GenerationRetryError, "task_sha256 does not replay"):
                build_tau3_generation_retry_source(
                    source_jsonl_paths=[source],
                    generation_manifest_paths=[manifest],
                    out_jsonl=root / "retry.jsonl",
                )

    def test_rejects_incomplete_final_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self._source_jsonl(root)
            manifest = self._generation_run(root, success_indexes=set())
            payload = read_json(manifest)
            payload["task_count"] += 1
            manifest.write_text(json.dumps(payload) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(Tau3GenerationRetryError, "task_count"):
                build_tau3_generation_retry_source(
                    source_jsonl_paths=[source],
                    generation_manifest_paths=[manifest],
                    out_jsonl=root / "retry.jsonl",
                )

    def test_rejects_receipt_status_that_does_not_replay(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self._source_jsonl(root)
            manifest = self._generation_run(root, success_indexes={0})
            payload = read_json(manifest)
            receipt_path = manifest.parent / payload["task_receipts"][0]["path"]
            receipt = read_json(receipt_path)
            receipt["exit_code"] = 1
            receipt_path.write_text(json.dumps(receipt) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(Tau3GenerationRetryError, "terminal status"):
                build_tau3_generation_retry_source(
                    source_jsonl_paths=[source],
                    generation_manifest_paths=[manifest],
                    out_jsonl=root / "retry.jsonl",
                )

    def test_rejects_timed_out_success_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self._source_jsonl(root)
            manifest = self._generation_run(root, success_indexes={0})
            payload = read_json(manifest)
            receipt_path = manifest.parent / payload["task_receipts"][0]["path"]
            receipt = read_json(receipt_path)
            receipt["timed_out"] = True
            receipt_path.write_text(json.dumps(receipt) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(Tau3GenerationRetryError, "timed-out receipt"):
                build_tau3_generation_retry_source(
                    source_jsonl_paths=[source],
                    generation_manifest_paths=[manifest],
                    out_jsonl=root / "retry.jsonl",
                )

    def test_rejects_interior_receipt_path_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self._source_jsonl(root)
            manifest = self._generation_run(root, success_indexes={0})
            payload = read_json(manifest)
            original = manifest.parent / payload["task_receipts"][0]["path"]
            real_dir = manifest.parent / "real"
            real_dir.mkdir()
            moved = real_dir / original.name
            original.rename(moved)
            link = manifest.parent / "link"
            try:
                link.symlink_to(real_dir, target_is_directory=True)
            except OSError as exc:
                self.skipTest(f"symlinks unavailable: {exc}")
            payload["task_receipts"][0]["path"] = f"link/{moved.name}"
            manifest.write_text(json.dumps(payload) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(Tau3GenerationRetryError, "symlink component"):
                build_tau3_generation_retry_source(
                    source_jsonl_paths=[source],
                    generation_manifest_paths=[manifest],
                    out_jsonl=root / "retry.jsonl",
                )

    def _source_jsonl(self, root: Path, *, split: str = "train") -> Path:
        path = root / "source.jsonl"
        rows = [self._source_row("airline", "task-1", split), self._source_row("retail", "task-2", split), self._source_row("telecom", "task-3", split)]
        path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
        return path

    def _source_row(self, domain: str, task_id: str, split: str) -> dict[str, Any]:
        task = {"id": task_id, "instruction": f"solve {task_id}"}
        return {
            "schema_version": "hfr.tau3_training_source.v1",
            "source_revision": REVISION,
            "domain": domain,
            "split": split,
            "task_family": hashlib.sha256(f"family-{task_id}".encode()).hexdigest(),
            "task_sha256": hashlib.sha256(
                json.dumps(task, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            "prompt_sha256": hashlib.sha256(f"prompt-{task_id}".encode()).hexdigest(),
            "task": task,
        }

    def _generation_run(self, root: Path, *, success_indexes: set[int]) -> Path:
        out = root / "generation"
        out.mkdir()
        results_dir = root / "results"
        results_dir.mkdir()
        source_rows = read_jsonl(root / "source.jsonl")
        protocol_path = root / "protocol.json"
        protocol_path.write_text("{}\n", encoding="utf-8")
        prelaunch_path = out / "prelaunch_receipt.json"
        prelaunch_path.write_text("{}\n", encoding="utf-8")
        refs = []
        success_count = 0
        failure_count = 0
        for index, row in enumerate(source_rows):
            status = "success" if index in success_indexes else "failed"
            reward = 1.0 if status == "success" else 0.0
            result_path = results_dir / f"result-{index}.json"
            result_path.write_text(
                json.dumps(
                    {
                        "info": {
                            "git_commit": REVISION,
                            "environment_info": {"domain_name": row["domain"]},
                        },
                        "simulations": [
                            {
                                "id": f"simulation-{index}",
                                "task_id": row["task"]["id"],
                                "termination_reason": "agent_stop",
                                "reward_info": {"reward": reward},
                                "messages": [],
                            }
                        ]
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            task = {
                "domain": row["domain"],
                "split": row["split"],
                "official_task_split": "train",
                "task_id": row["task"]["id"],
                "task_family": row["task_family"],
                "task_sha256": row["task_sha256"],
                "prompt_sha256": row["prompt_sha256"],
                "has_reference_actions": "false",
            }
            receipt = {
                "schema_version": "hfr.tau3_teacher_generation_run.v1",
                "phase": "task",
                "created_at": "2026-07-23T00:00:00Z",
                "task": task,
                "command": ["tau2", "run"],
                "result_path": str(result_path),
                "result_sha256": sha256(result_path),
                "reward": reward,
                "exit_code": 0 if status == "success" else 1,
                "timed_out": False,
                "duration_seconds": 1.0,
                "terminal_status": status,
                "training_started": False,
                "sealed_payload_accessed": False,
            }
            receipt_path = out / f"task-{index:04d}-{row['domain']}.json"
            receipt_path.write_text(json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8")
            refs.append({"path": receipt_path.name, "terminal_status": status, "result_sha256": receipt["result_sha256"]})
            if status == "success":
                success_count += 1
            else:
                failure_count += 1
        manifest = {
            "schema_version": "hfr.tau3_teacher_generation_run.v1",
            "phase": "final",
            "created_at": "2026-07-23T00:00:00Z",
            "tau_revision": REVISION,
            "source": {"path": str(root / "source.jsonl"), "size": (root / "source.jsonl").stat().st_size, "sha256": sha256(root / "source.jsonl")},
            "teacher": {"model": "teacher"},
            "user_simulator": {"model": "teacher"},
            "protocol": {"path": str(protocol_path), "size": protocol_path.stat().st_size, "sha256": sha256(protocol_path)},
            "config": {
                "agent": "auto",
                "communication_protocol_enforced": False,
                "max_errors": 10,
                "max_steps": 30,
                "max_tasks": len(source_rows),
                "seed": 101,
                "timeout_seconds": 600,
            },
            "task_count": len(source_rows),
            "prelaunch_receipt": {"path": prelaunch_path.name, "size": prelaunch_path.stat().st_size, "sha256": sha256(prelaunch_path)},
            "success_count": success_count,
            "failure_count": failure_count,
            "task_receipts": refs,
            "sealed_rows": 0,
            "test_rows": 0,
            "loopback_only": True,
            "training_started": False,
            "sealed_payload_accessed": False,
        }
        manifest_path = out / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")
        return manifest_path

    def _rebind_result(
        self,
        manifest_path: Path,
        index: int,
        result_path: Path,
        result: dict[str, Any],
    ) -> None:
        result_path.write_text(json.dumps(result, sort_keys=True) + "\n", encoding="utf-8")
        manifest = read_json(manifest_path)
        ref = manifest["task_receipts"][index]
        receipt_path = manifest_path.parent / ref["path"]
        receipt = read_json(receipt_path)
        digest = sha256(result_path)
        receipt["result_sha256"] = digest
        receipt_path.write_text(json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8")
        ref["result_sha256"] = digest
        manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
