import hashlib
import json
import stat
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

from flightrecorder.schema_registry import check_schema_file
from flightrecorder.tau3_candidate_identity import (
    Tau3CandidateIdentityError,
    build_tau3_candidate_identity,
    main,
)


class Tau3CandidateIdentityTests(unittest.TestCase):
    def test_builds_public_safe_identity_from_final_training_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            receipt = _write_training_fixture(root)
            out = root / "identity.json"

            result = build_tau3_candidate_identity(
                candidate_id="candidate-a",
                training_receipt_path=receipt,
                endpoint_model="qwen-local+candidate-a",
                output_path=out,
                created_at="2026-07-23T00:00:00Z",
            )

            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema_version"], "hfr.tau3_candidate_identity.v1")
            self.assertEqual(payload["candidate_id"], "candidate-a")
            self.assertEqual(payload["training_receipt_sha256"], _sha256_file(receipt))
            self.assertEqual(payload["final_training_receipt_sha256"], _sha256_file(receipt))
            self.assertEqual(
                payload["endpoint_model_sha256"],
                hashlib.sha256(b"qwen-local+candidate-a").hexdigest(),
            )
            self.assertEqual(payload["adapter_tree_sha256"], result["adapter_tree_sha256"])
            self.assertEqual(payload["adapter_tree_sha256"], payload["adapter_identity"]["adapter_tree_sha256"])
            self.assertEqual(payload["adapter_tree_sha256"], payload["adapter_identity"]["tree_sha256"])
            self.assertEqual(payload["adapter_identity"]["file_count"], 4)
            self.assertEqual(payload["adapter_identity"]["adapter_weight_file_count"], 1)
            self.assertEqual(payload["adapter_identity"]["declared_file_set_sha256"], payload["adapter_identity"]["replayed_file_set_sha256"])
            self.assertTrue(payload["governance"]["public_safe"])
            self.assertTrue(payload["governance"]["hashes_only"])
            self.assertFalse(payload["governance"]["local_paths_included"])
            rendered = out.read_text(encoding="utf-8")
            self.assertNotIn("qwen-local+candidate-a", rendered)
            self.assertNotIn(str(root), rendered)
            self.assertTrue(check_schema_file(out, "tau3_candidate_identity")["passed"])
            self.assertFalse(out.stat().st_mode & stat.S_IWUSR)

    def test_cli_writes_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            receipt = _write_training_fixture(root)
            out = root / "identity.json"
            stdout = StringIO()
            stderr = StringIO()

            with redirect_stdout(stdout), redirect_stderr(stderr):
                code = main([
                    "--candidate-id",
                    "candidate-a",
                    "--training-receipt",
                    str(receipt),
                    "--endpoint-model",
                    "qwen-local+candidate-a",
                    "--out",
                    str(out),
                ])

            self.assertEqual(code, 0, stderr.getvalue())
            summary = json.loads(stdout.getvalue())
            self.assertEqual(summary["candidate_id"], "candidate-a")
            self.assertTrue(check_schema_file(out, "tau3_candidate_identity")["passed"])

    def test_refuses_tampered_adapter_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            receipt = _write_training_fixture(root)
            (root / "adapter" / "adapter_model.safetensors").write_bytes(b"tampered")

            with self.assertRaisesRegex(Tau3CandidateIdentityError, "adapter file hash does not replay"):
                build_tau3_candidate_identity(
                    candidate_id="candidate-a",
                    training_receipt_path=receipt,
                    endpoint_model="qwen-local+candidate-a",
                    output_path=root / "identity.json",
                )

    def test_refuses_private_or_path_candidate_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            receipt = _write_training_fixture(root)

            for candidate_id in ("Candidate-A", "../candidate-a", "candidate/a", "sk-secret-token"):
                with self.subTest(candidate_id=candidate_id):
                    with self.assertRaisesRegex(Tau3CandidateIdentityError, "public-safe slug"):
                        build_tau3_candidate_identity(
                            candidate_id=candidate_id,
                            training_receipt_path=receipt,
                            endpoint_model="qwen-local+candidate-a",
                            output_path=root / f"{hashlib.sha256(candidate_id.encode()).hexdigest()}.json",
                        )

    def test_refuses_absolute_adapter_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            receipt = _write_training_fixture(root)
            payload = json.loads(receipt.read_text(encoding="utf-8"))
            payload["adapter"]["path"] = str(root / "adapter")
            receipt.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(Tau3CandidateIdentityError, "adapter.path must be a portable relative path"):
                build_tau3_candidate_identity(
                    candidate_id="candidate-a",
                    training_receipt_path=receipt,
                    endpoint_model="qwen-local+candidate-a",
                    output_path=root / "identity.json",
                )

    def test_refuses_output_inside_adapter_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            receipt = _write_training_fixture(root)

            with self.assertRaisesRegex(Tau3CandidateIdentityError, "outside the adapter directory"):
                build_tau3_candidate_identity(
                    candidate_id="candidate-a",
                    training_receipt_path=receipt,
                    endpoint_model="qwen-local+candidate-a",
                    output_path=root / "adapter" / "identity.json",
                )


def _write_training_fixture(root: Path) -> Path:
    adapter = root / "adapter"
    adapter.mkdir()
    checkpoint = adapter / "checkpoint-0001"
    checkpoint.mkdir()
    (adapter / "adapter_config.json").write_text("{}\n", encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter-weights")
    (adapter / "notes.txt").write_text("non-weight artifact\n", encoding="utf-8")
    (checkpoint / "adapter_model.safetensors").write_bytes(b"checkpoint-weights")
    files = []
    for path in sorted(adapter.rglob("*")):
        if path.is_file():
            rel = path.relative_to(adapter).as_posix()
            files.append({
                "path": rel,
                "size": path.stat().st_size,
                "sha256": _sha256_file(path),
                "kind": _fixture_fingerprint_kind(rel),
            })
    receipt = {
        "schema_version": "hfr.tau3_mlx_training_run.v1",
        "phase": "final",
        "created_at": "2026-07-23T00:00:00Z",
        "bundle": {"kind": "mixture"},
        "output_dir": ".",
        "prelaunch_receipt": {"path": "prelaunch_receipt.json", "sha256": "a" * 64, "read_only": True},
        "telemetry": {"path": "telemetry.jsonl", "sha256": "b" * 64, "event_count": 1},
        "command": ["mlx_lm.lora"],
        "config": {},
        "mlx_lora_config": {"path": "mlx_lora_config.json", "sha256": "c" * 64, "read_only": True},
        "training_binding": {
            "protocol": {
                "sha256": "0" * 64,
                "protocol_signature": "1" * 64,
                "protocol_signature_provenance": {
                    "source": "protocol_manifest.signature",
                    "algorithm": "sha256",
                },
                "model_freeze_sha256": "2" * 64,
                "recipe_space_sha256": "3" * 64,
                "mlx_qlora_plan_sha256": "4" * 64,
            },
            "model": {
                "identity_sha256": "5" * 64,
                "tree_sha256": "6" * 64,
            },
            "dataset": {
                "manifest_sha256": "7" * 64,
                "files_sha256": "8" * 64,
                "source_binding_sha256": "9" * 64,
            },
            "recipe": {
                "recipe_sha256": "a" * 64,
            },
        },
        "checks": [],
        "terminal_status": "success",
        "exit_code": 0,
        "timed_out": False,
        "interrupted": False,
        "elapsed_seconds": 1.0,
        "peak_child_rss_kb": 0,
        "losses": {},
        "adapter": {
            "path": "adapter",
            "file_count": len(files),
            "files": files,
            "tree_sha256": _training_tree_sha256(files),
        },
        "adapter_weight_file_count": 1,
        "weights_updated": True,
        "schema_checked": True,
    }
    path = root / "training_receipt.json"
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _training_tree_sha256(files):
    digest = hashlib.sha256()
    for record in files:
        digest.update(json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    return digest.hexdigest()


def _fixture_fingerprint_kind(rel: str) -> str:
    name = Path(rel).name
    if name in {"adapter_config.json", "config.json"}:
        return "config"
    if "checkpoint" in rel.lower():
        return "checkpoint"
    if Path(rel).suffix in {".safetensors", ".npz", ".bin"}:
        return "adapter"
    return "artifact"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    unittest.main()
