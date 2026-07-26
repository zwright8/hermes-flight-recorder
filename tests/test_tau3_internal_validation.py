from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from flightrecorder.mlx_internal_validation import (
    _load_resume_measurements,
)
from flightrecorder.schema_registry import check_schema_contract
from flightrecorder.tau3_exposure import (
    REQUIRED_BEHAVIORS,
    REQUIRED_DOMAINS,
)
from flightrecorder.tau3_internal_validation import (
    Tau3InternalValidationError,
    _canonical_sha256,
    _expected_run_binding,
    build_tau3_internal_validation,
    validate_tau3_internal_validation,
)


class Tau3InternalValidationTests(unittest.TestCase):
    def test_round_trip_replays_complete_loss_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _fixture(Path(tmp))

            artifact = build_tau3_internal_validation(
                dataset_path=paths["dataset"],
                measurements_path=paths["measurements"],
                run_binding_path=paths["run_binding"],
                training_receipt_path=paths["receipt"],
                protocol_path=paths["protocol"],
                model_identity_path=paths["identity"],
                output_path=paths["artifact"],
                max_seq_length=64,
                created_at="2026-07-26T00:00:00Z",
            )
            result = validate_tau3_internal_validation(
                paths["artifact"],
                dataset_path=paths["dataset"],
                training_receipt_path=paths["receipt"],
                protocol_path=paths["protocol"],
                model_identity_path=paths["identity"],
            )

            self.assertTrue(result["passed"], result["errors"])
            self.assertTrue(
                check_schema_contract(
                    artifact,
                    name_or_id="tau3_internal_validation",
                )["passed"]
            )
            self.assertEqual(
                artifact["coverage"]["row_count"],
                len(REQUIRED_BEHAVIORS),
            )
            self.assertEqual(
                {row["name"] for row in artifact["coverage"]["domains"]},
                set(REQUIRED_DOMAINS),
            )
            self.assertEqual(
                {
                    row["name"]
                    for row in artifact["coverage"]["behaviors"]
                },
                set(REQUIRED_BEHAVIORS),
            )
            self.assertEqual(
                artifact["aggregate"]["numerical_failure_count"],
                0,
            )

    def test_tampered_measurement_fails_independent_replay(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _fixture(Path(tmp))
            build_tau3_internal_validation(
                dataset_path=paths["dataset"],
                measurements_path=paths["measurements"],
                run_binding_path=paths["run_binding"],
                training_receipt_path=paths["receipt"],
                protocol_path=paths["protocol"],
                model_identity_path=paths["identity"],
                output_path=paths["artifact"],
                max_seq_length=64,
            )
            rows = _read_jsonl(paths["measurements"])
            rows[0]["mean_loss"] = 9.0
            _write_jsonl(paths["measurements"], rows)
            artifact = _read_json(paths["artifact"])
            artifact["measurements"]["sha256"] = _sha256_file(
                paths["measurements"]
            )
            artifact["measurements"]["size"] = (
                paths["measurements"].stat().st_size
            )
            paths["artifact"].chmod(0o600)
            _write_json(paths["artifact"], artifact)

            result = validate_tau3_internal_validation(
                paths["artifact"],
                dataset_path=paths["dataset"],
                training_receipt_path=paths["receipt"],
                protocol_path=paths["protocol"],
                model_identity_path=paths["identity"],
            )

            self.assertFalse(result["passed"])
            self.assertIn("loss_sum does not replay", " ".join(result["errors"]))

    def test_builder_rejects_incomplete_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _fixture(Path(tmp))
            rows = _read_jsonl(paths["measurements"])
            _write_jsonl(paths["measurements"], rows[:-1])

            with self.assertRaisesRegex(
                Tau3InternalValidationError,
                "exactly one row",
            ):
                build_tau3_internal_validation(
                    dataset_path=paths["dataset"],
                    measurements_path=paths["measurements"],
                    run_binding_path=paths["run_binding"],
                    training_receipt_path=paths["receipt"],
                    protocol_path=paths["protocol"],
                    model_identity_path=paths["identity"],
                    output_path=paths["artifact"],
                    max_seq_length=64,
                )

    def test_builder_rejects_dataset_manifest_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _fixture(Path(tmp))
            manifest = _read_json(paths["manifest"])
            manifest["files"]["valid"]["sha256"] = "0" * 64
            _write_json(paths["manifest"], manifest)

            with self.assertRaisesRegex(
                Tau3InternalValidationError,
                "valid split does not bind",
            ):
                build_tau3_internal_validation(
                    dataset_path=paths["dataset"],
                    measurements_path=paths["measurements"],
                    run_binding_path=paths["run_binding"],
                    training_receipt_path=paths["receipt"],
                    protocol_path=paths["protocol"],
                    model_identity_path=paths["identity"],
                    output_path=paths["artifact"],
                    max_seq_length=64,
                )

    def test_builder_rejects_dataset_not_bound_by_training_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _fixture(Path(tmp))
            receipt = _read_json(paths["receipt"])
            receipt["training_binding"]["dataset"]["manifest_sha256"] = (
                "0" * 64
            )
            _write_json(paths["receipt"], receipt)

            with self.assertRaisesRegex(
                Tau3InternalValidationError,
                "dataset manifest does not match the training receipt",
            ):
                build_tau3_internal_validation(
                    dataset_path=paths["dataset"],
                    measurements_path=paths["measurements"],
                    run_binding_path=paths["run_binding"],
                    training_receipt_path=paths["receipt"],
                    protocol_path=paths["protocol"],
                    model_identity_path=paths["identity"],
                    output_path=paths["artifact"],
                    max_seq_length=64,
                )

    def test_builder_rejects_run_binding_from_another_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _fixture(Path(tmp))
            binding = _read_json(paths["run_binding"])
            binding["bindings"]["adapter_tree_sha256"] = "f" * 64
            _write_json(paths["run_binding"], binding)

            with self.assertRaisesRegex(
                Tau3InternalValidationError,
                "run binding does not match",
            ):
                build_tau3_internal_validation(
                    dataset_path=paths["dataset"],
                    measurements_path=paths["measurements"],
                    run_binding_path=paths["run_binding"],
                    training_receipt_path=paths["receipt"],
                    protocol_path=paths["protocol"],
                    model_identity_path=paths["identity"],
                    output_path=paths["artifact"],
                    max_seq_length=64,
                )

    def test_resume_measurements_reject_changed_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _fixture(Path(tmp))
            dataset_rows = _read_jsonl(paths["dataset"])
            resumed = _load_resume_measurements(
                paths["measurements"],
                dataset_rows,
                max_seq_length=64,
            )
            self.assertEqual(len(resumed), len(dataset_rows))

            rows = _read_jsonl(paths["measurements"])
            rows[1]["row_sha256"] = "0" * 64
            _write_jsonl(paths["measurements"], rows)
            with self.assertRaisesRegex(
                ValueError,
                "row_sha256 mismatch",
            ):
                _load_resume_measurements(
                    paths["measurements"],
                    dataset_rows,
                    max_seq_length=64,
                )


def _fixture(root: Path) -> dict[str, Path]:
    dataset_dir = root / "dataset"
    dataset_dir.mkdir()
    dataset = dataset_dir / "valid.jsonl"
    dataset_rows = []
    for index, behavior in enumerate(REQUIRED_BEHAVIORS):
        tokens = [10 + index, 20 + index, 30 + index, 40 + index]
        prompt_tokens = 2
        dataset_rows.append(
            {
                "messages": [
                    {"role": "system", "content": "policy"},
                    {"role": "assistant", "content": "target"},
                ],
                "metadata": {
                    "domain": REQUIRED_DOMAINS[
                        index % len(REQUIRED_DOMAINS)
                    ],
                    "behavior": behavior,
                    "token_counts": {
                        "input_token_ids": tokens,
                        "input_token_ids_sha256": _canonical_sha256(tokens),
                        "prompt_tokens": prompt_tokens,
                        "supervised_tokens": len(tokens) - prompt_tokens,
                        "total_tokens": len(tokens),
                    },
                },
            }
        )
    _write_jsonl(dataset, dataset_rows)
    manifest = dataset_dir / "manifest.json"
    _write_json(
        manifest,
        {
            "schema_version": "hfr.tau3_competitive_dataset.v1",
            "files": {
                "valid": {
                    "path": "valid.jsonl",
                    "sha256": _sha256_file(dataset),
                    "bytes": dataset.stat().st_size,
                }
            },
        },
    )

    protocol = root / "protocol.json"
    _write_json(
        protocol,
        {"schema_version": "hfr.tau3_protocol_config.v1"},
    )
    identity = root / "model-identity.json"
    _write_json(
        identity,
        {
            "schema_version": "hfr.tau3_model_identity.v1",
            "model_id": "fixture/base",
            "revision": "a" * 40,
        },
    )
    receipt = root / "training-receipt.json"
    adapter_tree_sha256 = "b" * 64
    _write_json(
        receipt,
        {
            "schema_version": "hfr.tau3_mlx_training_run.v1",
            "phase": "final",
            "created_at": "2026-07-26T00:00:00Z",
            "bundle": {"kind": "mixture"},
            "output_dir": ".",
            "command": ["python", "-m", "mlx_lm", "lora"],
            "config": {},
            "checks": [],
            "terminal_status": "success",
            "weights_updated": True,
            "adapter": {
                "path": "adapter",
                "tree_sha256": adapter_tree_sha256,
            },
            "adapter_weight_file_count": 1,
            "training_binding": {
                "protocol": {
                    "sha256": _sha256_file(protocol),
                    "protocol_signature": "c" * 64,
                    "protocol_signature_provenance": {
                        "source": "protocol_file_sha256_content_seal",
                        "algorithm": "sha256",
                    },
                },
                "model": {
                    "identity_sha256": _sha256_file(identity),
                },
                "dataset": {
                    "manifest_sha256": _sha256_file(manifest),
                },
                "recipe": {"max_seq_length": 64},
            },
        },
    )

    validation_dir = root / "validation"
    validation_dir.mkdir()
    measurements = validation_dir / "measurements.jsonl"
    measurement_rows = []
    for index, row in enumerate(dataset_rows):
        metadata = row["metadata"]
        counts = metadata["token_counts"]
        targets = counts["input_token_ids"][counts["prompt_tokens"] :]
        mean_loss = 0.25 + index / 100
        measurement_rows.append(
            {
                "row_index": index,
                "row_sha256": _canonical_sha256(row),
                "domain": metadata["domain"],
                "behavior": metadata["behavior"],
                "prompt_tokens": counts["prompt_tokens"],
                "supervised_tokens": counts["supervised_tokens"],
                "input_token_ids_sha256": _canonical_sha256(
                    counts["input_token_ids"]
                ),
                "target_tokens_sha256": _canonical_sha256(targets),
                "mean_loss": mean_loss,
                "loss_sum": mean_loss * counts["supervised_tokens"],
                "finite": True,
            }
        )
    _write_jsonl(measurements, measurement_rows)
    run_binding = validation_dir / "run-binding.json"
    _write_json(
        run_binding,
        _expected_run_binding(
            dataset_file=dataset,
            dataset_manifest_file=manifest,
            receipt_file=receipt,
            adapter_tree_sha256=adapter_tree_sha256,
            protocol_file=protocol,
            identity_file=identity,
            max_seq_length=64,
        ),
    )
    return {
        "dataset": dataset,
        "manifest": manifest,
        "measurements": measurements,
        "run_binding": run_binding,
        "receipt": receipt,
        "protocol": protocol,
        "identity": identity,
        "artifact": validation_dir / "internal-validation.json",
    }


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, values: list[dict]) -> None:
    path.write_text(
        "".join(
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
            for value in values
        ),
        encoding="utf-8",
    )


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    unittest.main()
