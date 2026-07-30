from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from flightrecorder.schema_registry import check_schema_contract
from flightrecorder.tau3_benchmark_protocol_lineage import (
    Tau3BenchmarkProtocolLineageError,
    create_tau3_benchmark_protocol_lineage,
    create_tau3_blind_custody_receipt,
    validate_tau3_benchmark_protocol_lineage,
    validate_tau3_blind_custody_receipt,
)
from tests.test_tau3_mlx_training import _fake_model, _protocol_config

RETIRED_SEALED_SHA = "9" * 64
INCIDENT_SHA = "8" * 64
SOURCE_REVISION = "1" * 40
ROOT = Path(__file__).resolve().parents[1]


class Tau3BenchmarkProtocolLineageTests(unittest.TestCase):
    def test_custody_and_sealed_only_protocol_rotation_replay(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = _fixture(root)

            custody = create_tau3_blind_custody_receipt(
                custody_id="opaque-custody-handle",
                sealed_source_manifest=fixture["sealed"],
                generator_validation=fixture["generator"],
                fresh_contamination_replay=fixture["contamination"],
                retired_source_incident_sha256=INCIDENT_SHA,
                out=root / "custody.json",
                created_at="2026-07-30T00:00:00Z",
            )
            self.assertEqual(custody["task_count"], 100)
            self.assertEqual(sum(custody["domain_counts"].values()), 100)
            self.assertFalse(custody["custody"]["consumed"])
            self.assertEqual((root / "custody.json").stat().st_mode & 0o777, 0o444)
            custody_schema = check_schema_contract(
                custody,
                name_or_id="tau3_blind_custody_receipt",
            )
            self.assertTrue(custody_schema["passed"], custody_schema["errors"])
            custody_replay = validate_tau3_blind_custody_receipt(
                custody_receipt=root / "custody.json",
                sealed_source_manifest=fixture["sealed"],
                generator_validation=fixture["generator"],
                fresh_contamination_replay=fixture["contamination"],
                expected_retired_source_incident_sha256=INCIDENT_SHA,
            )
            self.assertTrue(custody_replay["passed"])

            lineage = create_tau3_benchmark_protocol_lineage(
                training_protocol=fixture["training"],
                benchmark_protocol=fixture["benchmark"],
                custody_receipt=root / "custody.json",
                sealed_source_manifest=fixture["sealed"],
                generator_validation=fixture["generator"],
                fresh_contamination_replay=fixture["contamination"],
                retired_source_incident_sha256=INCIDENT_SHA,
                out=root / "lineage.json",
                created_at="2026-07-30T00:01:00Z",
            )
            self.assertTrue(lineage["passed"])
            self.assertEqual(lineage["allowed_delta"]["change_count"], 4)
            self.assertNotEqual(
                lineage["training_protocol_sha256"],
                lineage["benchmark_protocol_sha256"],
            )
            self.assertEqual((root / "lineage.json").stat().st_mode & 0o777, 0o444)
            lineage_schema = check_schema_contract(
                lineage,
                name_or_id="tau3_benchmark_protocol_lineage",
            )
            self.assertTrue(lineage_schema["passed"], lineage_schema["errors"])
            replay = validate_tau3_benchmark_protocol_lineage(
                lineage=root / "lineage.json",
                training_protocol=fixture["training"],
                benchmark_protocol=fixture["benchmark"],
                custody_receipt=root / "custody.json",
                sealed_source_manifest=fixture["sealed"],
                generator_validation=fixture["generator"],
                fresh_contamination_replay=fixture["contamination"],
                retired_source_incident_sha256=INCIDENT_SHA,
            )
            self.assertTrue(replay["passed"])

    def test_protocol_rotation_rejects_nonsealed_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = _fixture(root)
            create_tau3_blind_custody_receipt(
                custody_id="opaque-custody-handle",
                sealed_source_manifest=fixture["sealed"],
                generator_validation=fixture["generator"],
                fresh_contamination_replay=fixture["contamination"],
                retired_source_incident_sha256=INCIDENT_SHA,
                out=root / "custody.json",
            )
            benchmark = _read_json(fixture["benchmark"])
            benchmark["harness_contract"]["no_test_time_search"] = False
            _write_json(fixture["benchmark"], benchmark)
            with self.assertRaisesRegex(
                Tau3BenchmarkProtocolLineageError,
                "outside the sealed-only allowlist",
            ):
                create_tau3_benchmark_protocol_lineage(
                    training_protocol=fixture["training"],
                    benchmark_protocol=fixture["benchmark"],
                    custody_receipt=root / "custody.json",
                    sealed_source_manifest=fixture["sealed"],
                    generator_validation=fixture["generator"],
                    fresh_contamination_replay=fixture["contamination"],
                    retired_source_incident_sha256=INCIDENT_SHA,
                    out=root / "lineage.json",
                )

    def test_custody_rejects_overlap_and_tampered_self_seal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = _fixture(root)
            create_tau3_blind_custody_receipt(
                custody_id="opaque-custody-handle",
                sealed_source_manifest=fixture["sealed"],
                generator_validation=fixture["generator"],
                fresh_contamination_replay=fixture["contamination"],
                retired_source_incident_sha256=INCIDENT_SHA,
                out=root / "custody.json",
            )
            custody_path = root / "custody.json"
            custody_path.chmod(0o600)
            custody = _read_json(custody_path)
            custody["task_count"] = 99
            _write_json(custody_path, custody)
            with self.assertRaisesRegex(
                Tau3BenchmarkProtocolLineageError,
                "registered schema|self-seal",
            ):
                validate_tau3_blind_custody_receipt(
                    custody_receipt=custody_path,
                    sealed_source_manifest=fixture["sealed"],
                    generator_validation=fixture["generator"],
                    fresh_contamination_replay=fixture["contamination"],
                )

            contamination = _read_json(fixture["contamination"])
            contamination["overlaps"]["training_task"] = 1
            _write_json(fixture["contamination"], contamination)
            with self.assertRaisesRegex(
                Tau3BenchmarkProtocolLineageError,
                "registered schema|overlap",
            ):
                create_tau3_blind_custody_receipt(
                    custody_id="second-handle",
                    sealed_source_manifest=fixture["sealed"],
                    generator_validation=fixture["generator"],
                    fresh_contamination_replay=fixture["contamination"],
                    retired_source_incident_sha256=INCIDENT_SHA,
                    out=root / "second-custody.json",
                )

    def test_custody_rejects_duplicate_hashes_and_unbalanced_domains(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = _fixture(root)
            sealed = _read_json(fixture["sealed"])
            sealed["entries"][1]["task_sha256"] = sealed["entries"][0]["task_sha256"]
            _write_json(fixture["sealed"], sealed)
            _rebind_reports(fixture)
            with self.assertRaisesRegex(
                Tau3BenchmarkProtocolLineageError,
                "100 unique hashes",
            ):
                create_tau3_blind_custody_receipt(
                    custody_id="duplicate-handle",
                    sealed_source_manifest=fixture["sealed"],
                    generator_validation=fixture["generator"],
                    fresh_contamination_replay=fixture["contamination"],
                    retired_source_incident_sha256=INCIDENT_SHA,
                    out=root / "custody.json",
                )

            fixture = _fixture(root / "unbalanced")
            sealed = _read_json(fixture["sealed"])
            sealed["domain_counts"] = {"airline": 98, "retail": 1, "telecom": 1}
            _write_json(fixture["sealed"], sealed)
            _rebind_reports(fixture)
            generator = _read_json(fixture["generator"])
            generator["domain_counts"] = dict(sealed["domain_counts"])
            _write_json(fixture["generator"], generator)
            with self.assertRaisesRegex(
                Tau3BenchmarkProtocolLineageError,
                "balanced to within one task",
            ):
                create_tau3_blind_custody_receipt(
                    custody_id="unbalanced-handle",
                    sealed_source_manifest=fixture["sealed"],
                    generator_validation=fixture["generator"],
                    fresh_contamination_replay=fixture["contamination"],
                    retired_source_incident_sha256=INCIDENT_SHA,
                    out=root / "unbalanced/custody.json",
                )

    def test_build_and_validate_clis_replay_the_same_lineage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = _fixture(root)
            custody = root / "custody.json"
            lineage = root / "lineage.json"
            common = [
                "--sealed-source-manifest",
                str(fixture["sealed"]),
                "--generator-validation",
                str(fixture["generator"]),
                "--fresh-contamination-replay",
                str(fixture["contamination"]),
                "--retired-source-incident-sha256",
                INCIDENT_SHA,
            ]
            build_custody = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/build_tau3_blind_custody_receipt.py"),
                    "--custody-id",
                    "opaque-cli-handle",
                    *common,
                    "--out",
                    str(custody),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(build_custody.returncode, 0, build_custody.stderr)
            self.assertEqual(json.loads(build_custody.stdout)["task_count"], 100)
            validate_custody = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/validate_tau3_blind_custody_receipt.py"),
                    "--custody-receipt",
                    str(custody),
                    *common,
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(validate_custody.returncode, 0, validate_custody.stderr)
            self.assertTrue(json.loads(validate_custody.stdout)["passed"])

            lineage_common = [
                "--training-protocol",
                str(fixture["training"]),
                "--benchmark-protocol",
                str(fixture["benchmark"]),
                "--custody-receipt",
                str(custody),
                *common,
            ]
            build_lineage = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/build_tau3_benchmark_protocol_lineage.py"),
                    *lineage_common,
                    "--out",
                    str(lineage),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(build_lineage.returncode, 0, build_lineage.stderr)
            self.assertTrue(json.loads(build_lineage.stdout)["passed"])
            validate_lineage = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/validate_tau3_benchmark_protocol_lineage.py"),
                    "--lineage",
                    str(lineage),
                    *lineage_common,
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(validate_lineage.returncode, 0, validate_lineage.stderr)
            self.assertTrue(json.loads(validate_lineage.stdout)["passed"])


def _fixture(root: Path) -> dict[str, Path]:
    root.mkdir(parents=True, exist_ok=True)
    _, identity = _fake_model(root)
    training = _protocol_config(root, identity)
    protocol = _read_json(training)
    protocol["tau_revision"].update(
        {
            "revision": SOURCE_REVISION,
            "split_hashes": {
                "train": "2" * 64,
                "development": "3" * 64,
                "sealed": RETIRED_SEALED_SHA,
            },
        }
    )
    protocol["split_manifest"].update(
        {
            "source_manifest": {
                "local_path": "local/tau3/source-v1/manifest.json",
                "sha256": "4" * 64,
            },
            "splits": {
                "train": {"local_path": "local/train.json", "sealed": False, "sha256": "2" * 64},
                "development": {
                    "local_path": "local/development.json",
                    "sealed": False,
                    "sha256": "3" * 64,
                },
                "sealed": {
                    "local_path": "local/retired-sealed.json",
                    "sealed": True,
                    "sha256": RETIRED_SEALED_SHA,
                },
            },
        }
    )
    protocol["sealed_manifest"].update(
        {
            "access_count": 0,
            "manifest_sha256": RETIRED_SEALED_SHA,
            "leakage_blocking_hashes": ["5" * 64],
            "prompt_template_hashes": ["6" * 64],
        }
    )
    _write_json(training, protocol)

    sealed = root / "fresh-sealed-source.json"
    entries = [
        {
            "domain": (
                "airline"
                if index < 34
                else "retail"
                if index < 67
                else "telecom"
            ),
            "task_id_sha256": hashlib.sha256(f"id-{index}".encode()).hexdigest(),
            "prompt_sha256": hashlib.sha256(f"prompt-{index}".encode()).hexdigest(),
            "task_sha256": hashlib.sha256(f"task-{index}".encode()).hexdigest(),
        }
        for index in range(100)
    ]
    _write_json(
        sealed,
        {
            "schema_version": "hfr.tau3_sealed_source_manifest.v1",
            "source_revision": SOURCE_REVISION,
            "hashes_only": True,
            "task_count": 100,
            "domain_counts": {"airline": 34, "retail": 33, "telecom": 33},
            "entries": entries,
        },
    )
    fresh_sha = _sha256(sealed)

    generator = root / "generator-validation.json"
    _write_json(
        generator,
        {
            "schema_version": "hfr.tau3_blind_generator_validation.v1",
            "created_at": "2026-07-30T00:00:00Z",
            "passed": True,
            "source_revision": SOURCE_REVISION,
            "sealed_source_manifest_sha256": fresh_sha,
            "task_count": 100,
            "domain_counts": {"airline": 34, "retail": 33, "telecom": 33},
            "generator_source": {"commit_sha": "7" * 40, "script_sha256": "7" * 64},
            "golden_replay": {
                "passed": True,
                "replayed_task_count": 100,
                "passed_task_count": 100,
                "failed_task_count": 0,
                "state_check_failure_count": 0,
            },
            "schema_validation_passed": True,
            "task_hashes_unique": True,
            "prompt_hashes_unique": True,
            "hashes_only": True,
            "local_paths_included": False,
            "raw_payload_included": False,
        },
    )
    contamination = root / "fresh-contamination.json"
    _write_json(
        contamination,
        {
            "schema_version": "hfr.tau3_fresh_contamination_replay.v1",
            "created_at": "2026-07-30T00:00:01Z",
            "passed": True,
            "training_dataset_sha256": "a" * 64,
            "development_source_sha256": "b" * 64,
            "retired_sealed_source_manifest_sha256": RETIRED_SEALED_SHA,
            "fresh_sealed_source_manifest_sha256": fresh_sha,
            "overlaps": {
                f"{source}_{kind}": 0
                for source in ("training", "development", "retired_sealed")
                for kind in ("task", "task_id", "prompt", "family")
            },
            "hashes_only": True,
            "local_paths_included": False,
            "raw_payload_included": False,
        },
    )

    benchmark = root / "benchmark-protocol.json"
    benchmark_payload = copy.deepcopy(protocol)
    benchmark_payload["tau_revision"]["split_hashes"]["sealed"] = fresh_sha
    benchmark_payload["split_manifest"]["source_manifest"] = {
        "local_path": "custody/fresh-source-manifest.json",
        "sha256": "c" * 64,
    }
    benchmark_payload["split_manifest"]["splits"]["sealed"] = {
        "local_path": "custody/fresh-sealed-source.json",
        "sealed": True,
        "sha256": fresh_sha,
    }
    benchmark_payload["sealed_manifest"] = {
        "schema_version": "hfr.tau3_sealed_manifest.v1",
        "quarantine_predates_generation": True,
        "access_count": 0,
        "manifest_sha256": fresh_sha,
        "leakage_blocking_hashes": sorted(
            {
                entry[key]
                for entry in entries
                for key in ("task_id_sha256", "prompt_sha256", "task_sha256")
            }
        ),
        "prompt_template_hashes": sorted(entry["prompt_sha256"] for entry in entries),
    }
    _write_json(benchmark, benchmark_payload)
    return {
        "training": training,
        "benchmark": benchmark,
        "sealed": sealed,
        "generator": generator,
        "contamination": contamination,
    }


def _rebind_reports(fixture: dict[str, Path]) -> None:
    fresh_sha = _sha256(fixture["sealed"])
    for key, field in (
        ("generator", "sealed_source_manifest_sha256"),
        ("contamination", "fresh_sealed_source_manifest_sha256"),
    ):
        payload = _read_json(fixture[key])
        payload[field] = fresh_sha
        _write_json(fixture[key], payload)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
