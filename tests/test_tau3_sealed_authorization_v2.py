from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from flightrecorder.schema_registry import check_schema_contract
from flightrecorder.tau3_benchmark_run import (
    Tau3BenchmarkConfig,
    _sealed_authorization_binding,
)
from flightrecorder.tau3_benchmark_protocol_lineage import (
    create_tau3_benchmark_protocol_lineage,
    create_tau3_blind_custody_receipt,
)
from flightrecorder.tau3_sealed_authorization import (
    Tau3SealedAuthorizationError,
    create_tau3_sealed_authorization,
    validate_tau3_sealed_authorization,
)
from flightrecorder.tau3_competitive_v3_training_evidence import (
    validate_tau3_competitive_v3_training_evidence,
)
from tests.test_tau3_benchmark_protocol_lineage import (
    INCIDENT_SHA,
    SOURCE_REVISION,
    _fixture,
)
from tests.test_tau3_competitive_v3 import (
    build_complete_bundle,
    canonical_sha256,
)


class Tau3SealedAuthorizationV2Tests(unittest.TestCase):
    def test_v2_authorization_replays_dual_protocol_and_fresh_custody(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = _v2_fixture(Path(tmp))

            result = create_tau3_sealed_authorization(
                candidate_lock=fixture["lock"],
                protocol=fixture["benchmark"],
                sealed_source_manifest=fixture["sealed"],
                out=fixture["authorization"],
                created_at="2026-07-30T00:03:00Z",
                training_protocol=fixture["training"],
                benchmark_protocol_lineage=fixture["lineage"],
                custody_receipt=fixture["custody"],
                generator_validation=fixture["generator"],
                fresh_contamination_replay=fixture["contamination"],
                retired_source_incident_sha256=INCIDENT_SHA,
                candidate_selection_report=fixture["selection"],
                qualified_training_evidence=fixture["training_evidence"],
            )

            self.assertTrue(result["authorized"])
            self.assertEqual(
                result["schema_version"],
                "hfr.tau3_sealed_authorization_result.v2",
            )
            authorization = _read(fixture["authorization"])
            self.assertTrue(
                check_schema_contract(
                    authorization,
                    name_or_id="tau3_sealed_authorization_v2",
                )["passed"]
            )
            self.assertEqual(
                authorization["candidate_lock"]["training_protocol_sha256"],
                _sha256(fixture["training"]),
            )
            self.assertEqual(
                authorization["protocol"]["sha256"],
                _sha256(fixture["benchmark"]),
            )
            self.assertEqual(
                authorization["protocol_lineage"]["sha256"],
                _sha256(fixture["lineage"]),
            )
            self.assertEqual(
                authorization["blind_custody"]["receipt_sha256"],
                _sha256(fixture["custody"]),
            )
            self.assertEqual(
                authorization["sealed_source"]["domain_counts"],
                {"airline": 34, "retail": 33, "telecom": 33},
            )
            replay = validate_tau3_sealed_authorization(
                authorization_path=fixture["authorization"],
                candidate_lock_path=fixture["lock"],
                protocol_path=fixture["benchmark"],
                sealed_source_manifest_path=fixture["sealed"],
                arm_id="adapter",
                seeds=(101, 202, 303, 404),
                expected_tau_revision=SOURCE_REVISION,
                training_protocol_path=fixture["training"],
                benchmark_protocol_lineage_path=fixture["lineage"],
                custody_receipt_path=fixture["custody"],
                generator_validation_path=fixture["generator"],
                fresh_contamination_replay_path=fixture["contamination"],
                retired_source_incident_sha256=INCIDENT_SHA,
                candidate_selection_report_path=fixture["selection"],
                qualified_training_evidence_path=fixture["training_evidence"],
            )
            self.assertTrue(replay["authorized"])
            self.assertEqual(
                replay["training_protocol_sha256"],
                _sha256(fixture["training"]),
            )

            out = Path(tmp) / "benchmark-arm"
            staged = out / "inputs/sealed_authorization.json"
            staged.parent.mkdir(parents=True)
            shutil.copyfile(fixture["authorization"], staged)
            staged_ref = {
                "path": "inputs/sealed_authorization.json",
                "sha256": _sha256(staged),
                "size": staged.stat().st_size,
            }
            binding = _sealed_authorization_binding(
                Tau3BenchmarkConfig(
                    mode="sealed",
                    arm_id="adapter",
                    protocol_path=fixture["benchmark"],
                    sealed_task_count_manifest=fixture["sealed"],
                    sealed_authorization=fixture["authorization"],
                    candidate_lock=fixture["lock"],
                    training_protocol=fixture["training"],
                    benchmark_protocol_lineage=fixture["lineage"],
                    custody_receipt=fixture["custody"],
                    generator_validation=fixture["generator"],
                    fresh_contamination_replay=fixture["contamination"],
                    retired_source_incident_sha256=INCIDENT_SHA,
                    candidate_selection_report=fixture["selection"],
                    qualified_training_evidence=fixture["training_evidence"],
                ),
                staged_ref,
                out=out,
                expected_tau_revision=SOURCE_REVISION,
            )
            self.assertEqual(
                binding["benchmark_protocol_lineage_sha256"],
                _sha256(fixture["lineage"]),
            )

    def test_v2_authorization_fails_without_full_lineage_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = _v2_fixture(Path(tmp))

            with self.assertRaisesRegex(
                Tau3SealedAuthorizationError,
                "requires training protocol",
            ):
                create_tau3_sealed_authorization(
                    candidate_lock=fixture["lock"],
                    protocol=fixture["benchmark"],
                    sealed_source_manifest=fixture["sealed"],
                    out=fixture["authorization"],
                    created_at="2026-07-30T00:03:00Z",
                )

            self.assertFalse(fixture["authorization"].exists())

    def test_v2_authorization_rejects_training_protocol_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = _v2_fixture(Path(tmp))
            training = _read(fixture["training"])
            training["budget"]["passed"] = False
            _write(fixture["training"], training)

            with self.assertRaisesRegex(
                Tau3SealedAuthorizationError,
                "lineage replay failed|candidate lock",
            ):
                create_tau3_sealed_authorization(
                    candidate_lock=fixture["lock"],
                    protocol=fixture["benchmark"],
                    sealed_source_manifest=fixture["sealed"],
                    out=fixture["authorization"],
                    created_at="2026-07-30T00:03:00Z",
                    training_protocol=fixture["training"],
                    benchmark_protocol_lineage=fixture["lineage"],
                    custody_receipt=fixture["custody"],
                    generator_validation=fixture["generator"],
                    fresh_contamination_replay=fixture["contamination"],
                    retired_source_incident_sha256=INCIDENT_SHA,
                    candidate_selection_report=fixture["selection"],
                    qualified_training_evidence=fixture["training_evidence"],
                )

            self.assertFalse(fixture["authorization"].exists())

    def test_v2_authorization_rejects_lock_outside_qualified_cohort(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = _v2_fixture(Path(tmp))
            evidence = _read(fixture["training_evidence"])
            evidence["qualified_candidates"][0]["candidate_id"] = (
                "candidate-c"
            )
            _write(fixture["training_evidence"], evidence)

            with self.assertRaisesRegex(
                Tau3SealedAuthorizationError,
                "outside the qualified training cohort",
            ):
                create_tau3_sealed_authorization(
                    candidate_lock=fixture["lock"],
                    protocol=fixture["benchmark"],
                    sealed_source_manifest=fixture["sealed"],
                    out=fixture["authorization"],
                    created_at="2026-07-30T00:03:00Z",
                    training_protocol=fixture["training"],
                    benchmark_protocol_lineage=fixture["lineage"],
                    custody_receipt=fixture["custody"],
                    generator_validation=fixture["generator"],
                    fresh_contamination_replay=fixture["contamination"],
                    retired_source_incident_sha256=INCIDENT_SHA,
                    candidate_selection_report=fixture["selection"],
                    qualified_training_evidence=fixture[
                        "training_evidence"
                    ],
                )

            self.assertFalse(fixture["authorization"].exists())


def _v2_fixture(root: Path) -> dict[str, Path]:
    fixture = _fixture(root)
    qualification_root = root / "qualification"
    build_complete_bundle(qualification_root)
    selection_path = qualification_root / "final/candidate-selection.json"
    training_evidence_path = qualification_root / "training-evidence.json"
    selection = _read(selection_path)
    selected_candidate_id = str(selection["selected_candidate_id"])
    selected_row = next(
        candidate
        for candidate in selection["candidates"]
        if candidate["candidate_id"] == selected_candidate_id
    )
    training_validation = (
        validate_tau3_competitive_v3_training_evidence(
            training_evidence_path
        )
    )
    qualified = training_validation["qualified_candidates"][
        selected_candidate_id
    ]
    common = {
        "harness_contract": {
            "domains": ["airline", "retail", "telecom"],
            "context_window": 16384,
            "decoding": {
                "temperature": 0.0,
                "top_p": 1.0,
                "max_output_tokens": 1024,
                "seeds": [101, 202, 303, 404],
            },
            "turn_limit": 30,
            "retry_policy": "none",
            "no_test_time_search": True,
            "test_time_search": False,
        },
        "model_freeze": {
            "base_model": {
                "name": "local/base",
                "revision": "1" * 40,
                "local_identity_sha256": "1" * 64,
            },
            "comparators": [
                {
                    "name": "local/comparator-1",
                    "revision": "2" * 40,
                    "local_identity_sha256": "2" * 64,
                },
                {
                    "name": "local/comparator-2",
                    "revision": "3" * 40,
                    "local_identity_sha256": "3" * 64,
                },
            ],
        },
        "budget": {"passed": True},
        "candidate_selection_contract": {"passed": True},
        "contamination_attestation": {"passed": True},
        "redaction_attestation": {"passed": True},
        "licenses": [
            {"status": "approved", "training_allowed": True}
            for _ in range(4)
        ],
    }
    for key in ("training", "benchmark"):
        protocol = _read(fixture[key])
        protocol.update(json.loads(json.dumps(common)))
        _write(fixture[key], protocol)

    custody = root / "custody.json"
    create_tau3_blind_custody_receipt(
        custody_id="opaque-v2-custody",
        sealed_source_manifest=fixture["sealed"],
        generator_validation=fixture["generator"],
        fresh_contamination_replay=fixture["contamination"],
        retired_source_incident_sha256=INCIDENT_SHA,
        out=custody,
        created_at="2026-07-30T00:01:00Z",
    )
    lineage = root / "lineage.json"
    create_tau3_benchmark_protocol_lineage(
        training_protocol=fixture["training"],
        benchmark_protocol=fixture["benchmark"],
        custody_receipt=custody,
        sealed_source_manifest=fixture["sealed"],
        generator_validation=fixture["generator"],
        fresh_contamination_replay=fixture["contamination"],
        retired_source_incident_sha256=INCIDENT_SHA,
        out=lineage,
        created_at="2026-07-30T00:02:00Z",
    )
    lock = root / "candidate-lock.json"
    _write(
        lock,
        {
            "schema_version": "hfr.tau3_candidate_lock.v2",
            "created_at": "2026-07-30T00:02:30Z",
            "selected_candidate_id_hash": canonical_sha256(
                selected_candidate_id
            ),
            "candidate_identity_sha256": selected_row[
                "candidate_identity"
            ]["sha256"],
            "development_selection_report_sha256": _sha256(selection_path),
            "development_benchmark_manifest_sha256": "7" * 64,
            "training_receipt_sha256": qualified[
                "training_receipt_sha256"
            ],
            "endpoint_model_sha256": "9" * 64,
            "evaluator_model_contract_sha256": "a" * 64,
            "adapter_tree_sha256": qualified["adapter_tree_sha256"],
            "recipe_sha256": qualified["recipe_sha256"],
            "base_identity_sha256": "1" * 64,
            "base_tree_sha256": "d" * 64,
            "dataset_manifest_sha256": "e" * 64,
            "dataset_files_sha256": "f" * 64,
            "source_binding_sha256": hashlib.sha256(b"source").hexdigest(),
            "training_protocol_sha256": _sha256(fixture["training"]),
            "training_protocol_signature": _sha256(fixture["training"]),
            "benchmark_protocol_sha256": _sha256(fixture["benchmark"]),
            "benchmark_protocol_signature": _sha256(fixture["benchmark"]),
            "benchmark_protocol_lineage_sha256": _sha256(lineage),
            "hashes_only": True,
            "sealed_access_authorized": True,
            "local_paths_included": False,
            "raw_payload_included": False,
        },
    )
    return {
        **fixture,
        "custody": custody,
        "lineage": lineage,
        "lock": lock,
        "selection": selection_path,
        "training_evidence": training_evidence_path,
        "authorization": root / "authorization.json",
    }


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    unittest.main()
