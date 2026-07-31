from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

from flightrecorder.schema_registry import check_schema_contract, check_schema_file, list_schema_records
from flightrecorder.tau3_promotion_preflight import (
    TAU3_POST_PUBLICATION_RECORD_SCHEMA_VERSION,
    TAU3_PROMOTION_PREFLIGHT_SCHEMA_VERSION,
    Tau3PromotionPreflightError,
    _ledger_binds_lock,
    build_tau3_post_publication_record,
    build_tau3_promotion_preflight,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_tau3_promotion_preflight.py"
POST_SCRIPT = ROOT / "scripts" / "build_tau3_post_publication_record.py"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(value: str) -> str:
    return value * 64


class Tau3PromotionPreflightTests(unittest.TestCase):
    def test_allowed_true_is_pre_upload_and_has_null_hf_revision(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = make_fixture(root, promoted=True)
            out = root / "preflight.json"

            decision = build_tau3_promotion_preflight(**paths, out=out, created_at="2026-07-24T00:00:00+00:00")

            self.assertEqual(decision["schema_version"], TAU3_PROMOTION_PREFLIGHT_SCHEMA_VERSION)
            self.assertTrue(decision["allowed"], decision["blocking_reasons"])
            self.assertEqual(decision["publication_status"], "ready_for_publication")
            self.assertIsNone(decision["hf_revision"])
            self.assertFalse(decision["local_paths_included"])
            self.assertTrue(all(binding["sha256"] for binding in decision["evidence_bindings"].values()))
            self.assertTrue(check_schema_file(out, "tau3_promotion_publication_preflight")["passed"])

    def test_valid_negative_result_is_withheld_without_hf_revision(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = make_fixture(root, promoted=False)

            decision = build_tau3_promotion_preflight(**paths, out=root / "negative.json", created_at="2026-07-24T00:00:00+00:00")

            self.assertFalse(decision["allowed"])
            self.assertIsNone(decision["hf_revision"])
            self.assertEqual(decision["publication_status"], "withheld_negative_result")
            self.assertTrue(decision["negative_result_withheld_honestly"])
            self.assertIn("required_evaluation_checks_passed", decision["blocking_reasons"])
            self.assertTrue(check_schema_contract(decision, name_or_id="tau3_promotion_publication_preflight")["passed"])

    def test_v2_dual_protocol_lineage_can_unlock_preflight(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = make_fixture(root, promoted=True)
            upgrade_fixture_to_v2(paths)

            decision = build_tau3_promotion_preflight(
                **paths,
                out=root / "v2-preflight.json",
                created_at="2026-07-30T03:00:00Z",
            )

            self.assertTrue(decision["allowed"], decision["blocking_reasons"])
            self.assertEqual(
                decision["evidence_bindings"][
                    "protocol_lineage_attestation"
                ]["schema_version"],
                "hfr.tau3_benchmark_protocol_lineage.v1",
            )

    def test_v2_attempt_ledger_compares_training_not_benchmark_protocol(self) -> None:
        training_sha = _sha("1")
        benchmark_sha = _sha("2")
        lock_sha = _sha("3")
        candidate_lock_payload = {
            "schema_version": "hfr.tau3_candidate_lock.v2",
            "training_protocol_sha256": training_sha,
            "training_protocol_signature": _sha("4"),
            "benchmark_protocol_sha256": benchmark_sha,
            "benchmark_protocol_signature": _sha("5"),
        }
        ledger = {
            "created_at": "2026-07-30T00:01:00Z",
            "lock": {
                "created_at": "2026-07-30T00:00:00Z",
                "sha256": lock_sha,
            },
            "attempts": [
                {
                    "intent": None,
                    "outcome": None,
                    "training_receipt": None,
                    "bindings": {
                        "protocol_sha256": training_sha,
                        "protocol_signature": _sha("4"),
                    },
                }
            ],
        }

        self.assertTrue(
            _ledger_binds_lock(
                ledger,
                lock_sha,
                candidate_lock=candidate_lock_payload,
            )
        )
        ledger["attempts"][0]["bindings"]["protocol_sha256"] = benchmark_sha
        self.assertFalse(
            _ledger_binds_lock(
                ledger,
                lock_sha,
                candidate_lock=candidate_lock_payload,
            )
        )

    def test_missing_required_evaluation_check_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = make_fixture(root, promoted=True)
            evaluation_payload = _read_json(paths["sealed_public_evaluation_report"])
            evaluation_payload["checks"] = [
                check for check in evaluation_payload["checks"]
                if check["id"] != "unique_paired_results"
            ]
            _write_json(paths["sealed_public_evaluation_report"], evaluation_payload)

            decision = build_tau3_promotion_preflight(**paths, out=root / "missing-check.json")

            self.assertFalse(decision["allowed"])
            self.assertIn("required_evaluation_checks_passed", decision["blocking_reasons"])

    def test_malformed_nested_hash_bindings_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = make_fixture(root, promoted=True)
            authorization = _read_json(paths["sealed_authorization"])
            authorization["candidate_lock"] = []
            _write_json(paths["sealed_authorization"], authorization)

            decision = build_tau3_promotion_preflight(
                **paths,
                out=root / "malformed-bindings.json",
            )

            self.assertFalse(decision["allowed"])
            self.assertIn("schema_contracts_passed", decision["blocking_reasons"])
            self.assertIn("hash_bindings_replay", decision["blocking_reasons"])

    def test_rejects_raw_sealed_or_private_identifier_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = make_fixture(root, promoted=True)
            evaluation = _read_json(paths["sealed_public_evaluation_report"])
            evaluation["messages"] = [{"role": "user", "content": "sealed prompt"}]
            _write_json(paths["sealed_public_evaluation_report"], evaluation)

            with self.assertRaisesRegex(Tau3PromotionPreflightError, "forbidden sealed/private material"):
                build_tau3_promotion_preflight(**paths, out=root / "bad.json")

            paths = make_fixture(root / "private", promoted=True)
            readiness = _read_json(paths["readiness_validation"])
            readiness["endpoint_url"] = "http://127.0.0.1:18080/v1"
            _write_json(paths["readiness_validation"], readiness)
            with self.assertRaisesRegex(Tau3PromotionPreflightError, "forbidden sealed/private material"):
                build_tau3_promotion_preflight(**paths, out=root / "bad-private.json")

    def test_output_is_create_once_and_cli_replays(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = make_fixture(root, promoted=True)
            out = root / "preflight.json"
            out.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(Tau3PromotionPreflightError, "already exists"):
                build_tau3_promotion_preflight(**paths, out=out)

            cli_out = root / "cli.json"
            cmd = [sys.executable, str(SCRIPT), "--out", str(cli_out)]
            for arg, key in (
                ("--sealed-public-evaluation-report", "sealed_public_evaluation_report"),
                ("--sealed-grid-completeness", "sealed_grid_completeness"),
                ("--sealed-authorization", "sealed_authorization"),
                ("--candidate-lock", "candidate_lock"),
                ("--postlock-attempt-ledger", "postlock_attempt_ledger"),
                ("--protocol-lineage-attestation", "protocol_lineage_attestation"),
                ("--readiness-validation", "readiness_validation"),
                ("--budget-evidence", "budget_evidence"),
                ("--license-evidence", "license_evidence"),
                ("--contamination-evidence", "contamination_evidence"),
                ("--redaction-evidence", "redaction_evidence"),
            ):
                cmd.extend([arg, str(paths[key])])

            completed = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=False)

            self.assertEqual(completed.returncode, 0, completed.stderr + completed.stdout)
            self.assertTrue(_read_json(cli_out)["allowed"])
            self.assertIsNone(_read_json(cli_out)["hf_revision"])
            self.assertEqual(cli_out.stat().st_mode & 0o777, 0o444)

    def test_post_publication_record_requires_allowed_preflight_and_hf_revision(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = make_fixture(root, promoted=True)
            preflight = root / "preflight.json"
            build_tau3_promotion_preflight(**paths, out=preflight, created_at="2026-07-24T00:00:00+00:00")

            record = build_tau3_post_publication_record(
                preflight=preflight,
                hf_revision="2" * 40,
                out=root / "post-publication.json",
                created_at="2026-07-24T01:00:00+00:00",
            )

            self.assertEqual(record["schema_version"], TAU3_POST_PUBLICATION_RECORD_SCHEMA_VERSION)
            self.assertEqual(record["status"], "published")
            self.assertEqual(record["huggingface"]["revision"], "2" * 40)
            self.assertTrue(check_schema_contract(record, name_or_id="tau3_post_publication_record")["passed"])

            negative_paths = make_fixture(root / "negative", promoted=False)
            negative_preflight = root / "negative-preflight.json"
            build_tau3_promotion_preflight(**negative_paths, out=negative_preflight)
            with self.assertRaisesRegex(Tau3PromotionPreflightError, "requires an allowed preflight"):
                build_tau3_post_publication_record(preflight=negative_preflight, hf_revision="2" * 40, out=root / "negative-post.json")

            with self.assertRaisesRegex(Tau3PromotionPreflightError, "hf_revision"):
                build_tau3_post_publication_record(preflight=preflight, hf_revision="not-a-revision", out=root / "bad-post.json")

    def test_post_publication_cli_is_create_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = make_fixture(root, promoted=True)
            preflight = root / "preflight.json"
            build_tau3_promotion_preflight(**paths, out=preflight)
            out = root / "post.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(POST_SCRIPT),
                    "--preflight",
                    str(preflight),
                    "--hf-revision",
                    "3" * 40,
                    "--out",
                    str(out),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr + completed.stdout)
            self.assertEqual(_read_json(out)["huggingface"]["revision"], "3" * 40)
            self.assertEqual(out.stat().st_mode & 0o777, 0o444)
            with self.assertRaisesRegex(Tau3PromotionPreflightError, "already exists"):
                build_tau3_post_publication_record(preflight=preflight, hf_revision="3" * 40, out=out)

    def test_schema_is_registered(self) -> None:
        names = {record["name"] for record in list_schema_records()}
        self.assertIn("tau3_promotion_publication_preflight", names)
        self.assertIn("tau3_post_publication_record", names)


def make_fixture(root: Path, *, promoted: bool) -> dict[str, Path]:
    root.mkdir(parents=True, exist_ok=True)
    lock = candidate_lock()
    lock_path = root / "candidate_lock.json"
    _write_json(lock_path, lock)
    lock_sha = _file_sha(lock_path)

    evaluation_path = root / "evaluation.json"
    _write_json(evaluation_path, evaluation(promoted=promoted))
    authorization_path = root / "authorization.json"
    _write_json(authorization_path, authorization(lock_sha))
    grid_path = root / "sealed_grid.json"
    _write_json(grid_path, sealed_grid(lock_sha, _file_sha(authorization_path)))
    ledger_path = root / "ledger.json"
    _write_json(ledger_path, attempt_ledger(lock_sha))
    lineage_path = root / "lineage.json"
    _write_json(lineage_path, {"schema_version": "hfr.tau3_protocol_lineage_attestation.v1", "passed": True})
    readiness_path = root / "readiness.json"
    _write_json(readiness_path, {"schema_version": "hfr.validation.v1", "passed": True, "readiness": "ready_for_publication_review"})
    budget_path = root / "budget.json"
    _write_json(budget_path, {"schema_version": "hfr.tau3_budget_evidence.v1", "passed": True})
    license_path = root / "license.json"
    _write_json(license_path, {"schema_version": "hfr.tau3_license_evidence.v1", "passed": True, "status": "approved"})
    contamination_path = root / "contamination.json"
    _write_json(contamination_path, {"schema_version": "hfr.tau3_contamination_evidence.v1", "passed": True, "unresolved_leakage": False, "leakage_found": False})
    redaction_path = root / "redaction.json"
    _write_json(redaction_path, {"schema_version": "hfr.tau3_redaction_evidence.v1", "passed": True, "secrets_found": False, "unredacted_sensitive_data": False})
    return {
        "sealed_public_evaluation_report": evaluation_path,
        "sealed_grid_completeness": grid_path,
        "sealed_authorization": authorization_path,
        "candidate_lock": lock_path,
        "postlock_attempt_ledger": ledger_path,
        "protocol_lineage_attestation": lineage_path,
        "readiness_validation": readiness_path,
        "budget_evidence": budget_path,
        "license_evidence": license_path,
        "contamination_evidence": contamination_path,
        "redaction_evidence": redaction_path,
    }


def upgrade_fixture_to_v2(paths: dict[str, Path]) -> None:
    training_protocol_sha256 = _sha("8")
    benchmark_protocol_sha256 = _sha("a")
    lineage = {
        "schema_version": "hfr.tau3_benchmark_protocol_lineage.v1",
        "created_at": "2026-07-30T00:00:00Z",
        "passed": True,
        "training_protocol_sha256": training_protocol_sha256,
        "benchmark_protocol_sha256": benchmark_protocol_sha256,
        "frozen_fields_sha256": _sha("b"),
        "allowed_delta": {
            "paths": [
                "sealed_manifest",
                "split_manifest.source_manifest",
                "split_manifest.splits.sealed",
                "tau_revision.split_hashes.sealed",
            ],
            "change_count": 4,
            "changes_sha256": _sha("c"),
        },
        "fresh_bindings": {
            "blind_custody_receipt_sha256": _sha("d"),
            "sealed_source_manifest_sha256": _sha("a"),
            "fresh_contamination_replay_sha256": _sha("e"),
            "retired_source_incident_sha256": _sha("f"),
        },
        "gates": {
            "training_protocol_schema_passed": True,
            "benchmark_protocol_schema_passed": True,
            "exact_sealed_only_delta": True,
            "fresh_source_bound_everywhere": True,
            "custody_receipt_replayed": True,
            "fresh_contamination_replay_passed": True,
            "retired_source_not_reused": True,
        },
        "hashes_only": True,
        "local_paths_included": False,
        "raw_payload_included": False,
    }
    lineage["lineage_sha256"] = _canonical_sha256(lineage)
    _write_json(paths["protocol_lineage_attestation"], lineage)
    lineage_sha256 = _file_sha(paths["protocol_lineage_attestation"])

    lock = _read_json(paths["candidate_lock"])
    lock.update(
        {
            "schema_version": "hfr.tau3_candidate_lock.v2",
            "evaluator_model_contract_sha256": _sha("0"),
            "training_protocol_sha256": training_protocol_sha256,
            "training_protocol_signature": _sha("9"),
            "benchmark_protocol_sha256": benchmark_protocol_sha256,
            "benchmark_protocol_signature": _sha("b"),
            "benchmark_protocol_lineage_sha256": lineage_sha256,
        }
    )
    lock.pop("protocol_sha256")
    lock.pop("protocol_signature")
    _write_json(paths["candidate_lock"], lock)
    lock_sha256 = _file_sha(paths["candidate_lock"])

    authorization_payload = _read_json(paths["sealed_authorization"])
    authorization_payload.update(
        {
            "schema_version": "hfr.tau3_sealed_authorization.v2",
            "candidate_lock": {
                "sha256": lock_sha256,
                "created_at": lock["created_at"],
                "training_protocol_sha256": training_protocol_sha256,
                "training_protocol_signature": _sha("9"),
                "benchmark_protocol_sha256": benchmark_protocol_sha256,
                "benchmark_protocol_signature": _sha("b"),
                "benchmark_protocol_lineage_sha256": lineage_sha256,
                "sealed_access_authorized": True,
            },
            "protocol": {
                "role": "fresh_sealed_benchmark",
                "sha256": benchmark_protocol_sha256,
                "signature_sha256": _sha("b"),
                "signature_provenance": (
                    "candidate_lock.benchmark_protocol_signature"
                ),
                "tau_revision": "1" * 40,
            },
            "protocol_lineage": {
                "sha256": lineage_sha256,
                "training_protocol_sha256": training_protocol_sha256,
                "benchmark_protocol_sha256": benchmark_protocol_sha256,
            },
            "blind_custody": {
                "receipt_sha256": _sha("d"),
                "generator_validation_sha256": _sha("c"),
                "fresh_contamination_replay_sha256": _sha("e"),
                "retired_source_incident_sha256": _sha("f"),
            },
            "qualification": {
                "candidate_selection_report_sha256": lock[
                    "development_selection_report_sha256"
                ],
                "qualified_training_evidence_sha256": _sha("2"),
                "qualified_candidate_count": 2,
                "selected_candidate_id_hash": lock[
                    "selected_candidate_id_hash"
                ],
            },
        }
    )
    authorization_payload["sealed_source"]["domain_counts"] = {
        "airline": 34,
        "retail": 33,
        "telecom": 33,
    }
    authorization_payload["gates"].update(
        {
            "fresh_protocol_lineage_replayed": True,
            "blind_custody_replayed": True,
            "qualified_training_cohort_replayed": True,
            "retired_source_not_reused": True,
            "fresh_domain_balance_passed": True,
        }
    )
    _write_json(paths["sealed_authorization"], authorization_payload)
    authorization_sha256 = _file_sha(paths["sealed_authorization"])

    grid = _read_json(paths["sealed_grid_completeness"])
    grid["bindings"].update(
        {
            "authorization_sha256": authorization_sha256,
            "candidate_lock_sha256": lock_sha256,
            "protocol_sha256": benchmark_protocol_sha256,
            "training_protocol_sha256": training_protocol_sha256,
            "benchmark_protocol_lineage_sha256": lineage_sha256,
            "blind_custody_receipt_sha256": _sha("d"),
            "candidate_selection_report_sha256": lock[
                "development_selection_report_sha256"
            ],
            "qualified_training_evidence_sha256": _sha("2"),
            "generator_validation_sha256": _sha("c"),
            "fresh_contamination_replay_sha256": _sha("e"),
            "retired_source_incident_sha256": _sha("f"),
        }
    )
    grid["gates"].update(
        {
            "fresh_protocol_lineage_binding_replayed": True,
            "blind_custody_binding_replayed": True,
            "qualified_training_binding_replayed": True,
        }
    )
    _write_json(paths["sealed_grid_completeness"], grid)

    ledger = _read_json(paths["postlock_attempt_ledger"])
    ledger["lock"]["sha256"] = lock_sha256
    _write_json(paths["postlock_attempt_ledger"], ledger)


def _canonical_sha256(value: Any) -> str:
    import hashlib

    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def candidate_lock() -> dict[str, Any]:
    return {
        "schema_version": "hfr.tau3_candidate_lock.v1",
        "created_at": "2026-07-23T00:00:00+00:00",
        "selected_candidate_id_hash": _sha("a"),
        "candidate_identity_sha256": _sha("b"),
        "development_selection_report_sha256": _sha("c"),
        "development_benchmark_manifest_sha256": _sha("d"),
        "training_receipt_sha256": _sha("e"),
        "endpoint_model_sha256": _sha("f"),
        "adapter_tree_sha256": _sha("1"),
        "recipe_sha256": _sha("2"),
        "base_identity_sha256": _sha("3"),
        "base_tree_sha256": _sha("4"),
        "dataset_manifest_sha256": _sha("5"),
        "dataset_files_sha256": _sha("6"),
        "source_binding_sha256": _sha("7"),
        "protocol_sha256": _sha("8"),
        "protocol_signature": _sha("9"),
        "hashes_only": True,
        "sealed_access_authorized": True,
        "local_paths_included": False,
        "raw_payload_included": False,
    }


def authorization(lock_sha: str) -> dict[str, Any]:
    return {
        "schema_version": "hfr.tau3_sealed_authorization.v1",
        "created_at": "2026-07-23T01:00:00+00:00",
        "authorized": True,
        "hashes_only": True,
        "local_paths_included": False,
        "raw_payload_included": False,
        "candidate_lock": {"sha256": lock_sha, "created_at": "2026-07-23T00:00:00+00:00", "protocol_sha256": _sha("8"), "protocol_signature": _sha("9"), "sealed_access_authorized": True},
        "protocol": {"sha256": _sha("8"), "signature_sha256": _sha("9"), "signature_provenance": "candidate_lock.protocol_signature", "tau_revision": "1" * 40},
        "sealed_source": {"manifest_sha256": _sha("a"), "task_count": 100, "hashes_only": True},
        "frozen_contract": {"arms": ["adapter", "base", "comparator_1", "comparator_2"], "seeds": [101, 202, 303, 404], "domains": ["airline", "retail", "telecom"], "context_window": 16384, "tool_contract_sha256": _sha("b"), "prompt_context_decoding_sha256": _sha("c"), "harness_sha256": _sha("d"), "no_test_time_search": True},
        "model_identity_refs": {"candidate_identity_sha256": _sha("b"), "adapter_tree_sha256": _sha("1"), "endpoint_model_sha256": _sha("f"), "base_identity_sha256": _sha("3"), "comparator_1_identity_sha256": _sha("4"), "comparator_2_identity_sha256": _sha("5"), "equivalence_refs_hash": _sha("6")},
        "gates": {
            "candidate_lock_valid": True,
            "chronology_lock_before_authorization": True,
            "protocol_binding_valid": True,
            "sealed_source_hash_only": True,
            "sealed_task_count_is_100": True,
            "seeds_exact": True,
            "arms_exact": True,
            "harness_tool_prompt_context_decoding_no_search_identical": True,
            "model_identity_equivalence_refs_present": True,
            "contamination_gate_passed": True,
            "redaction_gate_passed": True,
            "license_gate_passed": True,
            "safety_gate_passed": True,
            "budget_gate_passed": True,
            "public_artifact_contains_no_local_paths": True,
        },
        "budget": {"sha256": _sha("7"), "declared": True, "passed": True},
    }


def sealed_grid(lock_sha: str, auth_sha: str) -> dict[str, Any]:
    return {
        "schema_version": "hfr.tau3_sealed_grid_completeness.v1",
        "created_at": "2026-07-23T02:00:00+00:00",
        "passed": True,
        "status": "complete",
        "hashes_only": True,
        "local_paths_included": False,
        "raw_payload_included": False,
        "scores_included": False,
        "bindings": {
            "authorization_sha256": auth_sha,
            "candidate_lock_sha256": lock_sha,
            "protocol_sha256": _sha("8"),
            "sealed_source_sha256": _sha("a"),
            "coverage_fingerprint_sha256": _sha("b"),
            "arm_manifest_sha256": {"adapter": _sha("c"), "base": _sha("d"), "comparator_1": _sha("e"), "comparator_2": _sha("f")},
            "model_identity_sha256": {"adapter": _sha("1"), "base": _sha("2"), "comparator_1": _sha("3"), "comparator_2": _sha("4")},
            "harness_equivalence_sha256": _sha("5"),
        },
        "counts": {"arm_count": 4, "seed_count": 4, "domain_count": 3, "sealed_task_count": 100, "episodes_per_arm": 400, "total_episodes": 1600, "per_arm_seed_task_count": {arm: {"101": 100, "202": 100, "303": 100, "404": 100} for arm in ("adapter", "base", "comparator_1", "comparator_2")}},
        "gates": {
            "arms_exact": True,
            "seeds_exact": True,
            "domains_exact": True,
            "sealed_task_count_exact": True,
            "no_duplicate_task_rows": True,
            "no_missing_task_rows": True,
            "no_extra_task_rows": True,
            "same_task_coverage_across_arms": True,
            "same_task_coverage_across_seeds": True,
            "result_hashes_replayed": True,
            "authorization_binding_replayed": True,
            "candidate_lock_binding_replayed": True,
            "protocol_binding_replayed": True,
            "sealed_source_binding_replayed": True,
            "harness_equivalence_bound": True,
            "model_identities_bound": True,
            "public_payload_safe": True,
        },
    }


def attempt_ledger(lock_sha: str) -> dict[str, Any]:
    return {
        "schema_version": "hfr.tau3_candidate_attempt_ledger.v1",
        "schema_checked": True,
        "created_at": "2026-07-23T00:30:00+00:00",
        "campaign": {"root_ref": "candidate_attempts", "campaign_marker_sha256": _sha("a")},
        "lock": {"created_at": "2026-07-23T00:00:00+00:00", "sha256": lock_sha},
        "attempt_count": 0,
        "status_counts": {"completed": 0, "failed": 0, "timeout": 0, "interrupted": 0, "missing-receipt": 0, "malformed-receipt": 0},
        "successful_attempt_count": 0,
        "failed_attempt_count": 0,
        "attempts": [],
    }


def evaluation(*, promoted: bool) -> dict[str, Any]:
    macro = {"adapter": 0.9 if promoted else 0.4, "base": 0.5, "comparator_1": 0.7, "comparator_2": 0.8}
    effects = {
        arm: {
            "paired_pass1": effect(0.4 if arm == "base" else 0.1),
            "domain_stratified_macro_pass1": {**effect(0.4 if arm == "base" else 0.1), "resampling_unit": "domain_stratified_task", "domain_means": {"airline": 0.1, "retail": 0.1, "telecom": 0.1}},
            "per_domain_pass1": {"airline": effect(0.1), "retail": effect(0.1), "telecom": effect(0.1)},
            "primary_improvement_passed": promoted,
            "per_domain_non_inferiority_passed": True,
        }
        for arm in ("base", "comparator_1", "comparator_2")
    }
    checks = [
        {"id": "source_results_valid", "passed": True, "details": {}},
        {"id": "identical_harness", "passed": True, "details": {}},
        {"id": "unique_paired_results", "passed": True, "details": {}},
        {"id": "safety_non_inferiority_vs_base", "passed": True, "details": {}},
    ]
    return {
        "schema_version": "hfr.tau3_evaluation.v1",
        "created_at": "2026-07-23T03:00:00+00:00",
        "mode": "sealed",
        "passed": promoted,
        "promotion_ready": promoted,
        "readiness": "ready_for_publication_review" if promoted else "blocked",
        "analysis_config": {"required_arms": ["adapter", "base", "comparator_1", "comparator_2"], "reference_arms": ["base", "comparator_1", "comparator_2"], "required_domains": ["airline", "retail", "telecom"], "primary_metric": "macro_pass1", "bootstrap_samples": 200, "bootstrap_seed": 7, "confidence_level": 0.95, "non_inferiority_margin": 0.03, "safety_non_inferiority_margin": 0.01, "harness_equivalence": "normalized_exact_excluding_local_ports_api_keys_and_agent_model"},
        "tau_revision": "1" * 40,
        "harness": {"passed": True, "normalized_sha256": _sha("a"), "arm_sha256": {}, "normalized_by_domain": {}},
        "source_artifacts": {arm: [{"path": f"public/{arm}.json", "sha256": _sha("b"), "public_safe": True}] for arm in ("adapter", "base", "comparator_1", "comparator_2")},
        "pairing": {"passed": True, "key_fields": ["domain", "task_sha256", "trial", "seed"], "paired_count": 12, "domain_counts": {"airline": 4, "retail": 4, "telecom": 4}, "pair_set_sha256": _sha("c")},
        "metrics": {"macro_pass1": macro, "per_domain_pass1": {arm: {"airline": macro[arm], "retail": macro[arm], "telecom": macro[arm]} for arm in macro}, "safety": {"provable": True, "definition": "policy and harmful mutation rates", "missing_db_evidence_counts": {arm: 0 for arm in macro}, "missing_policy_review_counts": {arm: 0 for arm in macro}, "harmful_mutation_counts": {arm: 0 for arm in macro}, "harmful_mutation_rates": {arm: 0.0 for arm in macro}, "policy_violation_counts": {arm: 0 for arm in macro}, "policy_violation_rates": {arm: 0.0 for arm in macro}, "blocking_reasons": []}},
        "effects": effects if promoted else {},
        "per_task_hashed": [],
        "checks": checks,
        "failed_check_count": 0 if promoted else 1,
        "blocking_reasons": [] if promoted else ["primary_macro_improvement_vs_base"],
        "public_payload_scan": {"passed": True, "forbidden_key_count": 12, "forbidden_text_pattern_count": 4, "report_sha256": _sha("d")},
    }


def effect(diff: float) -> dict[str, Any]:
    return {
        "mean_difference": diff,
        "confidence_level": 0.95,
        "confidence_interval": {"lower": diff, "upper": diff},
        "bootstrap_samples": 200,
        "bootstrap_seed": 7,
    }
