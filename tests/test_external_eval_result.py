import copy
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path

from flightrecorder.external_eval import (
    build_external_eval_plan,
    write_external_eval_plan,
)
from flightrecorder.external_eval_result import (
    EXTERNAL_EVAL_RESULT_SCHEMA_VERSION,
    ExternalEvalResultError,
    build_external_eval_result,
    external_eval_result_digest,
    write_external_eval_result,
)
from flightrecorder.heldout_manifest import (
    build_heldout_manifest,
    write_heldout_manifest,
)
from flightrecorder.schema_registry import _validate_value
from flightrecorder.validation import validate_external_eval_result


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = (
    ROOT / "flightrecorder" / "schemas" / "external_eval_result.v1.schema.json"
)


class ExternalEvalResultTests(unittest.TestCase):
    def test_completed_local_mock_import_is_exact_and_keeps_benchmark_failure_separate(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan, heldout = _write_ready_inputs(root, ["case-a", "case-b"])
            raw = _write_json(
                root / "suite_summary.json",
                {
                    "schema_version": "hfr.run_suite.v1",
                    "total": 2,
                    "passed": 1,
                    "failed": 1,
                    "error_count": 0,
                    "errors": [],
                    "metrics": {},
                    "scenarios_dir": "scenarios",
                    "out_dir": "runs",
                    "artifacts": {},
                    "runs": [
                        {"scenario_id": "case-a", "passed": True, "score": 100},
                        {"scenario_id": "case-b", "passed": False, "score": 25},
                    ],
                },
            )
            metadata = _write_runner_metadata(root / "runner.json", exit_code=0)
            out = root / "external_eval_result.json"

            result = build_external_eval_result(
                plan_path=plan,
                heldout_manifest_path=heldout,
                raw_result_path=raw,
                runner_metadata_path=metadata,
                adapter_id="local_mock",
                execution_id="eval-001",
                model_id="local/mock-candidate",
                normalizer_id="hfr.local_mock.run_suite",
                normalizer_version="1",
                raw_format="hfr.run_suite.v1",
                execution_status="completed",
                out_path=out,
                created_at="2026-07-10T00:00:00+00:00",
            )

            self.assertEqual(
                result["schema_version"], EXTERNAL_EVAL_RESULT_SCHEMA_VERSION
            )
            self.assertTrue(result["integrity"]["passed"], result["integrity"])
            self.assertEqual(result["execution"]["status"], "completed")
            self.assertEqual(result["benchmark_outcome"]["status"], "failed")
            self.assertTrue(result["coverage"]["complete"])
            self.assertEqual(
                result["coverage"]["expected_case_ids"], ["case-a", "case-b"]
            )
            self.assertEqual(
                result["coverage"]["observed_case_ids"], ["case-a", "case-b"]
            )
            self.assertEqual(
                result["coverage"]["mapped_case_ids"], ["case-a", "case-b"]
            )
            self.assertEqual(result["coverage"]["missing_case_ids"], [])
            self.assertEqual(result["coverage"]["unexpected_case_ids"], [])
            self.assertEqual(result["coverage"]["unmapped_case_ids"], [])
            self.assertEqual(result["coverage"]["duplicate_case_ids"], [])
            self.assertEqual(result["coverage"]["unmapped_record_indexes"], [])
            self.assertEqual(
                [case["raw_record_index"] for case in result["cases"]], [0, 1]
            )
            for case in result["cases"]:
                self.assertRegex(case["raw_record_sha256"], r"^[0-9a-f]{64}$")
                self.assertNotIn("raw_record", case)
            self.assertEqual(result["governance"]["readiness"], "blocked")
            self.assertFalse(result["governance"]["external_eval_claims_allowed"])
            self.assertIn(
                "external evaluation benchmark outcome failed",
                result["governance"]["blocking_reasons"],
            )
            self.assertFalse(
                result["execution_boundary"]["benchmark_started_by_flight_recorder"]
            )
            self.assertTrue(result["execution_boundary"]["import_only"])
            self.assertEqual(result["identity"]["plan_sha256"], _sha256(plan))
            self.assertEqual(
                result["identity"]["heldout_manifest_sha256"], _sha256(heldout)
            )
            self.assertEqual(result["identity"]["raw_result_sha256"], _sha256(raw))
            self.assertEqual(
                result["identity"]["runner_metadata_sha256"], _sha256(metadata)
            )
            self.assertRegex(result["identity"]["digest_sha256"], r"^[0-9a-f]{64}$")
            self.assertEqual(
                result["identity"]["digest_sha256"],
                external_eval_result_digest(result),
            )
            tampered = copy.deepcopy(result)
            tampered["cases"][0]["score"] = 99
            self.assertNotEqual(
                result["identity"]["digest_sha256"],
                external_eval_result_digest(tampered),
            )

            write_external_eval_result(result, out)
            written = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(written, result)
            self.assertNotIn(str(root), out.read_text(encoding="utf-8"))
            _assert_schema_valid(self, written)

    def test_completed_import_fails_closed_on_incomplete_mapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan, heldout = _write_ready_inputs(root, ["case-a", "case-b"])
            raw = _write_suite(
                root / "suite_summary.json",
                [{"scenario_id": "case-a", "passed": True, "score": 100}],
            )

            result = _build(root, plan, heldout, raw, execution_status="completed")

            self.assertFalse(result["integrity"]["passed"])
            self.assertFalse(result["coverage"]["complete"])
            self.assertEqual(result["coverage"]["missing_case_ids"], ["case-b"])
            self.assertIn("completed_coverage_exact", _failed_checks(result))
            self.assertEqual(result["governance"]["readiness"], "blocked")
            self.assertFalse(result["governance"]["external_eval_claims_allowed"])
            _assert_schema_valid(self, result)

    def test_completed_import_rejects_aggregate_only_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan, heldout = _write_ready_inputs(root, ["case-a"])
            raw = _write_json(
                root / "lm_eval_results.json", {"results": {"task": {"acc": 1.0}}}
            )

            result = _build(
                root,
                plan,
                heldout,
                raw,
                execution_status="completed",
                normalizer_id="hfr.lm_eval.samples",
                raw_format="aggregate_json",
            )

            self.assertFalse(result["integrity"]["passed"])
            self.assertTrue(result["normalizer"]["aggregate_only"])
            self.assertEqual(result["normalizer"]["raw_record_count"], 0)
            self.assertEqual(result["benchmark_outcome"]["status"], "not_available")
            self.assertIn("completed_has_per_case_records", _failed_checks(result))
            self.assertEqual(result["governance"]["readiness"], "blocked")

    def test_completed_import_rejects_non_allowlisted_normalizer_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan, heldout = _write_ready_inputs(root, ["case-a"])
            raw = _write_suite(
                root / "suite_summary.json",
                [{"scenario_id": "case-a", "passed": True, "score": 100}],
            )

            result = _build(
                root,
                plan,
                heldout,
                raw,
                execution_status="completed",
                normalizer_id="hfr.local_mock.run_suite.v2",
            )

            self.assertFalse(result["integrity"]["passed"])
            self.assertIn("normalizer_contract_supported", _failed_checks(result))
            self.assertEqual(result["governance"]["readiness"], "blocked")

    def test_completed_local_mock_rejects_inconsistent_suite_aggregates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan, heldout = _write_ready_inputs(root, ["case-a"])
            raw = _write_suite(
                root / "suite_summary.json",
                [{"scenario_id": "case-a", "passed": True, "score": 100}],
            )
            payload = json.loads(raw.read_text(encoding="utf-8"))
            payload["passed"] = 0
            _write_json(raw, payload)

            result = _build(root, plan, heldout, raw, execution_status="completed")

            self.assertFalse(result["integrity"]["passed"])
            self.assertIn("completed_raw_result_readable", _failed_checks(result))
            self.assertIn(
                "passed count does not match",
                " ".join(result["normalizer"]["errors"]),
            )

    def test_duplicate_unexpected_and_unmapped_ids_are_reported_exactly(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan, heldout = _write_ready_inputs(root, ["case-a", "case-b"])
            raw = _write_suite(
                root / "suite_summary.json",
                [
                    {"scenario_id": "case-a", "passed": True, "score": 100},
                    {"scenario_id": "case-a", "passed": True, "score": 100},
                    {"scenario_id": "case-extra", "passed": True, "score": 100},
                    {"scenario_id": "case-b", "score": 50},
                    {"passed": True, "score": 100},
                ],
            )

            result = _build(root, plan, heldout, raw, execution_status="completed")

            coverage = result["coverage"]
            self.assertEqual(coverage["duplicate_case_ids"], ["case-a"])
            self.assertEqual(coverage["unexpected_case_ids"], ["case-extra"])
            self.assertEqual(coverage["unmapped_case_ids"], ["case-b"])
            self.assertEqual(coverage["missing_case_ids"], ["case-b"])
            self.assertEqual(coverage["unmapped_record_indexes"], [3, 4])
            self.assertFalse(coverage["complete"])
            self.assertFalse(result["integrity"]["passed"])

    def test_terminal_runner_failure_can_have_valid_integrity_without_claim_readiness(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan, heldout = _write_ready_inputs(root, ["case-a"])
            raw = _write_json(
                root / "runner_failure.json",
                {"error": "runner exited before samples were emitted"},
            )
            metadata = _write_runner_metadata(root / "runner.json", exit_code=17)

            result = build_external_eval_result(
                plan_path=plan,
                heldout_manifest_path=heldout,
                raw_result_path=raw,
                runner_metadata_path=metadata,
                adapter_id="local_mock",
                execution_id="eval-001",
                model_id="local/mock-candidate",
                normalizer_id="hfr.local_mock.aggregate_json",
                normalizer_version="1",
                raw_format="aggregate_json",
                execution_status="failed",
                failure_class="runner_error",
                failure_message="runner exited with status 17",
                out_path=root / "external_eval_result.json",
            )

            self.assertTrue(result["integrity"]["passed"], result["integrity"])
            self.assertEqual(result["execution"]["status"], "failed")
            self.assertEqual(result["benchmark_outcome"]["status"], "not_available")
            self.assertFalse(result["coverage"]["complete"])
            self.assertEqual(result["governance"]["readiness"], "blocked")
            self.assertFalse(result["governance"]["external_eval_claims_allowed"])
            _assert_schema_valid(self, result)

    def test_inline_runner_observation_is_digest_bound_when_metadata_file_is_absent(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan, heldout = _write_ready_inputs(root, ["case-a"])
            raw = _write_suite(
                root / "suite_summary.json",
                [{"scenario_id": "case-a", "passed": True, "score": 100}],
            )

            result = build_external_eval_result(
                plan_path=plan,
                heldout_manifest_path=heldout,
                raw_result_path=raw,
                runner_metadata_path=None,
                runner_observation=_runner_observation(exit_code=0),
                adapter_id="local_mock",
                execution_id="eval-001",
                model_id="local/mock-candidate",
                normalizer_id="hfr.local_mock.run_suite",
                normalizer_version="1",
                raw_format="hfr.run_suite.v1",
                execution_status="completed",
                out_path=root / "external_eval_result.json",
            )

            self.assertTrue(result["integrity"]["passed"], result["integrity"])
            self.assertIsNone(result["sources"]["runner_metadata"]["path"])
            self.assertIsNone(result["identity"]["runner_metadata_sha256"])
            self.assertRegex(
                result["identity"]["runner_observation_sha256"], r"^[0-9a-f]{64}$"
            )
            self.assertEqual(result["governance"]["readiness"], "ready_for_review")
            self.assertTrue(result["governance"]["external_eval_claims_allowed"])

    def test_claims_require_explicit_no_credential_recording_attestation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan, heldout = _write_ready_inputs(root, ["case-a"])
            raw = _write_suite(
                root / "suite_summary.json",
                [{"scenario_id": "case-a", "passed": True, "score": 100}],
            )
            observation = _runner_observation(exit_code=0)
            observation["side_effects"].pop("credential_values_recorded")

            result = build_external_eval_result(
                plan_path=plan,
                heldout_manifest_path=heldout,
                raw_result_path=raw,
                runner_metadata_path=None,
                runner_observation=observation,
                adapter_id="local_mock",
                execution_id="eval-001",
                model_id="local/mock-candidate",
                normalizer_id="hfr.local_mock.run_suite",
                normalizer_version="1",
                raw_format="hfr.run_suite.v1",
                execution_status="completed",
                out_path=root / "external_eval_result.json",
            )

            self.assertTrue(result["integrity"]["passed"], result["integrity"])
            self.assertEqual(result["benchmark_outcome"]["status"], "passed")
            self.assertEqual(result["governance"]["readiness"], "blocked")
            self.assertFalse(result["governance"]["external_eval_claims_allowed"])
            self.assertIn(
                "runner did not explicitly attest that credential values were not recorded",
                result["governance"]["blocking_reasons"],
            )

    def test_runner_metadata_extensions_replay_deterministically(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan, heldout = _write_ready_inputs(root, ["case-a"])
            raw = _write_suite(
                root / "suite_summary.json",
                [{"scenario_id": "case-a", "passed": True, "score": 100}],
            )
            metadata = _write_runner_metadata(root / "runner.json", exit_code=0)
            metadata_payload = json.loads(metadata.read_text(encoding="utf-8"))
            metadata_payload["runner_observation"]["ignored_public_extension"] = "ok"
            _write_json(metadata, metadata_payload)
            out = root / "external_eval_result.json"

            result = build_external_eval_result(
                plan_path=plan,
                heldout_manifest_path=heldout,
                raw_result_path=raw,
                runner_metadata_path=metadata,
                adapter_id="local_mock",
                execution_id="eval-001",
                model_id="local/mock-candidate",
                normalizer_id="hfr.local_mock.run_suite",
                normalizer_version="1",
                raw_format="hfr.run_suite.v1",
                execution_status="completed",
                out_path=out,
            )
            write_external_eval_result(result, out)

            self.assertEqual(validate_external_eval_result(out).errors, [])

    def test_semantically_invalid_upstream_sources_block_claims(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan, heldout = _write_ready_inputs(root, ["case-a"])
            raw = _write_suite(
                root / "suite_summary.json",
                [{"scenario_id": "case-a", "passed": True, "score": 100}],
            )
            plan_payload = json.loads(plan.read_text(encoding="utf-8"))
            plan_payload["adapter_count"] = 999
            _write_json(plan, plan_payload)

            invalid_plan_result = _build(
                root, plan, heldout, raw, execution_status="completed"
            )
            self.assertFalse(invalid_plan_result["integrity"]["passed"])
            self.assertIn("plan_semantically_valid", _failed_checks(invalid_plan_result))

            plan, heldout = _write_ready_inputs(root, ["case-a"])
            heldout_payload = json.loads(heldout.read_text(encoding="utf-8"))
            heldout_payload["source_count"] = 999
            _write_json(heldout, heldout_payload)
            invalid_heldout_result = _build(
                root, plan, heldout, raw, execution_status="completed"
            )
            self.assertFalse(invalid_heldout_result["integrity"]["passed"])
            self.assertIn("heldout_manifest_ready", _failed_checks(invalid_heldout_result))
            self.assertFalse(
                invalid_heldout_result["governance"]["external_eval_claims_allowed"]
            )

    def test_public_identifiers_and_metric_names_reject_path_shaped_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan, heldout = _write_ready_inputs(root, ["case-a"])
            raw = _write_suite(
                root / "suite_summary.json",
                [
                    {
                        "scenario_id": "case-a",
                        "passed": True,
                        "score": 100,
                        "metrics": {
                            "/Users/alice/private.txt": 1,
                            "C:/Users/alice/private-metric": 1,
                            "file:/Users/alice/private-metric": 1,
                        },
                    }
                ],
            )

            result = _build(root, plan, heldout, raw, execution_status="completed")

            self.assertFalse(result["integrity"]["passed"])
            rendered = json.dumps(result, sort_keys=True)
            self.assertNotIn("/Users/alice", rendered)
            self.assertNotIn("C:/Users", rendered)
            self.assertNotIn("file:/Users", rendered)
            self.assertIn("non-public metric name", rendered)
            unsafe_case_raw = _write_suite(
                root / "unsafe_case.json",
                [
                    {
                        "scenario_id": "/Users/alice/private-case",
                        "passed": True,
                        "score": 100,
                    }
                ],
            )
            unsafe_case_result = _build(
                root,
                plan,
                heldout,
                unsafe_case_raw,
                execution_status="completed",
            )
            self.assertFalse(unsafe_case_result["integrity"]["passed"])
            self.assertNotIn(
                "/Users/alice", json.dumps(unsafe_case_result, sort_keys=True)
            )
            for private_model in ("/Users/alice/private/model", "../private/model"):
                with self.subTest(private_model=private_model):
                    with self.assertRaisesRegex(
                        ExternalEvalResultError, "public identifiers"
                    ):
                        build_external_eval_result(
                            plan_path=plan,
                            heldout_manifest_path=heldout,
                            raw_result_path=raw,
                            runner_metadata_path=None,
                            runner_observation=_runner_observation(exit_code=0),
                            adapter_id="local_mock",
                            execution_id="eval-001",
                            model_id=private_model,
                            normalizer_id="hfr.local_mock.run_suite",
                            normalizer_version="1",
                            raw_format="hfr.run_suite.v1",
                            execution_status="completed",
                            out_path=root / "result.json",
                        )

    def test_output_must_not_alias_any_input_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan, heldout = _write_ready_inputs(root, ["case-a"])
            raw = _write_suite(
                root / "suite_summary.json",
                [{"scenario_id": "case-a", "passed": True, "score": 100}],
            )
            metadata = _write_runner_metadata(root / "runner.json", exit_code=0)

            with self.assertRaisesRegex(ExternalEvalResultError, "must not alias"):
                build_external_eval_result(
                    plan_path=plan,
                    heldout_manifest_path=heldout,
                    raw_result_path=raw,
                    runner_metadata_path=metadata,
                    adapter_id="local_mock",
                    execution_id="eval-001",
                    model_id="local/mock-candidate",
                    normalizer_id="hfr.local_mock.run_suite",
                    normalizer_version="1",
                    raw_format="hfr.run_suite.v1",
                    execution_status="completed",
                    out_path=raw,
                )

            hardlink = root / "raw-hardlink.json"
            os.link(raw, hardlink)
            with self.assertRaisesRegex(ExternalEvalResultError, "must not alias"):
                build_external_eval_result(
                    plan_path=plan,
                    heldout_manifest_path=heldout,
                    raw_result_path=raw,
                    runner_metadata_path=metadata,
                    adapter_id="local_mock",
                    execution_id="eval-001",
                    model_id="local/mock-candidate",
                    normalizer_id="hfr.local_mock.run_suite",
                    normalizer_version="1",
                    raw_format="hfr.run_suite.v1",
                    execution_status="completed",
                    out_path=hardlink,
                )

            safe_out = root / "safe-result.json"
            safe_result = build_external_eval_result(
                plan_path=plan,
                heldout_manifest_path=heldout,
                raw_result_path=raw,
                runner_metadata_path=metadata,
                adapter_id="local_mock",
                execution_id="eval-001",
                model_id="local/mock-candidate",
                normalizer_id="hfr.local_mock.run_suite",
                normalizer_version="1",
                raw_format="hfr.run_suite.v1",
                execution_status="completed",
                out_path=safe_out,
            )
            original_raw = raw.read_bytes()
            with self.assertRaisesRegex(
                ExternalEvalResultError, "exact output path used while building"
            ):
                write_external_eval_result(safe_result, raw)
            self.assertEqual(raw.read_bytes(), original_raw)

            sources = root / "sources"
            sources.mkdir()
            same_name_raw = _write_suite(
                sources / "same-name-result.json",
                [{"scenario_id": "case-a", "passed": True, "score": 100}],
            )
            intended = root / "same-name-result.json"
            same_name_result = build_external_eval_result(
                plan_path=plan,
                heldout_manifest_path=heldout,
                raw_result_path=same_name_raw,
                runner_metadata_path=metadata,
                adapter_id="local_mock",
                execution_id="eval-001",
                model_id="local/mock-candidate",
                normalizer_id="hfr.local_mock.run_suite",
                normalizer_version="1",
                raw_format="hfr.run_suite.v1",
                execution_status="completed",
                out_path=intended,
            )
            original_same_name_raw = same_name_raw.read_bytes()
            with self.assertRaisesRegex(
                ExternalEvalResultError, "exact output path used while building"
            ):
                write_external_eval_result(same_name_result, same_name_raw)
            self.assertEqual(same_name_raw.read_bytes(), original_same_name_raw)

    def test_url_endpoint_uses_separate_public_model_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _unused_plan, heldout = _write_ready_inputs(root, ["case-a"])
            plan = root / "url-endpoint-plan.json"
            plan_payload = build_external_eval_plan(
                adapters=["local_mock"],
                scenario_manifest=heldout,
                model_endpoint="https://provider.example/v1",
                model="org/public-model",
                allow_installed=True,
                output_base_dir=root,
            )
            write_external_eval_plan(plan_payload, plan)
            raw = _write_suite(
                root / "suite_summary.json",
                [{"scenario_id": "case-a", "passed": True, "score": 100}],
            )
            metadata = _write_json(
                root / "runner-url.json",
                {
                    "adapter_id": "local_mock",
                    "execution_id": "eval-url-001",
                    "model_id": "org/public-model",
                    "runner_observation": _runner_observation(exit_code=0),
                },
            )

            result = build_external_eval_result(
                plan_path=plan,
                heldout_manifest_path=heldout,
                raw_result_path=raw,
                runner_metadata_path=metadata,
                adapter_id="local_mock",
                execution_id="eval-url-001",
                model_id="org/public-model",
                normalizer_id="hfr.local_mock.run_suite",
                normalizer_version="1",
                raw_format="hfr.run_suite.v1",
                execution_status="completed",
                out_path=root / "url-result.json",
            )

            self.assertTrue(result["integrity"]["passed"], result["integrity"])
            self.assertTrue(result["governance"]["external_eval_claims_allowed"])

    def test_result_identity_must_match_declared_model_not_endpoint_alias(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _unused_plan, heldout = _write_ready_inputs(root, ["case-a"])
            plan = root / "declared-model-plan.json"
            write_external_eval_plan(
                build_external_eval_plan(
                    adapters=["local_mock"],
                    scenario_manifest=heldout,
                    model_endpoint="local/wrong-endpoint-identity",
                    model="org/actual-model",
                    allow_installed=True,
                    output_base_dir=root,
                ),
                plan,
            )
            raw = _write_suite(
                root / "suite_summary.json",
                [{"scenario_id": "case-a", "passed": True, "score": 100}],
            )
            metadata = _write_json(
                root / "runner.json",
                {
                    "adapter_id": "local_mock",
                    "execution_id": "eval-001",
                    "model_id": "local/wrong-endpoint-identity",
                    "runner_observation": _runner_observation(exit_code=0),
                },
            )

            result = build_external_eval_result(
                plan_path=plan,
                heldout_manifest_path=heldout,
                raw_result_path=raw,
                runner_metadata_path=metadata,
                adapter_id="local_mock",
                execution_id="eval-001",
                model_id="local/wrong-endpoint-identity",
                normalizer_id="hfr.local_mock.run_suite",
                normalizer_version="1",
                raw_format="hfr.run_suite.v1",
                execution_status="completed",
                out_path=root / "result.json",
            )

            self.assertFalse(result["integrity"]["passed"])
            self.assertIn("identity_is_public_and_plan_bound", _failed_checks(result))
            self.assertFalse(result["governance"]["external_eval_claims_allowed"])

    def test_rejects_symlinked_sources(self):
        if not hasattr(os, "symlink"):
            self.skipTest("symlinks are unavailable")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan, heldout = _write_ready_inputs(root, ["case-a"])
            raw = _write_suite(
                root / "suite_summary.json",
                [{"scenario_id": "case-a", "passed": True, "score": 100}],
            )
            linked = root / "linked_suite.json"
            linked.symlink_to(raw)

            with self.assertRaisesRegex(ExternalEvalResultError, "symlink"):
                _build(root, plan, heldout, linked, execution_status="completed")

    def test_rejects_private_identity_before_it_can_enter_public_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan, heldout = _write_ready_inputs(root, ["case-a"])
            raw = _write_suite(
                root / "suite_summary.json",
                [{"scenario_id": "case-a", "passed": True, "score": 100}],
            )

            with self.assertRaisesRegex(ExternalEvalResultError, "public identifiers"):
                build_external_eval_result(
                    plan_path=plan,
                    heldout_manifest_path=heldout,
                    raw_result_path=raw,
                    runner_metadata_path=None,
                    runner_observation=_runner_observation(exit_code=0),
                    adapter_id="local_mock",
                    execution_id="https://provider.example/jobs/private-id",
                    model_id="local/mock-candidate",
                    normalizer_id="hfr.local_mock.run_suite",
                    normalizer_version="1",
                    raw_format="hfr.run_suite.v1",
                    execution_status="completed",
                    out_path=root / "external_eval_result.json",
                )

    def test_invalid_json_duplicate_keys_and_non_finite_values_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan, heldout = _write_ready_inputs(root, ["case-a"])
            duplicate = root / "duplicate.json"
            duplicate.write_text(
                '{"schema_version":"hfr.run_suite.v1","runs":[],"runs":[]}',
                encoding="utf-8",
            )
            non_finite = root / "non_finite.json"
            non_finite.write_text(
                '[{"case_id":"case-a","passed":true,"score":NaN}]', encoding="utf-8"
            )

            duplicate_result = _build(
                root, plan, heldout, duplicate, execution_status="completed"
            )
            finite_result = _build(
                root,
                plan,
                heldout,
                non_finite,
                execution_status="completed",
                normalizer_id="hfr.external.per_case_json",
                raw_format="json",
            )

            self.assertFalse(duplicate_result["integrity"]["passed"])
            self.assertFalse(finite_result["integrity"]["passed"])
            self.assertIn(
                "duplicate JSON key", " ".join(duplicate_result["normalizer"]["errors"])
            )
            self.assertIn(
                "unsupported JSON value",
                " ".join(finite_result["normalizer"]["errors"]),
            )


def _build(
    root: Path,
    plan: Path,
    heldout: Path,
    raw: Path,
    *,
    execution_status: str,
    normalizer_id: str = "hfr.local_mock.run_suite",
    raw_format: str = "hfr.run_suite.v1",
):
    metadata = root / "runner.json"
    if not metadata.exists():
        _write_runner_metadata(
            metadata, exit_code=0 if execution_status == "completed" else 1
        )
    return build_external_eval_result(
        plan_path=plan,
        heldout_manifest_path=heldout,
        raw_result_path=raw,
        runner_metadata_path=metadata,
        adapter_id="local_mock",
        execution_id="eval-001",
        model_id="local/mock-candidate",
        normalizer_id=normalizer_id,
        normalizer_version="1",
        raw_format=raw_format,
        execution_status=execution_status,
        out_path=root / "external_eval_result.json",
    )


def _write_ready_inputs(root: Path, case_ids: list[str]) -> tuple[Path, Path]:
    heldout_source = _write_suite(
        root / "heldout_source_suite.json",
        [
            {"scenario_id": case_id, "passed": True, "score": 100}
            for case_id in case_ids
        ],
    )
    heldout = root / "heldout.json"
    write_heldout_manifest(
        build_heldout_manifest(suite_summary_specs=[f"source={heldout_source}"]),
        heldout,
    )
    plan = _write_json(
        root / "external_eval_plan.json",
        {
            "schema_version": "hfr.external_eval_adapters.v1",
            "generated_at": "2026-07-10T00:00:00+00:00",
            "ready": True,
            "adapter_count": 1,
            "ready_adapter_count": 1,
            "selected_adapters": ["local_mock"],
            "allow_installed": True,
            "inputs": {
                "scenario_manifest": {
                    "path": heldout.name,
                    "exists": True,
                    "sha256": _sha256(heldout),
                    "size_bytes": heldout.stat().st_size,
                    "schema_version": "hfr.heldout_scenario_manifest.v1",
                    "ready": True,
                    "scenario_count": len(case_ids),
                },
                "model_endpoint": "local/mock-candidate",
                "model": "local/mock-candidate",
                "tool_schema_set": None,
                "inspect_task_set": None,
                "lm_eval_task_list": [],
                "swe_bench_task_set": None,
                "sandbox_policy": None,
            },
            "adapters": [
                {
                    "id": "local_mock",
                    "name": "Local mock eval",
                    "full_name": "Flight Recorder local mock external eval",
                    "domain": "offline_agentic_tasks",
                    "suite_tags": ["agentic", "mock", "offline"],
                    "required_inputs": ["scenario_manifest", "model_endpoint", "model"],
                    "provided_inputs": ["scenario_manifest", "model_endpoint", "model"],
                    "dependency_status": {
                        "available": True,
                        "commands": {},
                        "imports": {},
                    },
                    "execution_contract": {
                        "requires_identical_heldout_scenarios": True,
                        "scenario_manifest_sha256": _sha256(heldout),
                        "boundary": "Import committed held-out fixtures only.",
                    },
                    "adapter_contract": {
                        "schema_version": "hfr.external_eval_adapter_contract.v1",
                        "adapter_id": "external_eval.local_mock.fail_closed.v1",
                        "external_adapter_id": "local_mock",
                        "receipt_types": [
                            "hfr.external_eval_adapters.v1",
                            "hfr.external_eval_receipt.v1",
                        ],
                        "dry_run_transport": "plan_and_receipt_only",
                        "live_benchmark_supported": False,
                        "provider_api_called_by_flight_recorder": False,
                        "model_downloads_started_by_flight_recorder": False,
                        "credential_values_recorded": False,
                        "cost_incurred_usd": 0,
                        "requires_identical_heldout_scenarios": True,
                        "requires_external_runner_receipt_for_live": True,
                        "requires_dependency_probe_before_live": True,
                        "requires_explicit_live_opt_in": True,
                    },
                    "ready": True,
                    "blocking_reasons": [],
                }
            ],
            "blocking_reasons": [],
            "governance_handoff": {
                "external_eval_claims_allowed": False,
                "requires_identical_heldout_scenarios": True,
                "recommendation": "ready to execute",
            },
        },
    )
    return plan, heldout


def _write_suite(path: Path, runs: list[dict]) -> Path:
    normalized_runs = []
    for index, run in enumerate(runs):
        scenario_id = str(run.get("scenario_id") or f"case-{index}")
        normalized_runs.append(
            {
                "scenario_id": scenario_id,
                "scenario_title": scenario_id,
                "task_family": "external_eval_test",
                "scenario_path": f"scenarios/{scenario_id}.json",
                "trace_path": f"traces/{scenario_id}.jsonl",
                "run_dir": f"runs/{scenario_id}",
                "report": f"runs/{scenario_id}/report.html",
                "report_sha256": "a" * 64,
                "report_size_bytes": 1,
                "scorecard": f"runs/{scenario_id}/scorecard.json",
                "scorecard_sha256": "b" * 64,
                "scorecard_size_bytes": 1,
                "run_digest": f"runs/{scenario_id}/run_digest.json",
                "run_digest_sha256": "c" * 64,
                "run_digest_size_bytes": 1,
                "lineage": f"runs/{scenario_id}/artifact_lineage.json",
                "lineage_sha256": "d" * 64,
                "lineage_size_bytes": 1,
                "failed_rules": [],
                "critical_failures": [],
                **run,
            }
        )
        scenario_bytes = (
            json.dumps(
                {
                    "id": scenario_id,
                    "policy": {},
                    "prompt": f"Complete the {scenario_id} external-eval task.",
                    "title": scenario_id,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
        scenario_path = path.parent / normalized_runs[-1]["scenario_path"]
        scenario_path.parent.mkdir(parents=True, exist_ok=True)
        scenario_path.write_bytes(scenario_bytes)
        if "scenario_sha256" not in run:
            normalized_runs[-1]["scenario_sha256"] = hashlib.sha256(scenario_bytes).hexdigest()
        if "scenario_id" not in run:
            normalized_runs[-1].pop("scenario_id")
    passed = sum(row.get("passed") is True for row in normalized_runs)
    failed = sum(row.get("passed") is False for row in normalized_runs)
    scores = [row["score"] for row in normalized_runs if isinstance(row.get("score"), int)]
    return _write_json(
        path,
        {
            "schema_version": "hfr.run_suite.v1",
            "total": len(normalized_runs),
            "passed": passed,
            "failed": failed,
            "error_count": 0,
            "errors": [],
            "metrics": {
                "pass_rate": passed / len(normalized_runs) if normalized_runs else 0,
                "average_score": sum(scores) / len(scores) if scores else 0,
                "min_score": min(scores) if scores else None,
                "max_score": max(scores) if scores else None,
                "failed_rule_counts": [],
                "critical_failure_counts": [],
                "task_families": [],
                "failed": failed,
                "passed": passed,
            },
            "scenarios_dir": "scenarios",
            "out_dir": "runs",
            "artifacts": {},
            "runs": normalized_runs,
        },
    )


def _write_runner_metadata(path: Path, *, exit_code: int) -> Path:
    return _write_json(
        path,
        {
            "adapter_id": "local_mock",
            "execution_id": "eval-001",
            "model_id": "local/mock-candidate",
            "runner_observation": _runner_observation(exit_code=exit_code),
        },
    )


def _runner_observation(*, exit_code: int) -> dict:
    return {
        "runner_id": "public-local-runner",
        "runner_version": "1",
        "started_at": "2026-07-10T00:00:00+00:00",
        "finished_at": "2026-07-10T00:01:00+00:00",
        "exit_code": exit_code,
        "cost": {"reported": True, "amount": 0, "currency": "USD"},
        "side_effects": {
            "network_access": "not_observed",
            "provider_api_calls": "not_observed",
            "model_downloads": "not_observed",
            "filesystem_writes": "observed",
            "credential_values_recorded": "not_observed",
        },
    }


def _write_json(path: Path, value: object) -> Path:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _failed_checks(result: dict) -> set[str]:
    return {
        check["id"] for check in result["integrity"]["checks"] if not check["passed"]
    }


def _assert_schema_valid(test: unittest.TestCase, result: dict) -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    errors: list[str] = []
    _validate_value(result, schema, "$", schema, errors)
    test.assertEqual(errors, [])
