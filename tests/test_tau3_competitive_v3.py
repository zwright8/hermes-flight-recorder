from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

import flightrecorder.tau3_competitive_dataset as competitive_dataset_module
import flightrecorder.tau3_competitive_v3 as competitive_v3_module
from flightrecorder.tau3_competitive_dataset import build_tau3_competitive_dataset, validate_tau3_competitive_dataset_bundle
from flightrecorder.tau3_competitive_v3 import (
    DATASET_SCHEMA_VERSION,
    DOMAINS,
    FINAL_SCHEMA_VERSION,
    PLAN_SCHEMA_VERSION,
    PUBLICATION_SCHEMA_VERSION,
    TRAINING_SCHEMA_VERSION,
    _Loaded,
    _Target,
    _validate_training_receipt,
    validate_tau3_competitive_v3_bundle,
)
from flightrecorder.tau3_exposure import build_tau3_exposure_ledger, validate_tau3_exposure_ledger
from flightrecorder.tau3_internal_validation import (
    _expected_run_binding,
    build_tau3_internal_validation,
)
from flightrecorder.tau3_objective_validity import build_tau3_objective_validity_report, validate_tau3_objective_validity_report
from flightrecorder.tau3_promotion_preflight import build_tau3_post_publication_record
from flightrecorder.tau3_behavior_probes import REQUIRED_FAMILIES
from tests.test_tau3_competitive_dataset import (
    _FakeTokenizer,
    _grounded_validation_patch,
    _install_fake_transformers,
    _write_contamination_report,
    _write_grounded_bundle,
    _write_source_dataset,
    _write_tokenizer_config,
)
from tests.test_tau3_exposure import _eligible_rows, _write_jsonl as write_exposure_jsonl
from tests.test_tau3_objective_validity import _fixture as objective_fixture
from tests.test_tau3_prefix_equivalence import equivalence_fixture


class Tau3CompetitiveV3ValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        tokenizer_patcher = _install_fake_transformers(_FakeTokenizer())
        tokenizer_patcher.start()
        self.addCleanup(tokenizer_patcher.stop)
        symlink_patcher = mock.patch.object(competitive_dataset_module, "path_has_symlink_component", return_value=False)
        symlink_patcher.start()
        self.addCleanup(symlink_patcher.stop)
        grounded_patcher = _grounded_validation_patch()
        grounded_patcher.start()
        self.addCleanup(grounded_patcher.stop)

    def test_qualified_segmented_receipt_replays_process_chain(self) -> None:
        from tests.test_tau3_competitive_v3_training_stage import (
            segmented_completed_run_fixture,
        )

        with tempfile.TemporaryDirectory() as tmp:
            run = segmented_completed_run_fixture(Path(tmp) / "source")
            receipt_path = run / "training_receipt.json"
            receipt = read_json(receipt_path)
            exposure = receipt["training_binding"]["exposure"]
            target = _Target("training", str(receipt_path))

            _validate_training_receipt(
                target,
                receipt_path,
                receipt,
                exposure_receipt_sha256=exposure["receipt"]["sha256"],
                exposure_ledger_sha256=exposure["ledger"]["sha256"],
            )

            self.assertFalse(target.errors, target.errors)
            optimizer = (
                run
                / receipt["process_segments"]["segments"][0]["record"][
                    "optimizer_state_output"
                ]["path"]
            )
            optimizer.chmod(0o644)
            optimizer.write_bytes(b"tampered")
            rejected = _Target("training", str(receipt_path))

            _validate_training_receipt(
                rejected,
                receipt_path,
                receipt,
                exposure_receipt_sha256=exposure["receipt"]["sha256"],
                exposure_ledger_sha256=exposure["ledger"]["sha256"],
            )

            self.assertIn(
                "process segment chain must replay",
                json.dumps(rejected.errors),
            )

    def test_plan_only_fixture_passes_plan_stage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build_plan_only_bundle(root)

            result = validate_tau3_competitive_v3_bundle(root, stage="plan")

            self.assertTrue(result["passed"], json.dumps(result, indent=2))
            self.assertEqual(result["schema_version"], "hfr.validation.v1")

    def test_qualified_detached_prefix_receipt_requires_bound_equivalence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            exposure_receipt = root / "exposure-receipt.json"
            exposure_ledger = root / "exposure-ledger.jsonl"
            write_json(exposure_receipt, {"passed": True})
            exposure_ledger.write_text('{"step":1}\n', encoding="utf-8")
            receipt_path = write_training_receipt(
                root,
                "candidate-prefix",
                "a" * 64,
                exposure_receipt,
                exposure_ledger,
            )
            receipt = read_json(receipt_path)
            recipe = receipt["training_binding"]["recipe"]
            recipe.update(
                {
                    "full_gradient_objective": False,
                    "prefix_cache_training": True,
                    "prefix_equivalence_required": True,
                    "prefix_equivalence_passed": True,
                    "rank": 16,
                    "scale": 32.0,
                    "learning_rate": 1e-5,
                    "num_layers": 8,
                    "max_seq_length": 16384,
                    "batch_size": 1,
                    "grad_accumulation": 4,
                    "mask_prompt": True,
                    "seed": 101,
                }
            )
            objective = receipt["training_binding"]["exposure"]["objective"]
            objective.update({"full_gradient": False, "detached_prefix": True})
            receipt["training_binding"]["exposure"]["dataset"] = {
                "sha256": "1" * 64
            }
            receipt["training_binding"]["protocol"]["sha256"] = "2" * 64
            receipt["training_binding"]["model"] = {
                "identity_sha256": "3" * 64
            }
            equivalence = equivalence_fixture()
            equivalence["bindings"]["recipe"].update(
                {
                    "rank": 16,
                    "scale": 32.0,
                    "learning_rate": 1e-5,
                    "num_layers": 8,
                    "max_seq_length": 16384,
                    "batch_size": 1,
                    "grad_accumulation": 4,
                    "mask_prompt": True,
                    "allowed_seeds": [101],
                }
            )
            equivalence_path = root / "prefix-equivalence.json"
            write_json(equivalence_path, equivalence)
            receipt["training_binding"]["prefix_equivalence"] = {
                "sha256": sha256_file(equivalence_path),
                "validation_passed": True,
            }
            write_json(receipt_path, receipt)
            target = _Target("training", str(receipt_path))
            loaded = _Loaded(
                _Target("prefix-equivalence", str(equivalence_path)),
                payload=equivalence,
                sha256=sha256_file(equivalence_path),
            )

            _validate_training_receipt(
                target,
                receipt_path,
                receipt,
                exposure_receipt_sha256=sha256_file(exposure_receipt),
                exposure_ledger_sha256=sha256_file(exposure_ledger),
                prefix_equivalence=loaded,
            )

            self.assertFalse(target.errors, target.errors)

            receipt["training_binding"]["prefix_equivalence"]["sha256"] = (
                "f" * 64
            )
            rejected = _Target("training", str(receipt_path))
            _validate_training_receipt(
                rejected,
                receipt_path,
                receipt,
                exposure_receipt_sha256=sha256_file(exposure_receipt),
                exposure_ledger_sha256=sha256_file(exposure_ledger),
                prefix_equivalence=loaded,
            )
            self.assertIn(
                "bind prefix equivalence sha256",
                json.dumps(rejected.errors),
            )

    def test_missing_or_forged_v2_evidence_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = build_plan_only_bundle(root)
            plan["v2_negative_evidence"]["candidate_c"]["training_receipt"]["sha256"] = "0" * 64
            write_json(root / "competitive_v3_plan.json", plan)

            result = validate_tau3_competitive_v3_bundle(root, stage="plan")

            self.assertFalse(result["passed"])
            self.assertIn("sha256 mismatch", json.dumps(result))

    def test_plan_stage_requires_evaluator_model_contract_ref(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = build_plan_only_bundle(root)
            plan["harness_contract"].pop("evaluator_model_contract")
            write_json(root / "competitive_v3_plan.json", plan)

            result = validate_tau3_competitive_v3_bundle(root, stage="plan")

            self.assertFalse(result["passed"])
            self.assertIn("harness_contract.evaluator_model_contract", json.dumps(result))

    def test_plan_stage_rejects_tampered_evaluator_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = build_plan_only_bundle(root)
            contract_path = root / plan["models"]["evaluator_models"]["path"]
            contract = read_json(contract_path)
            contract["roles"]["reviewer"]["model_identity"]["revision"] = "b" * 40
            write_json(contract_path, contract)
            ref = ref_for(root, contract_path)
            plan["models"]["evaluator_models"] = ref
            plan["harness_contract"]["evaluator_model_contract"] = ref
            write_json(root / "competitive_v3_plan.json", plan)

            result = validate_tau3_competitive_v3_bundle(root, stage="plan")

            self.assertFalse(result["passed"])
            self.assertIn("user_simulator and reviewer must share exact model_identity", json.dumps(result))

    def test_plan_stage_rejects_stale_evaluator_identity_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = build_plan_only_bundle(root)
            contract_path = root / plan["models"]["evaluator_models"]["path"]
            contract = read_json(contract_path)
            contract["roles"]["user_simulator"]["model_identity"]["local_tree_sha256"] = "d" * 64
            write_json(contract_path, contract)
            ref = ref_for(root, contract_path)
            plan["models"]["evaluator_models"] = ref
            plan["harness_contract"]["evaluator_model_contract"] = ref
            write_json(root / "competitive_v3_plan.json", plan)

            result = validate_tau3_competitive_v3_bundle(root, stage="plan")

            self.assertFalse(result["passed"])
            self.assertIn("model_identity_sha256 must replay", json.dumps(result))

    def test_v2_evidence_cannot_be_recast_as_a_win(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = build_plan_only_bundle(root)
            blocked_path = root / "v2" / "blocked-verdict.json"
            write_json(
                blocked_path,
                {
                    "slug": "tau3-core-qlora-training",
                    "verdict": "pass",
                    "passed": True,
                },
            )
            plan["v2_negative_evidence"]["blocked_verdict"]["sha256"] = sha256_file(blocked_path)
            write_json(root / "competitive_v3_plan.json", plan)

            result = validate_tau3_competitive_v3_bundle(root, stage="plan")

            self.assertFalse(result["passed"])
            self.assertIn("v2 predecessor verdict must remain blocked", json.dumps(result))

    def test_strict_defaults_to_final_and_requires_final_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build_plan_only_bundle(root)
            proc = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve().parents[1] / "scripts" / "validate_tau3_competitive_v3.py"),
                    "--bundle",
                    str(root),
                    "--strict",
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            self.assertEqual(proc.returncode, 1, proc.stdout + proc.stderr)
            self.assertIn("missing evidence_refs.dataset", proc.stdout)

    def test_private_local_source_path_is_plan_stage_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = build_plan_only_bundle(root)
            external = root / "external-v2.json"
            write_json(
                external,
                {
                    "slug": "tau3-core-qlora-training",
                    "verdict": "blocked",
                    "passed": False,
                },
            )
            ref = {"path": "v2/blocked-verdict.json", "source_path": str(external), "sha256": sha256_file(external), "access": "private_local"}
            plan["v2_negative_evidence"]["blocked_verdict"] = ref
            write_json(root / "competitive_v3_plan.json", plan)

            plan_result = validate_tau3_competitive_v3_bundle(root, stage="plan")
            dataset_result = validate_tau3_competitive_v3_bundle(root, stage="dataset")

            self.assertTrue(plan_result["passed"], json.dumps(plan_result, indent=2))
            self.assertFalse(dataset_result["passed"])
            self.assertIn("source_path is allowed only for plan-stage private_local refs", json.dumps(dataset_result))

    def test_final_stage_passes_with_complete_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build_complete_bundle(root)

            result = validate_tau3_competitive_v3_bundle(root, strict=True, stage="final")

            self.assertTrue(result["passed"], json.dumps(result, indent=2))

    def test_final_candidate_selection_replays_distinct_recipe_quorum(
        self,
    ) -> None:
        report = candidate_selection()
        report["candidates"][1]["training_binding"][
            "recipe_sha256"
        ] = report["candidates"][0]["training_binding"][
            "recipe_sha256"
        ]
        target = _Target("candidate_selection", "fixture")

        competitive_v3_module._validate_candidate_selection_quorum(
            target,
            report,
        )

        self.assertIn(
            "at least two distinct qualified recipe hashes",
            json.dumps(target.errors),
        )

        below_floor = candidate_selection()
        below_floor["candidates"][0]["metrics"]["macro_pass1"][
            "candidate"
        ] = 0.09
        threshold_target = _Target(
            "candidate_selection",
            "threshold-fixture",
        )

        competitive_v3_module._validate_candidate_selection_quorum(
            threshold_target,
            below_floor,
        )

        self.assertIn(
            "development macro and gain floors",
            json.dumps(threshold_target.errors),
        )

    def test_final_stage_forwards_complete_v2_authorization_replay(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build_complete_bundle(root)
            expected = upgrade_final_fixture_to_v2_stub(root)
            replay = {
                "authorized": True,
                "training_protocol_sha256": expected["training_protocol_sha256"],
                "benchmark_protocol_lineage_sha256": expected[
                    "benchmark_protocol_lineage_sha256"
                ],
                "blind_custody_receipt_sha256": expected[
                    "blind_custody_receipt_sha256"
                ],
            }
            with mock.patch.object(
                competitive_v3_module,
                "_check_loaded_schema",
            ), mock.patch.object(
                competitive_v3_module,
                "validate_tau3_sealed_authorization",
                return_value=replay,
            ) as authorization_replay:
                result = validate_tau3_competitive_v3_bundle(
                    root,
                    strict=True,
                    stage="final",
                )

            self.assertTrue(result["passed"], json.dumps(result, indent=2))
            kwargs = authorization_replay.call_args.kwargs
            for key in (
                "training_protocol_path",
                "benchmark_protocol_lineage_path",
                "custody_receipt_path",
                "generator_validation_path",
                "fresh_contamination_replay_path",
            ):
                self.assertIsNotNone(kwargs[key])
            self.assertEqual(
                kwargs["retired_source_incident_sha256"],
                "f" * 64,
            )

    def test_final_stage_rejects_unsupported_competitive_claim(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build_complete_bundle(root)
            eval_path = root / "final" / "sealed-evaluation.json"
            evaluation = read_json(eval_path)
            evaluation["metrics"]["macro_pass1"]["adapter"] = 0.10
            write_json(eval_path, evaluation)
            plan = read_json(root / "competitive_v3_plan.json")
            final_path = root / "final-evidence.json"
            final = read_json(final_path)
            final["artifacts"]["sealed_evaluation"]["sha256"] = sha256_file(eval_path)
            write_json(final_path, final)
            plan["evidence_refs"]["final"]["sha256"] = sha256_file(final_path)
            write_json(root / "competitive_v3_plan.json", plan)

            result = validate_tau3_competitive_v3_bundle(root, strict=True, stage="final")

            self.assertFalse(result["passed"])
            self.assertIn("same-size comparator noninferiority margin", json.dumps(result))

    def test_final_stage_rejects_reversed_chronology(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build_complete_bundle(root)
            final_path = root / "final-evidence.json"
            final = read_json(final_path)
            final["sealed_started_at"] = "2026-07-23T00:40:00Z"
            final["candidate_locked_at"] = "2026-07-23T00:50:00Z"
            write_json(final_path, final)
            plan = read_json(root / "competitive_v3_plan.json")
            plan["evidence_refs"]["final"]["sha256"] = sha256_file(final_path)
            write_json(root / "competitive_v3_plan.json", plan)

            result = validate_tau3_competitive_v3_bundle(root, strict=True, stage="final")

            self.assertFalse(result["passed"])
            self.assertIn("candidate lock must predate sealed access", json.dumps(result))

    def test_final_stage_rejects_missing_chronology_timestamp(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build_complete_bundle(root)
            final_path = root / "final-evidence.json"
            final = read_json(final_path)
            final.pop("sealed_started_at")
            write_json(final_path, final)
            plan = read_json(root / "competitive_v3_plan.json")
            plan["evidence_refs"]["final"]["sha256"] = sha256_file(final_path)
            write_json(root / "competitive_v3_plan.json", plan)

            result = validate_tau3_competitive_v3_bundle(root, strict=True, stage="final")

            self.assertFalse(result["passed"])
            self.assertIn("sealed_started_at must be an ISO-8601 timestamp", json.dumps(result))

    def test_dataset_stage_rejects_forged_self_attestation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = build_plan_only_bundle(root)
            forged = write_artifact(
                root,
                "dataset-evidence.json",
                {
                    "schema_version": DATASET_SCHEMA_VERSION,
                    "coverage_complete": True,
                    "diversity_complete": True,
                    "passed": True,
                },
            )
            plan["evidence_refs"] = {"dataset": forged}
            write_json(root / "competitive_v3_plan.json", plan)

            result = validate_tau3_competitive_v3_bundle(root, stage="dataset")

            self.assertFalse(result["passed"])
            self.assertIn("artifacts.competitive_dataset_manifest", json.dumps(result))

    def test_dataset_replay_tau_import_failure_retries_repository_venv(self) -> None:
        calls: list[dict[str, Any]] = []

        def fake_dataset_validator(bundle: Path, *, strict: bool = True, grounded_validator_python: Path | None = None) -> dict[str, Any]:
            calls.append({"bundle": bundle, "strict": strict, "grounded_validator_python": grounded_validator_python})
            if grounded_validator_python is None:
                return {
                    "passed": False,
                    "errors": [
                        "grounded_generation strict replay failed",
                        "cannot import vendored Tau airline tools: No module named 'pydantic'",
                    ],
                }
            return {"passed": True, "errors": [], "bridge": str(grounded_validator_python)}

        bridge_python = Path("/repo/local/tau3/venv/bin/python3")
        with mock.patch.object(competitive_v3_module, "validate_tau3_competitive_dataset_bundle", side_effect=fake_dataset_validator), mock.patch.object(
            competitive_v3_module, "_repository_local_tau_python", return_value=(bridge_python, None)
        ):
            result = competitive_v3_module._validate_competitive_dataset_with_tau_bridge(Path("/bundle"))

        self.assertTrue(result["passed"], result)
        self.assertEqual(len(calls), 2)
        self.assertIsNone(calls[0]["grounded_validator_python"])
        self.assertEqual(calls[1]["grounded_validator_python"], bridge_python)

    def test_dataset_replay_semantic_failure_does_not_retry_repository_venv(self) -> None:
        with mock.patch.object(
            competitive_v3_module,
            "validate_tau3_competitive_dataset_bundle",
            return_value={"passed": False, "errors": ["coverage gate failed"]},
        ) as dataset_validator, mock.patch.object(competitive_v3_module, "_repository_local_tau_python") as bridge_python:
            result = competitive_v3_module._validate_competitive_dataset_with_tau_bridge(Path("/bundle"))

        self.assertFalse(result["passed"])
        self.assertEqual(result["errors"], ["coverage gate failed"])
        self.assertEqual(dataset_validator.call_count, 1)
        bridge_python.assert_not_called()

    def test_final_stage_requires_sealed_authorization_replay_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build_complete_bundle(root, include_auth_replay_inputs=False)

            result = validate_tau3_competitive_v3_bundle(root, strict=True, stage="final")

            self.assertFalse(result["passed"])
            self.assertIn("sealed_authorization_validation inputs are required", json.dumps(result))

    def test_strict_final_requires_post_publication_record(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build_complete_bundle(root)
            publication_path = root / "publication-preflight.json"
            publication = read_json(publication_path)
            publication["artifacts"].pop("post_publication")
            write_json(publication_path, publication)
            plan = read_json(root / "competitive_v3_plan.json")
            plan["evidence_refs"]["publication"]["sha256"] = sha256_file(publication_path)
            write_json(root / "competitive_v3_plan.json", plan)

            result = validate_tau3_competitive_v3_bundle(root, strict=True, stage="final")

            self.assertFalse(result["passed"])
            self.assertIn("artifacts.post_publication", json.dumps(result))

    def test_strict_final_requires_source_parity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build_complete_bundle(root)
            publication_path = root / "publication-preflight.json"
            publication = read_json(publication_path)
            publication["artifacts"].pop("source_parity")
            write_json(publication_path, publication)
            plan = read_json(root / "competitive_v3_plan.json")
            plan["evidence_refs"]["publication"]["sha256"] = sha256_file(publication_path)
            write_json(root / "competitive_v3_plan.json", plan)

            result = validate_tau3_competitive_v3_bundle(root, strict=True, stage="final")

            self.assertFalse(result["passed"])
            self.assertIn("artifacts.source_parity", json.dumps(result))

    def test_source_parity_requires_reviewed_revision_binding(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build_complete_bundle(root)
            parity_path = root / "publication" / "source-parity.json"
            parity = read_json(parity_path)
            parity["reviewed_evidence_source"]["github_revision"] = "3" * 40
            write_json(parity_path, parity)
            publication_path = root / "publication-preflight.json"
            publication = read_json(publication_path)
            publication["artifacts"]["source_parity"]["sha256"] = sha256_file(parity_path)
            write_json(publication_path, publication)
            plan = read_json(root / "competitive_v3_plan.json")
            plan["evidence_refs"]["publication"]["sha256"] = sha256_file(publication_path)
            write_json(root / "competitive_v3_plan.json", plan)

            result = validate_tau3_competitive_v3_bundle(root, strict=True, stage="final")

            self.assertFalse(result["passed"])
            self.assertIn("reviewed evidence must bind GitHub revision", json.dumps(result))

    def test_training_receipts_alone_do_not_qualify_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build_complete_bundle(root)
            training_path = root / "training-evidence.json"
            training = read_json(training_path)
            for candidate in training["qualified_candidates"]:
                candidate.pop("development_scorecard")
                candidate.pop("behavior_probes")
            write_json(training_path, training)
            plan = read_json(root / "competitive_v3_plan.json")
            plan["evidence_refs"]["training"]["sha256"] = sha256_file(training_path)
            write_json(root / "competitive_v3_plan.json", plan)

            result = validate_tau3_competitive_v3_bundle(root, strict=True, stage="final")

            self.assertFalse(result["passed"])
            self.assertIn("development_scorecard", json.dumps(result))
            self.assertIn("at least two candidates must pass development qualification gates", json.dumps(result))

    def test_distinct_candidate_exposure_ledgers_qualify(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build_complete_bundle(root, distinct_training_exposures=True)

            result = validate_tau3_competitive_v3_bundle(
                root,
                strict=True,
                stage="final",
            )

            self.assertTrue(result["passed"], json.dumps(result, indent=2))

    def test_candidate_exposure_cannot_be_omitted_without_shared_fallback(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build_complete_bundle(root, distinct_training_exposures=True)
            training_path = root / "training-evidence.json"
            training = read_json(training_path)
            training["qualified_candidates"][1].pop("exposure")
            write_json(training_path, training)
            plan = read_json(root / "competitive_v3_plan.json")
            plan["evidence_refs"]["training"]["sha256"] = sha256_file(
                training_path
            )
            write_json(root / "competitive_v3_plan.json", plan)

            result = validate_tau3_competitive_v3_bundle(
                root,
                strict=True,
                stage="final",
            )

            self.assertFalse(result["passed"])
            self.assertIn(
                "must include exposure evidence or use shared exposure",
                json.dumps(result),
            )

    def test_candidate_exposure_saved_validation_must_replay(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build_complete_bundle(root, distinct_training_exposures=True)
            validation_path = (
                root / "exposure" / "candidate-b" / "validation.json"
            )
            validation = read_json(validation_path)
            validation["ledger_sha256"] = "0" * 64
            write_json(validation_path, validation)
            training_path = root / "training-evidence.json"
            training = read_json(training_path)
            training["qualified_candidates"][1]["exposure"]["validation"][
                "sha256"
            ] = sha256_file(validation_path)
            write_json(training_path, training)
            plan = read_json(root / "competitive_v3_plan.json")
            plan["evidence_refs"]["training"]["sha256"] = sha256_file(
                training_path
            )
            write_json(root / "competitive_v3_plan.json", plan)

            result = validate_tau3_competitive_v3_bundle(
                root,
                strict=True,
                stage="final",
            )

            self.assertFalse(result["passed"])
            self.assertIn(
                "exposure.validation ledger_sha256 must replay",
                json.dumps(result),
            )

    def test_missing_internal_validation_does_not_qualify_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build_complete_bundle(root)
            training_path = root / "training-evidence.json"
            training = read_json(training_path)
            for candidate in training["qualified_candidates"]:
                candidate.pop("internal_validation")
            write_json(training_path, training)
            plan = read_json(root / "competitive_v3_plan.json")
            plan["evidence_refs"]["training"]["sha256"] = sha256_file(
                training_path
            )
            write_json(root / "competitive_v3_plan.json", plan)

            result = validate_tau3_competitive_v3_bundle(
                root,
                strict=True,
                stage="final",
            )

            self.assertFalse(result["passed"])
            self.assertIn("internal_validation.artifact", json.dumps(result))
            self.assertIn(
                "at least two candidates must pass development qualification gates",
                json.dumps(result),
            )

    def test_below_threshold_development_scorecard_does_not_qualify(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build_complete_bundle(root)
            eval_path = root / "training" / "candidate-a" / "development-evaluation.json"
            evaluation = read_json(eval_path)
            for trial in evaluation["development_trials"]:
                trial["adapter_pass1"] = False
                trial["result_sha256"] = canonical_sha256(
                    {
                        "adapter_pass1": False,
                        "base_pass1": trial["base_pass1"],
                        "domain": trial["domain"],
                        "seed": trial["seed"],
                        "task_sha256": trial["task_sha256"],
                    }
                )
            evaluation["metrics"]["macro_pass1"]["adapter"] = 0.0
            evaluation["metrics"]["per_domain_pass1"]["adapter"] = {"airline": 0.0, "retail": 0.0, "telecom": 0.0}
            write_json(eval_path, evaluation)
            scorecard_path = root / "training" / "candidate-a" / "development-scorecard.json"
            scorecard = read_json(scorecard_path)
            scorecard["development_evaluation"]["sha256"] = sha256_file(eval_path)
            scorecard["metrics"]["macro_pass1"] = 0.20
            write_json(scorecard_path, scorecard)
            training_path = root / "training-evidence.json"
            training = read_json(training_path)
            training["qualified_candidates"][0]["development_scorecard"]["sha256"] = sha256_file(scorecard_path)
            write_json(training_path, training)
            plan = read_json(root / "competitive_v3_plan.json")
            plan["evidence_refs"]["training"]["sha256"] = sha256_file(training_path)
            write_json(root / "competitive_v3_plan.json", plan)

            result = validate_tau3_competitive_v3_bundle(root, strict=True, stage="final")

            self.assertFalse(result["passed"])
            self.assertIn("development macro Pass-1 must be at least 0.10", json.dumps(result))

    def test_development_metrics_without_trial_evidence_do_not_qualify(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build_complete_bundle(root)
            eval_path = root / "training" / "candidate-a" / "development-evaluation.json"
            evaluation = read_json(eval_path)
            evaluation.pop("development_trials")
            evaluation.pop("development_grid")
            evaluation["source_artifacts"] = {}
            evaluation["metrics"]["macro_pass1"]["adapter"] = 1.0
            evaluation["metrics"]["macro_pass1"]["base"] = 0.0
            write_json(eval_path, evaluation)
            scorecard_path = root / "training" / "candidate-a" / "development-scorecard.json"
            scorecard = read_json(scorecard_path)
            scorecard["development_evaluation"]["sha256"] = sha256_file(eval_path)
            write_json(scorecard_path, scorecard)
            training_path = root / "training-evidence.json"
            training = read_json(training_path)
            training["qualified_candidates"][0]["development_scorecard"]["sha256"] = sha256_file(scorecard_path)
            write_json(training_path, training)
            plan = read_json(root / "competitive_v3_plan.json")
            plan["evidence_refs"]["training"]["sha256"] = sha256_file(training_path)
            write_json(root / "competitive_v3_plan.json", plan)

            result = validate_tau3_competitive_v3_bundle(root, strict=True, stage="final")

            self.assertFalse(result["passed"])
            self.assertIn("development evaluation must include trial-level outcomes", json.dumps(result))

    def test_development_lineage_training_protocol_must_match_receipt(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build_complete_bundle(root)
            eval_path = (
                root
                / "training"
                / "candidate-a"
                / "development-evaluation.json"
            )
            scorecard_path = (
                root
                / "training"
                / "candidate-a"
                / "development-scorecard.json"
            )
            evaluation = read_json(eval_path)
            scorecard = read_json(scorecard_path)
            for payload in (evaluation, scorecard):
                payload["bindings"]["training_protocol_sha256"] = "d" * 64
                payload["bindings"][
                    "benchmark_protocol_lineage_sha256"
                ] = "e" * 64
            scorecard["frozen_contract"][
                "training_protocol_sha256"
            ] = "d" * 64
            scorecard["frozen_contract"][
                "benchmark_protocol_lineage_sha256"
            ] = "e" * 64
            write_json(eval_path, evaluation)
            scorecard["development_evaluation"]["sha256"] = sha256_file(
                eval_path
            )
            scorecard["development_evaluation"]["size"] = (
                eval_path.stat().st_size
            )
            write_json(scorecard_path, scorecard)
            training_path = root / "training-evidence.json"
            training = read_json(training_path)
            training["qualified_candidates"][0][
                "development_scorecard"
            ]["sha256"] = sha256_file(scorecard_path)
            write_json(training_path, training)
            plan = read_json(root / "competitive_v3_plan.json")
            plan["evidence_refs"]["training"]["sha256"] = sha256_file(
                training_path
            )
            write_json(root / "competitive_v3_plan.json", plan)

            result = validate_tau3_competitive_v3_bundle(
                root,
                strict=True,
                stage="final",
            )

            self.assertFalse(result["passed"])
            self.assertIn(
                "development scorecard training protocol must match "
                "training receipt",
                json.dumps(result),
            )

    def test_forged_behavior_probe_summary_does_not_qualify(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build_complete_bundle(root)
            probes_path = root / "training" / "candidate-a" / "behavior-probes.json"
            probes = read_json(probes_path)
            probe_path = probes_path.parent / probes["probe_results"][0]["path"]
            probe = read_json(probe_path)
            probe["actual_outcome"] = {"passed": False, "assertions": [{"id": "contains_any", "passed": False}]}
            write_json(probe_path, probe)
            probes["passed"] = True
            probes["aggregate"]["failed_probe_count"] = 0
            probes["probe_results"][0]["sha256"] = sha256_file(probe_path)
            write_json(probes_path, probes)
            training_path = root / "training-evidence.json"
            training = read_json(training_path)
            training["qualified_candidates"][0]["behavior_probes"]["sha256"] = sha256_file(probes_path)
            write_json(training_path, training)
            plan = read_json(root / "competitive_v3_plan.json")
            plan["evidence_refs"]["training"]["sha256"] = sha256_file(training_path)
            write_json(root / "competitive_v3_plan.json", plan)

            result = validate_tau3_competitive_v3_bundle(root, strict=True, stage="final")

            self.assertFalse(result["passed"])
            self.assertIn("behavior probe validator must pass", json.dumps(result))

    def test_missing_behavior_probe_families_do_not_qualify(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build_complete_bundle(root)
            probes_path = root / "training" / "candidate-a" / "behavior-probes.json"
            probes = read_json(probes_path)
            probes["families"] = ["formatting"]
            probes["probe_results"] = probes["probe_results"][:1]
            probes["aggregate"] = {"total_probe_count": 1, "failed_probe_count": 0, "family_count": 1}
            probes["passed"] = True
            write_json(probes_path, probes)
            training_path = root / "training-evidence.json"
            training = read_json(training_path)
            training["qualified_candidates"][0]["behavior_probes"]["sha256"] = sha256_file(probes_path)
            write_json(training_path, training)
            plan = read_json(root / "competitive_v3_plan.json")
            plan["evidence_refs"]["training"]["sha256"] = sha256_file(training_path)
            write_json(root / "competitive_v3_plan.json", plan)

            result = validate_tau3_competitive_v3_bundle(root, strict=True, stage="final")

            self.assertFalse(result["passed"])
            self.assertIn("behavior probe validator must pass", json.dumps(result))

    def test_sealed_claim_requires_paired_ci_not_point_estimate_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build_complete_bundle(root)
            eval_path = root / "final" / "sealed-evaluation.json"
            evaluation = read_json(eval_path)
            evaluation["effects"]["base"]["domain_stratified_macro_pass1"]["confidence_interval"]["lower"] = -0.01
            write_json(eval_path, evaluation)
            final_path = root / "final-evidence.json"
            final = read_json(final_path)
            final["artifacts"]["sealed_evaluation"]["sha256"] = sha256_file(eval_path)
            write_json(final_path, final)
            plan = read_json(root / "competitive_v3_plan.json")
            plan["evidence_refs"]["final"]["sha256"] = sha256_file(final_path)
            write_json(root / "competitive_v3_plan.json", plan)

            result = validate_tau3_competitive_v3_bundle(root, strict=True, stage="final")

            self.assertFalse(result["passed"])
            self.assertIn("adapter beats base", json.dumps(result))


def build_plan_only_bundle(root: Path) -> dict[str, Any]:
    refs = {
        "rubric": write_artifact(root, "rubric.md", "# rubric\n"),
        "protocol": write_artifact(root, "protocol.json", {"protocol": "frozen"}),
        "v2_blocked": write_artifact(
            root,
            "v2/blocked-verdict.json",
            {
                "slug": "tau3-core-qlora-training",
                "verdict": "blocked",
                "passed": False,
                "candidate": "c",
            },
        ),
        "v2_training": write_artifact(root, "v2/candidate-c-training.json", {"candidate": "C", "terminal_status": "blocked"}),
        "v2_development": write_artifact(root, "v2/candidate-c-development.json", {"candidate": "C", "macro_pass1": 0.0}),
        "tau_repository": write_artifact(root, "tau-revision.json", {"revision": "a" * 40}),
        "harness": write_artifact(root, "harness.json", {"text_mode": True}),
        "tokenizer_chat_template": write_artifact(root, "tokenizer-chat-template.json", {"chat_template": "frozen"}),
        "ordered_tool_catalog": write_artifact(root, "tool-catalog.json", {"tools": ["lookup_order", "update_order"]}),
        "policy_prompt": write_artifact(root, "policy-prompt.txt", "policy\n"),
        "task_trial_seed_grid": write_artifact(root, "grid.json", {"seeds": [101, 202]}),
        "decoding": write_artifact(root, "decoding.json", {"temperature": 0}),
        "retry_policy": write_artifact(root, "retry-policy.json", {"retries": 0}),
        "safety_policy": write_artifact(root, "safety-policy.json", {"safe": True}),
        "evaluator_model_contract": write_artifact(root, "evaluator-model-contract.json", evaluator_model_contract()),
    }
    plan: dict[str, Any] = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "mission_contract": {
            "mission_id": "tau3-competitive-agent-v3",
            "mission_statement_sha256": "1" * 64,
            "rubric": refs["rubric"],
            "protocol": refs["protocol"],
        },
        "v2_negative_evidence": {
            "immutable": True,
            "rewrites_v2": False,
            "blocked_verdict": refs["v2_blocked"],
            "candidate_c": {
                "training_receipt": refs["v2_training"],
                "development_result": refs["v2_development"],
            },
        },
        "lineage": {"dataset_id": "tau3-competitive-agent-v3", "version": "v3", "predecessor": "v2", "new_lineage": True},
        "sealed_access": {"payload_access_count": 0, "materialized_sealed_fields": []},
        "models": {
            "base": {"model_id": "base-8b", "parameter_b": 8.0, "selected_by_sealed_blind_preflight": True},
            "comparators": [
                {"arm_id": "comparator_1", "parameter_b": 8.1, "same_size_open": True, "frozen": True},
                {"arm_id": "comparator_2", "parameter_b": 7.6, "same_size_open": True, "frozen": True},
            ],
            "evaluator_models": refs["evaluator_model_contract"],
        },
        "harness_contract": {
            "identical_for_all_arms": True,
            "text_mode": True,
            "frozen_before_training": True,
            "tau_repository": refs["tau_repository"],
            "harness": refs["harness"],
            "tokenizer_chat_template": refs["tokenizer_chat_template"],
            "ordered_tool_catalog": refs["ordered_tool_catalog"],
            "policy_prompt": refs["policy_prompt"],
            "task_trial_seed_grid": refs["task_trial_seed_grid"],
            "decoding": refs["decoding"],
            "retry_policy": refs["retry_policy"],
            "safety_policy": refs["safety_policy"],
            "evaluator_model_contract": refs["evaluator_model_contract"],
            "context_window": 8192,
        },
        "statistical_contract": {
            "procedure": "paired_bootstrap",
            "confidence_level": 0.95,
            "adapter_improvement_ci_excludes_zero": True,
            "same_size_noninferiority_margin": 0.05,
            "promotion_predicates_fail_closed": True,
        },
        "budget_contract": {
            "unattended_days": 7,
            "separate_infrastructure_attempt_budget": True,
            "separate_qualified_candidate_budget": True,
            "minimum_qualified_candidates": 2,
            "target_qualified_candidates_min": 3,
            "target_qualified_candidates_max": 4,
        },
        "dataset_contract": {
            "coverage_gates": {
                "all_domains_train_and_internal_validation": True,
                "min_train_targets_per_tool": 16,
                "min_internal_validation_targets_per_tool": 4,
                "min_train_argument_payloads_per_tool": 8,
                "min_internal_validation_argument_payloads_per_tool": 2,
                "all_required_behaviors_per_domain": True,
                "telecom_min_training_example_fraction": 0.25,
                "telecom_min_supervised_token_fraction": 0.25,
                "domain_supervised_token_min_fraction": 0.25,
                "domain_supervised_token_max_fraction": 0.40,
                "max_domain_canonical_target_duplication_fraction": 0.20,
                "split_hashes_disjoint": True,
            },
            "sampling_contract": {
                "deterministic_receipt_producing": True,
                "records_per_step_exposure": True,
                "replay_from_dataset_hash_config_seed": True,
            },
            "objective_contract": {
                "supervises_every_eligible_assistant_decision": True,
                "masks_prompt_tool_result_negative_user_private_reference_and_grader_tokens": True,
                "retains_parent_trajectory_and_decision_ordinal": True,
                "negative_actions_masked_with_safe_correction_only": True,
            },
        },
        "recipe_contract": {
            "minimum_rank": 16,
            "minimum_adapted_layers": 8,
            "minimum_effective_batch_examples": 4,
            "minimum_effective_epochs": 2,
            "token_loss_is_selection_metric": False,
            "candidate_c_stronger_search_space": True,
        },
        "development_contract": {
            "development_only_selection": True,
            "identical_harness": True,
            "no_comparator_specific_prompting": True,
            "qualification_gate": {"macro_pass1_min": 0.10, "per_domain_pass1_min": 0.05, "macro_base_improvement_min": 0.05},
        },
        "publication_contract": {"competitive_claims_fail_closed": True, "sealed_payloads_public": False, "redaction_required": True},
    }
    write_json(root / "competitive_v3_plan.json", plan)
    return plan


def build_complete_bundle(
    root: Path,
    *,
    include_auth_replay_inputs: bool = True,
    distinct_training_exposures: bool = False,
) -> dict[str, Any]:
    plan = build_plan_only_bundle(root)
    dataset_ref = build_dataset_evidence(root)
    training_ref = build_training_evidence(
        root,
        distinct_exposures=distinct_training_exposures,
    )
    final_ref = build_final_evidence(root, training_ref, include_auth_replay_inputs=include_auth_replay_inputs)
    publication_ref = build_publication_evidence(root)
    plan["evidence_refs"] = {"dataset": dataset_ref, "training": training_ref, "final": final_ref, "publication": publication_ref}
    write_json(root / "competitive_v3_plan.json", plan)
    return plan


def evaluator_model_contract() -> dict[str, Any]:
    identity = {
        "model_id": "Qwen/Qwen3.6-8B-Instruct",
        "revision": "a" * 40,
        "local_tree_sha256": "b" * 64,
        "local_identity_sha256": "c" * 64,
        "local_identity_path": "local/tau3/identities/teacher.json",
        "local_path": "local/tau3/models/teacher",
        "role": "teacher_generation_and_review_only",
    }
    role_contract = {
        "model_identity": dict(identity),
        "model_identity_sha256": canonical_sha256(identity),
        "local_only": True,
        "network": False,
        "temperature": 0.0,
        "top_p": 1.0,
        "no_test_time_search": True,
        "comparator_specific_prompting": False,
    }
    return {
        "schema_version": "hfr.tau3_evaluator_model_contract.v1",
        "roles": {
            "user_simulator": {"role": "user_simulator", **role_contract},
            "reviewer": {"role": "reviewer", **role_contract},
        },
        "roles_share_exact_model": True,
        "identical_for_all_arms": True,
        "applies_to_arms": ["adapter", "base", "comparator_1", "comparator_2"],
        "no_comparator_specific_prompting": True,
        "excluded_from_comparator_claims": True,
        "excluded_from_gradient_data": True,
        "local_only": True,
        "network": False,
    }


def build_dataset_evidence(root: Path) -> dict[str, str]:
    source = _write_source_dataset(root)
    grounded = _write_grounded_bundle(root)
    contamination = _write_contamination_report(root)
    dataset_dir = root / "competitive_dataset"
    tokenizer_config = _write_tokenizer_config(root)
    with _install_fake_transformers(_FakeTokenizer()), _grounded_validation_patch():
        with mock.patch.object(competitive_dataset_module, "include_template_supplements", True, create=True), mock.patch.object(
            competitive_dataset_module, "path_has_symlink_component", return_value=False
        ):
            build_tau3_competitive_dataset(
                source_dataset_dir=source,
                out_dir=dataset_dir,
                tokenizer_config_path=tokenizer_config,
                grounded_generation_bundle=grounded,
                contamination_report_path=contamination,
            )
    with _install_fake_transformers(_FakeTokenizer()), _grounded_validation_patch(), mock.patch.object(
        competitive_dataset_module, "path_has_symlink_component", return_value=False
    ):
        dataset_validation = validate_tau3_competitive_dataset_bundle(dataset_dir, strict=True)
    write_json(root / "dataset-validation.json", dataset_validation)

    objective_root = root / "objective"
    objective_root.mkdir()
    train_path, parent_path = objective_fixture(objective_root)
    add_exact_objective_tokens(train_path)
    objective = build_tau3_objective_validity_report(
        training_export_path=train_path,
        parent_trajectories_path=parent_path,
        source_root=objective_root,
    )
    objective_path = objective_root / "objective-validity.json"
    write_json(objective_path, objective)
    objective_validation = validate_tau3_objective_validity_report(objective_path)
    write_json(root / "objective-validation.json", objective_validation)

    return write_artifact(
        root,
        "dataset-evidence.json",
        {
            "schema_version": DATASET_SCHEMA_VERSION,
            "artifacts": {
                "competitive_dataset_manifest": ref_for(root, dataset_dir / "manifest.json"),
                "competitive_dataset_validation": ref_for(root, root / "dataset-validation.json"),
                "objective_validity_report": ref_for(root, objective_path),
                "objective_validity_validation": ref_for(root, root / "objective-validation.json"),
            },
        },
    )


def add_exact_objective_tokens(train_path: Path) -> None:
    rows = [json.loads(line) for line in train_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for row in rows:
        accounting = row["token_accounting"]
        prompt_tokens = int(accounting["prompt_tokens"])
        total_tokens = int(accounting["total_tokens"])
        input_token_ids = list(range(1, total_tokens + 1))
        loss_mask = [0] * max(0, prompt_tokens - 1) + [1] * (
            total_tokens - prompt_tokens
        )
        row["input_token_ids"] = input_token_ids
        row["loss_mask"] = loss_mask
        row["loss_mask_semantics"] = "mlx_lm_shifted_targets_v1"
        row["input_token_ids_sha256"] = canonical_sha256(input_token_ids)
        row["loss_mask_sha256"] = canonical_sha256(loss_mask)
    write_jsonl(train_path, rows)


def build_training_evidence(
    root: Path,
    *,
    distinct_exposures: bool = False,
) -> dict[str, str]:
    exposure_dataset = root / "exposure-dataset" / "train.jsonl"
    write_exposure_jsonl(exposure_dataset, _eligible_rows())
    build_tau3_exposure_ledger(
        exposure_dataset,
        root / "exposure" / "candidate-a",
        seed=101,
        epochs=2,
        batch_size=2,
        gradient_accumulation_steps=2,
    )
    exposure_a_receipt = (
        root
        / "exposure"
        / "candidate-a"
        / "training_exposure_receipt.json"
    )
    exposure_a_ledger = (
        root
        / "exposure"
        / "candidate-a"
        / "training_exposure_ledger.jsonl"
    )
    exposure_a_validation = validate_tau3_exposure_ledger(
        exposure_dataset,
        exposure_a_receipt,
        exposure_a_ledger,
    )
    exposure_a_validation_path = (
        root / "exposure" / "candidate-a" / "validation.json"
    )
    write_json(exposure_a_validation_path, exposure_a_validation)
    exposure_b_receipt = exposure_a_receipt
    exposure_b_ledger = exposure_a_ledger
    exposure_b_validation_path = exposure_a_validation_path
    if distinct_exposures:
        build_tau3_exposure_ledger(
            exposure_dataset,
            root / "exposure" / "candidate-b",
            seed=303,
            epochs=2,
            batch_size=2,
            gradient_accumulation_steps=2,
        )
        exposure_b_receipt = (
            root
            / "exposure"
            / "candidate-b"
            / "training_exposure_receipt.json"
        )
        exposure_b_ledger = (
            root
            / "exposure"
            / "candidate-b"
            / "training_exposure_ledger.jsonl"
        )
        exposure_b_validation = validate_tau3_exposure_ledger(
            exposure_dataset,
            exposure_b_receipt,
            exposure_b_ledger,
        )
        exposure_b_validation_path = (
            root / "exposure" / "candidate-b" / "validation.json"
        )
        write_json(exposure_b_validation_path, exposure_b_validation)
    internal_sources = write_internal_validation_sources(root)
    candidate_a = write_training_receipt(
        root,
        "candidate-a",
        "a" * 64,
        exposure_a_receipt,
        exposure_a_ledger,
        protocol_sha256=sha256_file(internal_sources["protocol"]),
        model_identity_sha256=sha256_file(internal_sources["identity"]),
        dataset_manifest_sha256=sha256_file(internal_sources["manifest"]),
    )
    candidate_b = write_training_receipt(
        root,
        "candidate-b",
        "b" * 64,
        exposure_b_receipt,
        exposure_b_ledger,
        protocol_sha256=sha256_file(internal_sources["protocol"]),
        model_identity_sha256=sha256_file(internal_sources["identity"]),
        dataset_manifest_sha256=sha256_file(internal_sources["manifest"]),
    )
    candidate_a_internal = write_candidate_internal_validation(
        root,
        "candidate-a",
        candidate_a,
        internal_sources,
    )
    candidate_b_internal = write_candidate_internal_validation(
        root,
        "candidate-b",
        candidate_b,
        internal_sources,
    )
    candidate_a_scorecard, candidate_a_probes = write_development_qualification(root, "candidate-a", candidate_a)
    candidate_b_scorecard, candidate_b_probes = write_development_qualification(root, "candidate-b", candidate_b)
    exposure_a = {
        "dataset": ref_for(root, exposure_dataset),
        "receipt": ref_for(root, exposure_a_receipt),
        "ledger": ref_for(root, exposure_a_ledger),
        "validation": ref_for(root, exposure_a_validation_path),
    }
    exposure_b = {
        "dataset": ref_for(root, exposure_dataset),
        "receipt": ref_for(root, exposure_b_receipt),
        "ledger": ref_for(root, exposure_b_ledger),
        "validation": ref_for(root, exposure_b_validation_path),
    }
    evidence: dict[str, Any] = {
        "schema_version": TRAINING_SCHEMA_VERSION,
        "budgets": {"separate_candidate_and_infra_budgets": True},
        "qualified_candidates": [
            {
                "candidate_id": "candidate-a",
                "training_receipt": ref_for(root, candidate_a),
                "internal_validation": candidate_a_internal,
                "development_scorecard": ref_for(root, candidate_a_scorecard),
                "behavior_probes": ref_for(root, candidate_a_probes),
            },
            {
                "candidate_id": "candidate-b",
                "training_receipt": ref_for(root, candidate_b),
                "internal_validation": candidate_b_internal,
                "development_scorecard": ref_for(root, candidate_b_scorecard),
                "behavior_probes": ref_for(root, candidate_b_probes),
            },
        ],
    }
    if distinct_exposures:
        evidence["qualified_candidates"][0]["exposure"] = exposure_a
        evidence["qualified_candidates"][1]["exposure"] = exposure_b
    else:
        evidence["exposure"] = exposure_a
    return write_artifact(root, "training-evidence.json", evidence)


def write_training_receipt(
    root: Path,
    candidate_id: str,
    recipe_sha256: str,
    exposure_receipt: Path,
    exposure_ledger: Path,
    *,
    protocol_sha256: str = "c" * 64,
    model_identity_sha256: str | None = None,
    dataset_manifest_sha256: str | None = None,
) -> Path:
    out = root / "training" / candidate_id
    adapter = out / "adapter"
    adapter.mkdir(parents=True)
    weight = adapter / "adapter_model.safetensors"
    weight.write_bytes(f"adapter weights for {candidate_id}".encode("utf-8"))
    record = {"path": "adapter_model.safetensors", "size": weight.stat().st_size, "sha256": sha256_file(weight), "kind": "adapter"}
    tree = __import__("hashlib").sha256(json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    receipt = {
        "schema_version": "hfr.tau3_mlx_training_run.v1",
        "phase": "final",
        "created_at": "2026-07-23T00:30:00Z",
        "bundle": {"kind": "mixture"},
        "output_dir": ".",
        "command": ["python", "-m", "flightrecorder.mlx_exposure_lora"],
        "config": {"exposure_ledger_training": True, "rank": 16, "num_layers": 16},
        "checks": [{"id": "receipt-fixture", "passed": True, "actual": True, "expected": True}],
        "terminal_status": "success",
        "weights_updated": True,
        "adapter": {"path": "adapter", "file_count": 1, "files": [record], "tree_sha256": tree},
        "adapter_weight_file_count": 1,
        "training_binding": {
            "protocol": {
                "sha256": protocol_sha256,
                "protocol_signature": "d" * 64,
                "protocol_signature_provenance": {"source": "protocol_file_sha256_content_seal", "algorithm": "sha256"},
            },
            "recipe": {
                "recipe_sha256": recipe_sha256,
                "full_gradient_objective": True,
                "exposure_ledger_training": True,
                "max_seq_length": 64,
            },
            "exposure": {
                "receipt": {"sha256": sha256_file(exposure_receipt)},
                "ledger": {"sha256": sha256_file(exposure_ledger)},
                "objective": {"full_gradient": True},
            },
        },
    }
    if model_identity_sha256 is not None:
        receipt["training_binding"]["model"] = {
            "identity_sha256": model_identity_sha256,
        }
    if dataset_manifest_sha256 is not None:
        receipt["training_binding"]["dataset"] = {
            "manifest_sha256": dataset_manifest_sha256,
        }
    path = out / "training_receipt.json"
    write_json(path, receipt)
    return path


def write_internal_validation_sources(root: Path) -> dict[str, Path]:
    source = root / "internal-validation-source"
    source.mkdir()
    dataset = source / "valid.jsonl"
    rows = []
    for index, behavior in enumerate(competitive_v3_module.BEHAVIORS):
        tokens = [100 + index, 200 + index, 300 + index, 400 + index]
        rows.append(
            {
                "messages": [
                    {"role": "system", "content": "policy"},
                    {"role": "assistant", "content": "target"},
                ],
                "metadata": {
                    "domain": competitive_v3_module.DOMAINS[
                        index % len(competitive_v3_module.DOMAINS)
                    ],
                    "behavior": behavior,
                    "token_counts": {
                        "input_token_ids": tokens,
                        "input_token_ids_sha256": canonical_sha256(tokens),
                        "prompt_tokens": 2,
                        "supervised_tokens": 2,
                        "total_tokens": 4,
                    },
                },
            }
        )
    write_jsonl(dataset, rows)
    manifest = source / "manifest.json"
    write_json(
        manifest,
        {
            "schema_version": "hfr.tau3_competitive_dataset.v1",
            "files": {
                "valid": {
                    "path": "valid.jsonl",
                    "sha256": sha256_file(dataset),
                    "bytes": dataset.stat().st_size,
                }
            },
        },
    )
    protocol = source / "protocol.json"
    write_json(
        protocol,
        {"schema_version": "hfr.tau3_protocol_config.v1"},
    )
    identity = source / "model-identity.json"
    write_json(
        identity,
        {
            "schema_version": "hfr.tau3_model_identity.v1",
            "model_id": "fixture/base",
            "revision": "a" * 40,
        },
    )
    return {
        "dataset": dataset,
        "manifest": manifest,
        "protocol": protocol,
        "identity": identity,
    }


def write_candidate_internal_validation(
    root: Path,
    candidate_id: str,
    receipt_path: Path,
    sources: dict[str, Path],
) -> dict[str, dict[str, str]]:
    receipt = read_json(receipt_path)
    out = receipt_path.parent / "internal-validation"
    out.mkdir()
    dataset_rows = [
        json.loads(line)
        for line in sources["dataset"].read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    measurements = out / "measurements.jsonl"
    measurement_rows = []
    for index, row in enumerate(dataset_rows):
        metadata = row["metadata"]
        counts = metadata["token_counts"]
        targets = counts["input_token_ids"][counts["prompt_tokens"] :]
        mean_loss = 0.2 + index / 100
        measurement_rows.append(
            {
                "row_index": index,
                "row_sha256": canonical_sha256(row),
                "domain": metadata["domain"],
                "behavior": metadata["behavior"],
                "prompt_tokens": counts["prompt_tokens"],
                "supervised_tokens": counts["supervised_tokens"],
                "input_token_ids_sha256": canonical_sha256(
                    counts["input_token_ids"]
                ),
                "target_tokens_sha256": canonical_sha256(targets),
                "mean_loss": mean_loss,
                "loss_sum": mean_loss * counts["supervised_tokens"],
                "finite": True,
            }
        )
    write_jsonl(measurements, measurement_rows)
    run_binding = out / "run-binding.json"
    write_json(
        run_binding,
        _expected_run_binding(
            dataset_file=sources["dataset"],
            dataset_manifest_file=sources["manifest"],
            receipt_file=receipt_path,
            adapter_tree_sha256=receipt["adapter"]["tree_sha256"],
            protocol_file=sources["protocol"],
            identity_file=sources["identity"],
            max_seq_length=64,
        ),
    )
    artifact = out / "internal-validation.json"
    build_tau3_internal_validation(
        dataset_path=sources["dataset"],
        measurements_path=measurements,
        run_binding_path=run_binding,
        training_receipt_path=receipt_path,
        protocol_path=sources["protocol"],
        model_identity_path=sources["identity"],
        output_path=artifact,
        max_seq_length=64,
        created_at="2026-07-23T00:45:00Z",
    )
    return {
        "artifact": ref_for(root, artifact),
        "dataset": ref_for(root, sources["dataset"]),
        "protocol": ref_for(root, sources["protocol"]),
        "model_identity": ref_for(root, sources["identity"]),
    }


def write_development_qualification(root: Path, candidate_id: str, training_receipt_path: Path) -> tuple[Path, Path]:
    receipt = read_json(training_receipt_path)
    receipt_sha = sha256_file(training_receipt_path)
    adapter_sha = receipt["adapter"]["tree_sha256"]
    bindings = {
        "training_receipt_sha256": receipt_sha,
        "adapter_tree_sha256": adapter_sha,
        "candidate_identity_sha256": "7" * 64,
        "harness_sha256": "4" * 64,
        "protocol_sha256": receipt["training_binding"]["protocol"]["sha256"],
        "grid_sha256": "5" * 64,
        "base_identity_sha256": "6" * 64,
        "evaluator_model_contract_sha256": "8" * 64,
    }
    grid, trials, replayed_metrics = development_trial_grid()
    grid["evaluator_model_contract_sha256"] = "8" * 64
    safety = {
        "provable": True,
        "blocking_reasons": [],
    }
    evaluation = {
        "schema_version": "hfr.tau3_development_evaluation.v1",
        "created_at": "2026-07-23T00:46:00Z",
        "mode": "development",
        "passed": True,
        "tau_revision": "a" * 40,
        "bindings": bindings,
        "harness": {"passed": True, "identity_sha256": "4" * 64},
        "source_artifacts": {
            "adapter": {"manifest_sha256": "9" * 64},
            "base": {"manifest_sha256": "a" * 64},
        },
        "pairing": {
            "passed": True,
            "key_fields": ["domain", "task_sha256", "trial", "seed"],
            "paired_count": len(trials),
            "domain_counts": {
                "airline": 4,
                "retail": 4,
                "telecom": 4,
            },
            "pair_set_sha256": canonical_sha256(
                [
                    {
                        "domain": trial["domain"],
                        "seed": trial["seed"],
                        "task_sha256": trial["task_sha256"],
                    }
                    for trial in trials
                ]
            ),
        },
        "development_grid": grid,
        "development_trials": trials,
        "metrics": {
            "macro_pass1": replayed_metrics["macro_pass1"],
            "per_domain_pass1": {
                "adapter": replayed_metrics["per_domain_pass1"]["adapter"],
                "base": {
                    "airline": 0.0,
                    "retail": 0.0,
                    "telecom": 0.0,
                },
            },
            "safety": safety,
        },
        "effects": {"base": {}},
        "checks": [{"id": "fixture", "passed": True, "details": True}],
        "failed_check_count": 0,
        "blocking_reasons": [],
        "public_payload_scan": {"passed": True},
    }
    evaluation["public_payload_scan"]["report_sha256"] = canonical_sha256(
        evaluation
    )
    evaluation_path = root / "training" / candidate_id / "development-evaluation.json"
    write_json(evaluation_path, evaluation)
    scorecard = {
        "schema_version": "hfr.tau3_development_scorecard.v1",
        "schema_checked": True,
        "created_at": "2026-07-23T00:47:00Z",
        "passed": True,
        "completed": True,
        "bindings": bindings,
        "frozen_contract": {
            "harness_sha256": "4" * 64,
            "protocol_sha256": "c" * 64,
            "grid_sha256": "5" * 64,
            "base_identity_sha256": "6" * 64,
            "evaluator_model_contract_sha256": "8" * 64,
        },
        "development_evaluation": {
            **ref_for(root, evaluation_path),
            "size": evaluation_path.stat().st_size,
        },
        "metrics": {
            "macro_pass1": 1.0,
            "per_domain_pass1": {
                "airline": 1.0,
                "retail": 1.0,
                "telecom": 1.0,
            },
            "adapter_base_macro_gain": 1.0,
        },
        "blockers": {
            "safety": 0,
            "context_overflow": 0,
            "harness_mismatch": 0,
            "evaluator_mismatch": 0,
            "threshold": 0,
        },
    }
    scorecard_path = root / "training" / candidate_id / "development-scorecard.json"
    probes_path = root / "training" / candidate_id / "behavior-probes.json"
    write_json(scorecard_path, scorecard)
    write_behavior_probes(root, probes_path, bindings)
    return scorecard_path, probes_path


def development_trial_grid() -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    seeds = [101, 202]
    tasks_by_domain = {domain: [sha_text(f"{domain}-task-{index}") for index in range(2)] for domain in ("airline", "retail", "telecom")}
    trials: list[dict[str, Any]] = []
    adapter_total = 0
    base_total = 0
    per_domain_adapter = {domain: 0 for domain in ("airline", "retail", "telecom")}
    per_domain_count = {domain: 0 for domain in ("airline", "retail", "telecom")}
    for domain, tasks in tasks_by_domain.items():
        for task_sha256 in tasks:
            for seed in seeds:
                result = {
                    "adapter_pass1": True,
                    "base_pass1": False,
                    "domain": domain,
                    "seed": seed,
                    "task_sha256": task_sha256,
                }
                trials.append(
                    {
                        **result,
                        "source_sha256": sha_text(f"source:{domain}:{task_sha256}:{seed}"),
                        "result_sha256": canonical_sha256(result),
                    }
                )
                adapter_total += 1
                per_domain_adapter[domain] += 1
                per_domain_count[domain] += 1
    count = len(trials)
    metrics = {
        "macro_pass1": {"adapter": adapter_total / count, "base": base_total / count},
        "per_domain_pass1": {
            "adapter": {
                domain: per_domain_adapter[domain] / per_domain_count[domain]
                for domain in ("airline", "retail", "telecom")
            }
        },
    }
    return {"domains": ["airline", "retail", "telecom"], "seeds": seeds, "tasks_by_domain": tasks_by_domain, "harness_sha256": "4" * 64}, trials, metrics


def write_behavior_probes(root: Path, probes_path: Path, bindings: dict[str, str]) -> None:
    refs = []
    for index, family in enumerate(REQUIRED_FAMILIES):
        expected = {"passed": True, "assertions": [{"id": "contains_any", "passed": True}]}
        result = {
            "schema_version": "hfr.tau3_behavior_probe_result.v1",
            "probe_id": f"{family}-fixture",
            "family": family,
            "sequence_index": index,
            "prompt": safe_text_record(f"Probe {family}"),
            "endpoint": endpoint_record(),
            "assertions": [{"type": "contains_any", "values": ["ok"]}],
            "observation": {
                "content": safe_text_record("ok"),
                "tool_calls": [],
                "tool_call_count": 0,
                "transport_error": None,
            },
            "expected_outcome": expected,
            "actual_outcome": expected,
            "expected_outcome_sha256": canonical_sha256(expected),
        }
        path = probes_path.parent / f"probe-{family}.json"
        write_json(path, result)
        refs.append({"path": path.name, "sha256": sha256_file(path)})
    probes = {
        "schema_version": "hfr.tau3_behavior_probes.v1",
        "passed": True,
        "bindings": {
            "training_receipt_sha256": bindings["training_receipt_sha256"],
            "adapter_tree_sha256": bindings["adapter_tree_sha256"],
            "harness_sha256": bindings["harness_sha256"],
            "protocol_sha256": bindings["protocol_sha256"],
            "grid_sha256": bindings["grid_sha256"],
        },
        "endpoint": endpoint_record(),
        "families": list(REQUIRED_FAMILIES),
        "probe_results": refs,
        "aggregate": {"total_probe_count": len(refs), "failed_probe_count": 0, "family_count": len(REQUIRED_FAMILIES)},
    }
    write_json(probes_path, probes)


def endpoint_record() -> dict[str, str]:
    return {"base_url_sha256": sha_text("http://127.0.0.1/v1"), "model_sha256": sha_text("local-agent"), "configuration_sha256": sha_text("fixture-config")}


def safe_text_record(text: str) -> dict[str, Any]:
    return {"sha256": sha_text(text), "redacted": text, "redacted_sha256": sha_text(text), "length": len(text)}


def build_final_evidence(root: Path, training_ref: dict[str, str], *, include_auth_replay_inputs: bool = True) -> dict[str, str]:
    training = read_json(root / training_ref["path"])
    selected_receipt_path = root / training["qualified_candidates"][0]["training_receipt"]["path"]
    selected_receipt = read_json(selected_receipt_path)
    selected_receipt_sha = sha256_file(selected_receipt_path)
    adapter_sha = selected_receipt["adapter"]["tree_sha256"]
    identity = {
        "schema_version": "hfr.tau3_candidate_identity.v1",
        "created_at": "2026-07-23T00:40:00Z",
        "candidate_id": "candidate-a",
        "training_receipt_sha256": selected_receipt_sha,
        "final_training_receipt_sha256": selected_receipt_sha,
        "adapter_tree_sha256": adapter_sha,
        "endpoint_model_sha256": "e" * 64,
        "training_binding": {
            "protocol_sha256": "c" * 64,
            "protocol_signature": "d" * 64,
            "model_freeze_sha256": "1" * 64,
            "recipe_space_sha256": "2" * 64,
            "mlx_qlora_plan_sha256": "3" * 64,
            "base_identity_sha256": "4" * 64,
            "base_tree_sha256": "5" * 64,
            "dataset_manifest_sha256": "6" * 64,
            "dataset_files_sha256": "7" * 64,
            "source_binding_sha256": "8" * 64,
            "recipe_sha256": "a" * 64,
        },
        "adapter_identity": {
            "adapter_tree_sha256": adapter_sha,
            "tree_sha256": adapter_sha,
            "file_count": 1,
            "adapter_weight_file_count": 1,
            "declared_file_set_sha256": "9" * 64,
            "replayed_file_set_sha256": "9" * 64,
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
    }
    identity_ref = write_artifact(root, "final/candidate-identity.json", identity)
    selection_ref = write_artifact(root, "final/candidate-selection.json", candidate_selection())
    lock_ref = write_artifact(root, "final/candidate-lock.json", candidate_lock(identity_ref["sha256"], selection_ref["sha256"], selected_receipt_sha, adapter_sha))
    protocol_path, sealed_source_path, auth_ref = write_sealed_authorization_graph(root, lock_ref, identity_ref["sha256"], adapter_sha)
    grid_ref = write_artifact(root, "final/sealed-grid.json", sealed_grid(auth_ref["sha256"], lock_ref["sha256"]))
    eval_ref = write_artifact(root, "final/sealed-evaluation.json", sealed_evaluation())
    promotion_ref = write_artifact(root, "final/promotion-preflight.json", promotion_preflight(root, lock_ref["sha256"]))
    auth_inputs = (
        {
            "sealed_authorization_validation": {
                "protocol": ref_for(root, protocol_path),
                "sealed_source_manifest": ref_for(root, sealed_source_path),
                "arm_id": "adapter",
                "seeds": [101, 202, 303, 404],
                "expected_tau_revision": "a" * 40,
            }
        }
        if include_auth_replay_inputs
        else {}
    )
    return write_artifact(
        root,
        "final-evidence.json",
        {
            "schema_version": FINAL_SCHEMA_VERSION,
            "candidate_locked_at": "2026-07-23T00:50:00Z",
            "sealed_started_at": "2026-07-23T01:00:00Z",
            "publication_preflight_at": "2026-07-23T02:00:00Z",
            "artifacts": {
                "candidate_selection": selection_ref,
                "candidate_identity": identity_ref,
                "candidate_lock": lock_ref,
                "sealed_authorization": auth_ref,
                "sealed_grid_completeness": grid_ref,
                "sealed_evaluation": eval_ref,
                "promotion_preflight": promotion_ref,
            },
            **auth_inputs,
        },
    )


def build_publication_evidence(root: Path) -> dict[str, str]:
    promotion_ref = ref_for(root, root / "final" / "promotion-preflight.json")
    post_path = root / "publication" / "post-publication.json"
    build_tau3_post_publication_record(
        preflight=root / promotion_ref["path"],
        hf_revision="2" * 40,
        out=post_path,
        created_at="2026-07-23T03:00:00Z",
    )
    redaction_ref = write_artifact(
        root,
        "publication/redaction.json",
        {"schema_version": "hfr.redaction.v1", "passed": True, "contains_sealed_payloads": False, "contains_credentials": False, "contains_private_paths": False},
    )
    parity_ref = write_artifact(
        root,
        "publication/source-parity.json",
        {
            "schema_version": "hfr.tau3_source_parity.v1",
            "passed": True,
            "github_revision": "1" * 40,
            "hf_revision": "2" * 40,
            "reviewed_evidence_source": {"github_revision": "1" * 40, "hf_revision": "2" * 40},
            "artifact_hash_bindings": {
                "post_publication_record_sha256": sha256_file(post_path),
                "promotion_preflight_sha256": promotion_ref["sha256"],
                "evidence_bundle_sha256": "3" * 64,
            },
        },
    )
    return write_artifact(
        root,
        "publication-preflight.json",
        {
            "schema_version": PUBLICATION_SCHEMA_VERSION,
            "artifacts": {
                "redaction": redaction_ref,
                "post_publication": ref_for(root, post_path),
                "promotion_preflight": promotion_ref,
                "source_parity": parity_ref,
            },
        },
    )


def upgrade_final_fixture_to_v2_stub(root: Path) -> dict[str, str]:
    final_path = root / "final-evidence.json"
    final = read_json(final_path)
    artifacts = final["artifacts"]
    lock_path = root / artifacts["candidate_lock"]["path"]
    lock = read_json(lock_path)
    training_protocol_sha256 = "1" * 64
    benchmark_protocol_sha256 = "2" * 64
    benchmark_protocol_lineage_sha256 = "3" * 64
    blind_custody_receipt_sha256 = "4" * 64
    lock.update(
        {
            "schema_version": "hfr.tau3_candidate_lock.v2",
            "evaluator_model_contract_sha256": "5" * 64,
            "training_protocol_sha256": training_protocol_sha256,
            "training_protocol_signature": training_protocol_sha256,
            "benchmark_protocol_sha256": benchmark_protocol_sha256,
            "benchmark_protocol_signature": benchmark_protocol_sha256,
            "benchmark_protocol_lineage_sha256": (
                benchmark_protocol_lineage_sha256
            ),
        }
    )
    lock.pop("protocol_sha256")
    lock.pop("protocol_signature")
    write_json(lock_path, lock)
    artifacts["candidate_lock"]["sha256"] = sha256_file(lock_path)

    authorization_path = root / artifacts["sealed_authorization"]["path"]
    authorization = read_json(authorization_path)
    authorization["schema_version"] = "hfr.tau3_sealed_authorization.v2"
    authorization["candidate_lock"] = {
        "sha256": artifacts["candidate_lock"]["sha256"],
        "created_at": lock["created_at"],
        "training_protocol_sha256": training_protocol_sha256,
        "training_protocol_signature": training_protocol_sha256,
        "benchmark_protocol_sha256": benchmark_protocol_sha256,
        "benchmark_protocol_signature": benchmark_protocol_sha256,
        "benchmark_protocol_lineage_sha256": (
            benchmark_protocol_lineage_sha256
        ),
        "sealed_access_authorized": True,
    }
    authorization["protocol_lineage"] = {
        "sha256": benchmark_protocol_lineage_sha256,
        "training_protocol_sha256": training_protocol_sha256,
        "benchmark_protocol_sha256": benchmark_protocol_sha256,
    }
    write_json(authorization_path, authorization)
    artifacts["sealed_authorization"]["sha256"] = sha256_file(
        authorization_path
    )

    grid_path = root / artifacts["sealed_grid_completeness"]["path"]
    grid = read_json(grid_path)
    grid["bindings"].update(
        {
            "authorization_sha256": artifacts["sealed_authorization"][
                "sha256"
            ],
            "candidate_lock_sha256": artifacts["candidate_lock"]["sha256"],
            "protocol_sha256": benchmark_protocol_sha256,
            "training_protocol_sha256": training_protocol_sha256,
            "benchmark_protocol_lineage_sha256": (
                benchmark_protocol_lineage_sha256
            ),
            "blind_custody_receipt_sha256": (
                blind_custody_receipt_sha256
            ),
        }
    )
    write_json(grid_path, grid)
    artifacts["sealed_grid_completeness"]["sha256"] = sha256_file(grid_path)

    promotion_path = root / artifacts["promotion_preflight"]["path"]
    promotion = read_json(promotion_path)
    promotion["evidence_bindings"]["candidate_lock"]["sha256"] = artifacts[
        "candidate_lock"
    ]["sha256"]
    write_json(promotion_path, promotion)
    artifacts["promotion_preflight"]["sha256"] = sha256_file(promotion_path)
    publication_path = root / "publication-preflight.json"
    publication = read_json(publication_path)
    post_path = root / publication["artifacts"]["post_publication"]["path"]
    post = read_json(post_path)
    post["preflight"]["sha256"] = artifacts["promotion_preflight"]["sha256"]
    post["record_sha256"] = canonical_sha256(
        {key: value for key, value in post.items() if key != "record_sha256"}
    )
    post_path.chmod(0o644)
    write_json(post_path, post)
    publication["artifacts"]["post_publication"]["sha256"] = sha256_file(
        post_path
    )
    publication["artifacts"]["promotion_preflight"]["sha256"] = artifacts[
        "promotion_preflight"
    ]["sha256"]
    parity_path = root / publication["artifacts"]["source_parity"]["path"]
    parity = read_json(parity_path)
    parity["artifact_hash_bindings"]["post_publication_record_sha256"] = (
        sha256_file(post_path)
    )
    parity["artifact_hash_bindings"]["promotion_preflight_sha256"] = artifacts[
        "promotion_preflight"
    ]["sha256"]
    write_json(parity_path, parity)
    publication["artifacts"]["source_parity"]["sha256"] = sha256_file(
        parity_path
    )
    write_json(publication_path, publication)

    input_refs: dict[str, dict[str, str]] = {}
    for key in (
        "training_protocol",
        "benchmark_protocol_lineage",
        "custody_receipt",
        "generator_validation",
        "fresh_contamination_replay",
    ):
        path = root / "final" / f"{key}.json"
        write_json(path, {"fixture": key})
        input_refs[key] = ref_for(root, path)
    final["sealed_authorization_validation"].update(input_refs)
    final["sealed_authorization_validation"][
        "retired_source_incident_sha256"
    ] = "f" * 64
    write_json(final_path, final)
    plan_path = root / "competitive_v3_plan.json"
    plan = read_json(plan_path)
    plan["evidence_refs"]["final"]["sha256"] = sha256_file(final_path)
    plan["evidence_refs"]["publication"]["sha256"] = sha256_file(
        publication_path
    )
    write_json(plan_path, plan)
    return {
        "training_protocol_sha256": training_protocol_sha256,
        "benchmark_protocol_lineage_sha256": (
            benchmark_protocol_lineage_sha256
        ),
        "blind_custody_receipt_sha256": blind_custody_receipt_sha256,
    }


def candidate_selection() -> dict[str, Any]:
    return {
        "schema_version": "hfr.tau3_candidate_selection.v2",
        "schema_checked": True,
        "created_at": "2026-07-23T00:40:00Z",
        "passed": True,
        "selected_candidate_id": "candidate-a",
        "selection_policy": {
            "minimum_qualified_candidates": 2,
            "distinct_qualified_recipes_required": True,
            "minimum_macro_pass1": 0.10,
            "minimum_macro_gain": 0.05,
            "minimum_per_domain_pass1": 0.05,
        },
        "base": {},
        "candidates": [
            {
                "candidate_id": "candidate-a",
                "eligible": True,
                "training_binding": {"recipe_sha256": "a" * 64},
                "metrics": {
                    "macro_pass1": {"candidate": 1.0, "base": 0.0},
                    "per_domain_pass1": {
                        "candidate": {
                            domain: 1.0 for domain in DOMAINS
                        }
                    },
                },
            },
            {
                "candidate_id": "candidate-b",
                "eligible": True,
                "training_binding": {"recipe_sha256": "b" * 64},
                "metrics": {
                    "macro_pass1": {"candidate": 1.0, "base": 0.0},
                    "per_domain_pass1": {
                        "candidate": {
                            domain: 1.0 for domain in DOMAINS
                        }
                    },
                },
            },
        ],
        "eligible_candidate_count": 2,
        "eligible_recipe_count": 2,
        "selection": {},
    }


def candidate_lock(identity_sha: str, selection_sha: str, receipt_sha: str, adapter_sha: str) -> dict[str, Any]:
    return {
        "schema_version": "hfr.tau3_candidate_lock.v1",
        "created_at": "2026-07-23T00:50:00Z",
        "selected_candidate_id_hash": "0" * 64,
        "candidate_identity_sha256": identity_sha,
        "development_selection_report_sha256": selection_sha,
        "development_benchmark_manifest_sha256": "1" * 64,
        "training_receipt_sha256": receipt_sha,
        "endpoint_model_sha256": "e" * 64,
        "adapter_tree_sha256": adapter_sha,
        "recipe_sha256": "a" * 64,
        "base_identity_sha256": "4" * 64,
        "base_tree_sha256": "5" * 64,
        "dataset_manifest_sha256": "6" * 64,
        "dataset_files_sha256": "7" * 64,
        "source_binding_sha256": "8" * 64,
        "protocol_sha256": "c" * 64,
        "protocol_signature": "d" * 64,
        "hashes_only": True,
        "sealed_access_authorized": True,
        "local_paths_included": False,
        "raw_payload_included": False,
    }


def write_sealed_authorization_graph(root: Path, lock_ref: dict[str, str], identity_sha: str, adapter_sha: str) -> tuple[Path, Path, dict[str, str]]:
    sealed_source = {
        "schema_version": "hfr.tau3_sealed_source_manifest.v1",
        "source_revision": "a" * 40,
        "hashes_only": True,
        "task_count": 100,
        "entries": [{"task_id_sha256": f"{index:064x}", "prompt_sha256": "1" * 64, "task_sha256": "2" * 64} for index in range(100)],
    }
    sealed_source_path = root / "final" / "sealed-source.json"
    write_json(sealed_source_path, sealed_source)
    protocol = {
        "schema_version": "hfr.tau3_protocol_config.v1",
        "protocol_manifest": {},
        "tau_revision": {"revision": "a" * 40, "split_hashes": {"sealed": sha256_file(sealed_source_path)}},
        "split_manifest": {"splits": {"sealed": {"sha256": sha256_file(sealed_source_path)}}},
        "harness_contract": {
            "domains": ["airline", "retail", "telecom"],
            "context_window": 16384,
            "turn_limit": 30,
            "retry_policy": "none",
            "test_time_search": False,
            "no_test_time_search": True,
            "decoding": {"temperature": 0.0, "top_p": 1.0, "max_output_tokens": 1024, "seeds": [101, 202, 303, 404]},
            "domain_contracts": {},
            "prompt_contract": {},
        },
        "model_freeze": {
            "base_model": {"name": "base", "revision": "base-rev", "local_identity_sha256": "4" * 64},
            "comparators": [
                {"name": "comp1", "revision": "comp1-rev", "local_identity_sha256": "5" * 64},
                {"name": "comp2", "revision": "comp2-rev", "local_identity_sha256": "6" * 64},
            ],
        },
        "budget": {"passed": True},
        "sealed_manifest": {"manifest_sha256": sha256_file(sealed_source_path), "access_count": 0},
        "mlx_qlora_plan": {},
        "recipe_space": {},
        "candidate_selection_contract": {"passed": True},
        "contamination_attestation": {"passed": True},
        "redaction_attestation": {"passed": True},
        "licenses": [{"status": "approved", "training_allowed": True} for _ in range(4)],
        "environment_manifest": {},
    }
    protocol_path = root / "final" / "sealed-protocol.json"
    write_json(protocol_path, protocol)
    lock = read_json(root / lock_ref["path"])
    lock["protocol_sha256"] = sha256_file(protocol_path)
    write_json(root / lock_ref["path"], lock)
    lock_ref["sha256"] = sha256_file(root / lock_ref["path"])
    auth_payload = sealed_authorization_payload(lock_ref["sha256"], lock, sha256_file(protocol_path), sha256_file(sealed_source_path), identity_sha, adapter_sha)
    auth_ref = write_artifact(root, "final/sealed-authorization.json", auth_payload)
    return protocol_path, sealed_source_path, auth_ref


def sealed_authorization_payload(lock_sha: str, lock: dict[str, Any], protocol_sha: str, sealed_sha: str, identity_sha: str, adapter_sha: str) -> dict[str, Any]:
    true_gates = {
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
    }
    return {
        "schema_version": "hfr.tau3_sealed_authorization.v1",
        "created_at": "2026-07-23T01:00:00Z",
        "authorized": True,
        "hashes_only": True,
        "local_paths_included": False,
        "raw_payload_included": False,
        "candidate_lock": {
            "sha256": lock_sha,
            "created_at": lock["created_at"],
            "protocol_sha256": protocol_sha,
            "protocol_signature": lock["protocol_signature"],
            "sealed_access_authorized": True,
        },
        "protocol": {
            "sha256": protocol_sha,
            "signature_sha256": lock["protocol_signature"],
            "signature_provenance": "candidate_lock.protocol_signature",
            "tau_revision": "a" * 40,
        },
        "sealed_source": {"manifest_sha256": sealed_sha, "task_count": 100, "hashes_only": True},
        "frozen_contract": {
            "arms": ["adapter", "base", "comparator_1", "comparator_2"],
            "seeds": [101, 202, 303, 404],
            "domains": ["airline", "retail", "telecom"],
            "context_window": 16384,
            "tool_contract_sha256": canonical_sha256({
                "agent": "llm_agent",
                "user": "user_simulator",
                "auto_review": True,
                "review_mode": "full",
                "communication_protocol_enforced": True,
                "max_retries": 0,
                "hallucination_retries": 0,
                "test_time_search": False,
                "domain_contracts_sha256": canonical_sha256({}),
            }),
            "prompt_context_decoding_sha256": canonical_sha256({
                "context_window": 16384,
                "turn_limit": 30,
                "temperature": 0.0,
                "top_p": 1.0,
                "max_output_tokens": 1024,
                "seeds": [101, 202, 303, 404],
                "prompt_contract_sha256": canonical_sha256({}),
            }),
            "harness_sha256": canonical_sha256({
                "domains": ["airline", "retail", "telecom"],
                "context_window": 16384,
                "temperature": 0.0,
                "top_p": 1.0,
                "max_output_tokens": 1024,
                "seeds": [101, 202, 303, 404],
                "turn_limit": 30,
                "retry_policy": "none",
                "test_time_search": False,
                "no_test_time_search": True,
            }),
            "no_test_time_search": True,
        },
        "model_identity_refs": {
            "candidate_identity_sha256": identity_sha,
            "adapter_tree_sha256": adapter_sha,
            "endpoint_model_sha256": "e" * 64,
            "base_identity_sha256": "4" * 64,
            "comparator_1_identity_sha256": "5" * 64,
            "comparator_2_identity_sha256": "6" * 64,
            "equivalence_refs_hash": canonical_sha256({
                "candidate_identity_sha256": identity_sha,
                "adapter_tree_sha256": adapter_sha,
                "base": {"name_sha256": sha_text("base"), "revision_sha256": sha_text("base-rev"), "local_identity_sha256": "4" * 64},
                "comparator_1": {"name_sha256": sha_text("comp1"), "revision_sha256": sha_text("comp1-rev"), "local_identity_sha256": "5" * 64},
                "comparator_2": {"name_sha256": sha_text("comp2"), "revision_sha256": sha_text("comp2-rev"), "local_identity_sha256": "6" * 64},
            }),
        },
        "gates": true_gates,
        "budget": {"sha256": canonical_sha256({"passed": True}), "declared": True, "passed": True},
    }


def sealed_grid(auth_sha: str, lock_sha: str) -> dict[str, Any]:
    seed_counts = {str(seed): 100 for seed in (101, 202, 303, 404)}
    return {
        "schema_version": "hfr.tau3_sealed_grid_completeness.v1",
        "created_at": "2026-07-23T01:20:00Z",
        "passed": True,
        "status": "complete",
        "hashes_only": True,
        "local_paths_included": False,
        "raw_payload_included": False,
        "scores_included": False,
        "bindings": {
            "authorization_sha256": auth_sha,
            "candidate_lock_sha256": lock_sha,
            "protocol_sha256": "c" * 64,
            "sealed_source_sha256": "f" * 64,
            "coverage_fingerprint_sha256": "1" * 64,
            "arm_manifest_sha256": {arm: "2" * 64 for arm in ("adapter", "base", "comparator_1", "comparator_2")},
            "model_identity_sha256": {arm: "3" * 64 for arm in ("adapter", "base", "comparator_1", "comparator_2")},
            "harness_equivalence_sha256": "4" * 64,
        },
        "counts": {
            "arm_count": 4,
            "seed_count": 4,
            "domain_count": 3,
            "sealed_task_count": 100,
            "episodes_per_arm": 400,
            "total_episodes": 1600,
            "per_arm_seed_task_count": {arm: seed_counts for arm in ("adapter", "base", "comparator_1", "comparator_2")},
        },
        "gates": {key: True for key in (
            "arms_exact",
            "seeds_exact",
            "domains_exact",
            "sealed_task_count_exact",
            "no_duplicate_task_rows",
            "no_missing_task_rows",
            "no_extra_task_rows",
            "same_task_coverage_across_arms",
            "same_task_coverage_across_seeds",
            "result_hashes_replayed",
            "authorization_binding_replayed",
            "candidate_lock_binding_replayed",
            "protocol_binding_replayed",
            "sealed_source_binding_replayed",
            "harness_equivalence_bound",
            "model_identities_bound",
            "public_payload_safe",
        )},
    }


def sealed_authorization(lock_sha: str, identity_sha: str, adapter_sha: str) -> dict[str, Any]:
    true_gates = {
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
    }
    return {
        "schema_version": "hfr.tau3_sealed_authorization.v1",
        "created_at": "2026-07-23T01:00:00Z",
        "authorized": True,
        "hashes_only": True,
        "local_paths_included": False,
        "raw_payload_included": False,
        "candidate_lock": {
            "sha256": lock_sha,
            "created_at": "2026-07-23T00:50:00Z",
            "protocol_sha256": "c" * 64,
            "protocol_signature": "d" * 64,
            "sealed_access_authorized": True,
        },
        "protocol": {
            "sha256": "c" * 64,
            "signature_sha256": "d" * 64,
            "signature_provenance": "candidate_lock.protocol_signature",
            "tau_revision": "a" * 40,
        },
        "sealed_source": {"manifest_sha256": "f" * 64, "task_count": 100, "hashes_only": True},
        "frozen_contract": {
            "arms": ["adapter", "base", "comparator_1", "comparator_2"],
            "seeds": [101, 202, 303, 404],
            "domains": ["airline", "retail", "telecom"],
            "context_window": 16384,
            "tool_contract_sha256": "1" * 64,
            "prompt_context_decoding_sha256": "2" * 64,
            "harness_sha256": "3" * 64,
            "no_test_time_search": True,
        },
        "model_identity_refs": {
            "candidate_identity_sha256": identity_sha,
            "adapter_tree_sha256": adapter_sha,
            "endpoint_model_sha256": "e" * 64,
            "base_identity_sha256": "4" * 64,
            "comparator_1_identity_sha256": "5" * 64,
            "comparator_2_identity_sha256": "6" * 64,
            "equivalence_refs_hash": "7" * 64,
        },
        "gates": true_gates,
        "budget": {"sha256": "8" * 64, "declared": True, "passed": True},
    }


def sealed_evaluation() -> dict[str, Any]:
    arms = ("adapter", "base", "comparator_1", "comparator_2")
    domains = {"airline": 0.2, "retail": 0.2, "telecom": 0.2}
    zero_counts = {arm: 0 for arm in arms}
    zero_rates = {arm: 0.0 for arm in arms}
    effect = {
        "mean_difference": 0.01,
        "confidence_level": 0.95,
        "confidence_interval": {"lower": 0.0, "upper": 0.1},
        "bootstrap_samples": 200,
        "bootstrap_seed": 7,
    }
    ref_effect = {
        "domain_stratified_macro_pass1": effect
        | {
            "mean_difference": 0.1,
            "confidence_interval": {"lower": 0.01, "upper": 0.2},
            "resampling_unit": "domain_stratified_task",
            "domain_means": {"airline": 0.1, "retail": 0.1, "telecom": 0.1},
        },
        "paired_pass1": effect,
        "per_domain_non_inferiority_passed": True,
        "per_domain_pass1": {"airline": effect, "retail": effect, "telecom": effect},
        "primary_improvement_passed": True,
    }
    return {
        "schema_version": "hfr.tau3_evaluation.v1",
        "created_at": "2026-07-23T01:40:00Z",
        "mode": "sealed",
        "passed": True,
        "promotion_ready": True,
        "readiness": "ready_for_publication_review",
        "analysis_config": {
            "required_arms": ["adapter", "base", "comparator_1", "comparator_2"],
            "reference_arms": ["base", "comparator_1", "comparator_2"],
            "required_domains": ["airline", "retail", "telecom"],
            "primary_metric": "macro_pass1",
            "bootstrap_samples": 200,
            "bootstrap_seed": 7,
            "confidence_level": 0.95,
            "non_inferiority_margin": 0.05,
            "safety_non_inferiority_margin": 0.01,
            "harness_equivalence": "normalized_exact_excluding_local_ports_api_keys_and_agent_model",
        },
        "tau_revision": "a" * 40,
        "harness": {"passed": True, "normalized_sha256": "1" * 64, "arm_sha256": {}, "normalized_by_domain": None},
        "source_artifacts": {arm: [{"path": f"{arm}.json", "sha256": "2" * 64, "public_safe": True}] for arm in arms},
        "pairing": {
            "passed": True,
            "key_fields": ["domain", "task_sha256", "trial", "seed"],
            "paired_count": 12,
            "domain_counts": {"airline": 4, "retail": 4, "telecom": 4},
            "pair_set_sha256": "3" * 64,
        },
        "metrics": {
            "macro_pass1": {"adapter": 0.20, "base": 0.10, "comparator_1": 0.23, "comparator_2": 0.18},
            "per_domain_pass1": {arm: domains for arm in arms},
            "safety": {
                "provable": True,
                "definition": "fixture safety aggregate",
                "missing_db_evidence_counts": zero_counts,
                "missing_policy_review_counts": zero_counts,
                "harmful_mutation_counts": zero_counts,
                "harmful_mutation_rates": zero_rates,
                "policy_violation_counts": zero_counts,
                "policy_violation_rates": zero_rates,
                "blocking_reasons": [],
            },
        },
        "effects": {
            "base": ref_effect,
            "comparator_1": ref_effect,
            "comparator_2": ref_effect,
        },
        "per_task_hashed": [],
        "checks": [
            {"id": "fixture", "passed": True, "details": {}},
            {"id": "safety_non_inferiority_vs_base", "passed": True, "details": {}},
        ],
        "failed_check_count": 0,
        "blocking_reasons": [],
        "public_payload_scan": {"passed": True, "forbidden_key_count": 1, "forbidden_text_pattern_count": 4, "report_sha256": "4" * 64},
    }


def promotion_preflight(root: Path, lock_sha: str) -> dict[str, Any]:
    def binding(path: Path, schema: str) -> dict[str, Any]:
        return {"sha256": sha256_file(path), "size": path.stat().st_size, "schema_version": schema, "schema_passed": True}

    scratch: dict[str, Path] = {}
    for name in (
        "sealed_public_evaluation_report",
        "sealed_grid_completeness",
        "sealed_authorization",
        "postlock_attempt_ledger",
        "protocol_lineage_attestation",
        "readiness_validation",
        "budget_evidence",
        "license_evidence",
        "contamination_evidence",
        "redaction_evidence",
    ):
        path = root / "final" / f"{name}.binding.json"
        write_json(path, {"schema_version": "hfr.validation.v1", "passed": True})
        scratch[name] = path
    bindings = {
        name: binding(path, "hfr.validation.v1") for name, path in scratch.items()
    }
    bindings["candidate_lock"] = {"sha256": lock_sha, "size": 1, "schema_version": "hfr.tau3_candidate_lock.v1", "schema_passed": True}
    return {
        "schema_version": "hfr.tau3_promotion_publication_preflight.v1",
        "created_at": "2026-07-23T02:00:00Z",
        "allowed": True,
        "publication_status": "ready_for_publication",
        "hf_revision": None,
        "hashes_only": True,
        "local_paths_included": False,
        "raw_payload_included": False,
        "private_identifiers_included": False,
        "negative_result_withheld_honestly": False,
        "failed_predicate_count": 0,
        "blocking_reasons": [],
        "promotion_predicates": {"fixture": True},
        "notes": ["fixture promotion preflight"],
        "decision_sha256": "b" * 64,
        "evidence_bindings": bindings,
    }


def ref_for(root: Path, path: Path) -> dict[str, str]:
    return {"path": path.relative_to(root).as_posix(), "sha256": sha256_file(path)}


def canonical_sha256(value: Any) -> str:
    return __import__("hashlib").sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def sha_text(value: str) -> str:
    return __import__("hashlib").sha256(value.encode("utf-8")).hexdigest()


def old_dataset_evidence() -> dict[str, Any]:
    return {
        "schema_version": DATASET_SCHEMA_VERSION,
        "dataset_sha256": "2" * 64,
        "content_addressed_rows": True,
        "coverage_complete": True,
        "diversity_complete": True,
        "split_hashes_disjoint": True,
        "sealed_payload_access_count": 0,
        "sealed_payload_fields_materialized": [],
        "redaction_passed": True,
        "splits": {
            "train": {"domains": ["airline", "retail", "telecom"]},
            "internal_validation": {"domains": ["airline", "retail", "telecom"]},
        },
        "telecom_training_example_fraction": 0.26,
        "telecom_supervised_token_fraction": 0.27,
        "domain_supervised_token_fractions": {"airline": 0.34, "retail": 0.39, "telecom": 0.27},
        "max_domain_canonical_target_duplication_fraction": 0.10,
        "behavior_coverage": {
            domain: {"behaviors": list(competitive_v3_module.BEHAVIORS)}
            for domain in ("airline", "retail", "telecom")
        },
        "coverage_counts_are_supervised_targets": True,
    }


def training_evidence() -> dict[str, Any]:
    candidate = {
        "receipt_sha256": "3" * 64,
        "all_rows_seen": True,
        "effective_epochs": 2.0,
        "effective_batch_examples": 4,
    }
    return {
        "schema_version": TRAINING_SCHEMA_VERSION,
        "deterministic_exposure_receipts": True,
        "all_rows_seen_at_least_once": True,
        "effective_stratified_epochs": 2.0,
        "all_exposure_rows_replayed": True,
        "qualified_candidate_count": 2,
        "separate_candidate_and_infra_budgets": True,
        "no_required_stratum_skipped": True,
        "qualified_candidates": [
            {"candidate_id": "a", **candidate},
            {"candidate_id": "b", **candidate, "receipt_sha256": "4" * 64},
        ],
    }


def final_evidence() -> dict[str, Any]:
    return {
        "schema_version": FINAL_SCHEMA_VERSION,
        "immutable_lock_chronology": True,
        "post_lock_mutation": False,
        "one_shot_sealed_comparison": True,
        "sealed_arms": ["adapter", "base", "comparator_1", "comparator_2"],
        "identical_harness_all_arms": True,
        "sealed_access_after_lock_only": True,
        "sealed_run_count": 1,
        "candidate_locked_at": "2026-07-23T00:00:00Z",
        "sealed_started_at": "2026-07-23T01:00:00Z",
        "publication_preflight_at": "2026-07-23T02:00:00Z",
        "claims": {
            "adapter_beats_base": True,
            "adapter_minus_base_macro_pass1_ci": {"lower": 0.01, "upper": 0.10},
            "strongest_comparator_gap": -0.04,
            "per_domain_noninferiority": True,
            "safety_passed": True,
            "unsupported_competitive_claims": [],
        },
    }


def publication_evidence() -> dict[str, Any]:
    return {
        "schema_version": PUBLICATION_SCHEMA_VERSION,
        "redaction_passed": True,
        "contains_sealed_payloads": False,
        "contains_credentials": False,
        "contains_private_paths": False,
        "competitive_claim_replays_from_public_evidence": True,
        "claims_fail_closed": True,
    }


def write_artifact(root: Path, relative: str, payload: Any) -> dict[str, str]:
    path = root / relative
    if isinstance(payload, str):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding="utf-8")
    else:
        write_json(path, payload)
    return {"path": relative, "sha256": sha256_file(path)}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = __import__("hashlib").sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
