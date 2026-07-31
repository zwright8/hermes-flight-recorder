from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from flightrecorder.tau3_internal_validation import (
    _expected_run_binding,
    build_tau3_internal_validation,
)
from flightrecorder.tau3_competitive_v3_training_evidence import (
    Tau3CompetitiveV3TrainingEvidenceError,
    build_tau3_competitive_v3_training_evidence,
    validate_tau3_competitive_v3_training_evidence,
)
from tests.test_tau3_competitive_v3 import (
    build_complete_bundle,
    sha256_file,
)
from tests.test_tau3_prefix_equivalence import equivalence_fixture


ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = ("candidate-a", "candidate-b")


class Tau3CompetitiveV3TrainingEvidenceTests(unittest.TestCase):
    def test_builds_reference_only_qualified_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _prepare_conventional_bundle(root)

            result = build_tau3_competitive_v3_training_evidence(
                root,
                candidate_ids=list(CANDIDATES),
            )

            self.assertTrue(result["passed"])
            self.assertEqual(result["candidate_count"], 2)
            evidence = _read_json(root / "training-evidence.json")
            self.assertNotIn("exposure", evidence)
            self.assertEqual(
                [item["candidate_id"] for item in evidence["qualified_candidates"]],
                list(CANDIDATES),
            )
            for candidate in evidence["qualified_candidates"]:
                self.assertIn("exposure", candidate)
                self.assertNotIn("prefix_equivalence", candidate)
                self.assertEqual(
                    candidate["internal_validation"]["dataset"]["path"],
                    "dataset/valid.jsonl",
                )
            replay = validate_tau3_competitive_v3_training_evidence(
                root / "training-evidence.json"
            )
            self.assertTrue(replay["passed"])
            self.assertEqual(
                set(replay["qualified_candidates"]),
                set(CANDIDATES),
            )
            for bindings in replay["qualified_candidates"].values():
                self.assertEqual(
                    set(bindings),
                    {
                        "training_receipt_sha256",
                        "recipe_sha256",
                        "adapter_tree_sha256",
                    },
                )

    def test_rejects_fewer_and_duplicate_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _prepare_conventional_bundle(root)
            with self.assertRaisesRegex(
                Tau3CompetitiveV3TrainingEvidenceError,
                "at least two",
            ):
                build_tau3_competitive_v3_training_evidence(
                    root,
                    candidate_ids=["candidate-a"],
                )
            with self.assertRaisesRegex(
                Tau3CompetitiveV3TrainingEvidenceError,
                "distinct",
            ):
                build_tau3_competitive_v3_training_evidence(
                    root,
                    candidate_ids=["candidate-a", "candidate-a"],
                )

    def test_rejects_tampered_candidate_exposure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _prepare_conventional_bundle(root)
            ledger = (
                root
                / "training/candidates/candidate-b/exposure"
                / "training_exposure_ledger.jsonl"
            )
            ledger.write_text(
                ledger.read_text(encoding="utf-8") + '{"tampered":true}\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                Tau3CompetitiveV3TrainingEvidenceError,
                "replay failed",
            ):
                build_tau3_competitive_v3_training_evidence(
                    root,
                    candidate_ids=list(CANDIDATES),
                )
            self.assertFalse((root / "training-evidence.json").exists())

    def test_rejects_unsafe_nested_reference(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _prepare_conventional_bundle(root)
            scorecard_path = (
                root
                / "training/candidates/candidate-a/development"
                / "development-scorecard.json"
            )
            scorecard = _read_json(scorecard_path)
            scorecard["development_evaluation"]["path"] = (
                "sealed/development-evaluation.json"
            )
            _write_json(scorecard_path, scorecard)
            with self.assertRaisesRegex(
                Tau3CompetitiveV3TrainingEvidenceError,
                "unsafe/private/sealed/raw-log",
            ):
                build_tau3_competitive_v3_training_evidence(
                    root,
                    candidate_ids=list(CANDIDATES),
                )

    def test_rejects_outside_root_reference(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _prepare_conventional_bundle(root)
            scorecard_path = (
                root
                / "training/candidates/candidate-a/development"
                / "development-scorecard.json"
            )
            scorecard = _read_json(scorecard_path)
            scorecard["development_evaluation"]["path"] = "../escape.json"
            _write_json(scorecard_path, scorecard)
            with self.assertRaisesRegex(
                Tau3CompetitiveV3TrainingEvidenceError,
                "unsafe/private/sealed/raw-log",
            ):
                build_tau3_competitive_v3_training_evidence(
                    root,
                    candidate_ids=list(CANDIDATES),
                )

    def test_rejects_symlinked_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _prepare_conventional_bundle(root)
            probes = (
                root
                / "training/candidates/candidate-a/behavior"
                / "behavior-probes.json"
            )
            target = root / "training/candidate-a/behavior-probes.json"
            probes.unlink()
            probes.symlink_to(target)
            with self.assertRaisesRegex(
                Tau3CompetitiveV3TrainingEvidenceError,
                "symlink",
            ):
                build_tau3_competitive_v3_training_evidence(
                    root,
                    candidate_ids=list(CANDIDATES),
                )

    def test_full_gradient_rejects_dangling_prefix_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _prepare_conventional_bundle(root)
            prefix = (
                root
                / "training/candidates/candidate-a"
                / "prefix-equivalence.json"
            )
            prefix.symlink_to(root / "missing-prefix-equivalence.json")
            with self.assertRaisesRegex(
                Tau3CompetitiveV3TrainingEvidenceError,
                "must not include prefix equivalence",
            ):
                build_tau3_competitive_v3_training_evidence(
                    root,
                    candidate_ids=list(CANDIDATES),
                )

    def test_preserves_existing_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _prepare_conventional_bundle(root)
            output = root / "training-evidence.json"
            output.write_text('{"sentinel":true}\n', encoding="utf-8")
            with self.assertRaisesRegex(
                Tau3CompetitiveV3TrainingEvidenceError,
                "refusing to overwrite",
            ):
                build_tau3_competitive_v3_training_evidence(
                    root,
                    candidate_ids=list(CANDIDATES),
                )
            self.assertEqual(output.read_text(encoding="utf-8"), '{"sentinel":true}\n')

    def test_preserves_output_replaced_immediately_after_link(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _prepare_conventional_bundle(root)
            output = root / "training-evidence.json"
            real_link = os.link

            def replace_after_link(source: Path, destination: Path) -> None:
                real_link(source, destination)
                destination.unlink()
                destination.write_text('{"foreign":true}\n', encoding="utf-8")

            with mock.patch(
                "flightrecorder.tau3_competitive_v3_training_evidence.os.link",
                side_effect=replace_after_link,
            ):
                with self.assertRaisesRegex(
                    Tau3CompetitiveV3TrainingEvidenceError,
                    "replaced during atomic publication",
                ):
                    build_tau3_competitive_v3_training_evidence(
                        root,
                        candidate_ids=list(CANDIDATES),
                    )
            self.assertEqual(
                output.read_text(encoding="utf-8"),
                '{"foreign":true}\n',
            )

    def test_surfaces_owned_output_cleanup_failure_with_original_cause(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _prepare_conventional_bundle(root)
            output = root / "training-evidence.json"
            resolved_output = root.resolve() / "training-evidence.json"
            path_type = type(output)
            real_unlink = path_type.unlink

            def fail_output_cleanup(
                path: Path,
                missing_ok: bool = False,
            ) -> None:
                if path == resolved_output:
                    raise PermissionError("fixture cleanup denied")
                real_unlink(path, missing_ok=missing_ok)

            with mock.patch(
                "flightrecorder.tau3_competitive_v3_training_evidence."
                "_replay_evidence",
                side_effect=[
                    None,
                    Tau3CompetitiveV3TrainingEvidenceError(
                        "fixture post-write replay failure"
                    ),
                ],
            ), mock.patch.object(
                path_type,
                "unlink",
                new=fail_output_cleanup,
            ):
                with self.assertRaisesRegex(
                    Tau3CompetitiveV3TrainingEvidenceError,
                    "cleanup also failed",
                ) as caught:
                    build_tau3_competitive_v3_training_evidence(
                        root,
                        candidate_ids=list(CANDIDATES),
                    )
            self.assertIsInstance(
                caught.exception.__cause__,
                Tau3CompetitiveV3TrainingEvidenceError,
            )
            self.assertIn(
                "post-write replay failure",
                str(caught.exception.__cause__),
            )
            self.assertTrue(output.is_file())

    def test_valid_bound_detached_prefix_candidate_passes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _prepare_conventional_bundle(root)
            _make_candidate_detached(root, "candidate-a")

            result = build_tau3_competitive_v3_training_evidence(
                root,
                candidate_ids=list(CANDIDATES),
            )

            self.assertTrue(result["passed"])
            candidate = _read_json(root / "training-evidence.json")[
                "qualified_candidates"
            ][0]
            self.assertEqual(
                candidate["prefix_equivalence"]["path"],
                "training/candidates/candidate-a/prefix-equivalence.json",
            )

    def test_detached_prefix_candidate_rejects_missing_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _prepare_conventional_bundle(root)
            prefix = _make_candidate_detached(root, "candidate-a")
            prefix.unlink()

            with self.assertRaisesRegex(
                Tau3CompetitiveV3TrainingEvidenceError,
                "prefix equivalence file is missing",
            ):
                build_tau3_competitive_v3_training_evidence(
                    root,
                    candidate_ids=list(CANDIDATES),
                )

    def test_detached_prefix_candidate_rejects_tampered_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _prepare_conventional_bundle(root)
            prefix = _make_candidate_detached(root, "candidate-a")
            artifact = _read_json(prefix)
            artifact["passed"] = False
            _write_json(prefix, artifact)

            with self.assertRaisesRegex(
                Tau3CompetitiveV3TrainingEvidenceError,
                "replay failed",
            ):
                build_tau3_competitive_v3_training_evidence(
                    root,
                    candidate_ids=list(CANDIDATES),
                )

    def test_detached_prefix_candidate_rejects_wrong_binding(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _prepare_conventional_bundle(root)
            _make_candidate_detached(root, "candidate-a")
            receipt_path = (
                root
                / "training/candidates/candidate-a/run"
                / "training_receipt.json"
            )
            receipt = _read_json(receipt_path)
            receipt["training_binding"]["prefix_equivalence"]["sha256"] = (
                "f" * 64
            )
            _write_json(receipt_path, receipt)

            with self.assertRaisesRegex(
                Tau3CompetitiveV3TrainingEvidenceError,
                "bind prefix equivalence sha256",
            ):
                build_tau3_competitive_v3_training_evidence(
                    root,
                    candidate_ids=list(CANDIDATES),
                )

    def test_cli_emits_json_and_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _prepare_conventional_bundle(root)
            command = [
                sys.executable,
                str(
                    ROOT
                    / "scripts"
                    / "build_tau3_competitive_v3_training_evidence.py"
                ),
                "--bundle",
                str(root),
                "--candidate",
                "candidate-a",
                "--candidate",
                "candidate-b",
            ]
            completed = subprocess.run(
                command,
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue(json.loads(completed.stdout)["passed"])

            repeated = subprocess.run(
                command,
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(repeated.returncode, 2)
            self.assertFalse(json.loads(repeated.stdout)["passed"])


def _prepare_conventional_bundle(root: Path) -> None:
    build_complete_bundle(root, distinct_training_exposures=True)
    original = _read_json(root / "training-evidence.json")

    dataset = root / "dataset"
    evidence = root / "evidence"
    dataset.mkdir(exist_ok=True)
    evidence.mkdir(exist_ok=True)
    shutil.copy2(root / "exposure-dataset/train.jsonl", dataset / "train.jsonl")
    shutil.copy2(
        root / "internal-validation-source/valid.jsonl",
        dataset / "valid.jsonl",
    )
    shutil.copy2(
        root / "internal-validation-source/manifest.json",
        dataset / "manifest.json",
    )
    shutil.copy2(
        root / "internal-validation-source/protocol.json",
        evidence / "protocol.json",
    )
    shutil.copy2(
        root / "internal-validation-source/model-identity.json",
        evidence / "base-model-identity.json",
    )

    for candidate in original["qualified_candidates"]:
        candidate_id = candidate["candidate_id"]
        source_receipt = root / candidate["training_receipt"]["path"]
        destination = root / "training" / "candidates" / candidate_id
        run = destination / "run"
        run.mkdir(parents=True)
        shutil.copy2(source_receipt, run / "training_receipt.json")
        shutil.copytree(source_receipt.parent / "adapter", run / "adapter")

        exposure = destination / "exposure"
        exposure.mkdir()
        candidate_exposure = candidate["exposure"]
        shutil.copy2(
            root / candidate_exposure["receipt"]["path"],
            exposure / "training_exposure_receipt.json",
        )
        shutil.copy2(
            root / candidate_exposure["ledger"]["path"],
            exposure / "training_exposure_ledger.jsonl",
        )
        shutil.copy2(
            root / candidate_exposure["validation"]["path"],
            exposure / "validation.json",
        )

        internal = destination / "internal-validation"
        shutil.copytree(source_receipt.parent / "internal-validation", internal)

        development = destination / "development"
        development.mkdir()
        source_scorecard = root / candidate["development_scorecard"]["path"]
        scorecard = _read_json(source_scorecard)
        source_evaluation = root / scorecard["development_evaluation"]["path"]
        destination_evaluation = development / "development-evaluation.json"
        shutil.copy2(source_evaluation, destination_evaluation)
        scorecard["development_evaluation"].update(
            {
                "path": destination_evaluation.relative_to(root).as_posix(),
                "sha256": sha256_file(destination_evaluation),
                "size": destination_evaluation.stat().st_size,
            }
        )
        _write_json(development / "development-scorecard.json", scorecard)

        behavior = destination / "behavior"
        behavior.mkdir()
        source_probes = root / candidate["behavior_probes"]["path"]
        probes = _read_json(source_probes)
        shutil.copy2(source_probes, behavior / "behavior-probes.json")
        for ref in probes["probe_results"]:
            shutil.copy2(source_probes.parent / ref["path"], behavior / ref["path"])

    os.unlink(root / "training-evidence.json")


def _make_candidate_detached(root: Path, candidate_id: str) -> Path:
    candidate = root / "training" / "candidates" / candidate_id
    receipt_path = candidate / "run" / "training_receipt.json"
    receipt = _read_json(receipt_path)
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
            "max_seq_length": 64,
            "batch_size": 1,
            "grad_accumulation": 4,
            "mask_prompt": True,
            "seed": 101,
        }
    )
    receipt["training_binding"]["exposure"]["objective"].update(
        {"full_gradient": False, "detached_prefix": True}
    )
    receipt["training_binding"]["exposure"]["dataset"] = {
        "sha256": sha256_file(root / "dataset" / "train.jsonl")
    }

    artifact = equivalence_fixture()
    artifact["bindings"].update(
        {
            "dataset_file_sha256": sha256_file(
                root / "dataset" / "train.jsonl"
            ),
            "protocol_file_sha256": sha256_file(
                root / "evidence" / "protocol.json"
            ),
            "model_identity_file_sha256": sha256_file(
                root / "evidence" / "base-model-identity.json"
            ),
        }
    )
    artifact["bindings"]["recipe"].update(
        {
            "rank": 16,
            "scale": 32.0,
            "learning_rate": 1e-5,
            "num_layers": 8,
            "max_seq_length": 64,
            "batch_size": 1,
            "grad_accumulation": 4,
            "mask_prompt": True,
            "allowed_seeds": [101],
        }
    )
    prefix = candidate / "prefix-equivalence.json"
    _write_json(prefix, artifact)
    receipt["training_binding"]["prefix_equivalence"] = {
        "sha256": sha256_file(prefix),
        "validation_passed": True,
    }
    _write_json(receipt_path, receipt)
    receipt_sha256 = sha256_file(receipt_path)

    valid = root / "dataset" / "valid.jsonl"
    manifest = root / "dataset" / "manifest.json"
    protocol = root / "evidence" / "protocol.json"
    identity = root / "evidence" / "base-model-identity.json"
    internal = candidate / "internal-validation"
    run_binding = internal / "run-binding.json"
    _write_json(
        run_binding,
        _expected_run_binding(
            dataset_file=valid,
            dataset_manifest_file=manifest,
            receipt_file=receipt_path,
            adapter_tree_sha256=receipt["adapter"]["tree_sha256"],
            protocol_file=protocol,
            identity_file=identity,
            max_seq_length=64,
        ),
    )
    internal_artifact = internal / "internal-validation.json"
    internal_artifact.unlink()
    build_tau3_internal_validation(
        dataset_path=valid,
        measurements_path=internal / "measurements.jsonl",
        run_binding_path=run_binding,
        training_receipt_path=receipt_path,
        protocol_path=protocol,
        model_identity_path=identity,
        output_path=internal_artifact,
        max_seq_length=64,
        created_at="2026-07-23T00:45:00Z",
    )

    development = candidate / "development"
    scorecard_path = development / "development-scorecard.json"
    scorecard = _read_json(scorecard_path)
    evaluation_path = root / scorecard["development_evaluation"]["path"]
    evaluation = _read_json(evaluation_path)
    evaluation["bindings"]["training_receipt_sha256"] = receipt_sha256
    _write_json(evaluation_path, evaluation)
    scorecard["bindings"]["training_receipt_sha256"] = receipt_sha256
    scorecard["development_evaluation"].update(
        {
            "sha256": sha256_file(evaluation_path),
            "size": evaluation_path.stat().st_size,
        }
    )
    _write_json(scorecard_path, scorecard)

    probes_path = candidate / "behavior" / "behavior-probes.json"
    probes = _read_json(probes_path)
    probes["bindings"]["training_receipt_sha256"] = receipt_sha256
    _write_json(probes_path, probes)
    return prefix


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    unittest.main()
