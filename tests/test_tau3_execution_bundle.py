import json
import os
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

from flightrecorder.tau3_evaluation import analyze_tau3_evaluation
from flightrecorder.tau3_execution_bundle import (
    ArmInput,
    CandidateInput,
    Tau3ExecutionBundleError,
    build_tau3_execution_bundle,
)
from flightrecorder.tau3_execution_validation import (
    validate_tau3_benchmark_result_bundle,
    validate_tau3_training_result_bundle,
)
from tests.test_tau3_execution_validation import ARMS, build_execution_bundle, read_json, sha256_file


class Tau3ExecutionBundleTests(unittest.TestCase):
    def test_assembles_bundle_that_passes_training_and_benchmark_validators(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            build_execution_bundle(source)
            genuine_report = self._write_genuine_public_report(source)

            out = root / "bundle"
            manifest = self._assemble(source, out, public_report=genuine_report)

            training_manifest = manifest["training"]
            self.assertIsInstance(training_manifest, dict)
            self.assertEqual(manifest["schema_version"], "hfr.tau3_execution_bundle.v1")
            self.assertEqual(training_manifest["selected_candidate_id"], "candidate-a")
            self.assertFalse(str(manifest).count(str(source)))
            self.assertTrue((out / "training" / "candidate-a" / "training_receipt.json").is_file())
            self.assertTrue((out / "benchmark" / "sealed" / "adapter" / "manifest.json").is_file())
            self.assertEqual(oct(stat.S_IMODE((out / "manifest.json").stat().st_mode)), "0o444")

            training = validate_tau3_training_result_bundle(out, strict=True)
            self.assertTrue(training["passed"], json.dumps(training, indent=2))

            benchmark = validate_tau3_benchmark_result_bundle(out, strict=True)
            self.assertTrue(benchmark["passed"], json.dumps(benchmark, indent=2))

    def test_rejects_nonempty_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            build_execution_bundle(source)
            out = root / "bundle"
            out.mkdir()
            (out / "leftover").write_text("x", encoding="utf-8")
            with self.assertRaisesRegex(Tau3ExecutionBundleError, "must be empty"):
                self._assemble(source, out)

    def test_rejects_duplicate_candidate_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            build_execution_bundle(source)
            with self.assertRaisesRegex(Tau3ExecutionBundleError, "duplicate candidate"):
                build_tau3_execution_bundle(
                    out_dir=root / "bundle",
                    flight_recorder_git_commit="a" * 40,
                    tracked_worktree_clean=True,
                    protocol=source / "protocol.json",
                    selected_candidate_id="candidate-a",
                    candidate_dirs=[
                        CandidateInput("candidate-a", source / "training" / "candidate-a"),
                        CandidateInput("candidate-a", source / "training" / "candidate-a"),
                    ],
                    candidate_selection_report=source / "candidate-selection-report.json",
                    candidate_lock=source / "candidate-lock.json",
                    development_arm_dirs=self._arms(source, "development"),
                    sealed_arm_dirs=self._arms(source, "sealed"),
                    public_report=source / "public-evaluation-report.json",
                    make_read_only=False,
                )

    def test_rejects_missing_selected_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            build_execution_bundle(source)
            with self.assertRaisesRegex(Tau3ExecutionBundleError, "selected candidate"):
                build_tau3_execution_bundle(
                    out_dir=root / "bundle",
                    flight_recorder_git_commit="a" * 40,
                    tracked_worktree_clean=True,
                    protocol=source / "protocol.json",
                    selected_candidate_id="missing",
                    candidate_dirs=[CandidateInput("candidate-a", source / "training" / "candidate-a")],
                    candidate_selection_report=source / "candidate-selection-report.json",
                    candidate_lock=source / "candidate-lock.json",
                    development_arm_dirs=self._arms(source, "development"),
                    sealed_arm_dirs=self._arms(source, "sealed"),
                    public_report=source / "public-evaluation-report.json",
                    make_read_only=False,
                )

    def test_rejects_bad_source_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            build_execution_bundle(source)
            with self.assertRaisesRegex(Tau3ExecutionBundleError, "source hash mismatch"):
                self._assemble(source, root / "bundle", expected_source_hashes={"protocol": "0" * 64})

    @unittest.skipIf(os.name == "nt", "symlink behavior differs on Windows")
    def test_rejects_symlink_in_source_tree(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            build_execution_bundle(source)
            target = source / "training" / "candidate-a" / "adapter" / "adapter_model.safetensors"
            link = source / "training" / "candidate-a" / "adapter" / "adapter-link.safetensors"
            link.symlink_to(target)
            with self.assertRaisesRegex(Tau3ExecutionBundleError, "symlink"):
                self._assemble(source, root / "bundle")

    def test_cli_builds_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            build_execution_bundle(source)
            out = root / "bundle"
            proc = subprocess.run(
                [
                    sys.executable,
                    "scripts/build_tau3_execution_bundle.py",
                    "--out",
                    str(out),
                    "--git-commit",
                    "a" * 40,
                    "--tracked-worktree-clean",
                    "--protocol",
                    str(source / "protocol.json"),
                    "--selected-candidate-id",
                    "candidate-a",
                    "--candidate",
                    f"candidate-a={source / 'training' / 'candidate-a'}",
                    "--candidate-selection-report",
                    str(source / "candidate-selection-report.json"),
                    "--candidate-lock",
                    str(source / "candidate-lock.json"),
                    "--development-arm",
                    f"base={source / 'benchmark' / 'development' / 'base'}",
                    "--development-arm",
                    f"adapter={source / 'benchmark' / 'development' / 'adapter'}",
                    "--sealed-arm",
                    f"adapter={source / 'benchmark' / 'sealed' / 'adapter'}",
                    "--sealed-arm",
                    f"base={source / 'benchmark' / 'sealed' / 'base'}",
                    "--sealed-arm",
                    f"comparator_1={source / 'benchmark' / 'sealed' / 'comparator_1'}",
                    "--sealed-arm",
                    f"comparator_2={source / 'benchmark' / 'sealed' / 'comparator_2'}",
                    "--public-report",
                    str(source / "public-evaluation-report.json"),
                    "--keep-writable",
                ],
                cwd=Path(__file__).resolve().parents[1],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr + proc.stdout)
            self.assertTrue((out / "manifest.json").is_file())

    def _assemble(
        self,
        source: Path,
        out: Path,
        *,
        public_report: Path | None = None,
        expected_source_hashes: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        return build_tau3_execution_bundle(
            out_dir=out,
            flight_recorder_git_commit="a" * 40,
            tracked_worktree_clean=True,
            protocol=source / "protocol.json",
            selected_candidate_id="candidate-a",
            candidate_dirs=[CandidateInput("candidate-a", source / "training" / "candidate-a")],
            candidate_selection_report=source / "candidate-selection-report.json",
            candidate_lock=source / "candidate-lock.json",
            development_arm_dirs=self._arms(source, "development"),
            sealed_arm_dirs=self._arms(source, "sealed"),
            public_report=public_report or source / "public-evaluation-report.json",
            expected_source_hashes=expected_source_hashes,
        )

    def _arms(self, source: Path, mode: str) -> list[ArmInput]:
        return [ArmInput(arm, source / "benchmark" / mode / arm) for arm in sorted(p.name for p in (source / "benchmark" / mode).iterdir())]

    def _write_genuine_public_report(self, source: Path) -> Path:
        arm_paths: dict[str, list[Path]] = {arm: [] for arm in ARMS}
        for arm in ARMS:
            manifest_path = source / "benchmark" / "sealed" / arm / "manifest.json"
            manifest = read_json(manifest_path)
            for receipt_ref in manifest["run_receipts"]:
                receipt = read_json(manifest_path.parent / receipt_ref["path"])
                arm_paths[arm].append(manifest_path.parent / receipt["result_path"])
        out = source / "public-evaluation-report-genuine.json"
        report = analyze_tau3_evaluation(
            arm_result_paths=arm_paths,
            out_path=out,
            mode="sealed",
            expected_tau_revision="a" * 40,
            created_at="2026-07-23T01:05:00Z",
            bootstrap_samples=200,
            bootstrap_seed=7,
        )
        self.assertEqual(sha256_file(out), sha256_file(out))
        self.assertFalse(report["passed"])
        return out


if __name__ == "__main__":
    unittest.main()
