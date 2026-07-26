from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from flightrecorder.schema_registry import check_schema_contract
from flightrecorder.tau3_competitive_v3_training_stage import (
    Tau3CompetitiveV3TrainingStageError,
    stage_tau3_competitive_v3_training_run,
)
from tests.test_tau3_competitive_v3 import (
    sha256_file,
    write_json,
    write_training_receipt,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "stage_tau3_competitive_v3_training_run.py"


class Tau3CompetitiveV3TrainingStageTests(unittest.TestCase):
    def test_stages_only_governed_completed_run_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle = root / "bundle"
            bundle.mkdir()
            run = completed_run_fixture(root / "source")

            result = stage_tau3_competitive_v3_training_run(
                bundle,
                candidate_id="candidate-a",
                training_run=run,
            )
            staged = bundle / "training/candidates/candidate-a/run"

            self.assertTrue(result["passed"])
            self.assertFalse(result["raw_logs_included"])
            self.assertEqual(result["staged_file_count"], 5)
            self.assertTrue((staged / "training_receipt.json").is_file())
            self.assertTrue((staged / "prelaunch_receipt.json").is_file())
            self.assertTrue((staged / "telemetry.jsonl").is_file())
            self.assertTrue((staged / "mlx_lora_config.json").is_file())
            self.assertTrue(
                (staged / "adapter/adapter_model.safetensors").is_file()
            )
            self.assertFalse((staged / "child.stdout.log").exists())
            schema_result = check_schema_contract(
                read_json(staged / "staging_receipt.json"),
                name_or_id="tau3_competitive_v3_training_run_stage",
            )
            self.assertTrue(schema_result["passed"], schema_result["errors"])
            self.assertEqual(
                result["training_receipt"]["sha256"],
                sha256_file(staged / "training_receipt.json"),
            )

    def test_refuses_nonterminal_training_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle = root / "bundle"
            bundle.mkdir()
            run = completed_run_fixture(root / "source")
            receipt_path = run / "training_receipt.json"
            receipt = read_json(receipt_path)
            receipt["terminal_status"] = "crash"
            receipt["weights_updated"] = False
            write_json(receipt_path, receipt)

            with self.assertRaisesRegex(
                Tau3CompetitiveV3TrainingStageError,
                "final, successful",
            ):
                stage_tau3_competitive_v3_training_run(
                    bundle,
                    candidate_id="candidate-a",
                    training_run=run,
                )

    def test_refuses_tampered_adapter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle = root / "bundle"
            bundle.mkdir()
            run = completed_run_fixture(root / "source")
            (run / "adapter/adapter_model.safetensors").write_bytes(
                b"tampered"
            )

            with self.assertRaisesRegex(
                Tau3CompetitiveV3TrainingStageError,
                "fingerprint does not replay",
            ):
                stage_tau3_competitive_v3_training_run(
                    bundle,
                    candidate_id="candidate-a",
                    training_run=run,
                )

    def test_refuses_to_overwrite_staged_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle = root / "bundle"
            bundle.mkdir()
            run = completed_run_fixture(root / "source")
            stage_tau3_competitive_v3_training_run(
                bundle,
                candidate_id="candidate-a",
                training_run=run,
            )

            with self.assertRaisesRegex(
                Tau3CompetitiveV3TrainingStageError,
                "refusing to overwrite",
            ):
                stage_tau3_competitive_v3_training_run(
                    bundle,
                    candidate_id="candidate-a",
                    training_run=run,
                )

    def test_refuses_symlinked_source_artifact_escape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle = root / "bundle"
            bundle.mkdir()
            run = completed_run_fixture(root / "source")
            outside = root / "outside-telemetry.jsonl"
            outside.write_text('{"event":"outside"}\n', encoding="utf-8")
            telemetry = run / "telemetry.jsonl"
            telemetry.unlink()
            telemetry.symlink_to(outside)
            receipt_path = run / "training_receipt.json"
            receipt = read_json(receipt_path)
            receipt["telemetry"]["sha256"] = sha256_file(outside)
            receipt["telemetry"]["event_count"] = 1
            write_json(receipt_path, receipt)

            with self.assertRaisesRegex(
                Tau3CompetitiveV3TrainingStageError,
                "escapes its allowed root",
            ):
                stage_tau3_competitive_v3_training_run(
                    bundle,
                    candidate_id="candidate-a",
                    training_run=run,
                )

    def test_cli_emits_staging_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle = root / "bundle"
            bundle.mkdir()
            run = completed_run_fixture(root / "source")

            proc = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--bundle",
                    str(bundle),
                    "--candidate-id",
                    "candidate-a",
                    "--training-run",
                    str(run),
                ],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            result = json.loads(proc.stdout)

            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertTrue(result["passed"])
            self.assertEqual(result["candidate_id"], "candidate-a")


def completed_run_fixture(root: Path) -> Path:
    exposure_receipt = root / "exposure-receipt.json"
    exposure_ledger = root / "exposure-ledger.jsonl"
    exposure_receipt.parent.mkdir(parents=True)
    write_json(exposure_receipt, {"passed": True})
    exposure_ledger.write_text("{}\n", encoding="utf-8")
    receipt_path = write_training_receipt(
        root,
        "candidate-a",
        "a" * 64,
        exposure_receipt,
        exposure_ledger,
    )
    run = receipt_path.parent
    write_json(run / "mlx_lora_config.json", {"rank": 16})
    (run / "telemetry.jsonl").write_text(
        '{"event":"start"}\n{"event":"complete"}\n',
        encoding="utf-8",
    )
    (run / "child.stdout.log").write_text(
        "raw log must not be staged\n",
        encoding="utf-8",
    )
    receipt = read_json(receipt_path)
    prelaunch = {
        **receipt,
        "phase": "prelaunch",
        "terminal_status": "prelaunch",
        "weights_updated": False,
    }
    for field in (
        "adapter",
        "adapter_weight_file_count",
        "elapsed_seconds",
        "exit_code",
        "interrupted",
        "losses",
        "mlx_lora_config",
        "peak_child_rss_kb",
        "prelaunch_receipt",
        "schema_checked",
        "telemetry",
        "timed_out",
    ):
        prelaunch.pop(field, None)
    write_json(run / "prelaunch_receipt.json", prelaunch)
    receipt.update(
        {
            "schema_checked": True,
            "prelaunch_receipt": file_record(
                run / "prelaunch_receipt.json",
            ),
            "mlx_lora_config": file_record(
                run / "mlx_lora_config.json",
            ),
            "telemetry": {
                "path": "telemetry.jsonl",
                "sha256": sha256_file(run / "telemetry.jsonl"),
                "event_count": 2,
            },
        }
    )
    write_json(receipt_path, receipt)
    return run


def file_record(path: Path) -> dict[str, object]:
    return {
        "path": path.name,
        "sha256": sha256_file(path),
        "read_only": True,
    }


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


if __name__ == "__main__":
    unittest.main()
