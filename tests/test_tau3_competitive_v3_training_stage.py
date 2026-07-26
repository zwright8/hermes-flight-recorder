from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from flightrecorder.schema_registry import check_schema_contract
from flightrecorder.tau3_candidate_identity import (
    build_tau3_candidate_identity,
)
from flightrecorder.tau3_candidate_selection import (
    _training_receipt_eligible,
)
from flightrecorder.tau3_competitive_v3_training_stage import (
    Tau3CompetitiveV3TrainingStageError,
    stage_tau3_competitive_v3_training_run,
)
from flightrecorder.tau3_mlx_training import (
    Tau3MlxTrainingConfig,
    _run_process_segments,
    validate_tau3_process_segments,
)
from tests.test_tau3_competitive_v3 import (
    sha256_file,
    write_json,
    write_training_receipt,
)
from tests.tau3_process_segments_fixture import (
    write_process_segments_fixture,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "stage_tau3_competitive_v3_training_run.py"


class Tau3CompetitiveV3TrainingStageTests(unittest.TestCase):
    def test_real_segment_chain_replays_after_staging_and_can_qualify(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle = root / "bundle"
            bundle.mkdir()
            run = segmented_completed_run_fixture(root / "source")

            stage_tau3_competitive_v3_training_run(
                bundle,
                candidate_id="candidate-a",
                training_run=run,
            )

            staged = bundle / "training/candidates/candidate-a/run"
            receipt_path = staged / "training_receipt.json"
            receipt = read_json(receipt_path)
            replay = validate_tau3_process_segments(
                receipt["process_segments"],
                output_dir=staged,
                expected_config=receipt["config"],
            )
            self.assertTrue(replay["passed"], replay)
            eligibility = _training_receipt_eligible(
                receipt,
                receipt_path,
            )
            self.assertTrue(eligibility["passed"], eligibility)
            identity = build_tau3_candidate_identity(
                candidate_id="candidate-a",
                training_receipt_path=receipt_path,
                endpoint_model="local/candidate-a",
                output_path=root / "candidate-identity.json",
            )
            self.assertEqual(identity["candidate_id"], "candidate-a")
            self.assertFalse(
                (staged / "adapter").stat().st_mode & 0o222
            )
            self.assertFalse(
                (
                    staged / "process_segments/segments"
                ).stat().st_mode
                & 0o222
            )

    def test_resumed_segment_chain_preserves_partial_evidence_after_staging(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle = root / "bundle"
            bundle.mkdir()
            run = segmented_completed_run_fixture(
                root / "source",
                resumed=True,
            )

            stage_tau3_competitive_v3_training_run(
                bundle,
                candidate_id="candidate-a",
                training_run=run,
            )

            staged = bundle / "training/candidates/candidate-a/run"
            receipt = read_json(staged / "training_receipt.json")
            partials = receipt["process_segments"]["recovery"][
                "preserved_partial_artifact_trees"
            ]
            self.assertEqual(len(partials), 1)
            partial = staged / partials[0]["path"]
            self.assertTrue(partial.is_dir())
            self.assertFalse(partial.stat().st_mode & 0o222)
            replay = validate_tau3_process_segments(
                receipt["process_segments"],
                output_dir=staged,
                expected_config=receipt["config"],
            )
            self.assertTrue(replay["passed"], replay)

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

    def test_segmented_run_stages_only_validated_chain_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle = root / "bundle"
            bundle.mkdir()
            run = completed_run_fixture(root / "source")
            receipt_path = run / "training_receipt.json"
            receipt = read_json(receipt_path)
            receipt["config"]["process_segment_iters"] = 400
            receipt["process_segments"] = write_process_segments_fixture(run)
            write_json(receipt_path, receipt)

            with mock.patch(
                "flightrecorder.tau3_competitive_v3_training_stage.validate_tau3_process_segments",
                return_value={"passed": True, "errors": []},
            ):
                result = stage_tau3_competitive_v3_training_run(
                    bundle,
                    candidate_id="candidate-a",
                    training_run=run,
                )

            staged = bundle / "training/candidates/candidate-a/run"
            self.assertTrue(result["passed"])
            self.assertTrue((staged / "process_segments/plan.json").is_file())
            self.assertTrue(
                (
                    staged
                    / "process_segments/segments/segment-0001"
                    / "optimizer_state.safetensors"
                ).is_file()
            )

    def test_segmented_run_rejects_chain_replay_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle = root / "bundle"
            bundle.mkdir()
            run = completed_run_fixture(root / "source")
            receipt_path = run / "training_receipt.json"
            receipt = read_json(receipt_path)
            receipt["config"]["process_segment_iters"] = 400
            receipt["process_segments"] = write_process_segments_fixture(run)
            write_json(receipt_path, receipt)

            with (
                mock.patch(
                    "flightrecorder.tau3_competitive_v3_training_stage.validate_tau3_process_segments",
                    return_value={
                        "passed": False,
                        "errors": ["optimizer_state_output sha256 mismatch"],
                    },
                ),
                self.assertRaisesRegex(
                    Tau3CompetitiveV3TrainingStageError,
                    "process segment chain does not replay",
                ),
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


def segmented_completed_run_fixture(
    root: Path,
    *,
    resumed: bool = False,
) -> Path:
    run = completed_run_fixture(root)
    shutil.rmtree(run / "adapter")
    (run / "telemetry.jsonl").unlink()
    config = Tau3MlxTrainingConfig(
        iters=4,
        dropout=0.0,
        grad_accumulation=2,
        save_every=2,
        report_every=2,
        grad_checkpoint=False,
        disable_compile=True,
        prefix_cache_training=True,
        exposure_ledger_training=True,
        process_segment_iters=2,
        timeout_seconds=30,
    )
    command = [
        "python",
        "--adapter-path",
        str(run / "adapter"),
    ]
    losses: dict[str, list[float]] = {
        "train": [],
        "validation": [],
    }
    if resumed:
        calls = 0

        def interrupt_second(**kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                return _successful_segment_child(**kwargs)
            raise KeyboardInterrupt

        with mock.patch(
            "flightrecorder.tau3_mlx_training._run_child",
            side_effect=interrupt_second,
        ):
            try:
                _run_process_segments(
                    command=command,
                    cwd=root,
                    output_dir=run,
                    final_adapter_dir=run / "adapter",
                    aggregate_telemetry_path=run / "telemetry.jsonl",
                    cfg=config,
                    losses=losses,
                )
            except KeyboardInterrupt:
                pass
            else:
                raise AssertionError("fixture interruption did not occur")
        losses = {"train": [], "validation": []}
        with mock.patch(
            "flightrecorder.tau3_mlx_training._run_child",
            side_effect=_successful_segment_child,
        ):
            result = _run_process_segments(
                command=command,
                cwd=root,
                output_dir=run,
                final_adapter_dir=run / "adapter",
                aggregate_telemetry_path=run / "telemetry.jsonl",
                cfg=config,
                losses=losses,
                resume=True,
            )
    else:
        with mock.patch(
            "flightrecorder.tau3_mlx_training._run_child",
            side_effect=_successful_segment_child,
        ):
            result = _run_process_segments(
                command=command,
                cwd=root,
                output_dir=run,
                final_adapter_dir=run / "adapter",
                aggregate_telemetry_path=run / "telemetry.jsonl",
                cfg=config,
                losses=losses,
            )
    receipt_path = run / "training_receipt.json"
    receipt = read_json(receipt_path)
    receipt["config"].update(
        {
            "iters": config.iters,
            "process_segment_iters": config.process_segment_iters,
            "grad_accumulation": config.grad_accumulation,
            "report_every": config.report_every,
            "dropout": config.dropout,
        }
    )
    receipt["checks"] = [
        {
            "id": check_id,
            "passed": True,
            "actual": True,
            "expected": True,
        }
        for check_id in (
            "protocol_schema_passed",
            "recipe_within_protocol_recipe_space",
            "mixture_manifest_protocol_sha_matches",
            "mixture_no_sealed_or_test_rows",
        )
    ]
    receipt["process_segments"] = result["process_segments"]
    receipt["telemetry"] = {
        **result["process_segments"]["aggregate_telemetry"],
        "event_count": result["telemetry_event_count"],
    }
    receipt["losses"] = losses
    receipt["adapter"] = result["process_segments"]["final_adapter"]
    receipt["adapter_weight_file_count"] = sum(
        record.get("kind") == "adapter"
        for record in receipt["adapter"]["files"]
    )
    binding = receipt["training_binding"]
    binding["protocol"].update(
        {
            "model_freeze_sha256": "1" * 64,
            "recipe_space_sha256": "2" * 64,
            "mlx_qlora_plan_sha256": "3" * 64,
        }
    )
    binding["model"] = {
        "identity_sha256": "4" * 64,
        "tree_sha256": "5" * 64,
    }
    binding["dataset"] = {
        "manifest_sha256": "6" * 64,
        "files_sha256": "7" * 64,
        "source_binding_sha256": "8" * 64,
    }
    write_json(receipt_path, receipt)
    return run


def _successful_segment_child(**kwargs):
    command = kwargs["command"]
    adapter_dir = Path(command[command.index("--adapter-path") + 1])
    end = int(command[command.index("--hfr-child-segment-end") + 1])
    (adapter_dir / "adapter_config.json").write_text(
        "{}\n",
        encoding="utf-8",
    )
    (adapter_dir / "adapters.safetensors").write_bytes(
        f"adapter-{end}".encode()
    )
    (adapter_dir / f"{end:07d}_adapters.safetensors").write_bytes(
        f"checkpoint-{end}".encode()
    )
    optimizer = Path(
        command[
            command.index(
                "--hfr-child-segment-optimizer-state-output"
            )
            + 1
        ]
    )
    optimizer.write_bytes(f"optimizer-{end}".encode())
    loss = 1.0 / end
    kwargs["losses"]["train"].append(loss)
    kwargs["telemetry_path"].write_text(
        json.dumps(
            {
                "time": "2026-07-26T00:00:00Z",
                "stream": "stdout",
                "text": f"Iter {end}: Train loss {loss}",
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0, False, 1, 100 + end


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
