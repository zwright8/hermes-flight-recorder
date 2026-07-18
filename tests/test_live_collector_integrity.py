import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from flightrecorder.validation import validate_artifacts
from scripts import collect_live_hermes_data as collector


ROOT = Path(__file__).resolve().parents[1]


class LiveCollectorIntegrityTests(unittest.TestCase):
    def test_canonical_export_is_called_once_and_not_post_processed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            export_dir = root / "export"
            export_dir.mkdir()
            action_path = export_dir / "action_sft.jsonl"
            canonical = b'{"schema_version":"hfr.rl.action_sft.v1"}\n'
            action_path.write_bytes(canonical)
            manifest = {
                "action_sft_count": 1,
                "artifact_fingerprints": {
                    "action_sft": {
                        "path": "action_sft.jsonl",
                        "exists": True,
                        "size_bytes": len(canonical),
                        "sha256": hashlib.sha256(canonical).hexdigest(),
                    }
                },
            }

            with mock.patch.object(collector, "export_rl_dataset", return_value=manifest) as exporter:
                returned, integrity = collector.export_canonical_training_views(
                    root / "runs",
                    export_dir,
                    metadata={"collection": "fixture"},
                )

            exporter.assert_called_once()
            self.assertIs(returned, manifest)
            self.assertTrue(integrity["passed"], integrity)
            self.assertEqual(integrity["registered_row_count"], 1)
            self.assertEqual(integrity["actual_row_count"], 1)
            self.assertEqual(action_path.read_bytes(), canonical)

    def test_real_canonical_export_retains_native_actions_and_strict_integrity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = root / "runs"
            export_dir = root / "training_export"
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "flightrecorder",
                    "run",
                    "--scenario",
                    str(ROOT / "scenarios" / "email_reply_completion_good.json"),
                    "--out",
                    str(runs / "email"),
                ],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr + completed.stdout)

            manifest, integrity = collector.export_canonical_training_views(
                runs,
                export_dir,
                metadata={"collection": "live_collector_integrity_test"},
            )
            before = (export_dir / "action_sft.jsonl").read_bytes()
            validation = validate_artifacts(training_export_dir=export_dir, strict=True)

            self.assertEqual(manifest["action_sft_count"], 1)
            self.assertTrue((runs / "email" / "trajectory_v2.json").is_file())
            lineage = json.loads((runs / "email" / "artifact_lineage.json").read_text(encoding="utf-8"))
            self.assertIn("trajectory_v2", {row["name"] for row in lineage["outputs"]})
            self.assertTrue(integrity["passed"], integrity)
            self.assertEqual(integrity["actual_row_count"], 1)
            self.assertTrue(validation["passed"], validation)
            self.assertEqual((export_dir / "action_sft.jsonl").read_bytes(), before)
            row = json.loads(before)
            self.assertEqual(row["schema_version"], "hfr.rl.action_sft.v1")
            self.assertTrue(any(message.get("tool_calls") for message in row["messages"]))
            self.assertTrue(any(message.get("role") == "tool" for message in row["messages"]))
            self.assertTrue(row["tools"])

            lineage_path = runs / "email" / "artifact_lineage.json"
            original_lineage = lineage_path.read_bytes()
            scorecard_path = runs / "email" / "scorecard.json"
            trajectory_record = next(
                item for item in lineage["outputs"] if item["name"] == "trajectory_v2"
            )
            trajectory_record.update(
                {
                    "path": "scorecard.json",
                    "size_bytes": scorecard_path.stat().st_size,
                    "sha256": hashlib.sha256(scorecard_path.read_bytes()).hexdigest(),
                }
            )
            lineage_path.write_text(
                json.dumps(lineage, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            misbound_path_validation = validate_artifacts(runs_dir=runs, strict=True)
            self.assertFalse(misbound_path_validation["passed"])
            misbound_errors = "\n".join(
                error
                for target in misbound_path_validation["targets"]
                for error in target["errors"]
            )
            self.assertIn(
                "artifact_lineage.outputs.trajectory_v2.path must identify trajectory_v2.json",
                misbound_errors,
            )
            lineage_path.write_bytes(original_lineage)

            trajectory_path = runs / "email" / "trajectory_v2.json"
            trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
            trajectory.setdefault("metadata", {})["tampered_after_lineage"] = True
            trajectory_path.write_text(
                json.dumps(trajectory, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
            tampered_run_validation = validate_artifacts(runs_dir=runs, strict=True)
            self.assertFalse(tampered_run_validation["passed"])
            tamper_errors = "\n".join(
                error
                for target in tampered_run_validation["targets"]
                for error in target["errors"]
            )
            self.assertIn(
                "artifact_lineage.outputs.trajectory_v2.sha256 does not match the current file",
                tamper_errors,
            )
            self.assertIn(
                "artifact_lineage.outputs.trajectory_v2.size_bytes does not match the current file",
                tamper_errors,
            )

            retired_row = json.dumps({"schema_version": "hfr.live_hermes_action_sft.v1", "messages": []}) + "\n"
            (export_dir / "action_sft.jsonl").write_text(
                retired_row + retired_row,
                encoding="utf-8",
            )
            mutated_integrity = collector.registered_artifact_integrity(manifest, export_dir, "action_sft")
            mutated_validation = validate_artifacts(training_export_dir=export_dir, strict=True)
            self.assertFalse(mutated_integrity["passed"])
            self.assertIn("row count mismatch", mutated_integrity["failures"])
            self.assertIn("size mismatch", mutated_integrity["failures"])
            self.assertIn("sha256 mismatch", mutated_integrity["failures"])
            self.assertFalse(mutated_validation["passed"])


if __name__ == "__main__":
    unittest.main()
