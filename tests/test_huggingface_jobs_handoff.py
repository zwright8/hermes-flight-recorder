import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "scripts" / "build_agentic_finetune_experiment.py"
PREPARE = ROOT / "scripts" / "prepare_huggingface_jobs_handoff.py"


class HuggingFaceJobsHandoffTests(unittest.TestCase):
    def test_handoff_is_immutable_revision_ready_without_network_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            experiment = root / "experiment"
            handoff_dir = root / "hf_jobs"
            model_manifest = root / "model.json"
            model_manifest.write_text(
                json.dumps(
                    {
                        "schema_version": "hfr.model_candidate.v1",
                        "model_id": "Qwen/Qwen3-4B-Instruct-2507",
                        "license": {"status": "approved", "allow_training": True},
                        "compatibility": {"chat_template": "messages_and_tools"},
                    }
                ),
                encoding="utf-8",
            )
            built = subprocess.run(
                [
                    sys.executable,
                    str(BUILD),
                    "--runs-dir",
                    str(ROOT / "examples" / "agentic_training"),
                    "--out",
                    str(experiment),
                ],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(built.returncode, 0, built.stderr + built.stdout)

            prepared = subprocess.run(
                [
                    sys.executable,
                    str(PREPARE),
                    "--experiment-dir",
                    str(experiment),
                    "--model-manifest",
                    str(model_manifest),
                    "--dataset-repo",
                    "test-user/hermes-agentic-data",
                    "--hub-model-id",
                    "test-user/hermes-agentic-adapter",
                    "--out",
                    str(handoff_dir),
                ],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(prepared.returncode, 0, prepared.stderr + prepared.stdout)
            handoff = json.loads((handoff_dir / "handoff.json").read_text(encoding="utf-8"))
            request = json.loads((handoff_dir / "job_request.template.json").read_text(encoding="utf-8"))
            payload_manifest = json.loads(
                (handoff_dir / "payload" / "dataset_training_manifest.json").read_text(encoding="utf-8")
            )
            self.assertFalse(handoff["submitted"])
            self.assertFalse(handoff["network_writes_performed"])
            self.assertTrue(handoff["submission"]["requires_explicit_paid_network_approval"])
            self.assertEqual(request["secrets"], {"HF_TOKEN": "$HF_TOKEN"})
            self.assertIn("REPLACE_WITH_DATASET_COMMIT", request["script"])
            self.assertIn("REPLACE_WITH_DATASET_COMMIT", request["script_args"])
            self.assertEqual(request["timeout"], "4h")
            self.assertFalse((handoff_dir / "job_request.json").exists())
            self.assertNotIn("source_manifest", payload_manifest)
            self.assertTrue((handoff_dir / "payload" / "runtime" / "train_agentic_lora.py").exists())
            self.assertTrue((handoff_dir / "payload" / "runtime" / "hf_job.py").exists())
            self.assertTrue(handoff["payload_fingerprints"])


if __name__ == "__main__":
    unittest.main()
