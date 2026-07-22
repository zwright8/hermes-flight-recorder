from __future__ import annotations

import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "export_runtime_adapter_publication.py"
SPEC = importlib.util.spec_from_file_location(
    "export_runtime_adapter_publication_test_module", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
EXPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXPORT)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def evaluation_report(*, split: str, candidate: str, passing: bool) -> dict:
    task_id = f"{split}-task-1"
    passed = 1 if passing else 0
    failed = 0 if passing else 1
    metric = {"passed": passed, "failed": failed, "total": 1, "pass_rate": float(passed)}
    checks = [
        {"check_id": "tool_calls_functional_order", "passed": passing},
        {"check_id": "tool_calls_exact_order", "passed": False},
        {"check_id": "final_answer_exact", "passed": passing},
    ]
    return {
        "schema_version": "hfr.runtime_adapter_candidate_evaluation.v1",
        "base_model": {"id": "base", "revision": "revision"},
        "tokenizer": {"id": "base", "revision": "revision"},
        "chat_template": {"sha256": "a" * 64},
        "heldout": {"split": split, "sha256": "b" * 64, "row_count": 1},
        "candidate_reports": [
            {
                "candidate_id": candidate,
                "heldout_subset": {
                    "evaluation_scopes": ["browser"],
                    "row_count": 1,
                    "task_ids_sha256": "c" * 64,
                },
                "metrics": {
                    "overall": metric,
                    "check_pass_rates": {
                        "tool_calls_functional_order": metric,
                        "tool_calls_exact_order": {
                            "passed": 0,
                            "failed": 1,
                            "total": 1,
                            "pass_rate": 0.0,
                        },
                        "final_answer_exact": metric,
                    },
                    "critical_safety_failures": 0,
                },
                "promotion_eligible": passing,
                "scores": [
                    {
                        "task_id": task_id,
                        "passed": passing,
                        "checks": checks,
                    }
                ],
            }
        ],
    }


class RuntimeAdapterPublicationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.inputs = self.root / "inputs"
        self.inputs.mkdir()
        self.paths = {
            "development_adapter_report": self.inputs / "development_adapter.json",
            "development_base_report": self.inputs / "development_base.json",
            "sealed_adapter_report": self.inputs / "sealed_adapter.json",
            "sealed_base_report": self.inputs / "sealed_base.json",
            "training_result": self.inputs / "training_result.json",
            "trainer_state": self.inputs / "trainer_state.json",
            "validation": self.inputs / "validation.json",
            "dataset_manifest": self.inputs / "dataset_manifest.json",
            "model_manifest": self.inputs / "model_manifest.json",
            "adapter_config": self.inputs / "adapter_config.json",
            "chat_template": self.inputs / "chat_template.jinja",
            "output_dir": self.root / "public",
        }
        write_json(
            self.paths["development_adapter_report"],
            evaluation_report(split="development", candidate="adapter", passing=True),
        )
        write_json(
            self.paths["development_base_report"],
            evaluation_report(split="development", candidate="base", passing=False),
        )
        write_json(
            self.paths["sealed_adapter_report"],
            evaluation_report(split="sealed_final", candidate="adapter", passing=True),
        )
        write_json(
            self.paths["sealed_base_report"],
            evaluation_report(split="sealed_final", candidate="base", passing=False),
        )
        write_json(
            self.paths["training_result"],
            {
                "schema_version": "hfr.agentic_lora_training_result.v1",
                "status": "succeeded",
            },
        )
        write_json(
            self.paths["trainer_state"],
            {
                "log_history": [
                    {
                        "step": 1,
                        "epoch": 1.0,
                        "loss": 0.1,
                        "mean_token_accuracy": 1.0,
                        "learning_rate": 0.0001,
                        "grad_norm": 0.2,
                        "entropy": 0.01,
                        "num_tokens": 100,
                    }
                ]
            },
        )
        write_json(self.paths["dataset_manifest"], {"status": "safe"})
        write_json(
            self.paths["validation"],
            {"passed": True, "error_count": 0, "status": "passed"},
        )
        write_json(
            self.paths["model_manifest"],
            {
                "model_id": "Qwen/test-base",
                "source": {"revision": "pinned-revision"},
            },
        )
        write_json(
            self.paths["adapter_config"],
            {"base_model_name_or_path": "/Users/example/cache", "revision": None},
        )
        self.paths["chat_template"].write_text("{{ messages }}\n", encoding="utf-8")

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_exports_paired_reports_curve_and_checksums(self) -> None:
        result = EXPORT.export_bundle(**self.paths)

        self.assertEqual(result["development_task_count"], 1)
        self.assertEqual(result["sealed_task_count"], 1)
        output = self.paths["output_dir"]
        self.assertTrue((output / "metrics_summary.csv").is_file())
        self.assertTrue((output / "paired_task_scores.csv").is_file())
        self.assertTrue((output / "training_curve.csv").is_file())
        checksums = (output / "SHA256SUMS").read_text(encoding="utf-8")
        self.assertIn("sealed_adapter_evaluation.json", checksums)
        self.assertNotIn("SHA256SUMS", checksums)
        adapter_config = json.loads(
            (output / "adapter_config.json").read_text(encoding="utf-8")
        )
        self.assertEqual(adapter_config["base_model_name_or_path"], "Qwen/test-base")
        self.assertEqual(adapter_config["revision"], "pinned-revision")

    def test_rejects_mismatched_paired_task_identity(self) -> None:
        report = json.loads(
            self.paths["sealed_base_report"].read_text(encoding="utf-8")
        )
        report["candidate_reports"][0]["heldout_subset"][
            "task_ids_sha256"
        ] = "d" * 64
        write_json(self.paths["sealed_base_report"], report)

        with self.assertRaisesRegex(EXPORT.PublicationError, "subsets do not match"):
            EXPORT.export_bundle(**self.paths)

    def test_rejects_absolute_user_path_in_public_artifact(self) -> None:
        validation = copy.deepcopy(
            {
                "passed": True,
                "error_count": 0,
                "path": "/Users/example/private",
            }
        )
        write_json(self.paths["validation"], validation)

        with self.assertRaisesRegex(EXPORT.PublicationError, "forbidden text"):
            EXPORT.export_bundle(**self.paths)

    def test_rejects_failed_campaign_validation(self) -> None:
        write_json(
            self.paths["validation"],
            {"passed": False, "error_count": 1, "status": "failed"},
        )

        with self.assertRaisesRegex(EXPORT.PublicationError, "did not pass"):
            EXPORT.export_bundle(**self.paths)


if __name__ == "__main__":
    unittest.main()
