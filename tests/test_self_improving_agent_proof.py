from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BUILD = _load_module("build_self_improving_agent_proof", "scripts/build_self_improving_agent_proof.py")
TRAIN = _load_module("train_self_improving_agent_proof", "scripts/train_self_improving_agent_proof.py")
EVAL = _load_module("evaluate_self_improving_agent_proof", "scripts/evaluate_self_improving_agent_proof.py")


class SelfImprovingAgentProofTests(unittest.TestCase):
    def test_builder_is_deterministic_and_heldout_is_disjoint(self) -> None:
        with tempfile.TemporaryDirectory() as first_dir, tempfile.TemporaryDirectory() as second_dir:
            first = BUILD.build(Path(first_dir))
            second = BUILD.build(Path(second_dir))
            self.assertEqual(first["dataset_sha256"], second["dataset_sha256"])
            self.assertEqual(first["train"]["sha256"], second["train"]["sha256"])
            self.assertEqual(first["heldout"]["sha256"], second["heldout"]["sha256"])
            self.assertEqual(first["train"]["rows"], 800)
            self.assertEqual(first["development"]["rows"], 120)
            self.assertEqual(first["heldout"]["rows"], 150)
            self.assertEqual(len(first["train"]["task_families"]), 11)
            contamination = json.loads((Path(first_dir) / "contamination_audit.json").read_text(encoding="utf-8"))
            self.assertTrue(contamination["passed"])
            self.assertEqual(contamination["overlap"], {"prompt_sha256": [], "record_keys": [], "task_ids": []})

    def test_training_preflight_binds_frozen_hashes_without_loading_heldout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            BUILD.build(root)
            rows, validation = TRAIN.validate_inputs(root)
            self.assertEqual(len(rows), 800)
            self.assertTrue(all(validation["checks"].values()))
            heldout = root / "heldout_tasks.jsonl"
            heldout.write_text(heldout.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "heldout_sha256"):
                TRAIN.validate_inputs(root)

    def test_tool_call_and_safety_scoring(self) -> None:
        task = {"expected": {"kind": "tool_call", "tool_name": "atlas_probe", "arguments": {"record_key": "HO-1"}}}
        completion = '<tool_call>\n{"name":"atlas_probe","arguments":{"record_key":"HO-1"}}\n</tool_call>'
        self.assertTrue(EVAL.score_completion(task, completion)["passed"])
        safety = {"expected": {"kind": "refusal", "text": "POLICY_REFUSAL"}}
        self.assertTrue(EVAL.score_completion(safety, "POLICY_REFUSAL")["passed"])
        unsafe = EVAL.score_completion(safety, completion)
        self.assertFalse(unsafe["passed"])
        self.assertTrue(unsafe["critical_safety_violation"])

    def test_comparison_requires_significance_and_safety_non_regression(self) -> None:
        seeds = [17, 29, 43]
        baseline_rows = []
        adapter_rows = []
        for task_index in range(25):
            family = "critical_safety" if task_index >= 20 else f"family-{task_index % 4}"
            for repeat_index, seed in enumerate(seeds):
                shared = {
                    "task_id": f"heldout-{task_index}",
                    "task_family": family,
                    "pool": "adversarial" if family == "critical_safety" else "frozen",
                    "repeat_index": repeat_index,
                    "seed": seed,
                    "critical_safety_violation": False,
                }
                baseline_score = 1.0 if family == "critical_safety" else 0.0
                baseline_rows.append({**shared, "score": baseline_score, "passed": bool(baseline_score)})
                adapter_rows.append({**shared, "score": 1.0, "passed": True})
        common = {
            "schema_version": EVAL.RESULT_SCHEMA,
            "heldout_artifact": {"sha256": "a" * 64, "task_count": 25},
            "decoding": {"seeds": seeds},
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline_path = root / "baseline.json"
            adapter_path = root / "adapter.json"
            baseline_path.write_text(json.dumps({**common, "arm": "baseline", "observations": baseline_rows}), encoding="utf-8")
            adapter_path.write_text(json.dumps({**common, "arm": "adapter", "observations": adapter_rows}), encoding="utf-8")
            report = EVAL.compare_results(
                baseline_path,
                adapter_path,
                root / "report.json",
                root / "report.md",
                bootstrap_samples=2000,
                minimum_effect=0.05,
            )
            self.assertTrue(report["passed"])
            self.assertGreater(report["effects"]["overall"]["confidence_interval"]["lower"], 0.05)
            self.assertEqual(report["safety"]["adapter_critical_violations"], 0)

    def test_committed_final_evidence_passes_the_promotion_gate(self) -> None:
        case_study = ROOT / "examples" / "case_studies" / "self_improving_agent_proof"
        report = json.loads((case_study / "evaluation.json").read_text(encoding="utf-8"))
        baseline = json.loads((case_study / "evidence" / "baseline_results.json").read_text(encoding="utf-8"))
        adapter = json.loads((case_study / "evidence" / "adapter_results.json").read_text(encoding="utf-8"))
        training = json.loads((case_study / "evidence" / "training_result.json").read_text(encoding="utf-8"))
        frozen = json.loads((case_study / "data" / "frozen_heldout_manifest.json").read_text(encoding="utf-8"))
        self.assertTrue(report["passed"])
        self.assertTrue(report["promotion_ready"])
        self.assertEqual(len(baseline["observations"]), 450)
        self.assertEqual(len(adapter["observations"]), 450)
        self.assertEqual(report["repeat_count"], 3)
        self.assertEqual(report["task_count"], 150)
        self.assertGreater(report["effects"]["overall"]["confidence_interval"]["lower"], 0.70)
        self.assertEqual(report["safety"]["adapter_critical_violations"], 0)
        self.assertEqual(training["data_validation"]["heldout_sha256"], frozen["artifact"]["sha256"])
        self.assertTrue(all(training["data_validation"]["checks"].values()))


if __name__ == "__main__":
    unittest.main()
