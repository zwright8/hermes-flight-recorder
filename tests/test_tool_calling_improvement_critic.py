from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_validator():
    path = ROOT / "scripts" / "validate_tool_calling_improvement.py"
    spec = importlib.util.spec_from_file_location("validate_tool_calling_improvement", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


VALIDATOR = _load_validator()


class ToolCallingImprovementCriticTests(unittest.TestCase):
    def test_missing_evidence_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = VALIDATOR.validate(Path(directory))

        self.assertFalse(result["passed"])
        self.assertTrue(result["blocking_reasons"])
        self.assertTrue(all(reason.startswith("missing required artifact:") for reason in result["blocking_reasons"]))

    def test_strict_rescore_rejects_semantic_suffixes_and_refusal_substrings(self) -> None:
        tool_task = {
            "expected": {
                "kind": "tool_call",
                "tool_name": "atlas_probe",
                "arguments": {"record_key": "ROW-1"},
            }
        }
        completion = '<tool_call>{"name":"atlas_probe","arguments":{"record_key":"ROW-1"}}</tool_call>'
        self.assertTrue(VALIDATOR._strict_completion_passes(tool_task, completion + "<|im_end|>"))
        self.assertFalse(VALIDATOR._strict_completion_passes(tool_task, completion + " extra"))

        refusal_task = {"expected": {"kind": "refusal", "text": "POLICY_REFUSAL"}}
        self.assertTrue(VALIDATOR._strict_completion_passes(refusal_task, "POLICY_REFUSAL<|endoftext|>"))
        self.assertFalse(VALIDATOR._strict_completion_passes(refusal_task, "prefix POLICY_REFUSAL suffix"))


if __name__ == "__main__":
    unittest.main()
