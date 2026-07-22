from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "validate_runtime_adapter_router.py"


class RuntimeAdapterRouterCaseStudyTests(unittest.TestCase):
    def test_case_study_validator_proves_routes_and_dispatch_boundary(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(SCRIPT)],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        result = json.loads(completed.stdout)
        self.assertTrue(result["passed"])
        self.assertTrue(result["actual_evaluation_schema_passed"])
        self.assertFalse(result["actual_candidate_evaluation_passed"])
        self.assertEqual(result["actual_promotion_eligible_candidate_count"], 0)
        self.assertEqual(result["specialist_candidate"], "demo/browser-specialist")
        self.assertEqual(result["fallback_candidate"], "demo/generalist")
        self.assertEqual(result["read_dispatch"], "dispatched")
        self.assertEqual(result["denied_write"], "approval_missing")
        self.assertFalse(result["denied_write_handler_called"])
        self.assertEqual(result["authorized_write"], "dispatched")
        self.assertEqual(result["authorized_write_handler_count"], 1)


if __name__ == "__main__":
    unittest.main()
