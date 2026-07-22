#!/usr/bin/env python3
"""Validate the governed LoRA recipe-search mission and write its completion artifact."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

from flightrecorder.lora_recipe_search import validate_promotion_handoff, validate_search_result
from flightrecorder.repeated_eval import validate_promotion_evidence
from flightrecorder.schema_registry import check_schema_file
from run_lora_recipe_autoresearch_demo import run_demo


def validate_mission() -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    details: dict[str, Any] = {}
    try:
        with tempfile.TemporaryDirectory(prefix="hfr-autoresearch-validator-") as tmp:
            root = Path(tmp) / "demo"
            summary = run_demo(root)
            details = {
                key: summary[key]
                for key in (
                    "trial_count",
                    "kept_trial_count",
                    "discarded_trial_count",
                    "champion_trial_id",
                    "development_metric",
                    "promotion_ready",
                    "governance_ready",
                )
            }
            search = json.loads((root / "search_result.json").read_text(encoding="utf-8"))
            handoff = json.loads((root / "promotion_handoff.json").read_text(encoding="utf-8"))
            _add(checks, "demo_completed", summary["passed"], summary["passed"], True)
            _add(checks, "baseline_and_multiple_trials", summary["trial_count"] >= 3, summary["trial_count"], ">=3")
            _add(checks, "keep_decision_present", summary["kept_trial_count"] >= 2, summary["kept_trial_count"], ">=2 including baseline")
            _add(checks, "discard_decision_present", summary["discarded_trial_count"] >= 1, summary["discarded_trial_count"], ">=1")
            _add(
                checks,
                "development_only_selection",
                search["heldout_access"] == {"used_during_search": False, "artifact_count": 0, "artifacts": []},
                search["heldout_access"],
                {"used_during_search": False, "artifact_count": 0, "artifacts": []},
            )
            search_validation = validate_search_result(root / "search_result.json")
            _add(checks, "search_result_replays", search_validation["passed"], search_validation["errors"], [])
            evidence_validation = validate_promotion_evidence(root / "promotion_evidence.json")
            _add(checks, "promotion_evidence_replays", evidence_validation["passed"], evidence_validation["errors"], [])
            handoff_validation = validate_promotion_handoff(root / "promotion_handoff.json")
            _add(checks, "promotion_handoff_replays", handoff_validation["passed"], handoff_validation["errors"], [])
            _add(checks, "candidate_identity_bound", handoff["candidate_binding"]["matched"], handoff["candidate_binding"], {"matched": True})
            _add(checks, "promotion_not_applied", handoff["execution_boundary"]["promotion_applied"] is False, handoff["execution_boundary"]["promotion_applied"], False)
            schema_paths = [
                root / "search_plan.json",
                root / "search_result.json",
                root / "promotion_evidence.json",
                root / "promotion_handoff.json",
                *sorted(root.glob("trial-*.json")),
            ]
            schema_errors: list[str] = []
            for path in schema_paths:
                validation = check_schema_file(path)
                schema_errors.extend(f"{path.name}: {error}" for error in validation["errors"])
            _add(checks, "public_schemas_pass", not schema_errors, schema_errors, [])
    except Exception as exc:
        _add(checks, "validator_runtime", False, str(exc), "no exception")
    failed = [check for check in checks if not check["passed"]]
    return {
        "status": "passed" if not failed else "failed",
        "passed": not failed,
        "summary": (
            "Governed autoresearch-style LoRA recipe search and held-out promotion handoff replayed successfully."
            if not failed
            else f"Mission validator failed {len(failed)} check(s)."
        ),
        "output_artifact_path": "examples/case_studies/autoresearch_lora_optimizer/README.md",
        "check_count": len(checks),
        "failed_check_count": len(failed),
        "checks": checks,
        "details": details,
    }


def _add(checks: list[dict[str, Any]], check_id: str, passed: bool, actual: Any, expected: Any) -> None:
    checks.append({"id": check_id, "passed": bool(passed), "actual": actual, "expected": expected})


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = validate_mission()
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
