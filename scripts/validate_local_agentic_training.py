#!/usr/bin/env python3
"""Validate the offline, task-scoped local Agentic LoRA training contract."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from flightrecorder.schema_registry import check_schema_file
from train_agentic_lora import FixedTrainingTimeBudget, prepare_sft_rows

ROOT = Path(__file__).resolve().parents[1]
TRAINER = ROOT / "scripts" / "train_agentic_lora.py"
DOC = ROOT / "docs" / "local-agentic-training.md"


def validate_mission() -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    details: dict[str, Any] = {}
    try:
        with tempfile.TemporaryDirectory(prefix="hfr-local-training-validator-") as tmp:
            root = Path(tmp)
            fixture_dir = root / "fixture"
            fixture_process = _run(["--write-smoke-fixture", str(fixture_dir)])
            fixture = json.loads(fixture_process.stdout)
            _add(checks, "fixture_created", fixture_process.returncode == 0, fixture_process.returncode, 0)

            local_model = root / "local-model"
            local_model.mkdir()
            model_manifest_path = Path(fixture["model_manifest"])
            model_manifest = json.loads(model_manifest_path.read_text(encoding="utf-8"))
            model_manifest["model_id"] = str(local_model)
            model_manifest_path.write_text(json.dumps(model_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            out = root / "out"
            plan_process = _run(
                [
                    "--mode", "fr_action_sft",
                    "--dry-run",
                    "--local-training",
                    "--model", str(local_model),
                    "--model-manifest", str(model_manifest_path),
                    "--dataset-manifest", fixture["dataset_manifest"],
                    "--experiment-dir", str(fixture_dir),
                    "--output-dir", str(out),
                    "--task-family", "fixture",
                    "--device", "mps",
                    "--max-training-seconds", "7.5",
                    "--disable-trackio",
                ]
            )
            plan = json.loads(plan_process.stdout)
            plan_path = out / "fr_action_sft_plan.json"
            details = {
                "mode": plan.get("mode"),
                "task_scope": plan.get("task_scope"),
                "local_training": plan.get("local_training"),
                "prepared_counts": plan.get("prepared_counts"),
            }
            _add(checks, "dry_run_plan_passed", plan_process.returncode == 0 and plan.get("passed") is True, plan_process.returncode, 0)
            schema = check_schema_file(plan_path)
            _add(checks, "plan_schema_passed", schema["passed"], schema["errors"], [])
            _add(
                checks,
                "task_scope_exact",
                plan["task_scope"]["requested_task_families"] == ["fixture"]
                and plan["task_scope"]["selected_task_families"] == ["fixture"],
                plan["task_scope"],
                {"requested_task_families": ["fixture"], "selected_task_families": ["fixture"]},
            )
            boundary = plan["local_training"]
            _add(
                checks,
                "offline_local_boundary",
                boundary["local_files_only"] is True
                and boundary["network_allowed"] is False
                and boundary["hub_push_allowed"] is False
                and boundary["remote_tracking_allowed"] is False
                and boundary["execution_requested"] is False,
                boundary,
                "offline, local-only, unexecuted",
            )
            _add(
                checks,
                "fixed_time_and_device_bound",
                boundary["fixed_training_time_budget_seconds"] == 7.5 and boundary["device_order"] == ["mps"],
                boundary,
                {"fixed_training_time_budget_seconds": 7.5, "device_order": ["mps"]},
            )
            action_path = fixture_dir / "data" / "flightrecorder_action_sft.jsonl"
            action_rows = [json.loads(line) for line in action_path.read_text(encoding="utf-8").splitlines() if line]
            prepared = prepare_sft_rows(action_rows)
            native_tool_rows = [
                row
                for row in prepared
                if row.get("tools")
                and any(message.get("tool_calls") for message in row.get("messages", []) if isinstance(message, dict))
            ]
            _add(checks, "native_tool_calls_preserved", bool(native_tool_rows), len(native_tool_rows), ">=1")
            generated = sorted(path.name for path in out.iterdir())
            _add(checks, "no_training_started", generated == ["fr_action_sft_plan.json"], generated, ["fr_action_sft_plan.json"])

            now = [0.0]
            budget = FixedTrainingTimeBudget(3.0, clock=lambda: now[0])
            budget.begin_phase()
            now[0] = 3.0
            _add(checks, "time_budget_stops_fail_closed", budget.should_stop(), budget.elapsed_seconds, 3.0)

            missing_process = _run(
                [
                    "--mode", "fr_action_sft",
                    "--dry-run",
                    "--local-training",
                    "--model", str(local_model),
                    "--model-manifest", str(model_manifest_path),
                    "--dataset-manifest", fixture["dataset_manifest"],
                    "--experiment-dir", str(fixture_dir),
                    "--output-dir", str(root / "missing"),
                    "--task-family", "does-not-exist",
                    "--disable-trackio",
                ]
            )
            missing_plan = json.loads(missing_process.stdout)
            failed_ids = {check["id"] for check in missing_plan["checks"] if not check["passed"]}
            _add(
                checks,
                "missing_task_family_blocks",
                missing_process.returncode == 1 and "requested_task_families_available" in failed_ids,
                sorted(failed_ids),
                "requested_task_families_available",
            )

            docs = DOC.read_text(encoding="utf-8")
            _add(
                checks,
                "design_reference_documented",
                "miolini/autoresearch-macos@537c6e6" in docs and "No upstream model" in docs,
                str(DOC),
                "pinned design reference and clean-room boundary",
            )
    except Exception as exc:  # validator must archive its own failure
        _add(checks, "validator_runtime", False, str(exc), "no exception")
    failed = [check for check in checks if not check["passed"]]
    return {
        "status": "passed" if not failed else "failed",
        "passed": not failed,
        "summary": (
            "Offline task-scoped local Agentic LoRA training contract validated without starting training."
            if not failed
            else f"Local Agentic LoRA validator failed {len(failed)} check(s)."
        ),
        "output_artifact_path": "docs/local-agentic-training.md",
        "check_count": len(checks),
        "failed_check_count": len(failed),
        "checks": checks,
        "details": details,
    }


def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(TRAINER), *args],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def _add(checks: list[dict[str, Any]], check_id: str, passed: bool, actual: Any, expected: Any) -> None:
    checks.append({"id": check_id, "passed": bool(passed), "actual": actual, "expected": expected})


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", required=True, type=Path)
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
