#!/usr/bin/env python3
"""Reference external wrapper for a Flight Recorder trainer consumer plan.

This example validates a trainer_consumer_plan.json and emits a dry-run receipt.
It deliberately never executes the trainer command. Real training launchers can
reuse the same checks before they take responsibility for process execution.
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path
from typing import Any

SOURCE_ROOT = Path(__file__).resolve().parents[2]
if (SOURCE_ROOT / "flightrecorder").is_dir():
    sys.path.insert(0, str(SOURCE_ROOT))

from flightrecorder.trainer_consumer_plan import TRAINER_CONSUMER_PLAN_SCHEMA_VERSION
from flightrecorder.validation import validate_artifacts

WRAPPER_RECEIPT_SCHEMA_VERSION = "hfr.example_trainer_wrapper_dry_run.v1"


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    plan_path = Path(args.plan)
    try:
        plan = _read_json(plan_path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        _write_error(args.out, plan_path, exc)
        print(f"trainer-wrapper: error: {exc}", file=sys.stderr)
        return 2

    validation = validate_artifacts(trainer_consumer_plan_paths=[plan_path], strict=args.strict)
    receipt = build_dry_run_receipt(plan_path, plan, validation)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.print_command and receipt["passed"]:
        print(receipt["would_run"]["shell"])
    else:
        status = "READY" if receipt["passed"] else "BLOCKED"
        print(
            f"{status} trainer-wrapper-dry-run "
            f"inputs={receipt['metrics']['trainer_input_count']} "
            f"external_code={receipt['metrics']['external_code_file_count']}"
        )
    return 0 if receipt["passed"] else 1


def build_dry_run_receipt(plan_path: Path, plan: dict[str, Any], validation: dict[str, Any]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    execution = plan.get("execution") if isinstance(plan.get("execution"), dict) else {}
    handoff = plan.get("handoff_contract") if isinstance(plan.get("handoff_contract"), dict) else {}
    argv = _string_list(execution.get("command_argv"))
    trainer_inputs = [item for item in execution.get("trainer_inputs", []) if isinstance(item, dict)] if isinstance(execution.get("trainer_inputs"), list) else []
    external_code_files = (
        [item for item in execution.get("external_code_files", []) if isinstance(item, dict)]
        if isinstance(execution.get("external_code_files"), list)
        else []
    )

    _add_check(checks, "plan_validation_passed", validation.get("passed") is True, {"plan": str(plan_path)})
    _add_check(
        checks,
        "schema_supported",
        plan.get("schema_version") == TRAINER_CONSUMER_PLAN_SCHEMA_VERSION,
        {"schema_version": str(plan.get("schema_version") or "")},
    )
    _add_check(checks, "plan_passed", plan.get("passed") is True, {"plan": str(plan_path)})
    _add_check(
        checks,
        "recommendation_ready",
        plan.get("recommendation") == "ready_for_external_trainer",
        {"recommendation": str(plan.get("recommendation") or "")},
    )
    _add_check(
        checks,
        "handoff_boundary_external",
        handoff.get("flight_recorder_executed_command") is False and handoff.get("runner_owns_execution") is True,
        {"runner_owns_execution": str(handoff.get("runner_owns_execution"))},
    )
    _add_check(
        checks,
        "execution_cwd_archive_root",
        execution.get("execution_cwd") == "archive_root" and handoff.get("runner_must_run_from") == "archive_root",
        {"execution_cwd": str(execution.get("execution_cwd") or "")},
    )
    _add_check(checks, "command_available", bool(argv), {"argc": str(len(argv))})
    _add_check(
        checks,
        "trainer_inputs_ready",
        bool(trainer_inputs) and all(item.get("passed") is True for item in trainer_inputs),
        {"trainer_input_count": str(len(trainer_inputs))},
    )
    _add_check(
        checks,
        "external_code_ready",
        all(item.get("passed") is True for item in external_code_files),
        {"external_code_file_count": str(len(external_code_files))},
    )
    _add_check(checks, "no_execution_performed", True, {"mode": "dry_run"})

    failed_checks = sum(1 for check in checks if check.get("passed") is False)
    passed = failed_checks == 0
    return {
        "schema_version": WRAPPER_RECEIPT_SCHEMA_VERSION,
        "wrapper": "examples/trainer-wrapper/consume_trainer_plan.py",
        "plan_path": str(plan_path),
        "passed": passed,
        "readiness": "ready" if passed else "blocked",
        "recommendation": "dry_run_ready" if passed else "block_dry_run",
        "check_count": len(checks),
        "failed_check_count": failed_checks,
        "checks": checks,
        "validation": _validation_record(validation),
        "would_run": {
            "mode": "dry_run",
            "execution_cwd": str(execution.get("execution_cwd") or ""),
            "archive_root": str(execution.get("archive_root") or ""),
            "external_code_root": str(execution.get("external_code_root") or ""),
            "argv": argv,
            "shell": shlex.join(argv) if argv else "",
        },
        "inputs": {
            "trainer_inputs": [_input_record(item) for item in trainer_inputs],
            "external_code_files": [_external_code_record(item) for item in external_code_files],
        },
        "metrics": {
            "trainer_input_count": len(trainer_inputs),
            "trainer_input_ready_count": sum(1 for item in trainer_inputs if item.get("passed") is True),
            "external_code_file_count": len(external_code_files),
            "external_code_ready_count": sum(1 for item in external_code_files if item.get("passed") is True),
            "command_arg_count": len(argv),
        },
        "notes": [
            "This receipt proves the example wrapper parsed and validated a trainer consumer plan.",
            "The wrapper did not execute the command. A real trainer launcher must own process execution separately.",
        ],
    }


def _input_record(item: dict[str, Any]) -> dict[str, Any]:
    record: dict[str, Any] = {
        "artifact_name": str(item.get("artifact_name") or ""),
        "archive_path": str(item.get("archive_path") or ""),
        "resolved_path": str(item.get("resolved_path") or ""),
        "kind": str(item.get("kind") or ""),
        "sha256": str(item.get("sha256") or ""),
        "passed": item.get("passed") is True,
    }
    for field_name in ("size_bytes", "file_count"):
        if isinstance(item.get(field_name), int) and not isinstance(item.get(field_name), bool):
            record[field_name] = item[field_name]
    return record


def _external_code_record(item: dict[str, Any]) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": str(item.get("path") or ""),
        "resolved_path": str(item.get("resolved_path") or ""),
        "sha256": str(item.get("sha256") or ""),
        "passed": item.get("passed") is True,
    }
    if isinstance(item.get("size_bytes"), int) and not isinstance(item.get("size_bytes"), bool):
        record["size_bytes"] = item["size_bytes"]
    return record


def _validation_record(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "passed": summary.get("passed") is True,
        "strict": summary.get("strict") is True,
        "target_count": _int_value(summary.get("target_count")),
        "error_count": _int_value(summary.get("error_count")),
        "warning_count": _int_value(summary.get("warning_count")),
    }


def _add_check(checks: list[dict[str, Any]], check_id: str, passed: bool, scope: dict[str, str]) -> None:
    checks.append(
        {
            "id": check_id,
            "passed": passed,
            "actual": {"passed": passed},
            "expected": {"passed": True},
            "scope": scope,
            "summary": f"{check_id}: passed={passed}",
        }
    )


def _write_error(out: str | None, plan_path: Path, exc: Exception) -> None:
    if not out:
        return
    payload = {
        "schema_version": WRAPPER_RECEIPT_SCHEMA_VERSION,
        "wrapper": "examples/trainer-wrapper/consume_trainer_plan.py",
        "plan_path": str(plan_path),
        "passed": False,
        "readiness": "blocked",
        "recommendation": "block_dry_run",
        "check_count": 1,
        "failed_check_count": 1,
        "checks": [
            {
                "id": "plan_readable",
                "passed": False,
                "actual": {"passed": False},
                "expected": {"passed": True},
                "scope": {"plan": str(plan_path)},
                "summary": f"plan_readable: {exc}",
            }
        ],
        "validation": {"passed": False, "strict": False, "target_count": 0, "error_count": 1, "warning_count": 0},
        "would_run": {"mode": "dry_run", "execution_cwd": "", "archive_root": "", "external_code_root": "", "argv": [], "shell": ""},
        "inputs": {"trainer_inputs": [], "external_code_files": []},
        "metrics": {
            "trainer_input_count": 0,
            "trainer_input_ready_count": 0,
            "external_code_file_count": 0,
            "external_code_ready_count": 0,
            "command_arg_count": 0,
        },
        "notes": ["The wrapper did not execute anything."],
    }
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"plan must contain a JSON object: {path}")
    return payload


def _string_list(value: Any) -> list[str]:
    return [item for item in value if isinstance(item, str)] if isinstance(value, list) else []


def _int_value(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dry-run an external trainer wrapper handoff from trainer_consumer_plan.json")
    parser.add_argument("--plan", required=True, help="Path to trainer_consumer_plan.json")
    parser.add_argument("--out", help="Write a dry-run receipt JSON to this path")
    parser.add_argument("--strict", action="store_true", help="Treat validation warnings as blockers")
    parser.add_argument("--print-command", action="store_true", help="Print the shell-escaped command when the plan is ready")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
