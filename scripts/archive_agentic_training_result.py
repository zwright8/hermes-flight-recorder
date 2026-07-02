#!/usr/bin/env python3
"""Archive a side-effect-free result receipt for an external agentic trainer."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flightrecorder.agentic_training_result import (
    FAILURE_CLASSES,
    RESULT_STATUSES,
    AgenticTrainingResultError,
    build_agentic_training_result,
    write_agentic_training_result,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path, help="hfr.agentic_training_plan.v1 JSON plan")
    parser.add_argument(
        "--runtime-preflight",
        required=True,
        type=Path,
        help="hfr.agentic_training_runtime_preflight.v1 JSON runtime preflight",
    )
    parser.add_argument("--status", required=True, choices=RESULT_STATUSES, help="External runner training status")
    parser.add_argument("--failure-class", default="none", choices=FAILURE_CLASSES, help="Classified failure for non-completed runs")
    parser.add_argument("--failure-message", default="", help="Human-readable failure summary for non-completed runs")
    parser.add_argument("--runner-id", default="external", help="External runner identifier")
    parser.add_argument("--run-id", default="", help="External runner run id")
    parser.add_argument("--output-dir", default="", help="External runner output directory or registry target")
    parser.add_argument("--config", action="append", default=[], type=Path, help="Config file to fingerprint")
    parser.add_argument("--metrics", action="append", default=[], type=Path, help="Metrics file to fingerprint")
    parser.add_argument("--adapter", action="append", default=[], type=Path, help="Adapter artifact to fingerprint")
    parser.add_argument("--checkpoint", action="append", default=[], type=Path, help="Checkpoint artifact to fingerprint")
    parser.add_argument("--log", action="append", default=[], type=Path, help="Log file to fingerprint")
    parser.add_argument("--failure-report", action="append", default=[], type=Path, help="Failure report file to fingerprint")
    parser.add_argument("--created-at", help="Override created_at for reproducible sample artifacts")
    parser.add_argument("--out", required=True, type=Path, help="Destination training result receipt JSON")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    artifacts = {
        "config": args.config,
        "metrics": args.metrics,
        "adapter": args.adapter,
        "checkpoint": args.checkpoint,
        "log": args.log,
        "failure_report": args.failure_report,
    }
    try:
        result = build_agentic_training_result(
            plan_path=args.plan,
            runtime_preflight_path=args.runtime_preflight,
            out_path=args.out,
            status=args.status,
            failure_class=args.failure_class,
            failure_message=args.failure_message,
            runner_id=args.runner_id,
            run_id=args.run_id,
            output_dir=args.output_dir,
            artifacts=artifacts,
            created_at=args.created_at,
        )
        write_agentic_training_result(args.out, result)
    except AgenticTrainingResultError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
