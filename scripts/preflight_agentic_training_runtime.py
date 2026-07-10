#!/usr/bin/env python3
"""Preflight local runtime readiness for an agentic training plan."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flightrecorder.agentic_training_runtime import (  # noqa: E402 - repo bootstrap precedes local import
    AgenticTrainingRuntimePreflightError,
    build_agentic_training_runtime_preflight,
    write_agentic_training_runtime_preflight,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path, help="hfr.agentic_training_plan.v1 JSON plan")
    parser.add_argument("--out", required=True, type=Path, help="Destination runtime preflight JSON")
    parser.add_argument("--require-module", action="append", default=[], help="Additional Python module that must be discoverable")
    parser.add_argument("--skip-default-modules", action="store_true", help="Only check modules supplied with --require-module")
    parser.add_argument("--created-at", help="Override created_at for reproducible sample artifacts")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        preflight = build_agentic_training_runtime_preflight(
            plan_path=args.plan,
            out_path=args.out,
            require_modules=args.require_module,
            skip_default_modules=args.skip_default_modules,
            created_at=args.created_at,
        )
        write_agentic_training_runtime_preflight(args.out, preflight)
    except AgenticTrainingRuntimePreflightError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(preflight, indent=2, sort_keys=True))
    return 0 if preflight["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
