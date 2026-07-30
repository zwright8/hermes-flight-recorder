#!/usr/bin/env python3
"""Build or validate a non-qualifying Tau-3 development screening plan."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flightrecorder.tau3_development_screening import (  # noqa: E402
    Tau3DevelopmentScreeningError,
    build_tau3_development_screening,
    validate_tau3_development_screening,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--development-source", type=Path, required=True)
    build.add_argument("--out", type=Path, required=True)
    build.add_argument("--created-at")
    validate = subparsers.add_parser("validate")
    validate.add_argument("--screening", type=Path, required=True)
    validate.add_argument("--development-source", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        if args.command == "build":
            payload = build_tau3_development_screening(
                development_source=args.development_source,
                out_path=args.out,
                created_at=args.created_at,
            )
            result = {
                "wrote": str(args.out),
                "selected_task_set_sha256": payload[
                    "selected_task_set_sha256"
                ],
                "task_count": payload["task_count"],
                "expected_run_count": payload["expected_run_count"],
                "candidate_eligible": payload["candidate_eligible"],
            }
            return_code = 0
        else:
            result = validate_tau3_development_screening(
                screening=args.screening,
                development_source=args.development_source,
            )
            return_code = 0 if result["passed"] else 1
    except (OSError, Tau3DevelopmentScreeningError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
