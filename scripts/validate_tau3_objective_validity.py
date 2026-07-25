#!/usr/bin/env python3
"""Replay and validate a Tau-3 objective-validity artifact."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flightrecorder.tau3_objective_validity import validate_tau3_objective_validity_report  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True, help="Objective-validity report JSON path")
    parser.add_argument("--out", type=Path, help="Optional replay validation receipt path")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = validate_tau3_objective_validity_report(args.report)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "passed": result["passed"],
                "replayed_passed": result["replayed_passed"],
                "errors": result["error_count"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
