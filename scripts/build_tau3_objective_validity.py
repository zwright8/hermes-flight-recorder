#!/usr/bin/env python3
"""Build a replayable Tau-3 objective-validity artifact."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flightrecorder.tau3_objective_validity import (  # noqa: E402
    Tau3ObjectiveValidityError,
    write_tau3_objective_validity_report,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-export", type=Path, required=True, help="JSONL export with one row per supervised decision")
    parser.add_argument("--parent-trajectories", type=Path, required=True, help="Content-addressed parent trajectory JSONL evidence")
    parser.add_argument("--out", type=Path, required=True, help="Objective-validity report JSON path")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = write_tau3_objective_validity_report(
            training_export_path=args.training_export,
            parent_trajectories_path=args.parent_trajectories,
            out_path=args.out,
        )
    except Tau3ObjectiveValidityError as exc:
        print(json.dumps({"passed": False, "summary": str(exc)}, indent=2, sort_keys=True))
        return 2
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "failed": report["failed_check_count"],
                "eligible_decision_count": report["eligible_decision_count"],
                "supervised_row_count": report["supervised_row_count"],
                "summary": report["summary"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
