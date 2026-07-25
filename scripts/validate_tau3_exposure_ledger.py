#!/usr/bin/env python3
"""Validate a deterministic Tau-3 training exposure ledger."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flightrecorder.tau3_exposure import Tau3ExposureError, validate_tau3_exposure_ledger  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="Source training JSONL")
    parser.add_argument("--receipt", type=Path, required=True, help="training_exposure_receipt.json")
    parser.add_argument("--ledger", type=Path, help="Optional explicit training_exposure_ledger.jsonl")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = validate_tau3_exposure_ledger(args.dataset, args.receipt, args.ledger)
    except (OSError, Tau3ExposureError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps({
        "passed": result["passed"],
        "candidate_eligible": result["candidate_eligible"],
        "row_count": result["row_count"],
        "step_count": result["step_count"],
        "receipt_sha256": result["receipt_sha256"],
        "ledger_sha256": result["ledger_sha256"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
