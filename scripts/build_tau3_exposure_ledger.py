#!/usr/bin/env python3
"""Build a deterministic Tau-3 training exposure ledger."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flightrecorder.tau3_exposure import Tau3ExposureError, build_tau3_exposure_ledger  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="Training JSONL with required Tau-3 metadata")
    parser.add_argument("--out", type=Path, required=True, help="New or empty output directory")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--split", default="train")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        receipt = build_tau3_exposure_ledger(
            args.dataset,
            args.out,
            seed=args.seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            split=args.split,
        )
    except (OSError, Tau3ExposureError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps({
        "passed": receipt["passed"],
        "candidate_eligible": receipt["candidate_eligibility"]["passed"],
        "row_count": receipt["dataset"]["row_count"],
        "step_count": receipt["files"]["ledger"]["step_count"],
        "receipt": receipt["receipt_path"],
        "ledger_sha256": receipt["files"]["ledger"]["sha256"],
    }, indent=2, sort_keys=True))
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
