#!/usr/bin/env python3
"""Build a qualified Tau-3 competitive-v3 training-evidence wrapper."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flightrecorder.tau3_competitive_v3_training_evidence import (  # noqa: E402
    Tau3CompetitiveV3TrainingEvidenceError,
    build_tau3_competitive_v3_training_evidence,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        help="Qualified candidate ID; repeat at least twice",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = build_tau3_competitive_v3_training_evidence(
            args.bundle,
            candidate_ids=args.candidate,
        )
    except Tau3CompetitiveV3TrainingEvidenceError as exc:
        print(
            json.dumps(
                {"passed": False, "error": str(exc)},
                indent=2,
                sort_keys=True,
            )
        )
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
