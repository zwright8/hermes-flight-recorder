#!/usr/bin/env python3
"""Stage one completed Tau-3 v3 MLX run into a private evidence bundle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flightrecorder.tau3_competitive_v3_training_stage import (  # noqa: E402
    Tau3CompetitiveV3TrainingStageError,
    stage_tau3_competitive_v3_training_run,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--training-run", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = stage_tau3_competitive_v3_training_run(
            args.bundle,
            candidate_id=args.candidate_id,
            training_run=args.training_run,
        )
    except Tau3CompetitiveV3TrainingStageError as exc:
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
