#!/usr/bin/env python3
"""Immutably bind one content-addressed evidence stage to a Tau-3 v3 plan."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flightrecorder.tau3_competitive_v3 import (  # noqa: E402
    EVIDENCE_STAGES,
    Tau3CompetitiveV3BindingError,
    bind_tau3_competitive_v3_evidence,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--stage", choices=EVIDENCE_STAGES, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = bind_tau3_competitive_v3_evidence(
            args.bundle,
            stage=args.stage,
            evidence_path=args.evidence,
        )
    except Tau3CompetitiveV3BindingError as exc:
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
