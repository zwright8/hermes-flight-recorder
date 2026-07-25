#!/usr/bin/env python3
"""Validate a private Tau-3 competitive-agent v3 evidence bundle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flightrecorder.tau3_competitive_v3 import STAGES, validate_tau3_competitive_v3_bundle  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True, help="Tau-3 competitive v3 bundle directory")
    parser.add_argument("--strict", action="store_true", help="Require final-stage mission-critic checks")
    parser.add_argument("--stage", choices=STAGES, help="Validation stage; defaults to plan, or final with --strict")
    parser.add_argument("--out", type=Path, help="Optional JSON validation receipt path")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    stage = args.stage or ("final" if args.strict else "plan")
    result = validate_tau3_competitive_v3_bundle(args.bundle, strict=args.strict, stage=stage)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
