#!/usr/bin/env python3
"""Validate Tau-3 cross-domain QLoRA training-readiness bundles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from flightrecorder.tau3_training_artifacts import validate_tau3_training_bundle


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True, help="Bundle root containing manifest.json")
    parser.add_argument("--strict", action="store_true", help="Enable production readiness gates")
    parser.add_argument(
        "--allow-rehearsal",
        action="store_true",
        help="Allow bundle_mode=rehearsal to pass integrity/semantic checks while reporting ready_for_training=false",
    )
    parser.add_argument("--out", type=Path, help="Optional JSON validation receipt path")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = validate_tau3_training_bundle(
        args.bundle,
        strict=args.strict,
        allow_rehearsal=args.allow_rehearsal,
    )
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "passed": result["passed"],
                "bundle_mode": result.get("bundle_mode"),
                "ready_for_training": result.get("ready_for_training"),
                "failed": result.get("failed_check_count", 0),
                "summary": result.get("summary"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
