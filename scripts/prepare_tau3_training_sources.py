#!/usr/bin/env python3
"""Prepare deterministic Tau training-source partitions from a pinned checkout."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flightrecorder.tau3_source_partition import (  # noqa: E402
    Tau3SourcePartitionError,
    prepare_tau3_training_sources,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tau-repo", type=Path, required=True, help="Local official tau2-bench checkout")
    parser.add_argument("--expected-revision", required=True, help="Exact expected lowercase 40-hex Tau git HEAD")
    parser.add_argument("--out", type=Path, required=True, help="Absent output directory to create")
    parser.add_argument("--development-fraction", type=float, default=0.2)
    parser.add_argument("--salt", default="hfr-tau3-core-v1")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        summary = prepare_tau3_training_sources(
            args.tau_repo,
            args.expected_revision,
            args.out,
            development_fraction=args.development_fraction,
            salt=args.salt,
        )
    except (OSError, Tau3SourcePartitionError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
