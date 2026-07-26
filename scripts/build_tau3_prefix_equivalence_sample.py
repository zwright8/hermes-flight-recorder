#!/usr/bin/env python3
"""Build the frozen bounded dataset sample for Tau-3 prefix equivalence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flightrecorder.tau3_prefix_equivalence_sample import (  # noqa: E402
    Tau3PrefixEquivalenceSampleError,
    build_tau3_prefix_equivalence_sample,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        manifest = build_tau3_prefix_equivalence_sample(
            args.dataset,
            args.out,
        )
    except (OSError, Tau3PrefixEquivalenceSampleError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
