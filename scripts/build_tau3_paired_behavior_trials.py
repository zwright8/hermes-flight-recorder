#!/usr/bin/env python3
"""Build paired Tau-3 prefix-equivalence behavior trials."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flightrecorder.tau3_prefix_equivalence import (  # noqa: E402
    Tau3PrefixEquivalenceError,
    build_tau3_paired_behavior_trials,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-gradient-probes", type=Path, required=True)
    parser.add_argument("--detached-prefix-probes", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        artifact = build_tau3_paired_behavior_trials(
            full_gradient_probe_path=args.full_gradient_probes,
            detached_prefix_probe_path=args.detached_prefix_probes,
            output_path=args.out,
        )
    except (OSError, ValueError, Tau3PrefixEquivalenceError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
