#!/usr/bin/env python3
"""Replay a Tau-3 training-to-benchmark protocol lineage."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flightrecorder.tau3_benchmark_protocol_lineage import (  # noqa: E402
    Tau3BenchmarkProtocolLineageError,
    validate_tau3_benchmark_protocol_lineage,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lineage", type=Path, required=True)
    parser.add_argument("--training-protocol", type=Path, required=True)
    parser.add_argument("--benchmark-protocol", type=Path, required=True)
    parser.add_argument("--custody-receipt", type=Path, required=True)
    parser.add_argument("--sealed-source-manifest", type=Path, required=True)
    parser.add_argument("--generator-validation", type=Path, required=True)
    parser.add_argument("--fresh-contamination-replay", type=Path, required=True)
    parser.add_argument("--retired-source-incident-sha256", required=True)
    args = parser.parse_args(argv)
    try:
        result = validate_tau3_benchmark_protocol_lineage(
            lineage=args.lineage,
            training_protocol=args.training_protocol,
            benchmark_protocol=args.benchmark_protocol,
            custody_receipt=args.custody_receipt,
            sealed_source_manifest=args.sealed_source_manifest,
            generator_validation=args.generator_validation,
            fresh_contamination_replay=args.fresh_contamination_replay,
            retired_source_incident_sha256=args.retired_source_incident_sha256,
        )
    except (OSError, ValueError, Tau3BenchmarkProtocolLineageError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
