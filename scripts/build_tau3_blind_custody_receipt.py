#!/usr/bin/env python3
"""Build a create-once hash-only Tau-3 blind-custody receipt."""

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
    create_tau3_blind_custody_receipt,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--custody-id", required=True)
    parser.add_argument("--sealed-source-manifest", type=Path, required=True)
    parser.add_argument("--generator-validation", type=Path, required=True)
    parser.add_argument("--fresh-contamination-replay", type=Path, required=True)
    parser.add_argument("--retired-source-incident-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--created-at")
    args = parser.parse_args(argv)
    try:
        result = create_tau3_blind_custody_receipt(
            custody_id=args.custody_id,
            sealed_source_manifest=args.sealed_source_manifest,
            generator_validation=args.generator_validation,
            fresh_contamination_replay=args.fresh_contamination_replay,
            retired_source_incident_sha256=args.retired_source_incident_sha256,
            out=args.out,
            created_at=args.created_at,
        )
    except (OSError, ValueError, Tau3BenchmarkProtocolLineageError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
