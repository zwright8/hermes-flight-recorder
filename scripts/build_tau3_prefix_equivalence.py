#!/usr/bin/env python3
"""Build replayable Tau-3 detached-prefix qualification evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flightrecorder.tau3_prefix_equivalence import (  # noqa: E402
    build_tau3_prefix_equivalence,
)


def _read_object(path: Path, label: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return payload


def _read_trials(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("trials")
    if not isinstance(payload, list) or not all(
        isinstance(item, dict) for item in payload
    ):
        raise ValueError(
            "behavior trials must be a JSON list of objects or an object with trials"
        )
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bindings", type=Path, required=True)
    parser.add_argument(
        "--full-gradient-run",
        type=Path,
        action="append",
        required=True,
    )
    parser.add_argument(
        "--detached-prefix-run",
        type=Path,
        action="append",
        required=True,
    )
    parser.add_argument("--behavior-trials", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        artifact = build_tau3_prefix_equivalence(
            bindings=_read_object(args.bindings, "bindings"),
            full_gradient_runs=args.full_gradient_run,
            detached_prefix_runs=args.detached_prefix_run,
            behavior_trials=_read_trials(args.behavior_trials),
            output_path=args.out,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0 if artifact["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
