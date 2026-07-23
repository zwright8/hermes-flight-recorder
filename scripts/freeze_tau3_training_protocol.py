#!/usr/bin/env python3
"""Freeze a local production Tau-3 training protocol from verified evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flightrecorder.tau3_protocol_freeze import (  # noqa: E402
    Tau3ProtocolFreezeError,
    freeze_tau3_training_protocol,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tau-repo", type=Path, required=True)
    parser.add_argument("--tau-revision", required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--train-split", type=Path, required=True)
    parser.add_argument("--development-split", type=Path, required=True)
    parser.add_argument("--sealed-split", type=Path, required=True)
    parser.add_argument("--train-tasks", type=Path, required=True)
    parser.add_argument("--development-tasks", type=Path, required=True)
    parser.add_argument("--base-identity", type=Path, required=True)
    parser.add_argument("--base-model-path", type=Path, required=True)
    parser.add_argument("--comparator1-identity", type=Path, required=True)
    parser.add_argument("--comparator1-model-path", type=Path, required=True)
    parser.add_argument("--comparator2-identity", type=Path, required=True)
    parser.add_argument("--comparator2-model-path", type=Path, required=True)
    parser.add_argument("--teacher-identity", type=Path, required=True)
    parser.add_argument("--teacher-model-path", type=Path, required=True)
    parser.add_argument("--captures", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--created-at", default="2026-07-22T00:00:00+00:00")
    parser.add_argument("--hardware-class", default="local Apple M4 Max with 36 GB unified memory")
    parser.add_argument("--memory-gib", type=int, default=36)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        summary = freeze_tau3_training_protocol(
            tau_repo=args.tau_repo,
            tau_revision=args.tau_revision,
            source_manifest=args.source_manifest,
            train_split=args.train_split,
            development_split=args.development_split,
            sealed_split=args.sealed_split,
            train_tasks=args.train_tasks,
            development_tasks=args.development_tasks,
            base_identity=args.base_identity,
            base_model_path=args.base_model_path,
            comparator1_identity=args.comparator1_identity,
            comparator1_model_path=args.comparator1_model_path,
            comparator2_identity=args.comparator2_identity,
            comparator2_model_path=args.comparator2_model_path,
            teacher_identity=args.teacher_identity,
            teacher_model_path=args.teacher_model_path,
            captures=args.captures,
            out=args.out,
            created_at=args.created_at,
            hardware_class=args.hardware_class,
            memory_gib=args.memory_gib,
        )
    except (OSError, Tau3ProtocolFreezeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
