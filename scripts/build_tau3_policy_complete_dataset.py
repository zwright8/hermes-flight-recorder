#!/usr/bin/env python3
"""Build the family-split, policy-complete Tau-3 MLX dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flightrecorder.tau3_policy_complete_dataset import (  # noqa: E402
    Tau3PolicyCompleteDatasetError,
    build_tau3_policy_complete_dataset,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-corpus", type=Path, required=True)
    parser.add_argument("--captures", type=Path, required=True)
    parser.add_argument("--train-split", type=Path, required=True)
    parser.add_argument("--development-split", type=Path, required=True)
    parser.add_argument("--development-tasks", type=Path, required=True)
    parser.add_argument("--parent-protocol", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-seq-length", type=int, default=16384)
    parser.add_argument("--context-window", type=int, default=16384)
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--partition-salt", default="tau3-policy-complete-v2")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        manifest = build_tau3_policy_complete_dataset(
            teacher_corpus_dir=args.teacher_corpus,
            captures_path=args.captures,
            train_split_path=args.train_split,
            development_split_path=args.development_split,
            development_tasks_path=args.development_tasks,
            parent_protocol_path=args.parent_protocol,
            tokenizer_path=args.tokenizer,
            out_dir=args.out,
            max_seq_length=args.max_seq_length,
            context_window=args.context_window,
            validation_fraction=args.validation_fraction,
            partition_salt=args.partition_salt,
        )
    except (OSError, ValueError, Tau3PolicyCompleteDatasetError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "manifest": str(args.out / "manifest.json"),
                "manifest_sha256": manifest["manifest_sha256"],
                "counts": manifest["counts"],
                "passed": manifest["passed"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
