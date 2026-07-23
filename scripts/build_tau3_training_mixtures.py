#!/usr/bin/env python3
"""Build governed Tau-3 MLX training-mixture variants from clean views."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flightrecorder.tau3_training_mixture import (  # noqa: E402
    Tau3TrainingMixtureError,
    build_tau3_training_mixtures,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Existing training/input_export directory with train.jsonl and valid.jsonl")
    parser.add_argument("--out", type=Path, required=True, help="New or empty output directory")
    parser.add_argument("--tokenizer", type=Path, required=True, help="Pinned local base tokenizer/model directory")
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--context-window", type=int, default=8192)
    parser.add_argument("--max-action-repeat", type=int, default=3)
    parser.add_argument("--max-action-to-non-action-ratio", type=float, default=3.0)
    parser.add_argument(
        "--exclude-over-budget",
        action="store_true",
        help="Explicitly exclude and hash derived rows that exceed the sequence or context budget",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        manifest = build_tau3_training_mixtures(
            args.source,
            args.out,
            tokenizer_path=args.tokenizer,
            max_seq_length=args.max_seq_length,
            context_window=args.context_window,
            max_action_repeat=args.max_action_repeat,
            max_action_to_non_action_ratio=args.max_action_to_non_action_ratio,
            exclude_over_budget=args.exclude_over_budget,
        )
    except (OSError, Tau3TrainingMixtureError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps({
        "passed": manifest["passed"],
        "out": str(args.out),
        "variant_count": len(manifest["variants"]),
        "variants": manifest["variants"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
