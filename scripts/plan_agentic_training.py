#!/usr/bin/env python3
"""Write a registry-backed, side-effect-free agentic training plan."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flightrecorder.agentic_training_plan import (
    SUPPORTED_MODES,
    AgenticTrainingPlanError,
    build_agentic_training_plan,
    write_agentic_training_plan,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=SUPPORTED_MODES)
    parser.add_argument("--model-manifest", required=True, type=Path, help="Registered model manifest with license and compatibility metadata")
    parser.add_argument("--dataset-manifest", required=True, type=Path, help="Registered dataset manifest with redaction, license, and trainer-view metadata")
    parser.add_argument("--trainer-backend", default="external", help="External trainer backend identifier to record in the plan")
    parser.add_argument("--output-dir", default="", help="Intended external trainer output directory")
    parser.add_argument("--limit", type=int, help="Optional row limit for tiny smoke launches")
    parser.add_argument("--allow-future-rl", action="store_true", help="Allow future GRPO/RL planning modes to pass input gates")
    parser.add_argument("--out", required=True, type=Path, help="Destination plan JSON")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        plan = build_agentic_training_plan(
            out_path=args.out,
            mode=args.mode,
            model_manifest_path=args.model_manifest,
            dataset_manifest_path=args.dataset_manifest,
            trainer_backend=args.trainer_backend,
            output_dir=args.output_dir,
            limit=args.limit,
            allow_future_rl=args.allow_future_rl,
        )
    except AgenticTrainingPlanError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    write_agentic_training_plan(args.out, plan)
    print(json.dumps(plan, indent=2, sort_keys=True))
    return 0 if plan["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
