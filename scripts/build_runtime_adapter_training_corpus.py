#!/usr/bin/env python3
"""Build the runtime adapter router native tool-use training corpus."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flightrecorder.realistic_tool_corpus import DEFAULT_COUNT, write_runtime_adapter_corpus


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("examples/case_studies/runtime_adapter_router"),
        help="Directory where corpus data, controls, and manifests are written.",
    )
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT, help="Number of deterministic rows to write.")
    parser.add_argument("--seed", type=int, default=17, help="Deterministic corpus seed.")
    args = parser.parse_args()

    result = write_runtime_adapter_corpus(args.output_dir, count=args.count, seed=args.seed)
    print(
        json.dumps(
            {
                "output_dir": str(result.output_dir),
                "model_manifest": str(result.model_manifest),
                "dataset_manifest": str(result.dataset_manifest),
                "corpus_manifest": str(result.corpus_manifest),
                "all_action_sft_path": str(result.all_action_sft_path),
                "total_rows": result.total_rows,
                "split_counts": result.split_counts,
                "task_scope_counts": result.task_scope_counts,
                "task_family_counts": result.task_family_counts,
                "candidate_manifests": {scope: str(path) for scope, path in sorted(result.candidate_manifests.items())},
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
