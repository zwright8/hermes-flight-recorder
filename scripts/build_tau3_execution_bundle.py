#!/usr/bin/env python3
"""Build a private Tau-3 execution bundle from portable run artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flightrecorder.tau3_execution_bundle import (  # noqa: E402
    Tau3ExecutionBundleError,
    build_tau3_execution_bundle,
    parse_arm_arg,
    parse_candidate_arg,
    parse_expected_source_hash,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--git-commit", required=True, help="Exact clean 40-character Flight Recorder git revision.")
    parser.add_argument(
        "--tracked-worktree-clean",
        action="store_true",
        help="Explicit attestation that the supplied git revision had a clean tracked worktree.",
    )
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--selected-candidate-id", required=True)
    parser.add_argument("--candidate", action="append", required=True, help="candidate_id=/path/to/portable-training-output")
    parser.add_argument("--candidate-selection-report", type=Path, required=True)
    parser.add_argument("--candidate-lock", type=Path, required=True)
    parser.add_argument("--development-arm", action="append", required=True, help="arm_id=/path/to/development-arm-output")
    parser.add_argument("--sealed-arm", action="append", required=True, help="arm_id=/path/to/sealed-arm-output")
    parser.add_argument("--public-report", type=Path, required=True)
    parser.add_argument(
        "--expect-source-sha",
        action="append",
        default=[],
        help=(
            "Optional source hash guard as label=sha256. Labels: protocol, candidate_selection_report, "
            "candidate_lock, public_report, candidate:<id>, development:<arm>, sealed:<arm>."
        ),
    )
    parser.add_argument("--keep-writable", action="store_true", help="Do not chmod copied bundle files read-only after assembly.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        expected = dict(parse_expected_source_hash(item) for item in args.expect_source_sha)
        manifest = build_tau3_execution_bundle(
            out_dir=args.out,
            flight_recorder_git_commit=args.git_commit,
            tracked_worktree_clean=args.tracked_worktree_clean,
            protocol=args.protocol,
            selected_candidate_id=args.selected_candidate_id,
            candidate_dirs=[parse_candidate_arg(item) for item in args.candidate],
            candidate_selection_report=args.candidate_selection_report,
            candidate_lock=args.candidate_lock,
            development_arm_dirs=[parse_arm_arg(item) for item in args.development_arm],
            sealed_arm_dirs=[parse_arm_arg(item) for item in args.sealed_arm],
            public_report=args.public_report,
            expected_source_hashes=expected,
            make_read_only=not args.keep_writable,
        )
    except (OSError, Tau3ExecutionBundleError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps({"manifest": str(args.out / "manifest.json"), "schema_version": manifest["schema_version"]}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
