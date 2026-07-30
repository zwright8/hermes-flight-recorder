#!/usr/bin/env python3
"""Validate a hash-only fresh Tau-3 generator/custody evidence bundle."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flightrecorder.tau3_blind_source_bundle import (  # noqa: E402
    Tau3BlindSourceBundleError,
    validate_tau3_blind_source_bundle,
)
from flightrecorder.path_safety import path_has_symlink_component  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sealed-source-manifest", type=Path, required=True)
    parser.add_argument("--generator-validation", type=Path, required=True)
    parser.add_argument("--fresh-contamination-replay", type=Path, required=True)
    parser.add_argument("--generator-script", type=Path, required=True)
    parser.add_argument("--tau-repo", type=Path, required=True)
    parser.add_argument("--training-dataset", type=Path, required=True)
    parser.add_argument("--development-source", type=Path, required=True)
    parser.add_argument("--retired-sealed-source", type=Path, required=True)
    parser.add_argument("--expected-source-revision", required=True)
    parser.add_argument("--expected-generator-commit", required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    try:
        result = validate_tau3_blind_source_bundle(
            sealed_source_manifest=args.sealed_source_manifest,
            generator_validation=args.generator_validation,
            fresh_contamination_replay=args.fresh_contamination_replay,
            generator_script=args.generator_script,
            tau_repo=args.tau_repo,
            training_dataset=args.training_dataset,
            development_source=args.development_source,
            retired_sealed_source=args.retired_sealed_source,
            expected_source_revision=args.expected_source_revision,
            expected_generator_commit=args.expected_generator_commit,
        )
    except (OSError, Tau3BlindSourceBundleError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if args.out is not None:
        try:
            _write_json_new(args.out, result)
        except OSError as exc:
            print(str(exc), file=sys.stderr)
            return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _write_json_new(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise OSError(f"refusing to overwrite output: {path}")
    if not path.parent.is_dir():
        raise OSError(f"output parent is not a directory: {path.parent}")
    if path_has_symlink_component(path.parent, include_leaf=True):
        raise OSError("output path must not contain symlink components")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            descriptor = -1
            handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    os.chmod(path, 0o600)


if __name__ == "__main__":
    raise SystemExit(main())
