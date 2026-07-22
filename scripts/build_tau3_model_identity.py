#!/usr/bin/env python3
"""Build a content-addressed identity for one local Tau-3 study model tree."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flightrecorder.atomic_json import AtomicJsonError, atomic_write_json_cas  # noqa: E402
from flightrecorder.schema_registry import check_schema_contract  # noqa: E402
from flightrecorder.tau3_model_identity import (  # noqa: E402
    Tau3ModelIdentityError,
    build_tau3_model_identity,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True, help="Existing local model directory")
    parser.add_argument("--model-id", required=True, help="Frozen upstream or local model identifier")
    parser.add_argument("--revision", required=True, help="Immutable model revision")
    parser.add_argument("--out", type=Path, required=True, help="New identity JSON outside the model directory")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        target = _validated_target(args.out, args.model_path)
        identity = build_tau3_model_identity(
            args.model_path,
            model_id=args.model_id,
            revision=args.revision,
        )
        schema = check_schema_contract(identity, name_or_id="tau3_model_identity")
        if schema.get("passed") is not True:
            raise Tau3ModelIdentityError("identity schema failed: " + "; ".join(schema.get("errors", [])))
        identity_sha256 = atomic_write_json_cas(
            target,
            identity,
            expected_sha256=None,
            new_file_mode=0o600,
        )
    except (AtomicJsonError, OSError, Tau3ModelIdentityError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps({
        "file_count": identity["file_count"],
        "identity_file": target.name,
        "identity_sha256": identity_sha256,
        "model_id": identity["model_id"],
        "revision": identity["revision"],
        "total_size": identity["total_size"],
        "tree_sha256": identity["tree_sha256"],
    }, indent=2, sort_keys=True))
    return 0


def _validated_target(out: Path, model_path: Path) -> Path:
    root = model_path.resolve()
    target = out.parent.resolve() / out.name
    if target.is_relative_to(root):
        raise Tau3ModelIdentityError("identity output must be outside the model directory")
    if target.exists():
        raise Tau3ModelIdentityError("identity output already exists; refusing to overwrite it")
    return target


if __name__ == "__main__":
    raise SystemExit(main())
