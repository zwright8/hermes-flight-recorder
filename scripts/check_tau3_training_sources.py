#!/usr/bin/env python3
"""Preflight pinned local Tau, model, license, redaction, and MLX inputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flightrecorder.atomic_json import AtomicJsonError, atomic_write_json_cas  # noqa: E402
from scripts.build_tau3_training_artifacts import (  # noqa: E402
    Tau3BundleBuildError,
    _production_source_checks,
    _read_object,
    _trainer_command,
)


REQUIRED_CONFIG_OBJECTS = (
    "protocol_manifest",
    "tau_revision",
    "split_manifest",
    "harness_contract",
    "model_freeze",
    "budget",
    "sealed_manifest",
    "mlx_qlora_plan",
    "recipe_space",
    "candidate_selection_contract",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Local production protocol/source config")
    parser.add_argument("--out", type=Path, help="Optional new path for the portable preflight receipt")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config = _read_object(args.config, "production config")
        receipt = check_tau3_training_sources(config)
        if args.out is not None:
            if args.out.exists():
                raise Tau3BundleBuildError("source preflight output already exists; refusing to overwrite it")
            atomic_write_json_cas(
                args.out,
                receipt,
                expected_sha256=None,
                new_file_mode=0o600,
            )
    except (AtomicJsonError, OSError, Tau3BundleBuildError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps({
        "check_count": receipt["check_count"],
        "failed_check_count": receipt["failed_check_count"],
        "passed": receipt["passed"],
        "summary": receipt["summary"],
    }, indent=2, sort_keys=True))
    return 0 if receipt["passed"] else 1


def check_tau3_training_sources(config: dict[str, Any]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    for name in REQUIRED_CONFIG_OBJECTS:
        checks.append({
            "id": f"config_object_present:{name}",
            "passed": isinstance(config.get(name), dict),
            "actual": type(config.get(name)).__name__,
            "expected": "object",
        })
    try:
        command = _trainer_command(config, "production")
    except (KeyError, Tau3BundleBuildError, TypeError, ValueError) as exc:
        checks.append({
            "id": "production_command_frozen_and_local",
            "passed": False,
            "actual": str(exc),
            "expected": "portable local MLX-LM command",
        })
    else:
        checks.append({
            "id": "production_command_frozen_and_local",
            "passed": True,
            "actual": command,
            "expected": "portable local MLX-LM command",
        })
    checks.extend(_production_source_checks(config))
    failed = [check for check in checks if check.get("passed") is not True]
    return {
        "schema_version": "hfr.tau3_training_source_preflight.v1",
        "passed": not failed,
        "status": "passed" if not failed else "failed",
        "summary": (
            "Pinned local Tau/model/MLX sources passed preflight."
            if not failed
            else f"Pinned local Tau/model/MLX sources failed {len(failed)} check(s)."
        ),
        "check_count": len(checks),
        "failed_check_count": len(failed),
        "checks": checks,
        "execution_boundary": {
            "network_started": False,
            "model_downloads_started": False,
            "training_started": False,
            "sealed_evaluation_started": False,
        },
    }


if __name__ == "__main__":
    raise SystemExit(main())
