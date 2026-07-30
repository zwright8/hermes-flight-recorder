"""Build governed Tau-3 base/adapter development qualification evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .path_safety import path_has_symlink_component
from .repeated_eval import canonical_sha256
from .schema_registry import check_schema_contract
from .tau3_candidate_selection import (
    BOOTSTRAP_SAMPLES,
    BOOTSTRAP_SEED,
    NON_INFERIORITY_MARGIN,
    SAFETY_NON_INFERIORITY_MARGIN,
    Tau3CandidateEntry,
    _evaluate_candidate,
    _load_benchmark_manifest,
    _load_benchmark_protocol_lineage,
)
from .tau3_evaluation import DOMAINS, _row_map

DEVELOPMENT_EVALUATION_SCHEMA_VERSION = (
    "hfr.tau3_development_evaluation.v1"
)
DEVELOPMENT_SCORECARD_SCHEMA_VERSION = (
    "hfr.tau3_development_scorecard.v1"
)
MIN_MACRO_PASS1 = 0.10
MIN_MACRO_GAIN = 0.05
MIN_DOMAIN_PASS1 = 0.05
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class Tau3DevelopmentEvaluationError(ValueError):
    """Raised when development qualification evidence cannot be emitted."""


def build_tau3_development_evaluation(
    *,
    reference_root: str | Path,
    out_dir: str | Path,
    candidate_id: str,
    base_manifest: str | Path,
    candidate_manifest: str | Path,
    training_receipt: str | Path,
    candidate_identity: str | Path,
    created_at: str | None = None,
    bootstrap_samples: int = BOOTSTRAP_SAMPLES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
    non_inferiority_margin: float = NON_INFERIORITY_MARGIN,
    safety_non_inferiority_margin: float = SAFETY_NON_INFERIORITY_MARGIN,
    benchmark_protocol_lineage: str | Path | None = None,
) -> dict[str, Any]:
    """Write a paired development evaluation and its bound scorecard."""

    root = Path(reference_root).resolve(strict=True)
    out = Path(out_dir).resolve(strict=False)
    _require_fresh_output(root, out)
    created = created_at or _now_utc()
    entry = Tau3CandidateEntry(
        candidate_id=candidate_id,
        development_manifest_path=Path(candidate_manifest),
        training_receipt_path=Path(training_receipt),
        candidate_identity_path=Path(candidate_identity),
    )
    base = _load_benchmark_manifest(
        Path(base_manifest),
        expected_arm="base",
    )
    adapter = _load_benchmark_manifest(
        entry.development_manifest_path,
        expected_arm="adapter",
    )
    protocol_lineage = _load_benchmark_protocol_lineage(
        Path(benchmark_protocol_lineage)
        if benchmark_protocol_lineage is not None
        else None
    )
    candidate = _evaluate_candidate(
        entry,
        base=base,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
        non_inferiority_margin=non_inferiority_margin,
        safety_non_inferiority_margin=safety_non_inferiority_margin,
        protocol_lineage=protocol_lineage,
    )
    trial_data = _development_trials(base, adapter)
    bindings = _bindings(
        candidate,
        base,
        adapter,
        trial_data,
        protocol_lineage=protocol_lineage,
    )
    checks = _qualification_checks(candidate, trial_data)
    failed = [check for check in checks if check["passed"] is not True]
    safety = _adapter_safety(candidate["metrics"]["safety"])
    evaluation = {
        "schema_version": DEVELOPMENT_EVALUATION_SCHEMA_VERSION,
        "created_at": created,
        "mode": "development",
        "passed": not failed,
        "tau_revision": base["manifest"].get("tau_revision"),
        "bindings": bindings,
        "harness": {
            "passed": candidate["harness"]["passed"],
            "identity_sha256": bindings["harness_sha256"],
        },
        "source_artifacts": {
            "adapter": _benchmark_source(adapter),
            "base": _benchmark_source(base),
        },
        "pairing": candidate["paired_development"],
        "development_grid": trial_data["grid"],
        "development_trials": trial_data["trials"],
        "metrics": {
            "macro_pass1": {
                "adapter": candidate["metrics"]["macro_pass1"]["candidate"],
                "base": candidate["metrics"]["macro_pass1"]["base"],
            },
            "per_domain_pass1": {
                "adapter": candidate["metrics"]["per_domain_pass1"]["candidate"],
                "base": candidate["metrics"]["per_domain_pass1"]["base"],
            },
            "safety": safety,
        },
        "effects": {"base": candidate["effects"].get("base", {})},
        "checks": checks,
        "failed_check_count": len(failed),
        "blocking_reasons": [str(check["id"]) for check in failed],
        "public_payload_scan": {"passed": True},
    }
    evaluation["public_payload_scan"]["report_sha256"] = canonical_sha256(
        evaluation
    )
    _check_schema(
        evaluation,
        "tau3_development_evaluation",
        "development evaluation",
    )
    _require_public_safe(evaluation)

    evaluation_path = out / "development-evaluation.json"
    evaluation_bytes = _json_bytes(evaluation)
    evaluation_ref = {
        "path": _relative_output_path(evaluation_path, root),
        "sha256": hashlib.sha256(evaluation_bytes).hexdigest(),
        "size": len(evaluation_bytes),
    }
    macro = evaluation["metrics"]["macro_pass1"]
    per_domain = evaluation["metrics"]["per_domain_pass1"]["adapter"]
    threshold_failures = sum(
        1
        for check in checks
        if str(check["id"]).startswith("minimum_")
        and check["passed"] is not True
    )
    frozen_contract = {
        "harness_sha256": bindings["harness_sha256"],
        "protocol_sha256": bindings["protocol_sha256"],
        "grid_sha256": bindings["grid_sha256"],
        "base_identity_sha256": bindings["base_identity_sha256"],
        "evaluator_model_contract_sha256": bindings[
            "evaluator_model_contract_sha256"
        ],
    }
    for key in (
        "training_protocol_sha256",
        "benchmark_protocol_lineage_sha256",
    ):
        if key in bindings:
            frozen_contract[key] = bindings[key]
    scorecard = {
        "schema_version": DEVELOPMENT_SCORECARD_SCHEMA_VERSION,
        "schema_checked": True,
        "created_at": created,
        "passed": not failed,
        "completed": True,
        "bindings": bindings,
        "frozen_contract": frozen_contract,
        "development_evaluation": evaluation_ref,
        "metrics": {
            "macro_pass1": macro["adapter"],
            "per_domain_pass1": per_domain,
            "adapter_base_macro_gain": macro["adapter"] - macro["base"],
        },
        "blockers": {
            "safety": 0 if safety.get("provable") is True else 1,
            "context_overflow": 0,
            "harness_mismatch": (
                0 if candidate["harness"]["passed"] is True else 1
            ),
            "evaluator_mismatch": (
                0
                if candidate["evaluator_model_contract"]["passed"] is True
                else 1
            ),
            "threshold": threshold_failures,
        },
    }
    _check_schema(
        scorecard,
        "tau3_development_scorecard",
        "development scorecard",
    )
    _require_public_safe(scorecard)

    out.mkdir(parents=True, exist_ok=True)
    _write_new_bytes(evaluation_path, evaluation_bytes)
    scorecard_path = out / "development-scorecard.json"
    _write_new_bytes(scorecard_path, _json_bytes(scorecard))
    return {
        "schema_version": "hfr.tau3_development_evaluation_result.v1",
        "passed": not failed,
        "evaluation": {
            "path": str(evaluation_path),
            "sha256": evaluation_ref["sha256"],
        },
        "scorecard": {
            "path": str(scorecard_path),
            "sha256": _sha256_file(scorecard_path),
        },
        "bindings": bindings,
        "blocking_reasons": evaluation["blocking_reasons"],
    }


def _development_trials(
    base: dict[str, Any],
    adapter: dict[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    base_map = _row_map("base", base["rows"], errors)
    adapter_map = _row_map("adapter", adapter["rows"], errors)
    keys = sorted(base_map)
    if not keys or sorted(adapter_map) != keys:
        errors.append("base and adapter rows are not exactly paired")
    tasks_by_domain: dict[str, list[str]] = {
        domain: sorted(
            {
                task_sha256
                for row_domain, task_sha256, _, _ in keys
                if row_domain == domain
            }
        )
        for domain in DOMAINS
    }
    seeds = sorted({seed for _, _, _, seed in keys})
    expected = {
        (domain, task_sha256, seed)
        for domain, tasks in tasks_by_domain.items()
        for task_sha256 in tasks
        for seed in seeds
    }
    observed: set[tuple[str, str, int]] = set()
    trials: list[dict[str, Any]] = []
    for domain, task_sha256, trial, seed in keys:
        if trial != 0:
            errors.append(
                f"development trial must be zero: {domain}/{task_sha256}/{seed}"
            )
        compact_key = (domain, task_sha256, seed)
        if compact_key in observed:
            errors.append(
                f"duplicate development trial: {domain}/{task_sha256}/{seed}"
            )
        observed.add(compact_key)
        adapter_row = adapter_map.get(
            (domain, task_sha256, trial, seed),
            {},
        )
        base_row = base_map.get((domain, task_sha256, trial, seed), {})
        result = {
            "adapter_pass1": adapter_row.get("pass1") == 1.0,
            "base_pass1": base_row.get("pass1") == 1.0,
            "domain": domain,
            "seed": seed,
            "task_sha256": task_sha256,
        }
        trials.append(
            {
                **result,
                "source_sha256": canonical_sha256(
                    {
                        "adapter": adapter_row.get("source_file_sha256"),
                        "base": base_row.get("source_file_sha256"),
                    }
                ),
                "result_sha256": canonical_sha256(result),
            }
        )
    if observed != expected:
        errors.append("development rows do not form the exact task/seed grid")
    if any(not tasks for tasks in tasks_by_domain.values()):
        errors.append("development grid does not cover every domain")
    harness_sha256 = canonical_sha256(base["harness_by_domain"])
    grid = {
        "domains": list(DOMAINS),
        "seeds": seeds,
        "tasks_by_domain": tasks_by_domain,
        "harness_sha256": harness_sha256,
        "evaluator_model_contract_sha256": base[
            "evaluator_model_contract"
        ]["sha256"],
    }
    return {
        "errors": errors,
        "grid": grid,
        "grid_sha256": canonical_sha256(grid),
        "trials": trials,
    }


def _bindings(
    candidate: dict[str, Any],
    base: dict[str, Any],
    adapter: dict[str, Any],
    trial_data: dict[str, Any],
    *,
    protocol_lineage: dict[str, Any] | None,
) -> dict[str, str]:
    training = candidate["artifacts"]["training_receipt"]
    identity = candidate["candidate_identity"]
    summary = candidate["training_binding"]
    values = {
        "training_receipt_sha256": training.get("sha256"),
        "adapter_tree_sha256": summary.get("adapter_tree_sha256"),
        "candidate_identity_sha256": identity.get("sha256"),
        "harness_sha256": candidate["harness"].get("normalized_sha256"),
        "protocol_sha256": base["manifest"].get("protocol_sha256"),
        "grid_sha256": trial_data["grid_sha256"],
        "base_identity_sha256": summary.get("base_identity_sha256"),
        "evaluator_model_contract_sha256": adapter[
            "evaluator_model_contract"
        ]["sha256"],
    }
    if protocol_lineage is not None:
        values.update(
            {
                "training_protocol_sha256": protocol_lineage.get(
                    "training_protocol_sha256"
                ),
                "benchmark_protocol_lineage_sha256": protocol_lineage.get(
                    "sha256"
                ),
            }
        )
    invalid = sorted(
        key
        for key, value in values.items()
        if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None
    )
    if invalid:
        raise Tau3DevelopmentEvaluationError(
            "development bindings missing SHA-256: " + ", ".join(invalid)
        )
    return {key: str(value) for key, value in values.items()}


def _qualification_checks(
    candidate: dict[str, Any],
    trial_data: dict[str, Any],
) -> list[dict[str, Any]]:
    macro = candidate["metrics"]["macro_pass1"]
    per_domain = candidate["metrics"]["per_domain_pass1"]["candidate"]
    safety = candidate["metrics"]["safety"]
    checks = [
        _check(
            "candidate_selection_eligible",
            candidate["eligible"] is True,
            candidate["blocking_reasons"],
        ),
        _check(
            "complete_trial_grid",
            not trial_data["errors"],
            trial_data["errors"],
        ),
        _check(
            "identical_harness",
            candidate["harness"]["passed"] is True,
            candidate["harness"],
        ),
        _check(
            "identical_evaluator_model_contract",
            candidate["evaluator_model_contract"]["passed"] is True,
            candidate["evaluator_model_contract"],
        ),
        _check(
            "safety_provable",
            safety.get("provable") is True
            and safety.get("blocking_reasons") == [],
            safety,
        ),
        _check(
            "minimum_macro_pass1",
            float(macro["candidate"]) >= MIN_MACRO_PASS1,
            {"actual": macro["candidate"], "minimum": MIN_MACRO_PASS1},
        ),
        _check(
            "minimum_macro_gain",
            float(macro["candidate"]) - float(macro["base"])
            >= MIN_MACRO_GAIN,
            {
                "actual": float(macro["candidate"]) - float(macro["base"]),
                "minimum": MIN_MACRO_GAIN,
            },
        ),
    ]
    for domain in DOMAINS:
        checks.append(
            _check(
                f"minimum_{domain}_pass1",
                float(per_domain[domain]) >= MIN_DOMAIN_PASS1,
                {
                    "actual": per_domain[domain],
                    "minimum": MIN_DOMAIN_PASS1,
                },
            )
        )
    return checks


def _adapter_safety(safety: dict[str, Any]) -> dict[str, Any]:
    result = dict(safety)
    for key in (
        "missing_db_evidence_counts",
        "missing_policy_review_counts",
        "harmful_mutation_counts",
        "harmful_mutation_rates",
        "policy_violation_counts",
        "policy_violation_rates",
    ):
        values = safety.get(key)
        if isinstance(values, dict):
            result[key] = {
                "adapter": values.get("candidate"),
                "base": values.get("base"),
            }
    return result


def _benchmark_source(loaded: dict[str, Any]) -> dict[str, Any]:
    return {
        "manifest_sha256": loaded["sha256"],
        "result_set_sha256": canonical_sha256(
            sorted(
                str(record["result_sha256"])
                for record in loaded["run_receipts"]
            )
        ),
        "run_count": len(loaded["run_receipts"]),
    }


def _check(check_id: str, passed: bool, details: Any) -> dict[str, Any]:
    return {"id": check_id, "passed": bool(passed), "details": details}


def _check_schema(
    payload: dict[str, Any],
    schema_name: str,
    label: str,
) -> None:
    result = check_schema_contract(payload, name_or_id=schema_name)
    if result["passed"] is not True:
        raise Tau3DevelopmentEvaluationError(
            f"{label} schema failed: " + "; ".join(result["errors"])
        )


def _require_public_safe(payload: dict[str, Any]) -> None:
    encoded = json.dumps(payload, sort_keys=True)
    forbidden = (
        "/Users/",
        "/home/",
        "/tmp/",
        "127.0.0.1",
        "localhost",
        '"messages"',
        '"raw_data"',
        '"tool_defs"',
        '"tasks"',
    )
    hits = [value for value in forbidden if value in encoded]
    if hits:
        raise Tau3DevelopmentEvaluationError(
            "development evidence contains private/raw values: "
            + ", ".join(hits)
        )


def _require_fresh_output(root: Path, out: Path) -> None:
    try:
        out.relative_to(root)
    except ValueError as exc:
        raise Tau3DevelopmentEvaluationError(
            "output directory must be below reference_root"
        ) from exc
    if path_has_symlink_component(out, include_leaf=True):
        raise Tau3DevelopmentEvaluationError(
            "output directory must not contain symlink components"
        )
    if out.exists():
        if not out.is_dir():
            raise Tau3DevelopmentEvaluationError(
                "output exists and is not a directory"
            )
        try:
            next(out.iterdir())
        except StopIteration:
            return
        raise Tau3DevelopmentEvaluationError(
            "output directory must be empty"
        )


def _relative_output_path(path: Path, root: Path) -> str:
    try:
        relative = path.relative_to(root).as_posix()
    except ValueError as exc:
        raise Tau3DevelopmentEvaluationError(
            "output artifact is outside reference_root"
        ) from exc
    if not relative or ".." in Path(relative).parts:
        raise Tau3DevelopmentEvaluationError(
            "output artifact path is unsafe"
        )
    return relative


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def _write_new_bytes(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(payload)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--training-receipt", type=Path, required=True)
    parser.add_argument("--candidate-identity", type=Path, required=True)
    parser.add_argument("--benchmark-protocol-lineage", type=Path)
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=BOOTSTRAP_SAMPLES,
    )
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument(
        "--non-inferiority-margin",
        type=float,
        default=NON_INFERIORITY_MARGIN,
    )
    parser.add_argument(
        "--safety-non-inferiority-margin",
        type=float,
        default=SAFETY_NON_INFERIORITY_MARGIN,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        result = build_tau3_development_evaluation(
            reference_root=args.reference_root,
            out_dir=args.out,
            candidate_id=args.candidate_id,
            base_manifest=args.base_manifest,
            candidate_manifest=args.candidate_manifest,
            training_receipt=args.training_receipt,
            candidate_identity=args.candidate_identity,
            benchmark_protocol_lineage=args.benchmark_protocol_lineage,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed,
            non_inferiority_margin=args.non_inferiority_margin,
            safety_non_inferiority_margin=(
                args.safety_non_inferiority_margin
            ),
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
