#!/usr/bin/env python3
"""Export a public-safe, paired runtime-adapter evidence bundle."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any


REPORT_SCHEMA = "hfr.runtime_adapter_candidate_evaluation.v1"
TRAINING_RESULT_SCHEMA = "hfr.agentic_lora_training_result.v1"
CHECK_IDS = (
    "tool_calls_functional_order",
    "tool_calls_exact_order",
    "final_answer_exact",
)


class PublicationError(ValueError):
    """Raised when evidence is not safe or comparable enough to publish."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise PublicationError(f"expected JSON object: {path}")
    return value


def single_candidate(report: dict[str, Any], *, label: str) -> dict[str, Any]:
    if report.get("schema_version") != REPORT_SCHEMA:
        raise PublicationError(f"{label} has unsupported evaluation schema")
    candidates = report.get("candidate_reports")
    if not isinstance(candidates, list) or len(candidates) != 1:
        raise PublicationError(f"{label} must contain exactly one candidate report")
    candidate = candidates[0]
    if not isinstance(candidate, dict):
        raise PublicationError(f"{label} candidate report must be an object")
    return candidate


def score_checks(score: dict[str, Any]) -> dict[str, bool]:
    checks = score.get("checks")
    if not isinstance(checks, list):
        raise PublicationError("task score has no checks array")
    indexed = {
        str(item.get("check_id")): item.get("passed") is True
        for item in checks
        if isinstance(item, dict)
    }
    missing = [check_id for check_id in CHECK_IDS if check_id not in indexed]
    if missing:
        raise PublicationError(
            "task score is missing required checks: " + ", ".join(missing)
        )
    return indexed


def score_index(candidate: dict[str, Any]) -> dict[str, dict[str, Any]]:
    scores = candidate.get("scores")
    if not isinstance(scores, list):
        raise PublicationError("candidate report has no scores array")
    indexed: dict[str, dict[str, Any]] = {}
    for score in scores:
        if not isinstance(score, dict):
            raise PublicationError("candidate score must be an object")
        task_id = score.get("task_id")
        if not isinstance(task_id, str) or not task_id or task_id in indexed:
            raise PublicationError("candidate scores contain invalid or duplicate task IDs")
        indexed[task_id] = score
    return indexed


def metric(candidate: dict[str, Any], path: tuple[str, ...]) -> dict[str, Any]:
    value: Any = candidate.get("metrics")
    for key in path:
        if not isinstance(value, dict):
            raise PublicationError("candidate metrics have an invalid shape")
        value = value.get(key)
    if not isinstance(value, dict):
        raise PublicationError("candidate metric is missing: " + ".".join(path))
    return value


def validate_pair(
    base_report: dict[str, Any],
    adapter_report: dict[str, Any],
    *,
    expected_split: str,
) -> list[dict[str, Any]]:
    for identity_key in ("base_model", "tokenizer", "chat_template", "heldout"):
        if base_report.get(identity_key) != adapter_report.get(identity_key):
            raise PublicationError(
                f"{expected_split} reports do not match on {identity_key}"
            )
    heldout = base_report.get("heldout")
    if not isinstance(heldout, dict) or heldout.get("split") != expected_split:
        raise PublicationError(f"expected {expected_split} held-out split")

    base = single_candidate(base_report, label=f"{expected_split} base")
    adapter = single_candidate(adapter_report, label=f"{expected_split} adapter")
    if base.get("heldout_subset") != adapter.get("heldout_subset"):
        raise PublicationError(f"{expected_split} candidate subsets do not match")
    base_scores = score_index(base)
    adapter_scores = score_index(adapter)
    if base_scores.keys() != adapter_scores.keys():
        raise PublicationError(f"{expected_split} paired task IDs do not match")

    base_overall = metric(base, ("overall",))
    adapter_overall = metric(adapter, ("overall",))
    base_functional = metric(base, ("check_pass_rates", "tool_calls_functional_order"))
    adapter_functional = metric(
        adapter, ("check_pass_rates", "tool_calls_functional_order")
    )
    if int(adapter_overall.get("passed", 0)) <= int(base_overall.get("passed", 0)):
        raise PublicationError(f"{expected_split} adapter does not improve task success")
    if int(adapter_functional.get("passed", 0)) <= int(
        base_functional.get("passed", 0)
    ):
        raise PublicationError(
            f"{expected_split} adapter does not improve functional tool calls"
        )
    if adapter.get("promotion_eligible") is not True:
        raise PublicationError(f"{expected_split} adapter is not promotion eligible")
    if int(adapter.get("metrics", {}).get("critical_safety_failures", 0)) > int(
        base.get("metrics", {}).get("critical_safety_failures", 0)
    ):
        raise PublicationError(f"{expected_split} adapter has a safety regression")

    rows: list[dict[str, Any]] = []
    for task_id in sorted(base_scores):
        base_score = base_scores[task_id]
        adapter_score = adapter_scores[task_id]
        base_checks = score_checks(base_score)
        adapter_checks = score_checks(adapter_score)
        rows.append(
            {
                "split": expected_split,
                "task_id": task_id,
                "base_passed": base_score.get("passed") is True,
                "adapter_passed": adapter_score.get("passed") is True,
                "base_tool_calls_functional": base_checks[
                    "tool_calls_functional_order"
                ],
                "adapter_tool_calls_functional": adapter_checks[
                    "tool_calls_functional_order"
                ],
                "base_tool_calls_exact": base_checks["tool_calls_exact_order"],
                "adapter_tool_calls_exact": adapter_checks[
                    "tool_calls_exact_order"
                ],
                "base_final_answer_exact": base_checks["final_answer_exact"],
                "adapter_final_answer_exact": adapter_checks["final_answer_exact"],
            }
        )
    return rows


def aggregate_rows(
    *,
    split: str,
    base_report: dict[str, Any],
    adapter_report: dict[str, Any],
) -> list[dict[str, Any]]:
    base = single_candidate(base_report, label=f"{split} base")
    adapter = single_candidate(adapter_report, label=f"{split} adapter")
    definitions = (
        ("overall_task_success", ("overall",)),
        (
            "functional_tool_calls",
            ("check_pass_rates", "tool_calls_functional_order"),
        ),
        ("exact_tool_calls", ("check_pass_rates", "tool_calls_exact_order")),
        ("exact_final_answer", ("check_pass_rates", "final_answer_exact")),
    )
    rows: list[dict[str, Any]] = []
    for name, path in definitions:
        base_metric = metric(base, path)
        adapter_metric = metric(adapter, path)
        base_rate = float(base_metric.get("pass_rate", 0.0))
        adapter_rate = float(adapter_metric.get("pass_rate", 0.0))
        rows.append(
            {
                "split": split,
                "metric": name,
                "base_passed": int(base_metric.get("passed", 0)),
                "base_total": int(base_metric.get("total", 0)),
                "base_rate": base_rate,
                "adapter_passed": int(adapter_metric.get("passed", 0)),
                "adapter_total": int(adapter_metric.get("total", 0)),
                "adapter_rate": adapter_rate,
                "absolute_delta": adapter_rate - base_rate,
            }
        )
    return rows


def assert_public_safe(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    forbidden = (
        "/Users/",
        "HF_TOKEN",
        "GITHUB_TOKEN",
        "Authorization: Bearer",
        "BEGIN PRIVATE KEY",
    )
    matches = [item for item in forbidden if item in text]
    if matches:
        raise PublicationError(
            f"public artifact {path} contains forbidden text: {', '.join(matches)}"
        )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise PublicationError(f"refusing to write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def export_bundle(
    *,
    development_adapter_report: Path,
    development_base_report: Path,
    sealed_adapter_report: Path,
    sealed_base_report: Path,
    training_result: Path,
    trainer_state: Path,
    validation: Path,
    dataset_manifest: Path,
    model_manifest: Path,
    adapter_config: Path,
    chat_template: Path,
    output_dir: Path,
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise PublicationError(f"output directory is not empty: {output_dir}")

    reports = {
        "development_adapter_evaluation.json": development_adapter_report,
        "development_base_evaluation.json": development_base_report,
        "sealed_adapter_evaluation.json": sealed_adapter_report,
        "sealed_base_evaluation_posthoc.json": sealed_base_report,
    }
    loaded = {name: load_json_object(path) for name, path in reports.items()}
    paired_rows = validate_pair(
        loaded["development_base_evaluation.json"],
        loaded["development_adapter_evaluation.json"],
        expected_split="development",
    )
    paired_rows.extend(
        validate_pair(
            loaded["sealed_base_evaluation_posthoc.json"],
            loaded["sealed_adapter_evaluation.json"],
            expected_split="sealed_final",
        )
    )
    summary_rows = aggregate_rows(
        split="development",
        base_report=loaded["development_base_evaluation.json"],
        adapter_report=loaded["development_adapter_evaluation.json"],
    )
    summary_rows.extend(
        aggregate_rows(
            split="sealed_final",
            base_report=loaded["sealed_base_evaluation_posthoc.json"],
            adapter_report=loaded["sealed_adapter_evaluation.json"],
        )
    )

    training = load_json_object(training_result)
    if training.get("schema_version") != TRAINING_RESULT_SCHEMA:
        raise PublicationError("training result has unsupported schema")
    if training.get("status") != "succeeded":
        raise PublicationError("training result did not succeed")
    campaign_validation = load_json_object(validation)
    if campaign_validation.get("passed") is not True:
        raise PublicationError("campaign validation did not pass")
    if int(campaign_validation.get("error_count", 0)) != 0:
        raise PublicationError("campaign validation contains errors")
    model = load_json_object(model_manifest)
    model_id = model.get("model_id")
    model_revision = model.get("source", {}).get("revision")
    if not isinstance(model_id, str) or not model_id:
        raise PublicationError("model manifest has no model_id")
    if not isinstance(model_revision, str) or not model_revision:
        raise PublicationError("model manifest has no source revision")
    public_adapter_config = load_json_object(adapter_config)
    public_adapter_config["base_model_name_or_path"] = model_id
    public_adapter_config["revision"] = model_revision
    for safe_path in (
        development_adapter_report,
        development_base_report,
        sealed_adapter_report,
        sealed_base_report,
        training_result,
        validation,
        dataset_manifest,
        model_manifest,
        chat_template,
    ):
        assert_public_safe(safe_path)

    state = load_json_object(trainer_state)
    history = state.get("log_history")
    if not isinstance(history, list):
        raise PublicationError("trainer state has no log history")
    curve_rows = [
        {
            "step": int(item["step"]),
            "epoch": item.get("epoch"),
            "loss": item.get("loss"),
            "mean_token_accuracy": item.get("mean_token_accuracy"),
            "learning_rate": item.get("learning_rate"),
            "grad_norm": item.get("grad_norm"),
            "entropy": item.get("entropy"),
            "num_tokens": item.get("num_tokens"),
        }
        for item in history
        if isinstance(item, dict) and "step" in item and "loss" in item
    ]
    if not curve_rows:
        raise PublicationError("trainer state has no loss-bearing history rows")

    output_dir.mkdir(parents=True, exist_ok=True)
    copied = {
        **reports,
        "training_result.json": training_result,
        "campaign_validation.json": validation,
        "dataset_manifest.json": dataset_manifest,
        "model_manifest.json": model_manifest,
        "chat_template.jinja": chat_template,
    }
    for name, source in copied.items():
        shutil.copyfile(source, output_dir / name)
    (output_dir / "adapter_config.json").write_text(
        json.dumps(public_adapter_config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    assert_public_safe(output_dir / "adapter_config.json")
    write_csv(output_dir / "metrics_summary.csv", summary_rows)
    write_csv(output_dir / "paired_task_scores.csv", paired_rows)
    write_csv(output_dir / "training_curve.csv", curve_rows)

    published = sorted(path for path in output_dir.iterdir() if path.is_file())
    checksum_path = output_dir / "SHA256SUMS"
    checksum_path.write_text(
        "".join(f"{sha256_file(path)}  {path.name}\n" for path in published),
        encoding="utf-8",
    )
    return {
        "output_dir": str(output_dir),
        "file_count": len(published) + 1,
        "development_task_count": sum(
            row["split"] == "development" for row in paired_rows
        ),
        "sealed_task_count": sum(
            row["split"] == "sealed_final" for row in paired_rows
        ),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-adapter-report", type=Path, required=True)
    parser.add_argument("--development-base-report", type=Path, required=True)
    parser.add_argument("--sealed-adapter-report", type=Path, required=True)
    parser.add_argument("--sealed-base-report", type=Path, required=True)
    parser.add_argument("--training-result", type=Path, required=True)
    parser.add_argument("--trainer-state", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--adapter-config", type=Path, required=True)
    parser.add_argument("--chat-template", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = export_bundle(**vars(args))
    print(
        "RUNTIME_ADAPTER_PUBLICATION "
        f"files={result['file_count']} "
        f"development_tasks={result['development_task_count']} "
        f"sealed_tasks={result['sealed_task_count']} "
        f"out={result['output_dir']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
