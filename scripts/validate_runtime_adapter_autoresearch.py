#!/usr/bin/env python3
"""Validate a governed runtime-adapter autoresearch campaign outcome."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flightrecorder.lora_recipe_search import validate_search_result  # noqa: E402

SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from evaluate_runtime_adapter_candidates import validate_evaluation_report  # noqa: E402
from run_runtime_adapter_autoresearch import (  # noqa: E402
    CAMPAIGN_RECORD,
    SEALED_RECEIPT,
    SEALED_REPORT,
    atomic_write_json,
    champion_candidate_record,
    runner_paths,
    sha256_file,
)


def validate_campaign(
    *,
    campaign_dir: Path,
    out_path: Path | None = None,
    release_check_receipt: Path | None = None,
) -> dict[str, Any]:
    paths = runner_paths(campaign_dir)
    errors: list[str] = []
    search_validation = validate_search_result(paths.result)
    errors.extend(f"search_result: {error}" for error in search_validation["errors"])

    campaign = _load_json(paths.campaign_record, errors, "campaign_record")
    search = _load_json(paths.result, errors, "search_result")
    sealed_receipt = _load_json(paths.root / SEALED_RECEIPT, errors, "sealed_receipt")
    sealed_report = _load_json(paths.root / SEALED_REPORT, errors, "sealed_report")

    champion: dict[str, Any] = {}
    champion_candidate: dict[str, Any] = {}
    if isinstance(search, dict):
        champion_value = search.get("champion")
        if isinstance(champion_value, dict):
            champion = champion_value
        else:
            errors.append("search_result has no champion")
    if isinstance(campaign, dict) and champion:
        try:
            champion_record = champion_candidate_record(campaign, champion)
        except SystemExit as exc:
            errors.append(f"campaign_record: {exc}")
        else:
            candidate_value = champion_record.get("candidate")
            if isinstance(candidate_value, dict):
                champion_candidate = candidate_value
            else:
                errors.append("campaign_record champion has no candidate object")

    if isinstance(campaign, dict):
        records = campaign.get("candidate_records")
        if not isinstance(records, list):
            errors.append("campaign_record candidate_records is missing")
        else:
            for index, record in enumerate(records):
                if not isinstance(record, dict) or record.get("status") != "evaluated":
                    continue
                report_path = record.get("development_report")
                if not isinstance(report_path, str) or not report_path:
                    errors.append(
                        f"campaign_record candidate_records[{index}] has no development_report"
                    )
                    continue
                development_report = _load_json(
                    Path(report_path),
                    errors,
                    f"development_report[{index}]",
                )
                split = (
                    development_report.get("heldout", {}).get("split")
                    if isinstance(development_report, dict)
                    else None
                )
                if split != "development":
                    errors.append(
                        f"development_report[{index}] must declare development split"
                    )
                development_report_path = Path(report_path)
                if isinstance(development_report, dict):
                    semantic_errors = validate_evaluation_report(development_report)
                    errors.extend(
                        f"development_report[{index}]: {error}"
                        for error in semantic_errors
                    )
                    if development_report_path.is_file() and record.get(
                        "development_report_sha256"
                    ) != sha256_file(development_report_path):
                        errors.append(
                            f"development_report[{index}] hash does not match campaign record"
                        )

    if isinstance(sealed_report, dict):
        semantic_errors = validate_evaluation_report(sealed_report)
        errors.extend(f"sealed_report: {error}" for error in semantic_errors)
        if sealed_report.get("heldout", {}).get("split") != "sealed_final":
            errors.append("sealed_report must declare sealed_final split")
        eligible = sealed_report.get("promotion_eligible_candidates")
        if not isinstance(eligible, list) or len(eligible) != 1:
            errors.append("sealed_report must have exactly one promotion-eligible champion")
        elif champion_candidate and eligible[0] != champion_candidate.get("candidate_id"):
            errors.append("sealed_report promotion candidate does not match champion candidate_id")
        candidate_reports = [
            item
            for item in sealed_report.get("candidate_reports", [])
            if isinstance(item, dict)
        ]
        if len(candidate_reports) != 1:
            errors.append("sealed_report must contain exactly one candidate report")
        elif champion_candidate:
            report = candidate_reports[0]
            identity = report.get("identity", {}) if isinstance(report.get("identity"), dict) else {}
            adapter = identity.get("adapter", {}) if isinstance(identity.get("adapter"), dict) else {}
            if adapter.get("sha256") != champion_candidate.get("adapter_sha256"):
                errors.append("sealed_report adapter hash does not match champion adapter")
            metrics = report.get("metrics", {}) if isinstance(report.get("metrics"), dict) else {}
            if metrics.get("critical_safety_failures") != 0:
                errors.append("sealed_report has critical safety failures")
            if report.get("promotion_eligible") is not True:
                errors.append("sealed champion is not promotion eligible")

    if isinstance(sealed_receipt, dict) and champion_candidate:
        if sealed_receipt.get("status") != "completed":
            errors.append("sealed_receipt is not completed")
        if sealed_receipt.get("campaign_id") != campaign.get("campaign_id"):
            errors.append("sealed_receipt campaign_id does not match campaign")
        if sealed_receipt.get("candidate_identity_sha256") != champion_candidate.get("candidate_identity_sha256"):
            errors.append("sealed_receipt candidate identity does not match champion")
        expected_model_identity = {
            "model_id": champion_candidate.get("base_model"),
            "model_revision": champion_candidate.get("base_revision"),
            "tokenizer_revision": champion_candidate.get("tokenizer_revision"),
            "chat_template_sha256": champion_candidate.get("chat_template_sha256"),
        }
        if sealed_receipt.get("model_identity") != expected_model_identity:
            errors.append("sealed_receipt model identity does not match champion")
        sealed_report_path = paths.root / SEALED_REPORT
        if sealed_report_path.is_file() and sealed_receipt.get(
            "sealed_report_sha256"
        ) != sha256_file(sealed_report_path):
            errors.append("sealed_receipt report hash does not match sealed report")
        if sealed_receipt.get("sealed_access", {}).get("during_search") is not False:
            errors.append("sealed_receipt reports sealed access during search")

    if release_check_receipt is not None:
        release = _load_json(release_check_receipt, errors, "release_check_receipt")
        if isinstance(release, dict) and release.get("passed") is not True:
            errors.append("release_check_receipt did not pass")

    result = {
        "schema_version": "hfr.runtime_adapter_autoresearch_validation.v1",
        "passed": not errors,
        "status": "passed" if not errors else "failed",
        "campaign_dir": str(campaign_dir),
        "search_result_replayed": search_validation["passed"],
        "sealed_promotion_eligible": not errors
        and isinstance(sealed_report, dict)
        and sealed_report.get("passed") is True,
        "error_count": len(errors),
        "errors": errors,
    }
    if out_path is not None:
        atomic_write_json(out_path, result)
    return result


def _load_json(path: Path, errors: list[str], label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        errors.append(f"{label}: {exc}")
        return {}
    except json.JSONDecodeError as exc:
        errors.append(f"{label}: invalid JSON: {exc}")
        return {}
    if not isinstance(value, dict):
        errors.append(f"{label}: expected JSON object")
        return {}
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--release-check-receipt",
        type=Path,
        help="Optional JSON receipt from release_check.sh or an equivalent clean-worktree release gate.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = validate_campaign(
        campaign_dir=args.campaign_dir,
        out_path=args.out,
        release_check_receipt=args.release_check_receipt,
    )
    print(
        "RUNTIME_ADAPTER_AUTORESEARCH_VALIDATION "
        f"passed={result['passed']} errors={result['error_count']}"
    )
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
