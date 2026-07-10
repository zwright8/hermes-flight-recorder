"""Held-out scenario manifests for eval and external-adapter handoffs."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .schema_registry import check_schema_contract

HELDOUT_MANIFEST_SCHEMA_VERSION = "hfr.heldout_scenario_manifest.v1"
RUN_SUITE_SCHEMA_VERSION = "hfr.run_suite.v1"


class HeldoutManifestError(ValueError):
    """Raised when a held-out scenario manifest cannot be built."""


@dataclass(frozen=True)
class LabeledPath:
    label: str
    path: Path


def build_heldout_manifest(
    *,
    suite_summary_specs: list[str | Path],
    preserve_paths: bool = False,
) -> dict[str, Any]:
    """Build a manifest of held-out scenario IDs from one or more suite summaries."""
    specs = [_labeled_path(spec) for spec in suite_summary_specs]
    if not specs:
        raise HeldoutManifestError("At least one --suite-summary is required")
    sources = [_source_from_suite_summary(spec, preserve_paths) for spec in specs]
    status, scenario_ids, mismatches, blocking_reasons = _manifest_status(sources)
    ready = bool(scenario_ids) and not blocking_reasons
    identical = status == "identical"
    return {
        "schema_version": HELDOUT_MANIFEST_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "ready": ready,
        "status": status,
        "identical": identical,
        "cross_arm_claims_allowed": identical,
        "source_count": len(sources),
        "scenario_count": len(scenario_ids),
        "scenario_ids": scenario_ids,
        "sources": sources,
        "mismatches": mismatches,
        "blocking_reasons": blocking_reasons,
        "governance_handoff": {
            "external_adapter_manifest_allowed": ready,
            "cross_arm_claims_allowed": identical,
            "recommendation": _recommendation(status, ready),
        },
    }


def write_heldout_manifest(manifest: dict[str, Any], out_path: str | Path) -> None:
    """Write a held-out manifest as stable JSON."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.loads(json.dumps(manifest))
    for source in payload.get("sources", []):
        if isinstance(source, dict) and isinstance(source.get("path"), str):
            source["path"] = _output_relative_path(source.get("path"), path.parent)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _source_from_suite_summary(spec: LabeledPath, preserve_paths: bool) -> dict[str, Any]:
    summary = _read_object(spec.path, "suite summary")
    runs = summary.get("runs") if isinstance(summary.get("runs"), list) else []
    seen: set[str] = set()
    duplicates: set[str] = set()
    scenario_ids: list[str] = []
    scenario_fingerprints: dict[str, str] = {}
    for run in runs:
        if not isinstance(run, dict):
            continue
        scenario_id = run.get("scenario_id")
        if not isinstance(scenario_id, str) or not scenario_id:
            continue
        if scenario_id in seen:
            duplicates.add(scenario_id)
        seen.add(scenario_id)
        scenario_ids.append(scenario_id)
        scenario_sha = run.get("scenario_sha256")
        if isinstance(scenario_sha, str) and scenario_sha:
            scenario_fingerprints[scenario_id] = scenario_sha
    unique_scenarios = sorted(set(scenario_ids))
    blocking_reasons: list[str] = []
    schema_check = check_schema_contract(summary, name_or_id="run_suite")
    if summary.get("schema_version") != RUN_SUITE_SCHEMA_VERSION or schema_check["passed"] is not True:
        blocking_reasons.append("invalid_suite_summary_schema")
    if not unique_scenarios:
        blocking_reasons.append("empty_suite_summary")
    if int(summary.get("error_count", 0) or 0) > 0:
        blocking_reasons.append("suite_summary_errors")
    if duplicates:
        blocking_reasons.append("duplicate_scenario_ids")
    return {
        "label": spec.label,
        "path": _display_path(spec.path, preserve_paths),
        "schema_version": summary.get("schema_version"),
        "scenario_count": len(unique_scenarios),
        "scenario_ids": unique_scenarios,
        "scenario_fingerprints": dict(sorted(scenario_fingerprints.items())),
        "duplicate_scenario_ids": sorted(duplicates),
        "blocking_reasons": blocking_reasons,
    }


def _manifest_status(sources: list[dict[str, Any]]) -> tuple[str, list[str], list[dict[str, Any]], list[str]]:
    blocking_reasons = sorted({reason for source in sources for reason in source["blocking_reasons"]})
    reference = sources[0]["scenario_ids"] if sources else []
    mismatches: list[dict[str, Any]] = []
    for source in sources[1:]:
        current = source["scenario_ids"]
        if current != reference:
            mismatches.append(
                {
                    "label": source["label"],
                    "missing_from_source": sorted(set(reference) - set(current)),
                    "extra_in_source": sorted(set(current) - set(reference)),
                }
            )
    if not reference or any(not source["scenario_ids"] for source in sources):
        return "empty", [], [], sorted(set(blocking_reasons + ["empty_heldout_scenario_set"]))
    if blocking_reasons:
        return "blocked", reference, mismatches, blocking_reasons
    if len(sources) == 1:
        return "single_source", reference, [], []
    if mismatches:
        return "mismatched", reference, mismatches, ["heldout_scenario_set_mismatch"]
    return "identical", reference, [], []


def _recommendation(status: str, ready: bool) -> str:
    if ready and status == "identical":
        return "Held-out scenarios are identical across arms; cross-arm claims may use this manifest."
    if ready:
        return "Manifest can seed external adapter planning, but cross-arm claims still need another arm with the identical scenario set."
    if status == "mismatched":
        return "Do not use this manifest for promotion or external adapter claims until scenario sets match exactly."
    return "Resolve manifest blockers before using this held-out scenario set."


def _labeled_path(spec: str | Path) -> LabeledPath:
    text = str(spec)
    if "=" in text:
        label, raw_path = text.split("=", 1)
        if label and raw_path:
            return LabeledPath(label=label, path=Path(raw_path))
    path = Path(text)
    return LabeledPath(label=_default_label(path), path=path)


def _default_label(path: Path) -> str:
    if path.name == "suite_summary.json" and path.parent.name:
        return path.parent.name
    return path.stem or path.name or "heldout"


def _read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise HeldoutManifestError(f"{label} not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise HeldoutManifestError(f"Invalid JSON in {label} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise HeldoutManifestError(f"{label} must be a JSON object: {path}")
    return payload


def _display_path(path: Path, preserve_paths: bool) -> str:
    if preserve_paths:
        return str(path)
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except (OSError, ValueError):
        return str(path)


def _output_relative_path(value: Any, output_dir: Path) -> Any:
    if not isinstance(value, str) or not value:
        return value
    path = Path(value)
    if not path.is_absolute():
        if not path.exists():
            return value
        path = path.resolve()
    return os.path.relpath(path.resolve(), output_dir.resolve())
