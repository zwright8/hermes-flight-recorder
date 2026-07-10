"""Held-out scenario manifests for eval and external-adapter handoffs."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath
from typing import Any

from .schema_registry import check_schema_contract

HELDOUT_MANIFEST_SCHEMA_VERSION = "hfr.heldout_scenario_manifest.v1"
RUN_SUITE_SCHEMA_VERSION = "hfr.run_suite.v1"
MAX_SCENARIO_BYTES = 4 * 1024 * 1024


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
    public_sources = [
        {key: value for key, value in source.items() if not key.startswith("_")}
        for source in sources
    ]
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
        "sources": public_sources,
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
    return {
        "label": spec.label,
        "path": _display_path(spec.path, preserve_paths),
        "_source_identity": str(spec.path.resolve(strict=False)),
        "_source_content_identity": _sha256_file(spec.path),
        **_source_fields_from_suite_summary(summary, source_path=spec.path),
    }


def _source_fields_from_suite_summary(
    summary: dict[str, Any],
    *,
    source_path: Path | None = None,
) -> dict[str, Any]:
    """Derive the canonical held-out identity projection for a suite summary."""
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
        if _is_lowercase_sha256(scenario_sha):
            scenario_fingerprints[scenario_id] = scenario_sha
    unique_scenarios = sorted(set(scenario_ids))
    blocking_reasons: list[str] = []
    schema_check = check_schema_contract(summary, name_or_id="run_suite")
    if summary.get("schema_version") != RUN_SUITE_SCHEMA_VERSION or schema_check["passed"] is not True:
        blocking_reasons.append("invalid_suite_summary_schema")
    if not unique_scenarios:
        blocking_reasons.append("empty_suite_summary")
    error_count = summary.get("error_count", 0)
    if isinstance(error_count, int) and not isinstance(error_count, bool) and error_count > 0:
        blocking_reasons.append("suite_summary_errors")
    if duplicates:
        blocking_reasons.append("duplicate_scenario_ids")
    if set(scenario_fingerprints) != set(unique_scenarios):
        blocking_reasons.append("missing_scenario_fingerprints")
    if unique_scenarios and not _scenario_fingerprints_replay(summary, source_path):
        blocking_reasons.append("scenario_fingerprint_replay_failed")
    return {
        "schema_version": summary.get("schema_version"),
        "scenario_count": len(unique_scenarios),
        "scenario_ids": unique_scenarios,
        "scenario_fingerprints": dict(sorted(scenario_fingerprints.items())),
        "duplicate_scenario_ids": sorted(duplicates),
        "blocking_reasons": blocking_reasons,
    }


def _manifest_status(sources: list[dict[str, Any]]) -> tuple[str, list[str], list[dict[str, Any]], list[str]]:
    blocking_reasons = sorted({reason for source in sources for reason in source["blocking_reasons"]})
    labels = [source["label"] for source in sources]
    paths = [source.get("_source_identity", source["path"]) for source in sources]
    content_identities = [source.get("_source_content_identity") for source in sources]
    if len(set(labels)) != len(labels):
        blocking_reasons.append("duplicate_heldout_source_labels")
    if len(set(paths)) != len(paths):
        blocking_reasons.append("duplicate_heldout_source_paths")
    if (
        all(isinstance(identity, str) and identity for identity in content_identities)
        and len(set(content_identities)) != len(content_identities)
    ):
        blocking_reasons.append("duplicate_heldout_source_content")
    blocking_reasons = sorted(set(blocking_reasons))
    reference = sources[0]["scenario_ids"] if sources else []
    reference_fingerprints = sources[0]["scenario_fingerprints"] if sources else {}
    mismatches: list[dict[str, Any]] = []
    for source in sources[1:]:
        current = source["scenario_ids"]
        current_fingerprints = source["scenario_fingerprints"]
        fingerprint_mismatches = [
            {
                "scenario_id": scenario_id,
                "reference_sha256": reference_fingerprints.get(scenario_id),
                "source_sha256": current_fingerprints.get(scenario_id),
            }
            for scenario_id in sorted(set(reference) & set(current))
            if reference_fingerprints.get(scenario_id) != current_fingerprints.get(scenario_id)
        ]
        if current != reference or fingerprint_mismatches:
            mismatches.append(
                {
                    "label": source["label"],
                    "missing_from_source": sorted(set(reference) - set(current)),
                    "extra_in_source": sorted(set(current) - set(reference)),
                    "fingerprint_mismatches": fingerprint_mismatches,
                }
            )
    if not reference or any(not source["scenario_ids"] for source in sources):
        return "empty", [], [], sorted(set(blocking_reasons + ["empty_heldout_scenario_set"]))
    if blocking_reasons:
        return "blocked", reference, mismatches, blocking_reasons
    if len(sources) == 1:
        return "single_source", reference, [], []
    if mismatches:
        mismatch_reasons: list[str] = []
        if any(row["missing_from_source"] or row["extra_in_source"] for row in mismatches):
            mismatch_reasons.append("heldout_scenario_set_mismatch")
        if any(row["fingerprint_mismatches"] for row in mismatches):
            mismatch_reasons.append("heldout_scenario_fingerprint_mismatch")
        return "mismatched", reference, mismatches, mismatch_reasons
    return "identical", reference, [], []


def _recommendation(status: str, ready: bool) -> str:
    if ready and status == "identical":
        return "Held-out scenarios are identical across arms; cross-arm claims may use this manifest."
    if ready:
        return "Manifest can seed external adapter planning, but cross-arm claims still need another arm with the identical scenario set."
    if status == "mismatched":
        return "Do not use this manifest for promotion or external adapter claims until scenario sets and content fingerprints match exactly."
    return "Resolve manifest blockers before using this held-out scenario set."


def _is_lowercase_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _scenario_fingerprints_replay(summary: dict[str, Any], source_path: Path | None) -> bool:
    if source_path is None:
        return False
    runs = summary.get("runs") if isinstance(summary.get("runs"), list) else []
    for run in runs:
        if not isinstance(run, dict) or not isinstance(run.get("scenario_id"), str) or not run.get("scenario_id"):
            continue
        raw_path = run.get("scenario_path")
        expected_sha = run.get("scenario_sha256")
        if not isinstance(raw_path, str) or not raw_path or not _is_lowercase_sha256(expected_sha):
            return False
        if raw_path.startswith("<redacted:"):
            return False
        relative_path = Path(raw_path)
        windows_path = PureWindowsPath(raw_path)
        if (
            relative_path.is_absolute()
            or windows_path.is_absolute()
            or bool(windows_path.drive)
            or ".." in relative_path.parts
            or ".." in windows_path.parts
            or "\\" in raw_path
        ):
            return False
        scenario_path = source_path.parent / relative_path
        if _path_has_symlink_component(scenario_path, root=source_path.parent) or not scenario_path.is_file():
            return False
        try:
            if scenario_path.stat().st_size > MAX_SCENARIO_BYTES:
                return False
            actual_sha = _sha256_file(scenario_path)
        except OSError:
            return False
        if actual_sha != expected_sha:
            return False
    return True


def _path_has_symlink_component(path: Path, *, root: Path) -> bool:
    try:
        relative = path.relative_to(root)
        current = root
        for part in relative.parts:
            current /= part
            if current.is_symlink():
                return True
        return False
    except (OSError, ValueError):
        return True


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
