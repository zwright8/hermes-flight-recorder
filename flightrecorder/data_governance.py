"""Deterministic governance, erasure, and contamination controls for training data.

The core deliberately performs no provider calls.  It produces content-bound
receipts that an opt-in controller can use before launching external training.
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable

from .atomic_json import atomic_write_json_cas, json_file_sha256
from .path_safety import path_has_symlink_component

DATA_GOVERNANCE_RECEIPT_SCHEMA_VERSION = "hfr.data_governance_receipt.v1"
DATASET_CONTAMINATION_REPORT_SCHEMA_VERSION = "hfr.dataset_contamination_report.v1"
DATASET_DELETION_RECEIPT_SCHEMA_VERSION = "hfr.dataset_deletion_receipt.v1"
DATASET_DERIVATION_SCHEMA_VERSION = "hfr.dataset_derivation.v1"

_REDACTED_RE = re.compile(r"(?i)(?:\[REDACTED\]|<redacted:[^>]+>)")
_EMAIL_RE = re.compile(r"(?<![\w.+-])[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}(?![\w-])", re.IGNORECASE)
_PHONE_RE = re.compile(r"(?<!\w)(?:\+?1[ .-]?)?(?:\(?\d{3}\)?[ .-]?)\d{3}[ .-]\d{4}(?!\w)")
_ADDRESS_RE = re.compile(
    r"(?i)\b\d{1,6}\s+(?:[A-Z0-9][\w.'-]*\s+){0,5}"
    r"(?:street|st|avenue|ave|road|rd|boulevard|blvd|lane|ln|drive|dr|court|ct|way)\b"
)
_CUSTOMER_ID_RE = re.compile(
    r"(?i)\b(?:customer|account|patient|member|client|cust|acct)[\s_:#-]+"
    r"(?=[A-Z0-9_-]*\d)[A-Z0-9][A-Z0-9_-]{3,}\b"
)
_TITLED_NAME_RE = re.compile(r"\b(?:Mr|Mrs|Ms|Miss|Dr|Prof)\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}\b")
_NAME_KEY_RE = re.compile(r"(?i)(?:^|[_-])(?:full_?)?name(?:$|[_-])")
_IP_CANDIDATE_RE = re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?!\d)")
_TOKEN_RE = re.compile(r"[a-z0-9]+")

_REQUIRED_GOVERNANCE_FIELDS = (
    "owner",
    "tenant",
    "legal_basis",
    "allowed_purposes",
    "sensitivity",
    "jurisdiction",
    "retention_expires_at",
    "license",
    "provenance",
    "deletion_subject_ids",
)


class DataGovernanceError(ValueError):
    """Raised when a governance operation cannot be completed safely."""


def scan_personal_data(
    payload: Any,
    *,
    organization_entities: Iterable[str] = (),
) -> list[dict[str, Any]]:
    """Return redacted finding metadata for common personal-data classes.

    Matched values are intentionally never copied into findings.  Structured
    name fields complement conservative free-text patterns without requiring a
    probabilistic third-party recognizer.
    """

    organizations = tuple(sorted({str(value).strip() for value in organization_entities if str(value).strip()}))
    findings: list[dict[str, Any]] = []

    def append(kind: str, path: str, *, detector: str) -> None:
        key = (kind, path, detector)
        if any((row["kind"], row["path"], row["detector"]) == key for row in findings):
            return
        findings.append(
            {
                "kind": kind,
                "path": path,
                "detector": detector,
                "preview": f"{kind} value omitted",
            }
        )

    def visit(value: Any, path: str, key_name: str = "") -> None:
        if isinstance(value, dict):
            for key, child in sorted(value.items(), key=lambda item: str(item[0])):
                normalized = str(key).replace(".", "_")
                visit(child, f"{path}.{normalized}", normalized)
            return
        if isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, f"{path}[{index}]", key_name)
            return
        if not isinstance(value, str) or not value.strip():
            return
        scan_value = _REDACTED_RE.sub("", value)
        if not scan_value.strip():
            return
        if _NAME_KEY_RE.search(key_name) and len(scan_value.split()) >= 2:
            append("person_name", path, detector="structured_name_field")
        for regex, kind, detector in (
            (_EMAIL_RE, "email", "email_regex"),
            (_PHONE_RE, "phone", "north_american_phone_regex"),
            (_ADDRESS_RE, "postal_address", "street_address_regex"),
            (_CUSTOMER_ID_RE, "customer_id", "customer_identifier_regex"),
            (_TITLED_NAME_RE, "person_name", "titled_name_regex"),
        ):
            if regex.search(scan_value):
                append(kind, path, detector=detector)
        for candidate in _IP_CANDIDATE_RE.findall(scan_value):
            try:
                ipaddress.ip_address(candidate)
            except ValueError:
                continue
            append("ip_address", path, detector="ip_address_parser")
            break
        lowered = scan_value.casefold()
        for organization in organizations:
            if organization.casefold() in lowered:
                append("organization_entity", path, detector="configured_entity")
                break

    visit(payload, "$")
    return findings


def build_governance_receipt(
    records: list[dict[str, Any]],
    *,
    purpose: str,
    now: str | datetime | None = None,
    organization_entities: Iterable[str] = (),
    policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fail closed unless every record is governed, authorized, current, and clean."""

    instant = _datetime(now)
    baseline_policy = {
        "purpose": purpose,
        "required_fields": list(_REQUIRED_GOVERNANCE_FIELDS),
        "pii_policy": "block_unredacted",
        "unknown_license_allowed": False,
    }
    requested_policy = dict(policy or {})
    policy_violations: list[str] = []
    if "purpose" in requested_policy and requested_policy["purpose"] != purpose:
        policy_violations.append("purpose_override_forbidden")
    if "pii_policy" in requested_policy and requested_policy["pii_policy"] != "block_unredacted":
        policy_violations.append("pii_policy_weakening_forbidden")
    if requested_policy.get("unknown_license_allowed") not in {None, False}:
        policy_violations.append("unknown_license_policy_weakening_forbidden")
    if "required_fields" in requested_policy:
        requested_fields = requested_policy["required_fields"]
        if not isinstance(requested_fields, list) or not set(_REQUIRED_GOVERNANCE_FIELDS).issubset(
            {str(value) for value in requested_fields}
        ):
            policy_violations.append("required_governance_fields_weakening_forbidden")
    effective_policy = {
        **requested_policy,
        **baseline_policy,
    }
    statuses: list[dict[str, Any]] = []
    all_findings: list[dict[str, Any]] = []
    blocked_reasons: set[str] = set(policy_violations)
    authorized = 0
    for index, record in enumerate(records):
        record_id = str(record.get("episode_id") or record.get("row_id") or f"row-{index}")
        governance = record.get("governance") if isinstance(record.get("governance"), dict) else {}
        missing = [field for field in _REQUIRED_GOVERNANCE_FIELDS if not _present(governance.get(field))]
        reasons: list[str] = []
        if missing:
            reasons.append("missing_governance_metadata")
        allowed = governance.get("allowed_purposes") if isinstance(governance.get("allowed_purposes"), list) else []
        if purpose not in allowed:
            reasons.append("purpose_not_authorized")
        legal_basis = str(governance.get("legal_basis") or "").strip().casefold()
        if legal_basis == "consent" and not _present(governance.get("consent_id")):
            reasons.append("consent_identifier_missing")
        license_value = str(governance.get("license") or "").strip().casefold()
        if license_value in {"", "unknown", "none", "unlicensed"}:
            reasons.append("license_not_approved")
        expiry = _optional_datetime(governance.get("retention_expires_at"))
        if expiry is None:
            reasons.append("retention_invalid")
        elif expiry <= instant:
            reasons.append("retention_expired")

        trainable_payload = {key: value for key, value in record.items() if key != "governance"}
        findings = scan_personal_data(trainable_payload, organization_entities=organization_entities)
        for finding in findings:
            all_findings.append({"record_id": record_id, **finding})
        if findings and effective_policy.get("pii_policy") == "block_unredacted":
            reasons.append("unredacted_personal_data")

        reasons = sorted(set(reasons))
        blocked_reasons.update(reasons)
        if policy_violations:
            reasons.extend(policy_violations)
            reasons = sorted(set(reasons))
        passed = not reasons
        authorized += int(passed)
        statuses.append(
            {
                "record_id": record_id,
                "passed": passed,
                "blocked_reasons": reasons,
                "governance_fingerprint": _canonical_sha256(governance) if governance else "",
                "pii_finding_count": len(findings),
            }
        )

    policy_fingerprint = _canonical_sha256(effective_policy)
    scan_fingerprint = _canonical_sha256(
        [{key: value for key, value in row.items() if key != "preview"} for row in all_findings]
    )
    return {
        "schema_version": DATA_GOVERNANCE_RECEIPT_SCHEMA_VERSION,
        "passed": len(records) > 0 and not blocked_reasons,
        "purpose": purpose,
        "record_count": len(records),
        "authorized_record_count": authorized,
        "blocked_record_count": len(records) - authorized,
        "pii_finding_count": len(all_findings),
        "blocked_reasons": sorted(blocked_reasons) or ([] if records else ["no_records"]),
        "policy": effective_policy,
        "requested_policy": requested_policy,
        "policy_violations": policy_violations,
        "policy_fingerprint": policy_fingerprint,
        "scan_fingerprint": scan_fingerprint,
        "record_statuses": statuses,
        "findings": all_findings[:100],
        "evaluated_at": instant.isoformat(),
        "notes": [
            "Personal-data finding previews omit matched values.",
            "A passing receipt authorizes only the declared purpose and policy fingerprint.",
        ],
    }


def task_contract_fingerprint(row: dict[str, Any]) -> str:
    """Fingerprint observable task semantics for review and contamination checks."""

    messages = row.get("messages") if isinstance(row.get("messages"), list) else []
    system_messages = [
        message
        for message in messages
        if isinstance(message, dict) and message.get("role") in {"system", "developer"}
    ]
    payload = {
        "prompt": _normalized_text(str(row.get("prompt") or _first_user_message(messages))),
        "system_messages": system_messages,
        "tools": row.get("tools") if isinstance(row.get("tools"), list) else [],
        "environment": row.get("environment") if isinstance(row.get("environment"), dict) else {},
        "policy": row.get("policy") if isinstance(row.get("policy"), dict) else {},
        "scenario_contract": row.get("scenario_contract") if isinstance(row.get("scenario_contract"), dict) else {},
    }
    return _canonical_sha256(payload)


def build_contamination_report(
    rows: list[dict[str, Any]],
    *,
    protected_rows: list[dict[str, Any]] | None = None,
    similarity_threshold: float = 0.9,
) -> dict[str, Any]:
    """Cluster exact/near duplicates and detect split or protected-corpus leakage."""

    if not 0.0 < similarity_threshold <= 1.0:
        raise DataGovernanceError("similarity_threshold must be in (0, 1]")
    clusters = _duplicate_clusters(rows, similarity_threshold)
    cluster_rows: list[dict[str, Any]] = []
    cross_split: list[dict[str, Any]] = []
    for cluster_id, indexes in enumerate(clusters):
        ids = [_row_id(rows[index], index) for index in indexes]
        splits = sorted({str(rows[index].get("split") or "") for index in indexes if rows[index].get("split")})
        record = {
            "cluster_id": f"cluster-{cluster_id:06d}",
            "row_ids": ids,
            "row_count": len(indexes),
            "splits": splits,
            "exact_duplicate": len({_normalized_training_text(rows[index]) for index in indexes}) == 1 and len(indexes) > 1,
        }
        cluster_rows.append(record)
        if len(splits) > 1:
            cross_split.append(record)

    protected_matches: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        left = _normalized_training_text(row)
        for protected_index, protected in enumerate(protected_rows or []):
            similarity = _similarity(left, _normalized_training_text(protected))
            if similarity >= similarity_threshold:
                protected_matches.append(
                    {
                        "row_id": _row_id(row, index),
                        "protected_row_id": _row_id(protected, protected_index),
                        "similarity": round(similarity, 6),
                    }
                )

    identity = {
        "threshold": similarity_threshold,
        "clusters": cluster_rows,
        "cross_split": cross_split,
        "protected_matches": protected_matches,
    }
    return {
        "schema_version": DATASET_CONTAMINATION_REPORT_SCHEMA_VERSION,
        "passed": not cross_split and not protected_matches,
        "similarity_threshold": similarity_threshold,
        "row_count": len(rows),
        "cluster_count": len(cluster_rows),
        "duplicate_cluster_count": sum(1 for row in cluster_rows if row["row_count"] > 1),
        "cross_split_cluster_count": len(cross_split),
        "protected_match_count": len(protected_matches),
        "clusters": cluster_rows,
        "cross_split_clusters": cross_split,
        "protected_matches": protected_matches,
        "report_fingerprint": _canonical_sha256(identity),
        "blocking_reasons": [
            reason
            for condition, reason in (
                (bool(cross_split), "cross_split_duplicate_cluster"),
                (bool(protected_matches), "protected_corpus_contamination"),
            )
            if condition
        ],
    }


def assign_clustered_splits(
    rows: list[dict[str, Any]],
    *,
    similarity_threshold: float = 0.9,
    seed: str = "hfr-split-v2",
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
) -> list[dict[str, Any]]:
    """Assign duplicate clusters atomically to deterministic dataset splits."""

    if train_fraction <= 0 or validation_fraction < 0 or train_fraction + validation_fraction >= 1:
        raise DataGovernanceError("split fractions must leave a positive test fraction")
    output = [dict(row) for row in rows]
    for cluster_number, indexes in enumerate(_duplicate_clusters(rows, similarity_threshold)):
        cluster_identity = sorted(_row_id(rows[index], index) for index in indexes)
        digest = hashlib.sha256(f"{seed}:{_canonical_sha256(cluster_identity)}".encode("utf-8")).digest()
        position = int.from_bytes(digest[:8], "big") / float(2**64)
        if position < train_fraction:
            split = "train"
        elif position < train_fraction + validation_fraction:
            split = "validation"
        else:
            split = "test"
        cluster_id = f"cluster-{cluster_number:06d}-{_canonical_sha256(cluster_identity)[:12]}"
        for index in indexes:
            output[index]["split"] = split
            output[index]["duplicate_cluster_id"] = cluster_id
    return output


def build_dataset_derivation(
    *,
    parent_versions: list[str],
    recipe: dict[str, Any],
    selected_rows: list[dict[str, Any]],
    policy_fingerprint: str,
    scan_fingerprint: str,
    supersedes: list[str] | None = None,
) -> dict[str, Any]:
    """Build a content- and recipe-bound derivation record."""

    identity = {
        "parent_versions": sorted(set(parent_versions)),
        "recipe": recipe,
        "row_fingerprints": sorted(_canonical_sha256(row) for row in selected_rows),
        "policy_fingerprint": policy_fingerprint,
        "scan_fingerprint": scan_fingerprint,
        "supersedes": sorted(set(supersedes or [])),
    }
    digest = _canonical_sha256(identity)
    return {
        "schema_version": DATASET_DERIVATION_SCHEMA_VERSION,
        "derivation_id": f"hfrds-{digest[:16]}",
        "parent_versions": identity["parent_versions"],
        "recipe": recipe,
        "recipe_fingerprint": _canonical_sha256(recipe),
        "selected_row_count": len(selected_rows),
        "selected_content_fingerprint": _canonical_sha256(identity["row_fingerprints"]),
        "policy_fingerprint": policy_fingerprint,
        "scan_fingerprint": scan_fingerprint,
        "supersedes": identity["supersedes"],
        "identity_fingerprint": digest,
    }


def apply_deletion_request(
    *,
    deletion_subject_ids: list[str],
    dataset_entries: list[dict[str, Any]],
    model_entries: list[dict[str, Any]],
    output_dir: str | Path,
    request_id: str,
    erase_sources: bool = False,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Rebuild descendant JSONL datasets and quarantine lineage-dependent models.

    Rebuilt descendants are written first. Source files are erased only when
    the caller explicitly authorizes it; otherwise the receipt fails closed.
    """

    subjects = sorted({value.strip() for value in deletion_subject_ids if isinstance(value, str) and value.strip()})
    if not subjects:
        raise DataGovernanceError("at least one deletion subject id is required")
    if not dataset_entries:
        raise DataGovernanceError("at least one governed dataset entry is required")
    destination = Path(output_dir).expanduser()
    if path_has_symlink_component(destination, include_leaf=True):
        raise DataGovernanceError("deletion output directory must not traverse symlinks")
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    dataset_receipts: list[dict[str, Any]] = []
    entries_by_version: dict[str, dict[str, Any]] = {}
    source_by_version: dict[str, Path] = {}
    rows_by_version: dict[str, list[dict[str, Any]]] = {}
    parents_by_version: dict[str, list[str]] = {}
    direct_removed_by_version: dict[str, list[dict[str, Any]]] = {}
    model_versions_by_index: dict[int, set[str]] = {}
    source_paths: set[Path] = set()
    total_removed = 0

    for index, entry in enumerate(dataset_entries):
        source = Path(str(entry.get("path") or ""))
        version = str(entry.get("dataset_version") or "")
        if not version or not source.is_file() or path_has_symlink_component(source, include_leaf=True):
            raise DataGovernanceError(f"dataset entry {index} is not a safe regular file")
        source = source.resolve()
        if source == destination or source.is_relative_to(destination):
            raise DataGovernanceError(f"dataset entry {index} must be outside the deletion output directory")
        if source in source_paths:
            raise DataGovernanceError(f"dataset entry {index} aliases another source file")
        if version in entries_by_version:
            raise DataGovernanceError(f"dataset entry {index} duplicates dataset version {version!r}")
        source_paths.add(source)
        entries_by_version[version] = entry
        source_by_version[version] = source
        rows = _read_jsonl(source)
        raw_parent_versions = entry.get("parent_versions", [])
        if not isinstance(raw_parent_versions, list) or any(
            not isinstance(value, str) or not value.strip()
            for value in raw_parent_versions
        ):
            raise DataGovernanceError(
                f"dataset entry {index} parent_versions must be a list of non-empty strings"
            )
        parent_versions = sorted({value.strip() for value in raw_parent_versions})
        if version in parent_versions:
            raise DataGovernanceError(f"dataset version {version!r} cannot be its own parent")
        rows_by_version[version] = rows
        parents_by_version[version] = parent_versions
        direct_removed: list[dict[str, Any]] = []
        for row in rows:
            governance = row.get("governance") if isinstance(row.get("governance"), dict) else {}
            row_subjects = {
                value
                for value in governance.get("deletion_subject_ids", [])
                if isinstance(value, str)
            }
            if row_subjects.intersection(subjects):
                direct_removed.append(row)
        direct_removed_by_version[version] = direct_removed

    for index, model in enumerate(model_entries):
        model_id = str(model.get("model_id") or "").strip()
        if not model_id:
            raise DataGovernanceError(f"model entry {index} requires a model_id")
        raw_dataset_versions = model.get("dataset_versions", [])
        if not isinstance(raw_dataset_versions, list) or any(
            not isinstance(value, str) or not value.strip()
            for value in raw_dataset_versions
        ):
            raise DataGovernanceError(
                f"model entry {index} dataset_versions must be a list of non-empty strings"
            )
        model_versions_by_index[index] = {
            value.strip() for value in raw_dataset_versions
        }

    rebuild_order = _topological_dataset_versions(parents_by_version)
    directly_affected_versions = {
        version for version, removed in direct_removed_by_version.items() if removed
    }
    affected_versions = set(directly_affected_versions)
    affected_ancestors_by_version: dict[str, set[str]] = {
        version: set() for version in entries_by_version
    }
    for version in rebuild_order:
        affected_ancestors: set[str] = set()
        for parent_version in parents_by_version[version]:
            if parent_version not in entries_by_version:
                continue
            if parent_version in affected_versions:
                affected_ancestors.add(parent_version)
            affected_ancestors.update(affected_ancestors_by_version[parent_version])
        affected_ancestors_by_version[version] = affected_ancestors
        if affected_ancestors:
            affected_versions.add(version)

    closure_identity = {
        "rebuild_order": rebuild_order,
        "parent_versions": parents_by_version,
        "directly_affected_dataset_versions": sorted(directly_affected_versions),
        "affected_dataset_versions": sorted(affected_versions),
    }
    lineage_closure_fingerprint = _canonical_sha256(closure_identity)
    rebuilt_versions: dict[str, str] = {}

    for version in rebuild_order:
        source = source_by_version[version]
        rows = rows_by_version[version]
        direct_removed = direct_removed_by_version[version]
        direct_removed_fingerprints = {_canonical_sha256(row) for row in direct_removed}
        affected_ancestors = affected_ancestors_by_version[version]
        if affected_ancestors:
            removed = list(rows)
            kept: list[dict[str, Any]] = []
        else:
            removed = list(direct_removed)
            kept = [row for row in rows if _canonical_sha256(row) not in direct_removed_fingerprints]

        removed_fingerprints = {_canonical_sha256(row) for row in removed}
        source_parent_versions = parents_by_version[version]
        rebuilt_parent_versions = [
            rebuilt_versions.get(parent_version, parent_version)
            for parent_version in source_parent_versions
        ]
        output_path = destination / f"{version}-after-{_safe_component(request_id)}.jsonl"
        _atomic_write_jsonl(output_path, kept)
        written_rows = _read_jsonl(output_path)
        output_sha = _sha256_file(output_path)
        new_version = f"hfrds-{_canonical_sha256({'source_dataset_version': version, 'source_sha256': _sha256_file(source), 'parent_versions': rebuilt_parent_versions, 'output_sha256': output_sha, 'deletion_subject_fingerprints': [_canonical_sha256(value) for value in subjects]})[:16]}"
        rebuilt_versions[version] = new_version

        written_fingerprints = {_canonical_sha256(row) for row in written_rows}
        removed_fingerprints_absent = removed_fingerprints.isdisjoint(written_fingerprints)
        direct_subject_survivor_count = sum(
            1
            for row in written_rows
            if set(
                value
                for value in (
                    row.get("governance", {}).get("deletion_subject_ids", [])
                    if isinstance(row.get("governance"), dict)
                    else []
                )
                if isinstance(value, str)
            ).intersection(subjects)
        )
        affected_parent_version_survivors = sorted(
            set(rebuilt_parent_versions).intersection(affected_versions)
        )
        affected_lineage_survivor_count = (
            len(written_rows) if affected_ancestors else direct_subject_survivor_count
        )
        affected_lineage_survivor_count += len(affected_parent_version_survivors)
        affected_lineage_absent = affected_lineage_survivor_count == 0

        total_removed += len(removed)
        dataset_receipts.append(
            {
                "parent_dataset_version": version,
                "dataset_version": new_version,
                "source_path": f"<redacted:{source.name}>",
                "source_sha256": _sha256_file(source),
                "output_path": output_path.name,
                "output_sha256": output_sha,
                "input_row_count": len(rows),
                "output_row_count": len(written_rows),
                "removed_row_count": len(removed),
                "direct_subject_removed_row_count": len(direct_removed),
                "ancestor_lineage_removed_row_count": len(removed) - len(direct_removed),
                "removed_fingerprint_count": len(removed_fingerprints),
                "removed_fingerprints_absent": removed_fingerprints_absent,
                "source_parent_versions": source_parent_versions,
                "parent_versions": rebuilt_parent_versions,
                "directly_affected": version in directly_affected_versions,
                "lineage_affected": version in affected_versions,
                "affected_ancestor_versions": sorted(affected_ancestors),
                "affected_parent_version_survivors": affected_parent_version_survivors,
                "affected_lineage_survivor_count": affected_lineage_survivor_count,
                "affected_lineage_absent": affected_lineage_absent,
                "source_erased": False,
            }
        )

    quarantined: list[dict[str, Any]] = []
    for index, model in enumerate(model_entries):
        versions = model_versions_by_index[index]
        if versions.intersection(affected_versions):
            direct_versions = sorted(versions.intersection(directly_affected_versions))
            transitive_versions = sorted(versions.intersection(affected_versions - directly_affected_versions))
            quarantined.append(
                {
                    **model,
                    "status": "quarantined",
                    "quarantine_reason": "source_subject_deletion",
                    "deletion_request_id": request_id,
                    "affected_dataset_versions": sorted(versions.intersection(affected_versions)),
                    "directly_affected_dataset_versions": direct_versions,
                    "transitively_affected_dataset_versions": transitive_versions,
                    "quarantine_scope": (
                        "direct_and_transitive"
                        if direct_versions and transitive_versions
                        else "direct"
                        if direct_versions
                        else "transitive"
                    ),
                }
            )
    quarantined.sort(key=lambda row: str(row["model_id"]))
    quarantine_path = destination / "model_quarantine.json"
    _atomic_write_json(
        quarantine_path,
        {
            "schema_version": "hfr.model_quarantine.v1",
            "request_id": request_id,
            "affected_dataset_versions": sorted(affected_versions),
            "lineage_closure_fingerprint": lineage_closure_fingerprint,
            "model_count": len(quarantined),
            "models": quarantined,
        },
    )
    removed_fingerprints_absent = all(
        row["removed_fingerprints_absent"] for row in dataset_receipts
    )
    affected_lineage_survivor_count = sum(
        row["affected_lineage_survivor_count"] for row in dataset_receipts
    )
    affected_lineage_absent = affected_lineage_survivor_count == 0 and all(
        row["affected_lineage_absent"] for row in dataset_receipts
    )
    rebuild_integrity_passed = removed_fingerprints_absent and affected_lineage_absent
    erasure_errors: list[dict[str, str]] = []
    if erase_sources and rebuild_integrity_passed:
        for row in dataset_receipts:
            source = source_by_version[row["parent_dataset_version"]]
            try:
                source.unlink()
            except OSError as exc:
                erasure_errors.append(
                    {
                        "dataset_version": row["parent_dataset_version"],
                        "error_class": type(exc).__name__,
                    }
                )
            row["source_erased"] = not source.exists()
    blocked_reasons = []
    if not erase_sources:
        blocked_reasons.append("source_erasure_not_authorized")
    if erasure_errors or any(not row["source_erased"] for row in dataset_receipts):
        blocked_reasons.append("source_erasure_incomplete")
    if not removed_fingerprints_absent:
        blocked_reasons.append("removed_fingerprints_survived_rebuild")
    if not affected_lineage_absent:
        blocked_reasons.append("affected_lineage_survived_rebuild")
    receipt = {
        "schema_version": DATASET_DELETION_RECEIPT_SCHEMA_VERSION,
        "request_id": request_id,
        "created_at": created_at or datetime.now(timezone.utc).isoformat(),
        "passed": bool(dataset_receipts)
        and all(
            row["removed_fingerprints_absent"]
            and row["affected_lineage_absent"]
            and row["source_erased"]
            for row in dataset_receipts
        )
        and affected_lineage_absent
        and not erasure_errors,
        "blocked_reasons": blocked_reasons,
        "source_erasure_requested": erase_sources,
        "source_erasure_count": sum(1 for row in dataset_receipts if row["source_erased"]),
        "source_erasure_errors": erasure_errors,
        "deletion_subject_fingerprints": [_canonical_sha256(value) for value in subjects],
        "dataset_count": len(dataset_receipts),
        "removed_row_count": total_removed,
        "datasets": dataset_receipts,
        "rebuild_order": rebuild_order,
        "directly_affected_dataset_versions": sorted(directly_affected_versions),
        "affected_dataset_versions": sorted(affected_versions),
        "affected_parent_versions": sorted(affected_versions),
        "lineage_closure_fingerprint": lineage_closure_fingerprint,
        "affected_lineage_survivor_count": affected_lineage_survivor_count,
        "affected_lineage_absent": affected_lineage_absent,
        "quarantined_models": quarantined,
        "model_quarantine_path": quarantine_path.name,
        "notes": [
            "Rebuilt descendants are written before explicitly authorized source erasure.",
            "Deletion subject identifiers are represented by fingerprints in this public receipt.",
            "Affected lineage is closed transitively through declared parent_versions before any rebuild or model quarantine.",
            "Descendants of affected datasets are rebuilt without ancestor-derived rows and reference rebuilt parent versions.",
        ],
    }
    receipt["receipt_fingerprint"] = _canonical_sha256(receipt)
    _atomic_write_json(destination / "deletion_receipt.json", receipt)
    return receipt


def _duplicate_clusters(rows: list[dict[str, Any]], threshold: float) -> list[list[int]]:
    parent = list(range(len(rows)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    normalized = [_normalized_training_text(row) for row in rows]
    for left in range(len(rows)):
        for right in range(left + 1, len(rows)):
            if _similarity(normalized[left], normalized[right]) >= threshold:
                union(left, right)
    clusters: dict[int, list[int]] = {}
    for index in range(len(rows)):
        clusters.setdefault(find(index), []).append(index)
    return sorted(clusters.values(), key=lambda indexes: [_row_id(rows[index], index) for index in indexes])


def _topological_dataset_versions(parents_by_version: dict[str, list[str]]) -> list[str]:
    versions = set(parents_by_version)
    children: dict[str, set[str]] = {version: set() for version in versions}
    indegree = {version: 0 for version in versions}
    for child_version, parent_versions in parents_by_version.items():
        for parent_version in parent_versions:
            if parent_version not in versions:
                continue
            children[parent_version].add(child_version)
            indegree[child_version] += 1

    ready = sorted(version for version, count in indegree.items() if count == 0)
    ordered: list[str] = []
    while ready:
        version = ready.pop(0)
        ordered.append(version)
        for child_version in sorted(children[version]):
            indegree[child_version] -= 1
            if indegree[child_version] == 0:
                ready.append(child_version)
                ready.sort()
    if len(ordered) != len(versions):
        cycle_versions = sorted(version for version, count in indegree.items() if count > 0)
        raise DataGovernanceError(
            "dataset parent_versions contain a cycle: " + ", ".join(cycle_versions)
        )
    return ordered


def _similarity(left: str, right: str) -> float:
    if left == right:
        return 1.0
    if not left or not right:
        return 0.0
    left_tokens = set(_TOKEN_RE.findall(left))
    right_tokens = set(_TOKEN_RE.findall(right))
    union = left_tokens | right_tokens
    jaccard = len(left_tokens & right_tokens) / len(union) if union else 0.0
    sequence = SequenceMatcher(a=left, b=right, autojunk=False).ratio()
    return max(jaccard, sequence)


def _normalized_training_text(row: dict[str, Any]) -> str:
    contract = {
        "prompt": str(row.get("prompt") or _first_user_message(row.get("messages"))),
        "system": [
            message.get("content")
            for message in row.get("messages", [])
            if isinstance(message, dict) and message.get("role") in {"system", "developer"}
        ]
        if isinstance(row.get("messages"), list)
        else [],
        "tools": row.get("tools") if isinstance(row.get("tools"), list) else [],
    }
    return _normalized_text(json.dumps(contract, sort_keys=True, ensure_ascii=False, separators=(",", ":")))


def _normalized_text(value: str) -> str:
    return " ".join(_TOKEN_RE.findall(value.casefold()))


def _first_user_message(messages: Any) -> str:
    if not isinstance(messages, list):
        return ""
    for message in messages:
        if isinstance(message, dict) and message.get("role") == "user":
            return str(message.get("content") or "")
    return ""


def _row_id(row: dict[str, Any], index: int) -> str:
    return str(row.get("row_id") or row.get("episode_id") or row.get("scenario_id") or f"row-{index}")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DataGovernanceError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise DataGovernanceError(f"JSONL row must be an object at {path}:{line_number}")
            rows.append(row)
    return rows


def _atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":")))
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    atomic_write_json_cas(path, value, expected_sha256=json_file_sha256(path))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _datetime(value: str | datetime | None) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    if isinstance(value, datetime):
        parsed = value
    else:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _optional_datetime(value: Any) -> datetime | None:
    try:
        return _datetime(str(value)) if _present(value) else None
    except (TypeError, ValueError):
        return None


def _present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, dict, tuple, set)):
        return bool(value)
    return True


def _safe_component(value: str) -> str:
    rendered = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-.")
    if not rendered:
        raise DataGovernanceError("request_id must contain a safe path component")
    return rendered
