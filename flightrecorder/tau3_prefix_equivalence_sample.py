"""Build the bounded deterministic sample used for Tau-3 prefix equivalence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

SAMPLE_SCHEMA_VERSION = "hfr.tau3_prefix_equivalence_sample.v1"
SAMPLE_STRATA = (
    ("airline", "later_task_completion_actions"),
    ("airline", "safe_stopping"),
    ("retail", "clarification_refusal"),
    ("retail", "confirmation_before_mutation"),
    ("telecom", "error_result_recovery"),
    ("telecom", "successful_completion"),
)


class Tau3PrefixEquivalenceSampleError(ValueError):
    """Raised when the bounded sample cannot be built deterministically."""


def build_tau3_prefix_equivalence_sample(
    dataset_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Select one canonical row from each frozen domain/behavior stratum."""

    source = Path(dataset_path)
    out = Path(output_dir)
    if not source.is_file():
        raise Tau3PrefixEquivalenceSampleError(
            f"source dataset does not exist: {source}"
        )
    if out.exists():
        raise Tau3PrefixEquivalenceSampleError(
            f"sample output already exists: {out}"
        )
    rows = _read_jsonl(source)
    selected: list[dict[str, Any]] = []
    selected_hashes: list[str] = []
    for domain, behavior in SAMPLE_STRATA:
        matches = [
            row
            for row in rows
            if _metadata_value(row, "domain") == domain
            and _metadata_value(row, "behavior") == behavior
        ]
        if not matches:
            raise Tau3PrefixEquivalenceSampleError(
                f"missing sample stratum: {domain}/{behavior}"
            )
        ranked = sorted(
            ((_canonical_sha256(row), row) for row in matches),
            key=lambda item: item[0],
        )
        row_sha256, row = ranked[0]
        selected.append(row)
        selected_hashes.append(row_sha256)
    if len(set(selected_hashes)) != len(selected_hashes):
        raise Tau3PrefixEquivalenceSampleError(
            "sample strata selected duplicate rows"
        )

    out.mkdir(parents=True)
    train = out / "train.jsonl"
    valid = out / "valid.jsonl"
    _write_jsonl(train, selected)
    _write_jsonl(valid, selected)
    manifest = {
        "schema_version": SAMPLE_SCHEMA_VERSION,
        "purpose": "bounded_full_gradient_vs_detached_prefix_equivalence",
        "candidate_eligible": False,
        "source": {
            "path": str(source),
            "sha256": _sha256_file(source),
            "row_count": len(rows),
        },
        "selection": {
            "method": "minimum_canonical_row_sha256_per_frozen_stratum",
            "strata": [
                {"domain": domain, "behavior": behavior}
                for domain, behavior in SAMPLE_STRATA
            ],
        },
        "row_count": len(selected),
        "row_hashes": selected_hashes,
        "row_hashes_sha256": _canonical_sha256(selected_hashes),
        "files": {
            "train": {
                "path": "train.jsonl",
                "sha256": _sha256_file(train),
                "size": train.stat().st_size,
            },
            "valid": {
                "path": "valid.jsonl",
                "sha256": _sha256_file(valid),
                "size": valid.stat().st_size,
            },
        },
    }
    manifest_path = out / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for path in (train, valid, manifest_path):
        path.chmod(0o444)
    return manifest


def _metadata_value(row: dict[str, Any], field: str) -> Any:
    metadata = row.get("metadata")
    return metadata.get(field) if isinstance(metadata, dict) else None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise Tau3PrefixEquivalenceSampleError(
                f"line {line_number}: invalid JSON: {exc.msg}"
            ) from exc
        if not isinstance(row, dict):
            raise Tau3PrefixEquivalenceSampleError(
                f"line {line_number}: row must be an object"
            )
        rows.append(row)
    if not rows:
        raise Tau3PrefixEquivalenceSampleError(
            "source dataset contains no rows"
        )
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                )
                + "\n"
            )


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
