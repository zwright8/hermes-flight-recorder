"""Bundled JSON Schema contracts for public Flight Recorder artifacts."""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path
from typing import Any

SCHEMA_CATALOG_VERSION = "hfr.schema_catalog.v1"
SCHEMA_PACKAGE_DIR = "schemas"


class SchemaRegistryError(ValueError):
    """Raised when bundled schema contracts cannot be resolved or exported."""


def schema_catalog() -> dict[str, Any]:
    """Return the bundled schema catalog manifest."""
    catalog = _read_schema_file("manifest.json")
    if catalog.get("schema_version") != SCHEMA_CATALOG_VERSION:
        raise SchemaRegistryError(
            f"schema catalog version must be {SCHEMA_CATALOG_VERSION!r}; got {catalog.get('schema_version')!r}"
        )
    schemas = catalog.get("schemas")
    if not isinstance(schemas, list) or not schemas:
        raise SchemaRegistryError("schema catalog contains no schemas")
    return catalog


def list_schema_records() -> list[dict[str, Any]]:
    """Return public schema records sorted by stable schema name."""
    records = schema_catalog()["schemas"]
    return sorted((dict(record) for record in records if isinstance(record, dict)), key=lambda record: str(record.get("name") or ""))


def load_schema(name_or_id: str) -> dict[str, Any]:
    """Load a bundled JSON Schema by short name, filename, schema version, or $id."""
    record = _schema_record(name_or_id)
    return _read_schema_file(str(record["filename"]))


def write_schema_bundle(out_dir: str | Path, names: list[str] | None = None, *, force: bool = False) -> list[Path]:
    """Write selected bundled schemas plus the catalog manifest to a directory."""
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    selected = [_schema_record(name) for name in names] if names else list_schema_records()
    selected_names = {str(record["name"]) for record in selected}
    catalog = schema_catalog()
    catalog = {**catalog, "schemas": [record for record in catalog["schemas"] if record.get("name") in selected_names]}

    written: list[Path] = []
    manifest_path = target / "manifest.json"
    _write_schema_json(manifest_path, catalog, force)
    written.append(manifest_path)
    for record in selected:
        schema = _read_schema_file(str(record["filename"]))
        path = target / str(record["filename"])
        _write_schema_json(path, schema, force)
        written.append(path)
    return written


def _schema_record(name_or_id: str) -> dict[str, Any]:
    needle = name_or_id.strip()
    if not needle:
        raise SchemaRegistryError("schema name must be non-empty")
    for record in list_schema_records():
        values = {
            str(record.get("name") or ""),
            str(record.get("filename") or ""),
            str(record.get("artifact_schema_version") or ""),
            str(record.get("id") or ""),
        }
        if needle in values:
            return record
    available = ", ".join(record["name"] for record in list_schema_records())
    raise SchemaRegistryError(f"Unknown schema {name_or_id!r}; available schemas: {available}")


def _read_schema_file(filename: str) -> dict[str, Any]:
    try:
        raw = resources.files("flightrecorder").joinpath(SCHEMA_PACKAGE_DIR, filename).read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise SchemaRegistryError(f"Bundled schema file not found: {filename}") from exc
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise SchemaRegistryError(f"Bundled schema file must contain a JSON object: {filename}")
    return payload


def _write_schema_json(path: Path, payload: dict[str, Any], force: bool) -> None:
    if path.exists() and not force:
        raise SchemaRegistryError(f"Schema output already exists: {path}; pass --force to overwrite")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
