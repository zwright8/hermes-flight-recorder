"""Bundled JSON Schema contracts for public Flight Recorder artifacts."""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path
from typing import Any

SCHEMA_CATALOG_VERSION = "hfr.schema_catalog.v1"
SCHEMA_CHECK_VERSION = "hfr.schema_check.v1"
SCHEMA_JSONL_CHECK_VERSION = "hfr.schema_jsonl_check.v1"
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


def check_schema_file(path: str | Path, name_or_id: str | None = None) -> dict[str, Any]:
    """Check one JSON artifact against a bundled schema contract."""
    artifact_path = Path(path)
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    return check_schema_contract(payload, name_or_id=name_or_id, artifact_path=artifact_path)


def check_schema_jsonl_file(path: str | Path, name_or_id: str | None = None) -> dict[str, Any]:
    """Check each non-empty JSONL row against a bundled schema contract."""
    artifact_path = Path(path)
    errors: list[str] = []
    schema_counts: dict[str, int] = {}
    row_count = 0
    selected_schema = _schema_record(name_or_id) if name_or_id is not None else None
    for line_number, line in enumerate(artifact_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row_count += 1
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"line {line_number}: invalid JSON: {exc.msg}")
            continue
        try:
            result = check_schema_contract(payload, name_or_id=name_or_id)
        except SchemaRegistryError as exc:
            errors.append(f"line {line_number}: {exc}")
            continue
        schema_name = str(result.get("schema", {}).get("name") or "")
        if schema_name:
            schema_counts[schema_name] = schema_counts.get(schema_name, 0) + 1
        for error in result.get("errors", []):
            errors.append(f"line {line_number}: {error}")
    schema_record = (
        {key: selected_schema[key] for key in ("name", "artifact_schema_version", "filename", "id") if key in selected_schema}
        if selected_schema is not None
        else None
    )
    return {
        "schema_version": SCHEMA_JSONL_CHECK_VERSION,
        "schema": schema_record,
        "artifact_path": str(artifact_path),
        "row_count": row_count,
        "row_schema_counts": [{"name": name, "count": schema_counts[name]} for name in sorted(schema_counts)],
        "passed": not errors,
        "error_count": len(errors),
        "errors": errors,
        "notes": [
            "JSONL schema checks validate public row shape only.",
            "Use flightrecorder validate for semantic integrity checks over counts, hashes, evidence refs, and split assignments.",
        ],
    }


def check_schema_contract(
    payload: Any,
    *,
    name_or_id: str | None = None,
    artifact_path: str | Path | None = None,
) -> dict[str, Any]:
    """Check a JSON-compatible value against one bundled schema contract.

    This is a dependency-free conformance check for the subset of JSON Schema
    keywords Flight Recorder publishes in its bundled contracts. Use
    ``flightrecorder validate`` for richer artifact integrity checks.
    """
    record = _schema_record(name_or_id) if name_or_id is not None else _schema_record_for_payload(payload)
    schema = _read_schema_file(str(record["filename"]))
    errors: list[str] = []
    _validate_value(payload, schema, "$", schema, errors)
    return {
        "schema_version": SCHEMA_CHECK_VERSION,
        "schema": {key: record[key] for key in ("name", "artifact_schema_version", "filename", "id") if key in record},
        "artifact_path": str(artifact_path) if artifact_path is not None else None,
        "passed": not errors,
        "error_count": len(errors),
        "errors": errors,
        "notes": [
            "Schema checks validate public artifact shape only.",
            "Use flightrecorder validate for semantic integrity checks over hashes, counts, evidence refs, and lineage.",
        ],
    }


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


def _schema_record_for_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise SchemaRegistryError("Cannot infer schema for a non-object JSON artifact; pass --name")
    version = payload.get("schema_version")
    if isinstance(version, str):
        return _schema_record(version)
    raise SchemaRegistryError("Cannot infer schema because artifact has no schema_version; pass --name")


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


def _validate_value(value: Any, schema: Any, path: str, root: dict[str, Any], errors: list[str]) -> None:
    if schema is True:
        return
    if schema is False:
        errors.append(f"{path}: value is not allowed by schema")
        return
    if not isinstance(schema, dict):
        errors.append(f"{path}: schema is not an object")
        return

    if "$ref" in schema:
        _validate_value(value, _resolve_ref(str(schema["$ref"]), root), path, root, errors)

    for subschema in schema.get("allOf", []) if isinstance(schema.get("allOf"), list) else []:
        _validate_value(value, subschema, path, root, errors)

    one_of = schema.get("oneOf")
    if isinstance(one_of, list):
        matches = 0
        for subschema in one_of:
            sub_errors: list[str] = []
            _validate_value(value, subschema, path, root, sub_errors)
            if not sub_errors:
                matches += 1
        if matches != 1:
            errors.append(f"{path}: expected exactly one matching schema from oneOf, got {matches}")
        return

    if "const" in schema and value != schema["const"]:
        errors.append(f"{path}: expected constant {schema['const']!r}, got {value!r}")
    if "enum" in schema and isinstance(schema["enum"], list) and value not in schema["enum"]:
        errors.append(f"{path}: expected one of {schema['enum']!r}, got {value!r}")

    expected_type = schema.get("type")
    if expected_type is not None and not _matches_type(value, expected_type):
        errors.append(f"{path}: expected type {_type_label(expected_type)}, got {_value_type(value)}")
        return

    if isinstance(value, dict):
        _validate_object(value, schema, path, root, errors)
    if isinstance(value, list):
        _validate_array(value, schema, path, root, errors)
    if isinstance(value, str):
        _validate_string(value, schema, path, errors)
    if _is_number(value):
        _validate_number(value, schema, path, errors)


def _validate_object(value: dict[str, Any], schema: dict[str, Any], path: str, root: dict[str, Any], errors: list[str]) -> None:
    required = schema.get("required")
    if isinstance(required, list):
        for field in required:
            if isinstance(field, str) and field not in value:
                errors.append(f"{path}: missing required property {field!r}")

    properties = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
    for field, subschema in properties.items():
        if field in value:
            _validate_value(value[field], subschema, f"{path}.{field}", root, errors)

    additional = schema.get("additionalProperties", True)
    for field, item in value.items():
        if field in properties:
            continue
        if additional is False:
            errors.append(f"{path}.{field}: additional property is not allowed")
        elif isinstance(additional, dict):
            _validate_value(item, additional, f"{path}.{field}", root, errors)


def _validate_array(value: list[Any], schema: dict[str, Any], path: str, root: dict[str, Any], errors: list[str]) -> None:
    min_items = schema.get("minItems")
    if isinstance(min_items, int) and len(value) < min_items:
        errors.append(f"{path}: expected at least {min_items} item(s), got {len(value)}")
    items = schema.get("items")
    if items is not None:
        for index, item in enumerate(value):
            _validate_value(item, items, f"{path}[{index}]", root, errors)


def _validate_string(value: str, schema: dict[str, Any], path: str, errors: list[str]) -> None:
    min_length = schema.get("minLength")
    if isinstance(min_length, int) and len(value) < min_length:
        errors.append(f"{path}: expected length >= {min_length}, got {len(value)}")
    pattern = schema.get("pattern")
    if isinstance(pattern, str):
        import re

        if re.search(pattern, value) is None:
            errors.append(f"{path}: value does not match pattern {pattern!r}")


def _validate_number(value: int | float, schema: dict[str, Any], path: str, errors: list[str]) -> None:
    minimum = schema.get("minimum")
    if _is_number(minimum) and value < minimum:
        errors.append(f"{path}: expected value >= {minimum}, got {value}")
    maximum = schema.get("maximum")
    if _is_number(maximum) and value > maximum:
        errors.append(f"{path}: expected value <= {maximum}, got {value}")


def _resolve_ref(ref: str, root: dict[str, Any]) -> Any:
    if not ref.startswith("#/"):
        raise SchemaRegistryError(f"Unsupported schema reference {ref!r}; only local #/ references are supported")
    current: Any = root
    for raw_part in ref[2:].split("/"):
        part = raw_part.replace("~1", "/").replace("~0", "~")
        if not isinstance(current, dict) or part not in current:
            raise SchemaRegistryError(f"Schema reference {ref!r} cannot be resolved")
        current = current[part]
    return current


def _matches_type(value: Any, expected_type: Any) -> bool:
    if isinstance(expected_type, list):
        return any(_matches_type(value, item) for item in expected_type)
    if expected_type == "object":
        return isinstance(value, dict)
    if expected_type == "array":
        return isinstance(value, list)
    if expected_type == "string":
        return isinstance(value, str)
    if expected_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type == "number":
        return _is_number(value)
    if expected_type == "boolean":
        return isinstance(value, bool)
    if expected_type == "null":
        return value is None
    return True


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _type_label(expected_type: Any) -> str:
    if isinstance(expected_type, list):
        return " or ".join(str(item) for item in expected_type)
    return str(expected_type)


def _value_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__
