"""Trainer preflight manifests over passed Flight Recorder gates."""

from __future__ import annotations

import hashlib
import json
import shlex
from pathlib import Path
from typing import Any

TRAINER_PREFLIGHT_SCHEMA_VERSION = "hfr.trainer_preflight.v1"
TRAINER_LAUNCH_CHECK_SCHEMA_VERSION = "hfr.trainer_launch_check.v1"

_TRAINING_EXPORT_FILES = (
    "manifest.json",
    "dataset_metrics.json",
    "DATASET_CARD.md",
    "episodes.jsonl",
    "rewards.jsonl",
    "step_rewards.jsonl",
    "preferences.jsonl",
    "failure_modes.jsonl",
    "curriculum.json",
    "sft.jsonl",
    "dpo.jsonl",
    "reward_model.jsonl",
)
_COMPARE_EXPORT_FILES = (
    "manifest.json",
    "IMPROVEMENT_CARD.md",
    "improvement_pairs.jsonl",
    "improvement_dpo.jsonl",
)
_REVIEWED_EXPORT_FILES = (
    "manifest.json",
    "reviewed_labels.jsonl",
    "reviewed_sft.jsonl",
    "reviewed_reward_model.jsonl",
    "reviewed_preferences.jsonl",
    "reviewed_dpo.jsonl",
)
_VALIDATION_REQUIRED_GATE_SCHEMAS = {
    "hfr.training_gate.v1",
    "hfr.compare_gate.v1",
    "hfr.reviewed_gate.v1",
    "hfr.review_calibration.v1",
}


class TrainerPreflightError(ValueError):
    """Raised when a trainer preflight manifest cannot be built."""


def build_trainer_preflight(
    *,
    out_path: str | Path,
    gate_paths: list[str | Path],
    training_export_dir: str | Path | None = None,
    compare_export_dir: str | Path | None = None,
    reviewed_export_dir: str | Path | None = None,
    evidence_bundle_path: str | Path | None = None,
    require_gates: list[str] | None = None,
    trainer_command: str | None = None,
    allow_unvalidated_gates: bool = False,
    preserve_paths: bool = False,
    metadata: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build a launch guard over trainer-facing exports and gate outputs."""
    if not gate_paths:
        raise TrainerPreflightError("At least one --gate path is required.")
    if not any((training_export_dir, compare_export_dir, reviewed_export_dir, evidence_bundle_path)):
        raise TrainerPreflightError(
            "At least one trainer-facing artifact is required: --training-export, --compare-export, "
            "--reviewed-export, or --evidence-bundle."
        )

    checks: list[dict[str, Any]] = []
    gates = [_gate_record(Path(path), preserve_paths) for path in gate_paths]
    seen_gate_ids = {gate["id"] for gate in gates}
    for gate in gates:
        _add_bool_check(checks, "gate_passed", gate["passed"], {"gate": gate["id"], "path": gate["path"]})
        if _gate_requires_validation(gate) and not allow_unvalidated_gates:
            validation = gate.get("validation") if isinstance(gate.get("validation"), dict) else {}
            _add_bool_check(
                checks,
                "gate_validation_passed",
                bool(validation.get("available") and validation.get("passed")),
                {
                    "gate": gate["id"],
                    "validation_available": str(bool(validation.get("available"))).lower(),
                    "validation_error_count": str(_int_value(validation.get("error_count"))),
                },
            )

    for gate_id in require_gates or []:
        _add_bool_check(checks, "required_gate_present", gate_id in seen_gate_ids, {"gate": gate_id})

    artifacts: dict[str, Any] = {}
    if training_export_dir is not None:
        _add_export_artifacts(artifacts, checks, "training_export", Path(training_export_dir), _TRAINING_EXPORT_FILES, preserve_paths)
    if compare_export_dir is not None:
        _add_export_artifacts(artifacts, checks, "compare_export", Path(compare_export_dir), _COMPARE_EXPORT_FILES, preserve_paths)
    if reviewed_export_dir is not None:
        _add_export_artifacts(artifacts, checks, "reviewed_export", Path(reviewed_export_dir), _REVIEWED_EXPORT_FILES, preserve_paths)
    if evidence_bundle_path is not None:
        bundle_path = Path(evidence_bundle_path)
        artifacts["evidence_bundle"] = _file_record(bundle_path, preserve_paths)
        bundle = _read_json_optional(bundle_path)
        _add_bool_check(checks, "evidence_bundle_exists", bundle_path.exists() and bundle_path.is_file(), {"artifact": "evidence_bundle"})
        _add_bool_check(
            checks,
            "evidence_bundle_passed",
            bool(isinstance(bundle, dict) and bundle.get("passed") is True),
            {"artifact": "evidence_bundle"},
        )

    command = _trainer_command_record(trainer_command)
    failed_checks = sum(1 for check in checks if check.get("passed") is False)
    passed = failed_checks == 0
    preflight: dict[str, Any] = {
        "schema_version": TRAINER_PREFLIGHT_SCHEMA_VERSION,
        "preflight_path": _display_path(Path(out_path), preserve_paths),
        "passed": passed,
        "readiness": "ready" if passed else "blocked",
        "recommendation": "launch_allowed" if passed else "block_launch",
        "gate_count": len(gates),
        "passed_gate_count": sum(1 for gate in gates if gate.get("passed") is True),
        "required_gates": list(dict.fromkeys(require_gates or [])),
        "check_count": len(checks),
        "failed_check_count": failed_checks,
        "checks": checks,
        "gates": gates,
        "artifacts": artifacts,
        "trainer_command": command,
        "notes": [
            "Trainer preflight records evidence for a downstream launch guard; it does not train, sandbox, or execute commands.",
            "A ready preflight means the referenced gates and artifacts passed the configured handoff checks.",
        ],
    }
    clean_metadata = _metadata(metadata)
    if clean_metadata:
        preflight["metadata"] = clean_metadata
    return preflight


def build_trainer_launch_check(
    *,
    preflight_path: str | Path,
    preflight: dict[str, Any],
    validation_summary: dict[str, Any],
    require_gates: list[str] | None = None,
    require_metadata: dict[str, str] | None = None,
    preserve_paths: bool = False,
) -> dict[str, Any]:
    """Build a side-effect-free trainer launch decision from a preflight artifact."""
    if not isinstance(preflight, dict):
        raise TrainerPreflightError(f"trainer preflight must contain a JSON object: {preflight_path}")

    checks: list[dict[str, Any]] = []
    display_preflight_path = _display_path(Path(preflight_path), preserve_paths)
    validation = _validation_record(validation_summary)
    gates = _launch_gate_records(preflight.get("gates"))
    metadata = _metadata(preflight.get("metadata") if isinstance(preflight.get("metadata"), dict) else None)
    required_gates = list(dict.fromkeys(require_gates or []))
    required_metadata = _metadata(require_metadata)
    command = _approved_command_record(preflight.get("trainer_command"))

    _add_bool_check(checks, "preflight_validation_passed", validation["passed"], {"preflight": display_preflight_path})
    _add_bool_check(
        checks,
        "preflight_schema_supported",
        preflight.get("schema_version") == TRAINER_PREFLIGHT_SCHEMA_VERSION,
        {"schema_version": str(preflight.get("schema_version") or "")},
    )
    _add_bool_check(checks, "preflight_passed", preflight.get("passed") is True, {"preflight": display_preflight_path})
    _add_bool_check(
        checks,
        "recommendation_launch_allowed",
        preflight.get("recommendation") == "launch_allowed",
        {"recommendation": str(preflight.get("recommendation") or "")},
    )
    _add_bool_check(
        checks,
        "trainer_command_ready",
        bool(command["provided"] and command["parseable"] and command["argv"]),
        {"provided": str(command["provided"]).lower(), "parseable": str(command["parseable"]).lower()},
    )
    for gate_id in required_gates:
        matching_gate = next((gate for gate in gates if gate["id"] == gate_id), None)
        _add_bool_check(checks, "required_gate_present", matching_gate is not None, {"gate": gate_id})
        _add_bool_check(
            checks,
            "required_gate_passed",
            bool(matching_gate and matching_gate.get("passed") is True),
            {"gate": gate_id},
        )
    for key, expected in required_metadata.items():
        actual = metadata.get(key)
        _add_bool_check(
            checks,
            "required_metadata_matches",
            actual == expected,
            {"key": key, "expected": expected, "actual": "" if actual is None else actual},
        )

    failed_checks = sum(1 for check in checks if check.get("passed") is False)
    passed = failed_checks == 0
    command["approved"] = passed
    artifacts = preflight.get("artifacts") if isinstance(preflight.get("artifacts"), dict) else {}
    return {
        "schema_version": TRAINER_LAUNCH_CHECK_SCHEMA_VERSION,
        "preflight_path": display_preflight_path,
        "passed": passed,
        "readiness": "ready" if passed else "blocked",
        "recommendation": "launch_allowed" if passed else "block_launch",
        "check_count": len(checks),
        "failed_check_count": failed_checks,
        "checks": checks,
        "validation": validation,
        "required_gates": required_gates,
        "required_metadata": required_metadata,
        "gates": gates,
        "gate_count": len(gates),
        "passed_gate_count": sum(1 for gate in gates if gate.get("passed") is True),
        "artifacts": {"count": len(artifacts), "names": sorted(str(name) for name in artifacts.keys())},
        "approved_command": command,
        "notes": [
            "Trainer launch check validates the preflight and emits the approved command; it does not execute it.",
            "A ready launch check means the preflight artifact, referenced hashes, required gates, and required metadata passed.",
        ],
    }


def _gate_record(path: Path, preserve_paths: bool) -> dict[str, Any]:
    gate = _read_json_required(path, "gate")
    schema_version = gate.get("schema_version") if isinstance(gate.get("schema_version"), str) else "unknown"
    metrics = gate.get("metrics") if isinstance(gate.get("metrics"), dict) else {}
    validation = metrics.get("validation") if isinstance(metrics.get("validation"), dict) else {}
    record: dict[str, Any] = {
        "id": _gate_id(gate, path.stem),
        "path": _display_path(path, preserve_paths),
        "exists": path.exists(),
        "schema_version": schema_version,
        "passed": gate.get("passed") is True,
    }
    if path.exists() and path.is_file():
        record["size_bytes"] = path.stat().st_size
        record["sha256"] = _sha256(path)
    if validation:
        record["validation"] = {
            "available": bool(validation.get("available")),
            "passed": bool(validation.get("passed")),
            "strict": bool(validation.get("strict")),
            "error_count": _int_value(validation.get("error_count")),
            "warning_count": _int_value(validation.get("warning_count")),
        }
    return record


def _gate_id(gate: dict[str, Any], fallback: str) -> str:
    schema = str(gate.get("schema_version") or fallback)
    return schema.removeprefix("hfr.").removesuffix(".v1")


def _gate_requires_validation(gate: dict[str, Any]) -> bool:
    return gate.get("schema_version") in _VALIDATION_REQUIRED_GATE_SCHEMAS


def _add_export_artifacts(
    artifacts: dict[str, Any],
    checks: list[dict[str, Any]],
    name: str,
    root: Path,
    files: tuple[str, ...],
    preserve_paths: bool,
) -> None:
    artifacts[name] = _dir_record(root, preserve_paths)
    _add_bool_check(checks, "artifact_dir_exists", root.exists() and root.is_dir(), {"artifact": name})
    _add_bool_check(checks, "artifact_dir_regular", _is_regular_dir(root), {"artifact": name})
    for relative in files:
        key = f"{name}_{_artifact_key(relative)}"
        path = root / relative
        artifacts[key] = _file_record(path, preserve_paths)
        _add_bool_check(checks, "artifact_file_exists", path.exists() and path.is_file(), {"artifact": name, "file": relative})
        _add_bool_check(checks, "artifact_file_regular", _is_regular_file(path), {"artifact": name, "file": relative})


def _trainer_command_record(value: str | None) -> dict[str, Any]:
    if not value:
        return {"provided": False, "raw": "", "argv": []}
    try:
        argv = shlex.split(value)
    except ValueError:
        argv = []
    return {"provided": True, "raw": value, "argv": argv, "parseable": bool(argv)}


def _approved_command_record(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {"approved": False, "provided": False, "raw": "", "argv": [], "parseable": False, "shell": ""}
    argv = value.get("argv") if isinstance(value.get("argv"), list) else []
    argv = [item for item in argv if isinstance(item, str)]
    raw = value.get("raw") if isinstance(value.get("raw"), str) else ""
    parseable = bool(value.get("parseable")) if "parseable" in value else bool(argv)
    return {
        "approved": False,
        "provided": value.get("provided") is True,
        "raw": raw,
        "argv": argv,
        "parseable": parseable,
        "shell": shlex.join(argv) if argv else raw,
    }


def _validation_record(summary: dict[str, Any]) -> dict[str, Any]:
    targets = summary.get("targets") if isinstance(summary.get("targets"), list) else []
    errors: list[str] = []
    warnings: list[str] = []
    for target in targets:
        if not isinstance(target, dict):
            continue
        errors.extend(str(error) for error in target.get("errors", []) if isinstance(error, str))
        warnings.extend(str(warning) for warning in target.get("warnings", []) if isinstance(warning, str))
    return {
        "passed": summary.get("passed") is True,
        "strict": summary.get("strict") is True,
        "target_count": _int_value(summary.get("target_count")),
        "error_count": _int_value(summary.get("error_count")),
        "warning_count": _int_value(summary.get("warning_count")),
        "errors": errors,
        "warnings": warnings,
    }


def _launch_gate_records(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    records: list[dict[str, Any]] = []
    for gate in value:
        if not isinstance(gate, dict):
            continue
        records.append(
            {
                "id": str(gate.get("id") or ""),
                "path": str(gate.get("path") or ""),
                "schema_version": str(gate.get("schema_version") or ""),
                "passed": gate.get("passed") is True,
            }
        )
    return records


def _add_bool_check(checks: list[dict[str, Any]], check_id: str, passed: bool, scope: dict[str, str]) -> None:
    checks.append(
        {
            "id": check_id,
            "passed": passed,
            "actual": {"passed": passed},
            "expected": {"passed": True},
            "scope": scope,
            "summary": f"{check_id}: passed={passed}",
        }
    )


def _read_json_required(path: Path, label: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TrainerPreflightError(f"{label} must contain a JSON object: {path}")
    return payload


def _read_json_optional(path: Path) -> dict[str, Any] | None:
    if not path.exists() or not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else None


def _file_record(path: Path, preserve_paths: bool) -> dict[str, Any]:
    regular_file = _is_regular_file(path)
    record: dict[str, Any] = {
        "path": _display_path(path, preserve_paths),
        "exists": path.exists(),
        "kind": "file",
        "regular_file": regular_file,
        "symlink": path.is_symlink(),
    }
    if regular_file:
        record["size_bytes"] = path.stat().st_size
        record["sha256"] = _sha256(path)
    return record


def _dir_record(path: Path, preserve_paths: bool) -> dict[str, Any]:
    regular_directory = _is_regular_dir(path)
    record: dict[str, Any] = {
        "path": _display_path(path, preserve_paths),
        "exists": path.exists(),
        "kind": "directory",
        "regular_directory": regular_directory,
        "symlink": path.is_symlink(),
    }
    if regular_directory:
        record["entry_count"] = sum(1 for _ in path.iterdir())
    return record


def _is_regular_file(path: Path) -> bool:
    return path.exists() and path.is_file() and not path.is_symlink()


def _is_regular_dir(path: Path) -> bool:
    return path.exists() and path.is_dir() and not path.is_symlink()


def _artifact_key(value: str) -> str:
    return value.lower().replace(".", "_").replace("-", "_")


def _metadata(value: dict[str, str] | None) -> dict[str, str]:
    if not value:
        return {}
    return {str(key): str(raw_value) for key, raw_value in sorted(value.items()) if str(key)}


def _int_value(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path: Path, preserve_paths: bool = False) -> str:
    raw = str(path)
    if preserve_paths:
        return raw
    if _is_windows_absolute(raw):
        return f"<redacted:{_basename(raw)}>"
    resolved = path.resolve()
    cwd = Path.cwd().resolve()
    try:
        return str(resolved.relative_to(cwd))
    except ValueError:
        return f"<redacted:{resolved.name}>"


def _is_windows_absolute(value: str) -> bool:
    normalized = value.replace("/", "\\")
    return (len(normalized) >= 3 and normalized[1:3] == ":\\" and normalized[0].isalpha()) or normalized.startswith("\\\\")


def _basename(value: str) -> str:
    return value.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1] or "path"
