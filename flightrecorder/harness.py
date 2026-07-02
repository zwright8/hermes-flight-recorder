"""Offline harness runner contracts for Flight Recorder scenario execution."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .schema import load_scenario

HARNESS_MANIFEST_SCHEMA_VERSION = "hfr.harness_run_manifest.v1"
HARNESS_RESULT_SCHEMA_VERSION = "hfr.harness_run_result.v1"
HARNESS_NAME = "hermes_harness"
HARNESS_MODE = "offline_mock"
DEFAULT_PROVIDER = "mock"
DEFAULT_MODEL = "hfr-mock"

EXPECTED_RUN_ARTIFACTS = [
    "normalized_trace",
    "scorecard",
    "task_completion",
    "run_digest",
    "report",
    "lineage",
]


class HarnessError(ValueError):
    """Raised when the offline harness cannot produce a valid run handoff."""


def run_mock_harness(
    scenario_path: str | Path,
    out_dir: str | Path,
    *,
    mock_response: str,
    provider: str = DEFAULT_PROVIDER,
    model: str = DEFAULT_MODEL,
    preserve_paths: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    """Run a scenario through a deterministic offline mock harness.

    The harness does not launch Hermes or call a model provider. It writes a
    minimal observer JSONL trace from the scenario prompt and supplied mock
    response, then routes that trace through the normal Flight Recorder scoring
    path. The manifest/result artifacts give downstream infrastructure a stable
    contract to validate before training, eval, or demo layers consume the run.
    """
    if provider != DEFAULT_PROVIDER:
        raise HarnessError(f"unsupported harness provider {provider!r}; only {DEFAULT_PROVIDER!r} is available")
    if not isinstance(mock_response, str) or not mock_response.strip():
        raise HarnessError("--mock-response must be a non-empty string")

    scenario_path = Path(scenario_path)
    out_path = Path(out_dir)
    _prepare_output_dir(out_path, force=force)

    scenario = load_scenario(scenario_path)
    source_trace_path = out_path / "source_trace.observer.jsonl"
    _write_mock_observer_trace(source_trace_path, scenario=scenario, model=model, mock_response=mock_response)

    manifest_path = out_path / "harness_manifest.json"
    manifest = _build_manifest(
        scenario_path=scenario_path,
        scenario=scenario,
        source_trace_path=source_trace_path,
        out_dir=out_path,
        provider=provider,
        model=model,
        mock_response=mock_response,
        preserve_paths=preserve_paths,
    )
    _write_json(manifest_path, manifest)

    from .cli import _run_scenario_artifacts

    run = _run_scenario_artifacts(
        scenario_path,
        out_path,
        trace_override=source_trace_path,
        trace_format="observer_jsonl",
        preserve_paths=preserve_paths,
    )
    result = _build_result(
        manifest_path=manifest_path,
        manifest=manifest,
        source_trace_path=source_trace_path,
        run=run,
        out_dir=out_path,
        provider=provider,
        model=model,
        preserve_paths=preserve_paths,
    )
    result_path = out_path / "harness_result.json"
    result["result_path"] = _display_path(result_path, preserve_paths, base_dir=out_path)
    _write_json(result_path, result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a deterministic offline Hermes harness scenario")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Write harness manifest/result artifacts for one scenario")
    run.add_argument("--scenario", required=True, help="Scenario JSON/YAML contract to run")
    run.add_argument("--out", required=True, help="Output directory for harness and Flight Recorder artifacts")
    run.add_argument(
        "--mock-response",
        required=True,
        help="Deterministic assistant response to score against the scenario contract",
    )
    run.add_argument("--provider", default=DEFAULT_PROVIDER, choices=[DEFAULT_PROVIDER])
    run.add_argument("--model", default=DEFAULT_MODEL)
    run.add_argument("--preserve-paths", action="store_true", help="Allow absolute paths in generated artifacts")
    run.add_argument("--force", action="store_true", help="Replace an existing non-empty output directory")

    args = parser.parse_args(argv)
    if args.command == "run":
        result = run_mock_harness(
            args.scenario,
            args.out,
            mock_response=args.mock_response,
            provider=args.provider,
            model=args.model,
            preserve_paths=args.preserve_paths,
            force=args.force,
        )
        print(
            f"{'PASS' if result['passed'] else 'FAIL'} harness {result['scenario_id']} "
            f"score={result['score']['score']} result={args.out}/harness_result.json"
        )
        return 0 if result["passed"] else 1
    return 2


def _prepare_output_dir(path: Path, *, force: bool) -> None:
    if path.exists() and not path.is_dir():
        raise HarnessError(f"harness output path is not a directory: {path}")
    if path.exists() and any(path.iterdir()):
        if not force:
            raise HarnessError(f"harness output directory is not empty: {path}; pass --force to replace it")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _write_mock_observer_trace(path: Path, *, scenario: dict[str, Any], model: str, mock_response: str) -> None:
    session_id = f"harness-{_safe_id(str(scenario['id']))}"
    rows = [
        {
            "hook": "pre_llm_call",
            "payload": {
                "session_id": session_id,
                "model": model,
                "user_message": str(scenario.get("prompt") or ""),
            },
        },
        {
            "hook": "post_llm_call",
            "payload": {
                "session_id": session_id,
                "model": model,
                "assistant_response": mock_response,
            },
        },
    ]
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _build_manifest(
    *,
    scenario_path: Path,
    scenario: dict[str, Any],
    source_trace_path: Path,
    out_dir: Path,
    provider: str,
    model: str,
    mock_response: str,
    preserve_paths: bool,
) -> dict[str, Any]:
    return {
        "schema_version": HARNESS_MANIFEST_SCHEMA_VERSION,
        "manifest_path": _display_path(out_dir / "harness_manifest.json", preserve_paths, base_dir=out_dir),
        "created_at": _utc_now(),
        "harness": {
            "name": HARNESS_NAME,
            "mode": HARNESS_MODE,
            "provider": provider,
            "model": model,
            "network": "disabled",
        },
        "scenario": {
            "id": str(scenario.get("id") or ""),
            "title": str(scenario.get("title") or ""),
            "path": _display_path(scenario_path, preserve_paths, base_dir=out_dir),
            "sha256": _sha256(scenario_path),
            "prompt_sha256": _hash_text(str(scenario.get("prompt") or "")),
        },
        "source_trace": _file_record(source_trace_path, preserve_paths, base_dir=out_dir),
        "output_dir": {
            "path": ".",
            "kind": "directory",
            "exists": out_dir.exists(),
            "expected_artifacts": [
                "harness_manifest.json",
                "harness_result.json",
                "source_trace.observer.jsonl",
                "normalized_trace.json",
                "scorecard.json",
                "task_completion.json",
                "run_digest.json",
                "report.html",
                "artifact_lineage.json",
            ],
        },
        "tool_policy": {
            "network": "disabled",
            "allowed_tools": [],
            "write_scope": _display_path(out_dir, preserve_paths, base_dir=out_dir),
        },
        "mock_response": {
            "sha256": _hash_text(mock_response),
            "length_chars": len(mock_response),
        },
    }


def _build_result(
    *,
    manifest_path: Path,
    manifest: dict[str, Any],
    source_trace_path: Path,
    run: dict[str, Any],
    out_dir: Path,
    provider: str,
    model: str,
    preserve_paths: bool,
) -> dict[str, Any]:
    scorecard = run["scorecard"]
    paths = run["paths"]
    artifacts = {
        "harness_manifest": _json_file_record(manifest_path, preserve_paths, base_dir=out_dir),
        "source_trace": _file_record(source_trace_path, preserve_paths, base_dir=out_dir),
    }
    for name in EXPECTED_RUN_ARTIFACTS:
        path = paths.get(name)
        if isinstance(path, Path):
            artifacts[name] = _json_file_record(path, preserve_paths, base_dir=out_dir)

    checks = [
        {
            "id": "manifest_schema_version",
            "passed": manifest.get("schema_version") == HARNESS_MANIFEST_SCHEMA_VERSION,
            "summary": "Harness manifest uses the expected schema version.",
        },
        {
            "id": "scorecard_passed",
            "passed": bool(scorecard.get("passed")),
            "summary": "Generated scorecard passed the scenario contract.",
        },
        {
            "id": "run_artifacts_present",
            "passed": all(name in artifacts and artifacts[name].get("exists") for name in EXPECTED_RUN_ARTIFACTS),
            "summary": "Normal Flight Recorder run artifacts were written.",
        },
    ]
    failed_checks = [check for check in checks if not check["passed"]]
    return {
        "schema_version": HARNESS_RESULT_SCHEMA_VERSION,
        "created_at": _utc_now(),
        "passed": not failed_checks,
        "harness": {
            "name": HARNESS_NAME,
            "mode": HARNESS_MODE,
            "provider": provider,
            "model": model,
        },
        "scenario_id": str(scorecard.get("scenario_id") or manifest.get("scenario", {}).get("id") or ""),
        "score": {
            "score": int(scorecard.get("score") or 0),
            "passed": bool(scorecard.get("passed")),
            "pass_threshold": int(scorecard.get("pass_threshold") or 0),
        },
        "check_count": len(checks),
        "failed_check_count": len(failed_checks),
        "checks": checks,
        "manifest": {
            "path": _display_path(manifest_path, preserve_paths, base_dir=out_dir),
            "schema_version": manifest.get("schema_version"),
            "sha256": _sha256(manifest_path),
        },
        "artifacts": artifacts,
        "summary": (
            "Offline harness run is ready for downstream evidence handoff."
            if not failed_checks
            else "Offline harness run is blocked by failed checks."
        ),
    }


def _json_file_record(path: Path, preserve_paths: bool, *, base_dir: Path) -> dict[str, Any]:
    record = _file_record(path, preserve_paths, base_dir=base_dir)
    if path.exists() and path.is_file() and path.suffix == ".json":
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            record["schema_version"] = payload.get("schema_version")
            if isinstance(payload.get("passed"), bool):
                record["passed"] = payload["passed"]
    return record


def _file_record(path: Path, preserve_paths: bool, *, base_dir: Path) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": _display_path(path, preserve_paths, base_dir=base_dir),
        "kind": "file",
        "exists": path.exists(),
    }
    if path.exists() and path.is_file():
        record["size_bytes"] = path.stat().st_size
        record["sha256"] = _sha256(path)
    return record


def _display_path(path: Path, preserve_paths: bool, *, base_dir: Path) -> str:
    raw = str(path)
    if preserve_paths:
        return raw
    if _is_windows_absolute(raw):
        return f"<redacted:{_basename(raw)}>"
    resolved = path.resolve()
    for root in (base_dir.resolve(), Path.cwd().resolve()):
        try:
            return str(resolved.relative_to(root)) or "."
        except ValueError:
            continue
    return f"<redacted:{resolved.name}>"


def _safe_id(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in "._-" else "_" for char in value).strip("._-")
    return cleaned or "scenario"


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_windows_absolute(value: str) -> bool:
    normalized = value.replace("/", "\\")
    return (len(normalized) >= 3 and normalized[1:3] == ":\\" and normalized[0].isalpha()) or normalized.startswith("\\\\")


def _basename(value: str) -> str:
    return value.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
