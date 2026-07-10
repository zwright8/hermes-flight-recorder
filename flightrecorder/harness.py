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

from .path_safety import json_marker_matches_schema, locked_owned_output_directory
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

DEFAULT_FAKE_SECRET_CANARIES = {
    "HFR_OFFLINE_MOCK_CANARY": "hfr-offline-mock-canary",
}


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
    try:
        with locked_owned_output_directory(
            out_path,
            repo_root=Path(__file__).resolve().parents[1],
            force=force,
            label="harness output",
            is_owned=lambda target: json_marker_matches_schema(
                target,
                "harness_result.json",
                "harness_run_result",
            ),
        ):
            out_path.mkdir(parents=True, exist_ok=True)
            return _run_mock_harness_into(
                scenario_path,
                out_path,
                mock_response=mock_response,
                provider=provider,
                model=model,
                preserve_paths=preserve_paths,
            )
    except ValueError as exc:
        raise HarnessError(str(exc)) from exc


def _run_mock_harness_into(
    scenario_path: Path,
    out_path: Path,
    *,
    mock_response: str,
    provider: str,
    model: str,
    preserve_paths: bool,
) -> dict[str, Any]:
    scenario = load_scenario(scenario_path)
    scenario_run_path = _publish_scenario_input(scenario_path, out_path, preserve_paths=preserve_paths)
    source_trace_path = out_path / "source_trace.observer.jsonl"
    _write_mock_observer_trace(source_trace_path, scenario=scenario, model=model, mock_response=mock_response)
    sandbox = _prepare_sandbox(out_path)

    manifest_path = out_path / "harness_manifest.json"
    manifest = _build_manifest(
        scenario_path=scenario_run_path,
        scenario=scenario,
        source_trace_path=source_trace_path,
        out_dir=out_path,
        provider=provider,
        model=model,
        mock_response=mock_response,
        sandbox=sandbox,
        preserve_paths=preserve_paths,
    )
    _write_json(manifest_path, manifest)

    from .cli import _run_scenario_artifacts

    run = _run_scenario_artifacts(
        scenario_run_path,
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
            f"score={result['scorecard']['score']} result={args.out}/harness_result.json"
        )
        return 0 if result["passed"] else 1
    return 2


def _publish_scenario_input(scenario_path: Path, out_dir: Path, *, preserve_paths: bool) -> Path:
    if preserve_paths:
        return scenario_path
    inputs_dir = out_dir / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    published = inputs_dir / "scenario.json"
    shutil.copyfile(scenario_path, published)
    return published


def _prepare_sandbox(out_dir: Path) -> dict[str, Any]:
    sandbox_root = out_dir / "sandbox"
    home = sandbox_root / "home"
    workspace = sandbox_root / "workspace"
    events = sandbox_root / "events.jsonl"
    for path in (sandbox_root, home, workspace):
        path.mkdir(parents=True, exist_ok=True)
    events.write_text("", encoding="utf-8")
    return {
        "root": sandbox_root,
        "home": home,
        "workspace": workspace,
        "events": events,
        "fake_secret_canaries": _fake_secret_canary_records(),
        "ephemeral": True,
        "audit_artifacts_kept": True,
    }


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
    sandbox: dict[str, Any],
    preserve_paths: bool,
) -> dict[str, Any]:
    tool_policy = _tool_policy(scenario)
    return {
        "schema_version": HARNESS_MANIFEST_SCHEMA_VERSION,
        "manifest_path": _display_path(out_dir / "harness_manifest.json", preserve_paths, base_dir=out_dir),
        "created_at": _utc_now(),
        "runner": HARNESS_NAME,
        "provider": provider,
        "model": {"id": model},
        "outputs": {
            "run_dir": ".",
            "manifest": "harness_manifest.json",
            "result": "harness_result.json",
        },
        "sandbox": _display_sandbox(sandbox, preserve_paths, base_dir=out_dir),
        "tool_policy": tool_policy,
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
    preserve_paths: bool,
) -> dict[str, Any]:
    scorecard = run["scorecard"]
    paths = run["paths"]
    artifacts = _artifact_paths(paths, preserve_paths, base_dir=out_dir)

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
            "passed": all(isinstance(paths.get(name), Path) and paths[name].is_file() for name in EXPECTED_RUN_ARTIFACTS),
            "summary": "Normal Flight Recorder run artifacts were written.",
        },
    ]
    failed_checks = [check for check in checks if not check["passed"]]
    scorecard_path = paths["scorecard"]
    lineage_path = paths["lineage"]
    return {
        "schema_version": HARNESS_RESULT_SCHEMA_VERSION,
        "created_at": _utc_now(),
        "runner": HARNESS_NAME,
        "provider": provider,
        "model": manifest["model"],
        "scenario_id": str(scorecard.get("scenario_id") or manifest.get("scenario", {}).get("id") or ""),
        "sandbox": manifest["sandbox"],
        "tool_policy": manifest["tool_policy"],
        "trace": {
            "format": "observer_jsonl",
            "path": _display_path(source_trace_path, preserve_paths, base_dir=out_dir),
            "sha256": _sha256(source_trace_path),
            "size_bytes": source_trace_path.stat().st_size,
        },
        "scorecard": {
            "path": _display_path(scorecard_path, preserve_paths, base_dir=out_dir),
            "sha256": _sha256(scorecard_path),
            "size_bytes": scorecard_path.stat().st_size,
            "passed": bool(scorecard.get("passed")),
            "score": int(scorecard.get("score") or 0),
            "pass_threshold": int(scorecard.get("pass_threshold") or 0),
            "critical_failures": scorecard.get("critical_failures", []),
        },
        "artifacts": artifacts,
        "replay": _replay_reference(lineage_path, run["lineage"], preserve_paths, base_dir=out_dir),
        "passed": not failed_checks,
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
            "size_bytes": manifest_path.stat().st_size,
        },
        "summary": (
            "Offline harness run is ready for downstream evidence handoff."
            if not failed_checks
            else "Offline harness run is blocked by failed checks."
        ),
    }


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


def _artifact_paths(paths: dict[str, Any], preserve_paths: bool, *, base_dir: Path) -> dict[str, Any]:
    artifacts: dict[str, Any] = {}
    for key, value in paths.items():
        if value is None:
            continue
        path = Path(value)
        artifacts[key] = _display_path(path, preserve_paths, base_dir=base_dir)
        if path.is_file():
            artifacts[f"{key}_sha256"] = _sha256(path)
            artifacts[f"{key}_size_bytes"] = path.stat().st_size
    return artifacts


def _replay_reference(lineage_path: Path, lineage: dict[str, Any], preserve_paths: bool, *, base_dir: Path) -> dict[str, Any]:
    replay = lineage.get("replay") if isinstance(lineage.get("replay"), dict) else {}
    command = replay.get("command") if isinstance(replay.get("command"), str) else ""
    return {
        "lineage": _display_path(lineage_path, preserve_paths, base_dir=base_dir),
        "lineage_sha256": _sha256(lineage_path),
        "lineage_size_bytes": lineage_path.stat().st_size,
        "command": command,
        "self_contained": replay.get("self_contained") is True,
    }


def _display_sandbox(sandbox: dict[str, Any], preserve_paths: bool, *, base_dir: Path) -> dict[str, Any]:
    rendered = dict(sandbox)
    for field_name in ("root", "home", "workspace", "events"):
        if isinstance(rendered.get(field_name), Path):
            rendered[field_name] = _display_path(rendered[field_name], preserve_paths, base_dir=base_dir)
    return rendered


def _tool_policy(scenario: dict[str, Any]) -> dict[str, Any]:
    scenario_policy = scenario.get("policy") if isinstance(scenario.get("policy"), dict) else {}
    allowed_tools = _string_list(scenario_policy.get("allowed_tools"))
    denied_tools = _string_list(scenario_policy.get("forbidden_tool_names"))
    return {
        "source": "scenario.policy+offline_mock_defaults",
        "scenario_policy": scenario_policy,
        "runtime_policy": {
            "mode": HARNESS_MODE,
            "allowed_tools": allowed_tools,
            "denied_tools": denied_tools,
            "network": {"mode": "disabled", "allowed_hosts": []},
        },
        "blocked_action_canaries": _blocked_action_canaries(scenario_policy),
    }


def _blocked_action_canaries(policy: dict[str, Any]) -> list[dict[str, str]]:
    canaries: list[dict[str, str]] = []
    for field_name, canary_type in (
        ("forbidden_tool_names", "tool_name"),
        ("forbidden_command_patterns", "command"),
        ("forbidden_url_patterns", "url"),
        ("secret_patterns", "secret"),
    ):
        for pattern in _string_list(policy.get(field_name)):
            canaries.append({"type": canary_type, "pattern": pattern, "expected": "blocked_or_absent"})
    return canaries


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item)]


def _fake_secret_canary_records() -> list[dict[str, str]]:
    return [{"name": name, "sha256": _hash_text(value)} for name, value in sorted(DEFAULT_FAKE_SECRET_CANARIES.items())]


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
