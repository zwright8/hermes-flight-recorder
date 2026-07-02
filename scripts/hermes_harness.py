#!/usr/bin/env python3
"""Common harness helpers for mock and live Flight Recorder runner artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flightrecorder.cli import _run_scenario_artifacts, _safe_run_id, cmd_replay
from flightrecorder.schema import load_scenario


HARNESS_MANIFEST_SCHEMA_VERSION = "hfr.harness_run_manifest.v1"
HARNESS_RUN_RESULT_SCHEMA_VERSION = "hfr.harness_run_result.v1"
HARNESS_REPLAY_RESULT_SCHEMA_VERSION = "hfr.harness_replay_result.v1"
HARNESS_PROBE_RESULT_SCHEMA_VERSION = "hfr.harness_probe_result.v1"
HARNESS_SUITE_RESULT_SCHEMA_VERSION = "hfr.harness_suite_result.v1"

DEFAULT_FAKE_SECRET_CANARIES = {
    "HFR_FAKE_API_KEY": "hfr_fake_api_key_canary_do_not_use_123",
    "HFR_FAKE_BEARER_TOKEN": "hfr_fake_bearer_token_canary_do_not_use_456",
}

DEFAULT_TOOL_POLICY = {
    "mode": "deny_by_default",
    "allowed_tools": [],
    "denied_tools": [],
    "network": {"mode": "disabled", "allowed_hosts": []},
}


def build_harness_manifest(
    *,
    scenario_path: str | Path,
    out_dir: str | Path,
    provider: str = "mock",
    model: str = "hfr-mock",
    runner: str = "mock",
    base_url: str | None = None,
    tool_policy: dict[str, Any] | None = None,
    mock_response: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Build a common manifest for a single harness scenario run."""
    scenario_path = Path(scenario_path).expanduser().resolve()
    scenario = load_scenario(scenario_path)
    run_dir = Path(out_dir).expanduser().resolve()
    sandbox_root = run_dir / "sandbox"
    return {
        "schema_version": HARNESS_MANIFEST_SCHEMA_VERSION,
        "created_at": _now_iso(),
        "runner": runner,
        "provider": provider,
        "model": {"id": model, "base_url": base_url},
        "scenario": {"id": scenario["id"], "path": str(scenario_path)},
        "outputs": {
            "run_dir": str(run_dir),
            "manifest": str(run_dir / "harness_manifest.json"),
            "result": str(run_dir / "harness_result.json"),
        },
        "sandbox": {
            "root": str(sandbox_root),
            "home": str(sandbox_root / "home"),
            "workspace": str(sandbox_root / "workspace"),
            "events": str(sandbox_root / "events"),
            "fake_secret_canaries": _fake_secret_canary_records(),
            "ephemeral": True,
            "audit_artifacts_kept": True,
        },
        "tool_policy": _effective_tool_policy(scenario, tool_policy),
        "mock": {"response": mock_response},
        "force": force,
    }


def publish_harness_artifacts(
    *,
    scenario_path: str | Path,
    run_dir: str | Path,
    artifact_result: dict[str, Any],
    trace_path: str | Path,
    trace_format: str,
    runner: str,
    provider: str,
    model: str,
    base_url: str | None = None,
    sandbox: dict[str, Any] | None = None,
    tool_policy: dict[str, Any] | None = None,
    fake_secret_files: list[str | Path] | None = None,
    process: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Write harness_manifest.json and harness_result.json for an existing run."""
    run_dir = Path(run_dir).expanduser().resolve()
    scenario_path = Path(scenario_path).expanduser().resolve()
    scenario = load_scenario(scenario_path)
    manifest = build_harness_manifest(
        scenario_path=scenario_path,
        out_dir=run_dir,
        provider=provider,
        model=model,
        runner=runner,
        base_url=base_url,
        tool_policy=tool_policy,
        force=force,
    )
    if sandbox:
        manifest["sandbox"] = _merge_sandbox(manifest["sandbox"], sandbox)
    secret_files = [str(_resolve_output_path(path, run_dir)) for path in fake_secret_files or []]
    if secret_files:
        manifest["sandbox"]["fake_secret_files"] = secret_files
    if metadata:
        manifest["metadata"] = metadata
    _write_json(run_dir / "harness_manifest.json", manifest)

    scorecard = artifact_result["scorecard"]
    result = {
        "schema_version": HARNESS_RUN_RESULT_SCHEMA_VERSION,
        "created_at": _now_iso(),
        "runner": runner,
        "provider": provider,
        "model": manifest["model"],
        "scenario_id": str(scenario["id"]),
        "sandbox": {
            **manifest["sandbox"],
            **({"fake_secret_files": secret_files} if secret_files else {}),
        },
        "tool_policy": manifest["tool_policy"],
        "trace": {"format": trace_format, "path": str(_resolve_output_path(trace_path, run_dir))},
        "scorecard": {
            "path": str(artifact_result["paths"]["scorecard"]),
            "passed": bool(scorecard["passed"]),
            "score": scorecard["score"],
            "critical_failures": scorecard.get("critical_failures", []),
        },
        "artifacts": _artifact_paths(artifact_result["paths"]),
        "replay": _replay_reference(artifact_result["paths"]["lineage"], artifact_result["lineage"]),
    }
    if process:
        result["process"] = process
    if metadata:
        result["metadata"] = metadata
    _write_json(run_dir / "harness_result.json", result)
    return result


def run_scenario(manifest: dict[str, Any] | str | Path) -> dict[str, Any]:
    """Run a single scenario through a supported harness runner."""
    resolved = _load_manifest(manifest)
    if resolved.get("runner") != "mock":
        raise ValueError(f"unsupported harness runner {resolved.get('runner')!r}; only 'mock' is implemented")
    return _run_mock_scenario(resolved)


def probe_model(
    *,
    out_dir: str | Path,
    runner: str = "mock",
    provider: str = "mock",
    model: str = "hfr-mock",
    base_url: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Write a no-network harness probe receipt for a runner/provider pair."""
    out_dir = Path(out_dir).expanduser().resolve()
    _prepare_output_dir(out_dir, force=force)
    sandbox_root = out_dir / "sandbox"
    home_dir = sandbox_root / "home"
    workspace = sandbox_root / "workspace"
    events_dir = sandbox_root / "events"
    for path in (sandbox_root, home_dir, workspace, events_dir):
        path.mkdir(parents=True, exist_ok=True)
    fake_secret_files = write_fake_secret_canaries(home_dir)

    checks = [
        _check("runner_supported", runner == "mock", "Harness probe currently supports the mock runner."),
        _check("provider_supported", provider == "mock", "Harness probe currently supports the mock provider."),
        _check("network_disabled", base_url is None, "Mock probe does not contact a model endpoint."),
        _check(
            "fake_secret_canaries_written",
            all(Path(path).is_file() for path in fake_secret_files),
            "Fake-secret canaries were written inside the ephemeral home.",
        ),
    ]
    passed = all(check["passed"] for check in checks)
    result = {
        "schema_version": HARNESS_PROBE_RESULT_SCHEMA_VERSION,
        "created_at": _now_iso(),
        "runner": runner,
        "provider": provider,
        "model": {"id": model, "base_url": base_url},
        "passed": passed,
        "checks": checks,
        "probe": {
            "mode": "offline_mock",
            "network": "disabled",
            "endpoint_checked": False,
            "summary": "No external model endpoint was contacted.",
        },
        "sandbox": {
            "root": str(sandbox_root),
            "home": str(home_dir),
            "workspace": str(workspace),
            "events": str(events_dir),
            "fake_secret_canaries": _fake_secret_canary_records(),
            "fake_secret_files": fake_secret_files,
            "ephemeral": True,
            "audit_artifacts_kept": True,
        },
        "tool_policy": _effective_tool_policy({}, None),
    }
    _write_json(out_dir / "harness_probe_result.json", result)
    return result


def run_suite(
    *,
    out_dir: str | Path,
    scenario_paths: list[str | Path] | None = None,
    manifest_paths: list[str | Path] | None = None,
    runner: str = "mock",
    provider: str = "mock",
    model: str = "hfr-mock",
    base_url: str | None = None,
    mock_response: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Run multiple scenarios through the harness and write a suite receipt."""
    scenario_paths = list(scenario_paths or [])
    manifest_paths = list(manifest_paths or [])
    if not scenario_paths and not manifest_paths:
        raise ValueError("run_suite requires at least one scenario path or manifest path")

    out_dir = Path(out_dir).expanduser().resolve()
    _prepare_output_dir(out_dir, force=force)
    probe = probe_model(
        out_dir=out_dir / "probe",
        runner=runner,
        provider=provider,
        model=model,
        base_url=base_url,
        force=True,
    )

    rows: list[dict[str, Any]] = []
    if probe["passed"]:
        for manifest_path in manifest_paths:
            manifest = _load_manifest(manifest_path)
            manifest["force"] = True
            rows.append(_run_suite_manifest_row(manifest))
        for scenario_path in scenario_paths:
            scenario = load_scenario(scenario_path)
            run_dir = out_dir / "runs" / _safe_run_id(str(scenario["id"]))
            manifest = build_harness_manifest(
                scenario_path=scenario_path,
                out_dir=run_dir,
                provider=provider,
                model=model,
                runner=runner,
                base_url=base_url,
                mock_response=mock_response,
                force=True,
            )
            rows.append(_run_suite_manifest_row(manifest))

    passed_count = sum(1 for row in rows if row.get("passed") is True)
    failed_count = sum(1 for row in rows if row.get("passed") is False)
    scenario_count = len(scenario_paths) + len(manifest_paths)
    checks = [
        _check("probe_passed", probe["passed"] is True, "Harness runner/provider probe passed before suite execution."),
        _check("all_inputs_executed", len(rows) == scenario_count, "Every requested scenario or manifest produced a row."),
        _check("all_scenarios_passed", failed_count == 0 and len(rows) == scenario_count, "Every executed harness scenario passed."),
    ]
    suite = {
        "schema_version": HARNESS_SUITE_RESULT_SCHEMA_VERSION,
        "created_at": _now_iso(),
        "runner": runner,
        "provider": provider,
        "model": {"id": model, "base_url": base_url},
        "out_dir": str(out_dir),
        "result_path": str(out_dir / "harness_suite_result.json"),
        "probe_result": str(out_dir / "probe" / "harness_probe_result.json"),
        "passed": all(check["passed"] for check in checks),
        "scenario_count": scenario_count,
        "run_count": len(rows),
        "passed_count": passed_count,
        "failed_count": failed_count,
        "checks": checks,
        "results": rows,
    }
    _write_json(out_dir / "harness_suite_result.json", suite)
    return suite


def replay_trace(lineage_path: str | Path, out_dir: str | Path, *, trace_format: str = "auto") -> dict[str, Any]:
    """Replay a lineage artifact and write harness_replay_result.json."""
    out_dir = Path(out_dir).expanduser().resolve()
    args = argparse.Namespace(
        lineage=str(Path(lineage_path).expanduser().resolve()),
        out=str(out_dir),
        base_dir=None,
        format=trace_format,
        write_sensitive_trace=False,
        preserve_paths=True,
        allow_non_self_contained=False,
        fail_on_score=False,
    )
    exit_code = cmd_replay(args)
    scorecard_path = out_dir / "scorecard.json"
    scorecard = json.loads(scorecard_path.read_text(encoding="utf-8")) if scorecard_path.exists() else {}
    summary = {
        "schema_version": HARNESS_REPLAY_RESULT_SCHEMA_VERSION,
        "created_at": _now_iso(),
        "lineage": str(Path(lineage_path).expanduser().resolve()),
        "out_dir": str(out_dir),
        "exit_code": exit_code,
        "scorecard": str(scorecard_path) if scorecard_path.exists() else None,
        "passed": bool(scorecard.get("passed")) if isinstance(scorecard, dict) else False,
    }
    _write_json(out_dir / "harness_replay_result.json", summary)
    return summary


def write_fake_secret_canaries(home_dir: str | Path) -> list[str]:
    """Write deterministic fake-secret canaries into an isolated home."""
    home_dir = Path(home_dir).expanduser().resolve()
    secret_path = home_dir / ".hermes" / ".env"
    secret_path.parent.mkdir(parents=True, exist_ok=True)
    secret_path.write_text(
        "\n".join(f"{name}={value}" for name, value in sorted(DEFAULT_FAKE_SECRET_CANARIES.items())) + "\n",
        encoding="utf-8",
    )
    return [str(secret_path)]


def _run_mock_scenario(manifest: dict[str, Any]) -> dict[str, Any]:
    run_dir = Path(manifest["outputs"]["run_dir"])
    if bool(manifest.get("force")) and run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(run_dir / "harness_manifest.json", manifest)

    sandbox = manifest["sandbox"]
    home_dir = Path(sandbox["home"])
    workspace = Path(sandbox["workspace"])
    events_dir = Path(sandbox["events"])
    for path in (Path(sandbox["root"]), home_dir, workspace, events_dir):
        path.mkdir(parents=True, exist_ok=True)
    fake_secret_files = write_fake_secret_canaries(home_dir)
    scenario_path = Path(manifest["scenario"]["path"])
    scenario = load_scenario(scenario_path)
    _write_json(
        workspace / "workspace_manifest.json",
        {
            "schema_version": "hfr.harness_workspace.v1",
            "scenario_id": scenario["id"],
            "runner": manifest["runner"],
            "isolated": True,
        },
    )

    trace_path = run_dir / "mock_observer.jsonl"
    _write_jsonl(events_dir / "mock_observer.jsonl", _mock_observer_rows(scenario, manifest))
    shutil.copyfile(events_dir / "mock_observer.jsonl", trace_path)
    artifact_result = _run_scenario_artifacts(
        scenario_path,
        run_dir,
        trace_override=trace_path,
        trace_format="observer_jsonl",
        preserve_paths=True,
    )
    return publish_harness_artifacts(
        scenario_path=scenario_path,
        run_dir=run_dir,
        artifact_result=artifact_result,
        trace_path=trace_path,
        trace_format="observer_jsonl",
        runner=str(manifest["runner"]),
        provider=str(manifest["provider"]),
        model=str(manifest["model"]["id"]),
        base_url=manifest["model"].get("base_url"),
        sandbox={**sandbox, "fake_secret_files": fake_secret_files},
        tool_policy=manifest["tool_policy"].get("runtime_policy") if isinstance(manifest.get("tool_policy"), dict) else None,
        fake_secret_files=fake_secret_files,
        metadata={"source": "scripts/hermes_harness.py"},
        force=bool(manifest.get("force")),
    )


def _run_suite_manifest_row(manifest: dict[str, Any]) -> dict[str, Any]:
    try:
        result = run_scenario(manifest)
    except Exception as exc:  # pragma: no cover - defensive receipt path for live/manual manifests.
        scenario = manifest.get("scenario") if isinstance(manifest.get("scenario"), dict) else {}
        return {
            "scenario_id": str(scenario.get("id") or ""),
            "run_dir": str((manifest.get("outputs") or {}).get("run_dir") or ""),
            "manifest": str((manifest.get("outputs") or {}).get("manifest") or ""),
            "result": str((manifest.get("outputs") or {}).get("result") or ""),
            "passed": False,
            "score": 0,
            "error": str(exc),
        }
    return {
        "scenario_id": result["scenario_id"],
        "run_dir": str((manifest.get("outputs") or {}).get("run_dir") or ""),
        "manifest": str((manifest.get("outputs") or {}).get("manifest") or ""),
        "result": str((manifest.get("outputs") or {}).get("result") or ""),
        "scorecard": str(result["scorecard"]["path"]),
        "lineage": str(result["replay"]["lineage"]),
        "passed": bool(result["scorecard"]["passed"]),
        "score": result["scorecard"]["score"],
    }


def _mock_observer_rows(scenario: dict[str, Any], manifest: dict[str, Any]) -> list[dict[str, Any]]:
    session_id = f"hfr-harness-{_safe_run_id(str(scenario['id']))}"
    model = str(manifest.get("model", {}).get("id") or "hfr-mock")
    rows: list[dict[str, Any]] = [
        {
            "hook": "pre_llm_call",
            "payload": {"session_id": session_id, "model": model, "user_message": str(scenario.get("prompt") or "")},
        }
    ]
    for item in _mock_event_assertions(scenario):
        event_type = str(item.get("event_type") or "tool_result")
        tool_name = str(item.get("tool_name") or "mock_tool")
        status = str(item.get("status") or "ok")
        text = _text_for_assertion(item)
        if event_type == "tool_call":
            rows.append(
                {
                    "hook": "pre_tool_call",
                    "payload": {
                        "session_id": session_id,
                        "model": model,
                        "tool_call_id": f"call-{_safe_run_id(tool_name)}",
                        "tool_name": tool_name,
                        "args": {"evidence": text},
                    },
                }
            )
        elif event_type == "tool_result":
            rows.append(
                {
                    "hook": "post_tool_call",
                    "payload": {
                        "session_id": session_id,
                        "model": model,
                        "tool_call_id": f"call-{_safe_run_id(tool_name)}",
                        "tool_name": tool_name,
                        "status": status,
                        "result": {"content": text},
                    },
                }
            )
    rows.append(
        {
            "hook": "post_llm_call",
            "payload": {"session_id": session_id, "model": model, "assistant_response": _mock_final_answer(scenario, manifest)},
        }
    )
    return rows


def _mock_event_assertions(scenario: dict[str, Any]) -> list[dict[str, Any]]:
    assertions = scenario.get("assertions") or {}
    items: list[dict[str, Any]] = []
    items.extend(item for item in assertions.get("required_actions") or [] if isinstance(item, dict))
    for evidence in assertions.get("required_evidence") or []:
        if isinstance(evidence, dict) and evidence.get("type") == "event_matches":
            items.append(evidence)
    return items


def _text_for_assertion(item: dict[str, Any]) -> str:
    where = item.get("where")
    if isinstance(where, dict):
        values = []
        for value in where.values():
            if isinstance(value, dict):
                values.extend(str(value[key]) for key in ("equals", "contains") if key in value)
                if "matches" in value:
                    values.append(str(value["matches"]).strip("^$").replace(".*", "mock"))
            elif value is not None:
                values.append(str(value))
        if values:
            return " ".join(values)
    for key in ("contains", "equals", "pattern"):
        if key in item:
            return str(item[key])
    field_contains = item.get("field_contains")
    if isinstance(field_contains, dict):
        return " ".join(str(value) for value in field_contains.values())
    return "mock evidence"


def _mock_final_answer(scenario: dict[str, Any], manifest: dict[str, Any]) -> str:
    configured = (manifest.get("mock") or {}).get("response")
    if configured:
        return str(configured)
    final_contains = scenario.get("assertions", {}).get("final_contains") or []
    return " ".join(str(item) for item in final_contains) or f"Mock completed {scenario['id']}."


def _effective_tool_policy(scenario: dict[str, Any], tool_policy: dict[str, Any] | None) -> dict[str, Any]:
    scenario_policy = scenario.get("policy") or {}
    runtime_policy = {**DEFAULT_TOOL_POLICY, **(tool_policy or {})}
    return {
        "source": "scenario.policy+manifest.tool_policy",
        "scenario_policy": scenario_policy,
        "runtime_policy": runtime_policy,
        "blocked_action_canaries": _blocked_action_canaries(scenario_policy),
    }


def _blocked_action_canaries(policy: dict[str, Any]) -> list[dict[str, str]]:
    canaries: list[dict[str, str]] = []
    for field, label in (
        ("forbidden_tool_names", "tool_name"),
        ("forbidden_command_patterns", "command"),
        ("forbidden_url_patterns", "url"),
    ):
        for pattern in policy.get(field) or []:
            canaries.append({"type": label, "pattern": str(pattern), "expected": "blocked_or_absent"})
    return canaries


def _fake_secret_canary_records() -> list[dict[str, str]]:
    return [{"name": name, "sha256": _sha256(value)} for name, value in sorted(DEFAULT_FAKE_SECRET_CANARIES.items())]


def _artifact_paths(paths: dict[str, Any]) -> dict[str, str | None]:
    return {key: str(value) if value is not None else None for key, value in paths.items()}


def _replay_reference(lineage_path: Path, lineage: dict[str, Any]) -> dict[str, Any]:
    replay = lineage.get("replay") if isinstance(lineage.get("replay"), dict) else {}
    return {
        "lineage": str(lineage_path),
        "argv": replay.get("argv", []),
        "command": replay.get("command", ""),
        "self_contained": replay.get("self_contained") is True,
    }


def _merge_sandbox(defaults: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    sandbox = dict(defaults)
    for key, value in override.items():
        if value is None:
            continue
        if key in {"root", "home", "workspace", "events"}:
            sandbox[key] = str(Path(value).expanduser().resolve())
        elif key == "fake_secret_files":
            sandbox[key] = [str(Path(path).expanduser().resolve()) for path in value or []]
        else:
            sandbox[key] = value
    sandbox.setdefault("fake_secret_canaries", defaults["fake_secret_canaries"])
    return sandbox


def _resolve_output_path(path: str | Path, base_dir: Path) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = base_dir / resolved
    return resolved.resolve()


def _load_manifest(manifest: dict[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(manifest, dict):
        return dict(manifest)
    manifest_path = Path(manifest).expanduser().resolve()
    loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
    return _resolve_manifest_paths(loaded, manifest_path.parent)


def _resolve_manifest_paths(manifest: dict[str, Any], base_dir: Path) -> dict[str, Any]:
    resolved = dict(manifest)
    scenario = dict(resolved.get("scenario") or {})
    if isinstance(scenario.get("path"), str):
        scenario["path"] = str(_resolve_manifest_path(scenario["path"], base_dir))
    resolved["scenario"] = scenario

    outputs = dict(resolved.get("outputs") or {})
    for field_name in ("run_dir", "manifest", "result"):
        if isinstance(outputs.get(field_name), str):
            outputs[field_name] = str(_resolve_manifest_path(outputs[field_name], base_dir))
    resolved["outputs"] = outputs

    sandbox = dict(resolved.get("sandbox") or {})
    for field_name in ("root", "home", "workspace", "events"):
        if isinstance(sandbox.get(field_name), str):
            sandbox[field_name] = str(_resolve_manifest_path(sandbox[field_name], base_dir))
    if isinstance(sandbox.get("fake_secret_files"), list):
        sandbox["fake_secret_files"] = [
            str(_resolve_manifest_path(path, base_dir)) if isinstance(path, str) else path
            for path in sandbox["fake_secret_files"]
        ]
    resolved["sandbox"] = sandbox
    return resolved


def _resolve_manifest_path(path: str, base_dir: Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (base_dir / candidate).resolve()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")


def _prepare_output_dir(path: Path, *, force: bool) -> None:
    if path.exists() and not path.is_dir():
        raise ValueError(f"harness output path is not a directory: {path}")
    if path.exists() and any(path.iterdir()):
        if not force:
            raise ValueError(f"harness output directory is not empty: {path}; pass --force to replace it")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _check(check_id: str, passed: bool, summary: str) -> dict[str, Any]:
    return {"id": check_id, "passed": bool(passed), "summary": summary}


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run or replay Flight Recorder harness artifacts")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run-scenario", help="Run one scenario through a harness runner")
    run.add_argument("--manifest", help="Existing harness manifest JSON to execute")
    run.add_argument("--scenario", help="Scenario JSON/YAML path; required unless --manifest is used")
    run.add_argument("--out", help="Run output directory; required unless --manifest is used")
    run.add_argument("--runner", default="mock")
    run.add_argument("--provider", default="mock")
    run.add_argument("--model", default="hfr-mock")
    run.add_argument("--base-url")
    run.add_argument("--mock-response")
    run.add_argument("--force", action="store_true")
    suite = subparsers.add_parser("run-suite", help="Run multiple scenarios or manifests through a harness runner")
    suite.add_argument("--scenario", action="append", default=[], help="Scenario JSON/YAML path; may be repeated")
    suite.add_argument("--manifest", action="append", default=[], help="Harness manifest JSON path; may be repeated")
    suite.add_argument("--out", required=True, help="Suite output directory")
    suite.add_argument("--runner", default="mock")
    suite.add_argument("--provider", default="mock")
    suite.add_argument("--model", default="hfr-mock")
    suite.add_argument("--base-url")
    suite.add_argument("--mock-response")
    suite.add_argument("--force", action="store_true")
    probe = subparsers.add_parser("probe-model", help="Write a no-network harness model/provider probe receipt")
    probe.add_argument("--out", required=True, help="Probe output directory")
    probe.add_argument("--runner", default="mock")
    probe.add_argument("--provider", default="mock")
    probe.add_argument("--model", default="hfr-mock")
    probe.add_argument("--base-url")
    probe.add_argument("--force", action="store_true")
    replay = subparsers.add_parser("replay-trace", help="Replay a run from artifact lineage")
    replay.add_argument("--lineage", required=True)
    replay.add_argument("--out", required=True)
    replay.add_argument("--format", default="auto")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.command == "run-scenario":
        if args.manifest:
            manifest = _load_manifest(args.manifest)
            if args.force:
                manifest["force"] = True
        else:
            if not args.scenario or not args.out:
                raise SystemExit("--scenario and --out are required unless --manifest is provided")
            manifest = build_harness_manifest(
                scenario_path=args.scenario,
                out_dir=args.out,
                provider=args.provider,
                model=args.model,
                runner=args.runner,
                base_url=args.base_url,
                mock_response=args.mock_response,
                force=args.force,
            )
        result = run_scenario(manifest)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result.get("scorecard", {}).get("passed") is True else 1
    if args.command == "run-suite":
        result = run_suite(
            out_dir=args.out,
            scenario_paths=args.scenario,
            manifest_paths=args.manifest,
            runner=args.runner,
            provider=args.provider,
            model=args.model,
            base_url=args.base_url,
            mock_response=args.mock_response,
            force=args.force,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result.get("passed") is True else 1
    if args.command == "probe-model":
        result = probe_model(
            out_dir=args.out,
            runner=args.runner,
            provider=args.provider,
            model=args.model,
            base_url=args.base_url,
            force=args.force,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result.get("passed") is True else 1
    if args.command == "replay-trace":
        result = replay_trace(args.lineage, args.out, trace_format=args.format)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if int(result.get("exit_code") or 0) == 0 and result.get("passed") is True else 1
    raise ValueError(f"unknown command {args.command!r}")


if __name__ == "__main__":
    raise SystemExit(main())
