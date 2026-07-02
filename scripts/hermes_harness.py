#!/usr/bin/env python3
"""Common harness helpers for mock and live Flight Recorder runner artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flightrecorder.cli import _run_scenario_artifacts, _safe_run_id, cmd_replay
from flightrecorder.lineage import REPLAY_BUNDLE_SCHEMA_VERSION
from flightrecorder.schema import load_scenario


HARNESS_MANIFEST_SCHEMA_VERSION = "hfr.harness_run_manifest.v1"
HARNESS_RUN_RESULT_SCHEMA_VERSION = "hfr.harness_run_result.v1"
HARNESS_REPLAY_RESULT_SCHEMA_VERSION = "hfr.harness_replay_result.v1"
HARNESS_SUITE_RESULT_SCHEMA_VERSION = "hfr.harness_suite_result.v1"
HARNESS_MODEL_PROBE_SCHEMA_VERSION = "hfr.harness_model_probe.v1"

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
    preserve_paths: bool = True,
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
    _write_json(run_dir / "harness_manifest.json", _display_harness_manifest(manifest, run_dir, preserve_paths))

    scorecard = artifact_result["scorecard"]
    result = {
        "schema_version": HARNESS_RUN_RESULT_SCHEMA_VERSION,
        "created_at": _now_iso(),
        "runner": runner,
        "provider": provider,
        "model": manifest["model"],
        "scenario_id": str(scenario["id"]),
        "sandbox": _display_sandbox(
            {**manifest["sandbox"], **({"fake_secret_files": secret_files} if secret_files else {})},
            run_dir,
            preserve_paths,
        ),
        "tool_policy": manifest["tool_policy"],
        "trace": {
            "format": trace_format,
            "path": _display_path(_resolve_output_path(trace_path, run_dir), run_dir, preserve_paths),
        },
        "scorecard": {
            "path": _display_path(artifact_result["paths"]["scorecard"], run_dir, preserve_paths),
            "passed": bool(scorecard["passed"]),
            "score": scorecard["score"],
            "critical_failures": scorecard.get("critical_failures", []),
        },
        "artifacts": _artifact_paths(artifact_result["paths"], run_dir, preserve_paths),
        "replay": _replay_reference(artifact_result["paths"]["lineage"], artifact_result["lineage"], run_dir, preserve_paths),
    }
    if process:
        result["process"] = process
    if metadata:
        result["metadata"] = metadata
    _write_json(run_dir / "harness_result.json", result)
    return result


def publish_trace_run(
    *,
    scenario_path: str | Path,
    trace_path: str | Path,
    out_dir: str | Path,
    trace_format: str = "auto",
    runner: str = "recorded_trace",
    provider: str = "recorded",
    model: str | None = None,
    base_url: str | None = None,
    tool_policy: dict[str, Any] | None = None,
    force: bool = False,
    preserve_paths: bool = True,
) -> dict[str, Any]:
    """Publish a recorded trace as common harness manifest/result artifacts."""
    run_dir = Path(out_dir).expanduser().resolve()
    if force and run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    resolved_trace = Path(trace_path).expanduser().resolve()
    resolved_scenario = Path(scenario_path).expanduser().resolve()
    published_scenario = resolved_scenario
    published_trace = resolved_trace
    relative_replay_inputs: dict[str, Path] = {}
    if not preserve_paths:
        inputs_dir = run_dir / "inputs"
        inputs_dir.mkdir(parents=True, exist_ok=True)
        published_scenario = inputs_dir / "scenario.json"
        published_trace = inputs_dir / _portable_trace_name(resolved_trace)
        shutil.copyfile(resolved_scenario, published_scenario)
        shutil.copyfile(resolved_trace, published_trace)
        relative_replay_inputs = {
            "scenario": published_scenario,
            "source_trace": published_trace,
        }
    artifact_result = _run_scenario_artifacts(
        published_scenario,
        run_dir,
        trace_override=published_trace,
        trace_format=trace_format,
        preserve_paths=preserve_paths,
    )
    if relative_replay_inputs:
        artifact_result["lineage"] = _rewrite_relative_replay_lineage(
            artifact_result["lineage"],
            run_dir,
            relative_replay_inputs,
        )
        _write_json(artifact_result["paths"]["lineage"], artifact_result["lineage"])
    sandbox_root = run_dir / "sandbox"
    for path in (sandbox_root, sandbox_root / "home", sandbox_root / "workspace", sandbox_root / "events"):
        path.mkdir(parents=True, exist_ok=True)
    fake_secret_files = write_fake_secret_canaries(sandbox_root / "home")
    return publish_harness_artifacts(
        scenario_path=published_scenario,
        run_dir=run_dir,
        artifact_result=artifact_result,
        trace_path=published_trace,
        trace_format=_published_trace_format(artifact_result, trace_format),
        runner=runner,
        provider=provider,
        model=model or _published_trace_model(artifact_result),
        base_url=base_url,
        sandbox={
            "root": sandbox_root,
            "home": sandbox_root / "home",
            "workspace": sandbox_root / "workspace",
            "events": sandbox_root / "events",
            "ephemeral": True,
            "audit_artifacts_kept": True,
        },
        tool_policy=tool_policy,
        fake_secret_files=fake_secret_files,
        process={"exit_code": 0, "mode": "recorded_trace"},
        metadata={"source": "scripts/hermes_harness.py", "interface": "publish_trace_run"},
        force=force,
        preserve_paths=preserve_paths,
    )


def run_scenario(manifest: dict[str, Any] | str | Path, *, preserve_paths: bool = True) -> dict[str, Any]:
    """Run a single scenario through a supported harness runner."""
    resolved = _load_manifest(manifest)
    if resolved.get("runner") != "mock":
        raise ValueError(f"unsupported harness runner {resolved.get('runner')!r}; only 'mock' is implemented")
    return _run_mock_scenario(resolved, preserve_paths=preserve_paths)


def run_suite(
    *,
    scenarios_dir: str | Path,
    out_dir: str | Path,
    pattern: str = "*.json",
    recursive: bool = False,
    provider: str = "mock",
    model: str = "hfr-mock",
    runner: str = "mock",
    base_url: str | None = None,
    mock_response: str | None = None,
    force: bool = False,
    preserve_paths: bool = True,
) -> dict[str, Any]:
    """Run a directory of scenarios through the mock harness runner."""
    scenario_paths = _discover_scenario_paths(scenarios_dir, pattern=pattern, recursive=recursive)
    if not scenario_paths:
        raise ValueError(f"no scenarios matched {pattern!r} under {Path(scenarios_dir)}")
    suite_dir = Path(out_dir).expanduser().resolve()
    if force and suite_dir.exists():
        shutil.rmtree(suite_dir)
    suite_dir.mkdir(parents=True, exist_ok=True)

    runs: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    seen_run_ids: dict[str, Path] = {}
    for scenario_path in scenario_paths:
        try:
            scenario = load_scenario(scenario_path)
            run_id = _safe_run_id(str(scenario["id"]))
            if run_id in seen_run_ids:
                raise ValueError(
                    f"duplicate scenario id/run directory {scenario['id']!r}: "
                    f"{scenario_path} conflicts with {seen_run_ids[run_id]}"
                )
            seen_run_ids[run_id] = scenario_path
            run_dir = suite_dir / run_id
            manifest = build_harness_manifest(
                scenario_path=scenario_path,
                out_dir=run_dir,
                provider=provider,
                model=model,
                runner=runner,
                base_url=base_url,
                mock_response=mock_response,
                force=force,
            )
            result = run_scenario(manifest, preserve_paths=preserve_paths)
            runs.append(_suite_run_record(result, run_dir, suite_dir, preserve_paths))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append({"scenario_path": _display_path(scenario_path, suite_dir, preserve_paths), "error": str(exc)})

    passed = sum(1 for run in runs if run["passed"])
    failed = len(runs) - passed
    summary = {
        "schema_version": HARNESS_SUITE_RESULT_SCHEMA_VERSION,
        "created_at": _now_iso(),
        "runner_interface": "run_suite",
        "scenarios_dir": _display_path(scenarios_dir, suite_dir, preserve_paths),
        "out_dir": _display_path(suite_dir, suite_dir, preserve_paths),
        "pattern": pattern,
        "recursive": recursive,
        "runner": runner,
        "provider": provider,
        "model": {"id": model, "base_url": base_url},
        "total": len(runs),
        "passed": passed,
        "failed": failed,
        "error_count": len(errors),
        "errors": errors,
        "runs": runs,
        "artifacts": {"suite_result": _display_path(suite_dir / "harness_suite_result.json", suite_dir, preserve_paths)},
    }
    _write_json(suite_dir / "harness_suite_result.json", summary)
    return summary


def probe_model(
    *,
    out_dir: str | Path,
    provider: str = "mock",
    model: str = "hfr-mock",
    runner: str = "mock",
    base_url: str | None = None,
    allow_network: bool = False,
    tool_policy: dict[str, Any] | None = None,
    preserve_paths: bool = True,
) -> dict[str, Any]:
    """Write a no-network harness probe receipt for a model endpoint selection."""
    probe_dir = Path(out_dir).expanduser().resolve()
    sandbox_root = probe_dir / "sandbox"
    home_dir = sandbox_root / "home"
    workspace = sandbox_root / "workspace"
    events_dir = sandbox_root / "events"
    for path in (sandbox_root, home_dir, workspace, events_dir):
        path.mkdir(parents=True, exist_ok=True)
    fake_secret_files = write_fake_secret_canaries(home_dir)

    runtime_policy = {**DEFAULT_TOOL_POLICY, **(tool_policy or {})}
    probes = [
        _probe_check("model_declared", bool(str(model).strip()), {"model": model}),
        _probe_check("runner_declared", bool(str(runner).strip()), {"runner": runner}),
        _probe_check("endpoint_declared", provider == "mock" or bool(base_url), {"provider": provider, "base_url": base_url}),
        _probe_check("network_policy_captured", isinstance(runtime_policy.get("network"), dict), {"network": runtime_policy.get("network")}),
        _probe_check("fake_secret_canaries_declared", bool(DEFAULT_FAKE_SECRET_CANARIES), {"canary_count": len(DEFAULT_FAKE_SECRET_CANARIES)}),
        _probe_check(
            "external_network_not_contacted",
            provider == "mock" or not allow_network,
            {"allow_network": allow_network, "summary": "Harness probe records endpoint metadata only."},
        ),
    ]
    failed = [probe["id"] for probe in probes if probe["passed"] is not True]
    receipt = {
        "schema_version": HARNESS_MODEL_PROBE_SCHEMA_VERSION,
        "created_at": _now_iso(),
        "runner_interface": "probe_model",
        "passed": not failed,
        "readiness": "mock_verified" if provider == "mock" and not failed else ("metadata_recorded" if not failed else "blocked"),
        "recommendation": "ready_for_mock_harness" if provider == "mock" and not failed else ("run_live_smoke" if not failed else "fix_probe_inputs"),
        "runner": runner,
        "provider": provider,
        "model": {"id": model, "base_url": base_url},
        "sandbox": {
            "root": _display_path(sandbox_root, probe_dir, preserve_paths),
            "home": _display_path(home_dir, probe_dir, preserve_paths),
            "workspace": _display_path(workspace, probe_dir, preserve_paths),
            "events": _display_path(events_dir, probe_dir, preserve_paths),
            "fake_secret_canaries": _fake_secret_canary_records(),
            "fake_secret_files": [_display_path(path, probe_dir, preserve_paths) for path in fake_secret_files],
            "ephemeral": True,
            "audit_artifacts_kept": True,
        },
        "tool_policy": {
            "source": "harness.probe_model",
            "scenario_policy": {},
            "runtime_policy": runtime_policy,
            "blocked_action_canaries": [],
        },
        "probes": probes,
        "failed_probes": failed,
        "notes": [
            "No external endpoint is contacted by default.",
            "Run a live smoke script for provider-backed verification.",
        ],
    }
    _write_json(probe_dir / "harness_model_probe.json", receipt)
    return receipt


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


def _run_mock_scenario(manifest: dict[str, Any], *, preserve_paths: bool = True) -> dict[str, Any]:
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
        preserve_paths=preserve_paths,
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
        preserve_paths=preserve_paths,
    )


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


def _discover_scenario_paths(scenarios_dir: str | Path, *, pattern: str, recursive: bool) -> list[Path]:
    root = Path(scenarios_dir).expanduser().resolve()
    paths = root.rglob(pattern) if recursive else root.glob(pattern)
    return sorted(path for path in paths if path.is_file())


def _suite_run_record(result: dict[str, Any], run_dir: Path, suite_dir: Path, preserve_paths: bool) -> dict[str, Any]:
    return {
        "scenario_id": result["scenario_id"],
        "runner": result["runner"],
        "provider": result["provider"],
        "model": result["model"],
        "run_dir": _display_path(run_dir, suite_dir, preserve_paths),
        "manifest": _display_path(run_dir / "harness_manifest.json", suite_dir, preserve_paths),
        "result": _display_path(run_dir / "harness_result.json", suite_dir, preserve_paths),
        "trace": result["trace"],
        "scorecard": result["scorecard"],
        "replay": result["replay"],
        "passed": result["scorecard"]["passed"] is True,
    }


def _published_trace_format(artifact_result: dict[str, Any], fallback: str) -> str:
    trace = _published_normalized_trace(artifact_result)
    session = trace.get("session") if isinstance(trace.get("session"), dict) else {}
    source_format = session.get("source_format")
    if isinstance(source_format, str) and source_format:
        return source_format
    return fallback


def _published_trace_model(artifact_result: dict[str, Any]) -> str:
    trace = _published_normalized_trace(artifact_result)
    session = trace.get("session") if isinstance(trace.get("session"), dict) else {}
    model = session.get("model")
    return str(model) if isinstance(model, str) and model else "unknown"


def _published_normalized_trace(artifact_result: dict[str, Any]) -> dict[str, Any]:
    path = artifact_result.get("paths", {}).get("normalized_trace") if isinstance(artifact_result.get("paths"), dict) else None
    if isinstance(path, Path) and path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    return {}


def _portable_trace_name(path: Path) -> str:
    suffixes = "".join(path.suffixes)
    stem = path.name[: -len(suffixes)] if suffixes else path.name
    safe_stem = _safe_run_id(stem)
    return f"{safe_stem}{suffixes}" if suffixes else safe_stem


def _rewrite_relative_replay_lineage(
    lineage: dict[str, Any],
    run_dir: Path,
    replay_inputs: dict[str, Path],
) -> dict[str, Any]:
    rendered = json.loads(json.dumps(lineage))
    relative_inputs = {
        name: {
            "path": _display_path(path, run_dir, preserve_paths=False),
            "exists": path.exists(),
            "sha256": _sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for name, path in replay_inputs.items()
    }

    inputs = rendered.get("inputs")
    if isinstance(inputs, list):
        for record in inputs:
            if not isinstance(record, dict):
                continue
            name = record.get("name")
            relative = relative_inputs.get(name) if isinstance(name, str) else None
            if relative is None:
                continue
            record.update(relative)

    replay = rendered.get("replay")
    if not isinstance(replay, dict):
        replay = {}
        rendered["replay"] = replay
    argv = replay.get("argv")
    if isinstance(argv, list) and all(isinstance(item, str) for item in argv):
        argv = list(argv)
    else:
        argv = [
            "python",
            "-m",
            "flightrecorder",
            "run",
            "--scenario",
            "",
            "--trace",
            "",
            "--out",
            "replay",
        ]
    _replace_replay_flag(argv, "--scenario", relative_inputs["scenario"]["path"])
    _replace_replay_flag(argv, "--trace", relative_inputs["source_trace"]["path"])
    _replace_replay_flag(argv, "--out", "replay")
    replay["argv"] = argv
    replay["command"] = shlex.join(argv)
    replay["self_contained"] = True

    fingerprints = replay.get("input_fingerprints")
    if not isinstance(fingerprints, dict):
        fingerprints = {}
        replay["input_fingerprints"] = fingerprints
    for name, relative in relative_inputs.items():
        record = fingerprints.get(name)
        if not isinstance(record, dict):
            record = {}
            fingerprints[name] = record
        for field_name in ("path", "exists", "sha256"):
            record[field_name] = relative[field_name]

    summary = rendered.get("summary")
    if isinstance(summary, dict):
        summary["self_contained_replay"] = True
    rendered["portable_replay_bundle"] = {
        "schema_version": REPLAY_BUNDLE_SCHEMA_VERSION,
        "input_count": len(relative_inputs),
        "notes": [
            "Replay input paths are relative to the directory containing artifact_lineage.json.",
            "The harness copied recorded-trace inputs before scoring so public receipts can be replayed without source paths.",
        ],
    }
    return rendered


def _replace_replay_flag(argv: list[str], flag: str, value: str) -> None:
    if flag not in argv:
        argv.extend([flag, value])
        return
    index = argv.index(flag)
    if index + 1 >= len(argv):
        argv.append(value)
        return
    argv[index + 1] = value


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


def _artifact_paths(paths: dict[str, Any], base_dir: Path, preserve_paths: bool) -> dict[str, str | None]:
    return {key: _display_path(value, base_dir, preserve_paths) if value is not None else None for key, value in paths.items()}


def _replay_reference(lineage_path: Path, lineage: dict[str, Any], base_dir: Path, preserve_paths: bool) -> dict[str, Any]:
    replay = lineage.get("replay") if isinstance(lineage.get("replay"), dict) else {}
    argv = replay.get("argv", [])
    display_argv = _display_argv(argv, base_dir, preserve_paths) if isinstance(argv, list) else []
    return {
        "lineage": _display_path(lineage_path, base_dir, preserve_paths),
        "argv": display_argv,
        "command": shlex.join(display_argv) if display_argv else "",
        "self_contained": replay.get("self_contained") is True,
    }


def _display_harness_manifest(manifest: dict[str, Any], base_dir: Path, preserve_paths: bool) -> dict[str, Any]:
    rendered = json.loads(json.dumps(manifest))
    scenario = rendered.get("scenario") if isinstance(rendered.get("scenario"), dict) else {}
    if isinstance(scenario.get("path"), str):
        scenario["path"] = _display_path(scenario["path"], base_dir, preserve_paths)

    outputs = rendered.get("outputs") if isinstance(rendered.get("outputs"), dict) else {}
    for field_name in ("run_dir", "manifest", "result"):
        if isinstance(outputs.get(field_name), str):
            outputs[field_name] = _display_path(outputs[field_name], base_dir, preserve_paths)

    sandbox = rendered.get("sandbox") if isinstance(rendered.get("sandbox"), dict) else {}
    rendered["sandbox"] = _display_sandbox(sandbox, base_dir, preserve_paths)
    return rendered


def _display_sandbox(sandbox: dict[str, Any], base_dir: Path, preserve_paths: bool) -> dict[str, Any]:
    rendered = dict(sandbox)
    for field_name in ("root", "home", "workspace", "events"):
        if isinstance(rendered.get(field_name), str):
            rendered[field_name] = _display_path(rendered[field_name], base_dir, preserve_paths)
    if isinstance(rendered.get("fake_secret_files"), list):
        rendered["fake_secret_files"] = [
            _display_path(path, base_dir, preserve_paths) if isinstance(path, str) else path
            for path in rendered["fake_secret_files"]
        ]
    return rendered


def _display_argv(argv: list[Any], base_dir: Path, preserve_paths: bool) -> list[str]:
    rendered: list[str] = []
    for item in argv:
        value = str(item)
        rendered.append(_display_path(value, base_dir, preserve_paths) if _looks_like_path(value) else value)
    return rendered


def _display_path(path: str | Path, base_dir: Path, preserve_paths: bool) -> str:
    resolved = Path(path).expanduser().resolve()
    if preserve_paths:
        return str(resolved)
    for root in (base_dir.expanduser().resolve(), Path.cwd().resolve()):
        try:
            relative = os.path.relpath(resolved, root)
        except ValueError:
            continue
        if relative == "." or not relative.startswith(".."):
            return Path(relative).as_posix()
    return f"<redacted:{resolved.name}>"


def _looks_like_path(value: str) -> bool:
    if value.startswith(("http://", "https://")):
        return False
    candidate = Path(value).expanduser()
    return candidate.is_absolute() or "/" in value or "\\" in value or value.startswith(".")


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


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _probe_check(check_id: str, passed: bool, details: dict[str, Any]) -> dict[str, Any]:
    return {"id": check_id, "passed": bool(passed), "details": details}


def _now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run or replay Flight Recorder harness artifacts")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command_name, help_text in (
        ("run-scenario", "Run one scenario through a harness runner"),
        ("run", "Alias for run-scenario"),
    ):
        run = subparsers.add_parser(command_name, help=help_text)
        run.add_argument("--manifest", help="Existing harness manifest JSON to execute")
        run.add_argument("--scenario", help="Scenario JSON/YAML path; required unless --manifest is used")
        run.add_argument("--out", help="Run output directory; required unless --manifest is used")
        run.add_argument("--runner", default="mock")
        run.add_argument("--provider", default="mock")
        run.add_argument("--model", default="hfr-mock")
        run.add_argument("--base-url")
        run.add_argument("--mock-response")
        run.add_argument("--force", action="store_true")
        run.add_argument("--relative-paths", action="store_true", help="Write artifact paths relative to the run or repo root")
    suite = subparsers.add_parser("run-suite", help="Run a scenario directory through the mock harness runner")
    suite.add_argument("--scenarios", required=True)
    suite.add_argument("--out", required=True)
    suite.add_argument("--pattern", default="*.json")
    suite.add_argument("--recursive", action="store_true")
    suite.add_argument("--runner", default="mock")
    suite.add_argument("--provider", default="mock")
    suite.add_argument("--model", default="hfr-mock")
    suite.add_argument("--base-url")
    suite.add_argument("--mock-response")
    suite.add_argument("--force", action="store_true")
    suite.add_argument("--fail-on-failed", action="store_true")
    suite.add_argument("--relative-paths", action="store_true", help="Write artifact paths relative to the suite or repo root")
    probe = subparsers.add_parser("probe-model", help="Write a no-network harness model probe receipt")
    probe.add_argument("--out", required=True)
    probe.add_argument("--runner", default="mock")
    probe.add_argument("--provider", default="mock")
    probe.add_argument("--model", default="hfr-mock")
    probe.add_argument("--base-url")
    probe.add_argument("--allow-network", action="store_true")
    probe.add_argument("--relative-paths", action="store_true", help="Write artifact paths relative to the probe output root")
    publish = subparsers.add_parser("publish-trace", help="Publish a recorded trace as harness artifacts")
    publish.add_argument("--scenario", required=True)
    publish.add_argument("--trace", required=True)
    publish.add_argument("--out", required=True)
    publish.add_argument("--format", default="auto")
    publish.add_argument("--runner", default="recorded_trace")
    publish.add_argument("--provider", default="recorded")
    publish.add_argument("--model")
    publish.add_argument("--base-url")
    publish.add_argument("--force", action="store_true")
    publish.add_argument("--relative-paths", action="store_true", help="Write artifact paths relative to the run or repo root")
    replay = subparsers.add_parser("replay-trace", help="Replay a run from artifact lineage")
    replay.add_argument("--lineage", required=True)
    replay.add_argument("--out", required=True)
    replay.add_argument("--format", default="auto")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.command in {"run", "run-scenario"}:
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
        result = run_scenario(manifest, preserve_paths=not args.relative_paths)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result.get("scorecard", {}).get("passed") is True else 1
    if args.command == "run-suite":
        result = run_suite(
            scenarios_dir=args.scenarios,
            out_dir=args.out,
            pattern=args.pattern,
            recursive=bool(args.recursive),
            provider=args.provider,
            model=args.model,
            runner=args.runner,
            base_url=args.base_url,
            mock_response=args.mock_response,
            force=bool(args.force),
            preserve_paths=not args.relative_paths,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        if result["error_count"]:
            return 1
        if args.fail_on_failed and result["failed"]:
            return 1
        return 0
    if args.command == "probe-model":
        result = probe_model(
            out_dir=args.out,
            provider=args.provider,
            model=args.model,
            runner=args.runner,
            base_url=args.base_url,
            allow_network=bool(args.allow_network),
            preserve_paths=not args.relative_paths,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["passed"] else 1
    if args.command == "publish-trace":
        result = publish_trace_run(
            scenario_path=args.scenario,
            trace_path=args.trace,
            out_dir=args.out,
            trace_format=args.format,
            runner=args.runner,
            provider=args.provider,
            model=args.model,
            base_url=args.base_url,
            force=bool(args.force),
            preserve_paths=not args.relative_paths,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result.get("scorecard", {}).get("passed") is True else 1
    if args.command == "replay-trace":
        result = replay_trace(args.lineage, args.out, trace_format=args.format)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if int(result.get("exit_code") or 0) == 0 and result.get("passed") is True else 1
    raise ValueError(f"unknown command {args.command!r}")


if __name__ == "__main__":
    raise SystemExit(main())
