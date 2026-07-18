"""Shared runtime harness helpers for live and mock Flight Recorder scripts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import unquote

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flightrecorder.cli import _run_scenario_artifacts, _safe_run_id, cmd_replay
from flightrecorder.hermes_plugin import HOOKS
from flightrecorder.schema import load_scenario


PLUGIN_NAME = "flight_recorder_live"
HARNESS_MANIFEST_SCHEMA_VERSION = "hfr.harness_run_manifest.v1"
HARNESS_RUN_RESULT_SCHEMA_VERSION = "hfr.harness_run_result.v1"
HARNESS_REPLAY_RESULT_SCHEMA_VERSION = "hfr.harness_replay_result.v1"
HARNESS_SUITE_SUMMARY_SCHEMA_VERSION = "hfr.harness_suite_summary.v1"
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


def default_hermes_root(anchor: str | Path) -> str:
    return str(Path(anchor).resolve().parents[2] / "upstream-hermes-agent")


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
    """Build a common harness-run manifest for a single scenario."""
    scenario_path = Path(scenario_path).expanduser().resolve()
    scenario = load_scenario(scenario_path)
    run_dir = Path(out_dir).expanduser().resolve()
    sandbox_root = run_dir / "sandbox"
    effective_tool_policy = _effective_tool_policy(scenario, tool_policy)
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
            "fake_secret_canaries": [
                {"name": name, "sha256": _sha256(value)}
                for name, value in sorted(DEFAULT_FAKE_SECRET_CANARIES.items())
            ],
            "ephemeral": True,
            "audit_artifacts_kept": True,
        },
        "tool_policy": effective_tool_policy,
        "mock": {"response": mock_response},
        "force": force,
    }


def validate_harness_manifest(manifest: dict[str, Any]) -> None:
    """Raise ValueError when a harness manifest is missing required fields."""
    if manifest.get("schema_version") != HARNESS_MANIFEST_SCHEMA_VERSION:
        raise ValueError(f"manifest schema_version must be {HARNESS_MANIFEST_SCHEMA_VERSION!r}")
    for field in ("runner", "provider", "model", "scenario", "outputs", "sandbox", "tool_policy"):
        if field not in manifest:
            raise ValueError(f"manifest missing required field: {field}")
    if not isinstance(manifest["model"], dict) or not manifest["model"].get("id"):
        raise ValueError("manifest.model.id is required")
    if not Path(str(manifest["scenario"].get("path", ""))).exists():
        raise ValueError(f"manifest scenario path does not exist: {manifest['scenario'].get('path')}")
    for field in ("run_dir", "manifest", "result"):
        if field not in manifest["outputs"]:
            raise ValueError(f"manifest.outputs.{field} is required")
    for field in ("root", "home", "workspace", "events"):
        if field not in manifest["sandbox"]:
            raise ValueError(f"manifest.sandbox.{field} is required")


def run_scenario(manifest: dict[str, Any] | str | Path) -> dict[str, Any]:
    """Run one scenario through the configured harness runner."""
    resolved = _load_manifest(manifest)
    validate_harness_manifest(resolved)
    runner = str(resolved.get("runner") or "")
    if runner == "mock":
        return _run_mock_scenario(resolved)
    raise ValueError(f"unsupported harness runner {runner!r}; only 'mock' is available through this facade")


def run_suite(
    *,
    scenario_paths: list[str | Path],
    out_dir: str | Path,
    provider: str = "mock",
    model: str = "hfr-mock",
    runner: str = "mock",
    base_url: str | None = None,
    tool_policy: dict[str, Any] | None = None,
    mock_response: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Run a list of scenarios through the common harness facade."""
    suite_dir = Path(out_dir).expanduser().resolve()
    if force and suite_dir.exists():
        shutil.rmtree(suite_dir)
    suite_dir.mkdir(parents=True, exist_ok=True)
    runs: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for scenario_path in scenario_paths:
        try:
            scenario = load_scenario(scenario_path)
            run_id = _safe_run_id(str(scenario["id"]))
            manifest = build_harness_manifest(
                scenario_path=scenario_path,
                out_dir=suite_dir / run_id,
                provider=provider,
                model=model,
                runner=runner,
                base_url=base_url,
                tool_policy=tool_policy,
                mock_response=mock_response,
                force=force,
            )
            runs.append(run_scenario(manifest))
        except Exception as exc:  # pragma: no cover - defensive suite bookkeeping
            errors.append({"scenario_path": str(scenario_path), "error": str(exc)})
    summary = {
        "schema_version": HARNESS_SUITE_SUMMARY_SCHEMA_VERSION,
        "created_at": _now_iso(),
        "runner": runner,
        "provider": provider,
        "model": model,
        "out_dir": str(suite_dir),
        "total": len(runs) + len(errors),
        "passed": sum(1 for run in runs if run.get("scorecard", {}).get("passed") is True),
        "failed": sum(1 for run in runs if run.get("scorecard", {}).get("passed") is False),
        "error_count": len(errors),
        "runs": runs,
        "errors": errors,
    }
    _write_json(suite_dir / "harness_suite_summary.json", summary)
    return summary


def probe_model(
    *,
    provider: str = "mock",
    model: str = "hfr-mock",
    base_url: str | None = None,
    tool_policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a harness compatibility probe without requiring live network calls."""
    mock = provider == "mock" or (base_url or "").startswith("mock://")
    return {
        "schema_version": HARNESS_MODEL_PROBE_SCHEMA_VERSION,
        "created_at": _now_iso(),
        "provider": provider,
        "model": model,
        "base_url": base_url,
        "tool_policy": tool_policy or json.loads(json.dumps(DEFAULT_TOOL_POLICY)),
        "capabilities": {
            "chat_completions": True,
            "tool_calls": True,
            "trace_capture": True,
            "mock_endpoint": mock,
        },
        "status": "ok" if mock else "manifest_only",
        "notes": [
            "Mock probes are deterministic and do not contact a model endpoint.",
            "Non-mock probes record compatibility metadata only; live endpoint checks belong in smoke/eval scripts.",
        ],
    }


def replay_trace(
    lineage_path: str | Path,
    out_dir: str | Path,
    *,
    base_dir: str | Path | None = None,
    trace_format: str = "auto",
    allow_non_self_contained: bool = False,
    preserve_paths: bool = False,
) -> dict[str, Any]:
    """Replay a run from artifact lineage and return a compact replay summary."""
    out_dir = Path(out_dir).expanduser().resolve()
    args = argparse.Namespace(
        lineage=str(Path(lineage_path).expanduser().resolve()),
        out=str(out_dir),
        base_dir=str(Path(base_dir).expanduser().resolve()) if base_dir is not None else None,
        format=trace_format,
        write_sensitive_trace=False,
        preserve_paths=preserve_paths,
        allow_non_self_contained=allow_non_self_contained,
        fail_on_score=False,
    )
    exit_code = cmd_replay(args)
    scorecard_path = out_dir / "scorecard.json"
    scorecard = json.loads(scorecard_path.read_text(encoding="utf-8")) if scorecard_path.exists() else None
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
    """Publish common harness manifest/result files for an already-scored trace."""
    run_dir = Path(run_dir).expanduser().resolve()
    scenario_path = Path(scenario_path).expanduser().resolve()
    trace_path = _resolve_output_path(trace_path, run_dir)
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
        "trace": {"format": trace_format, "path": str(trace_path)},
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


def require_hermes_checkout(root: Path) -> dict[str, Any]:
    if not (root / "pyproject.toml").exists():
        raise SystemExit(f"Hermes checkout not found: {root}")
    if shutil.which("uv") is None:
        raise SystemExit("uv is required to run the Hermes checkout")
    status = hermes_git_info(root)
    if int(status.get("behind") or 0) > 0:
        upstream = status.get("upstream") or "upstream"
        print(
            f"warning: Hermes checkout {root} is {status['behind']} commit(s) behind {upstream}; "
            "pass --hermes-root pointing at a current mainline checkout for release evidence.",
            file=sys.stderr,
        )
    return status


def write_observer_plugin(plugin_dir: Path, *, description: str) -> None:
    plugin_dir.mkdir(parents=True, exist_ok=True)
    (plugin_dir / "plugin.yaml").write_text(
        "\n".join(
            [
                f"name: {PLUGIN_NAME}",
                'version: "0.1"',
                "kind: standalone",
                f"description: {_yaml_string(description)}",
                "provides_hooks:",
                *[f"  - {hook}" for hook in HOOKS],
                "",
            ]
        ),
        encoding="utf-8",
    )
    (plugin_dir / "__init__.py").write_text(
        "from flightrecorder.hermes_plugin import register as register_flight_recorder\n\n"
        "def register(ctx):\n"
        "    return register_flight_recorder(ctx)\n",
        encoding="utf-8",
    )


def write_runtime_config(path: Path, *, provider: str, model: str, base_url: str, api_key: str, max_turns: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "model:",
                f"  provider: {_yaml_string(provider)}",
                f"  default: {_yaml_string(model)}",
                f"  base_url: {_yaml_string(base_url)}",
                f"  api_key: {_yaml_string(api_key)}",
                "  api_mode: chat_completions",
                "plugins:",
                "  enabled:",
                f"    - {PLUGIN_NAME}",
                "agent:",
                f"  max_turns: {int(max_turns)}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def hermes_chat_command(
    *,
    hermes_root: Path,
    prompt: str,
    provider: str,
    model: str,
    max_turns: int,
    source: str,
    toolsets: str = "",
    yolo: bool = False,
) -> list[str]:
    cmd = [
        "uv",
        "run",
        "--project",
        str(hermes_root),
        "hermes",
        "chat",
        "--query",
        prompt,
        "--provider",
        provider,
        "--model",
        model,
        "--quiet",
        "--ignore-rules",
        "--max-turns",
        str(max_turns),
        "--source",
        source,
    ]
    if toolsets:
        cmd.extend(["--toolsets", toolsets])
    if yolo:
        cmd.append("--yolo")
    return cmd


def hermes_run_env(
    *,
    flight_root: Path,
    hermes_root: Path,
    hermes_home: Path,
    home_dir: Path,
    events_dir: Path,
    timeout: int,
    max_field_chars: int,
) -> dict[str, str]:
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    env.update(
        {
            "HOME": str(home_dir),
            "HERMES_HOME": str(hermes_home),
            "HERMES_FLIGHT_RECORDER_OUTPUT_DIR": str(events_dir),
            "HERMES_FLIGHT_RECORDER_MAX_FIELD_CHARS": str(max_field_chars),
            "HERMES_API_TIMEOUT": str(timeout),
            "HERMES_STREAM_READ_TIMEOUT": str(timeout),
            "HERMES_STREAM_RETRIES": "0",
            "HERMES_DISABLE_UPDATE_CHECK": "1",
            "PYTHONPATH": f"{flight_root}:{hermes_root}:{existing_pythonpath}",
        }
    )
    return env


def run_hermes_chat(
    *,
    args: Any,
    hermes_root: Path,
    flight_root: Path,
    hermes_home: Path,
    home_dir: Path,
    events_dir: Path,
    workspace: Path,
    prompt: str,
    source: str,
    max_field_chars: int = 40000,
) -> subprocess.CompletedProcess[str]:
    env = hermes_run_env(
        flight_root=flight_root,
        hermes_root=hermes_root,
        hermes_home=hermes_home,
        home_dir=home_dir,
        events_dir=events_dir,
        timeout=int(args.timeout),
        max_field_chars=max_field_chars,
    )
    cmd = hermes_chat_command(
        hermes_root=hermes_root,
        prompt=prompt,
        provider=str(args.provider),
        model=str(args.model),
        max_turns=int(args.max_turns),
        source=source,
        toolsets=str(getattr(args, "toolsets", "") or ""),
        yolo=bool(getattr(args, "yolo", False)),
    )
    return subprocess.run(
        cmd,
        cwd=workspace,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=int(args.timeout) + 30,
        check=False,
    )


def model_probe_payload(path: str, model: str, *, version: str) -> dict[str, Any] | None:
    normalized = path.split("?", 1)[0].rstrip("/") or "/"
    if normalized in {"/api/v1/models", "/v1/models", "/models"}:
        return {"object": "list", "data": [{"id": model, "object": "model"}]}
    if normalized in {"/v1/props", "/props", "/api/v1/props"}:
        return {"model": model, "context_length": 32768}
    if normalized == "/api/tags":
        return {"models": [{"name": model, "model": model}]}
    if normalized == "/api/show":
        return model_details(model)
    if normalized == "/version":
        return {"version": version}
    for prefix in ("/api/v1/models/", "/v1/models/", "/models/"):
        if normalized.startswith(prefix):
            requested = unquote(normalized[len(prefix) :])
            if not requested or requested == model:
                return {"id": model, "object": "model"}
    return None


def model_details(model: str) -> dict[str, Any]:
    return {
        "id": model,
        "model": model,
        "object": "model",
        "details": {"family": "mock"},
        "model_info": {},
        "capabilities": ["completion", "tools"],
        "context_length": 32768,
    }


def send_json(handler: Any, payload: dict[str, Any], *, status: int = 200) -> None:
    raw = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("content-type", "application/json")
    handler.send_header("connection", "close")
    handler.send_header("content-length", str(len(raw)))
    handler.end_headers()
    handler.wfile.write(raw)


def send_stream(handler: Any, chunks: list[dict[str, Any]]) -> None:
    text = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
    text += "data: [DONE]\n\n"
    raw = text.encode("utf-8")
    handler.send_response(200)
    handler.send_header("content-type", "text/event-stream")
    handler.send_header("cache-control", "no-cache")
    handler.send_header("connection", "close")
    handler.send_header("content-length", str(len(raw)))
    handler.end_headers()
    handler.wfile.write(raw)


def environment_summary(hermes_root: Path, flight_root: Path) -> dict[str, Any]:
    hermes_git = hermes_git_info(hermes_root)
    flight_git = hermes_git_info(flight_root)
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "hermes_root": str(hermes_root),
        "hermes_git_commit": hermes_git["commit"],
        "hermes_git_dirty": hermes_git["dirty"],
        "hermes_git_branch": hermes_git["branch"],
        "hermes_git_upstream": hermes_git["upstream"],
        "hermes_git_ahead": hermes_git["ahead"],
        "hermes_git_behind": hermes_git["behind"],
        "flight_recorder_root": str(flight_root),
        "flight_recorder_git_commit": flight_git["commit"],
        "flight_recorder_git_dirty": flight_git["dirty"],
    }


def _run_mock_scenario(manifest: dict[str, Any]) -> dict[str, Any]:
    run_dir = Path(manifest["outputs"]["run_dir"])
    if bool(manifest.get("force")) and run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    sandbox = manifest["sandbox"]
    sandbox_root = Path(sandbox["root"])
    home_dir = Path(sandbox["home"])
    workspace = Path(sandbox["workspace"])
    events_dir = Path(sandbox["events"])
    for path in (sandbox_root, home_dir, workspace, events_dir):
        path.mkdir(parents=True, exist_ok=True)

    scenario_path = Path(manifest["scenario"]["path"])
    scenario = load_scenario(scenario_path)
    _write_fake_secret_canaries(home_dir)
    _write_json(
        workspace / "workspace_manifest.json",
        {
            "schema_version": "hfr.harness_workspace.v1",
            "scenario_id": scenario["id"],
            "runner": manifest["runner"],
            "provider": manifest["provider"],
            "isolated": True,
        },
    )
    _write_json(run_dir / "harness_manifest.json", manifest)

    rows = _mock_observer_rows(scenario, manifest)
    events_trace = events_dir / "mock_observer.jsonl"
    trace_path = run_dir / "mock_observer.jsonl"
    _write_jsonl(events_trace, rows)
    shutil.copyfile(events_trace, trace_path)

    artifact_result = _run_scenario_artifacts(
        scenario_path,
        run_dir,
        trace_override=trace_path,
        trace_format="observer_jsonl",
        preserve_paths=True,
    )
    scorecard = artifact_result["scorecard"]
    result = {
        "schema_version": HARNESS_RUN_RESULT_SCHEMA_VERSION,
        "created_at": _now_iso(),
        "runner": manifest["runner"],
        "provider": manifest["provider"],
        "model": manifest["model"],
        "scenario_id": scenario["id"],
        "sandbox": {
            **sandbox,
            "fake_secret_files": [str(home_dir / ".hermes" / ".env")],
        },
        "tool_policy": manifest["tool_policy"],
        "trace": {"format": "observer_jsonl", "path": str(trace_path)},
        "scorecard": {
            "path": str(artifact_result["paths"]["scorecard"]),
            "passed": bool(scorecard["passed"]),
            "score": scorecard["score"],
            "critical_failures": scorecard.get("critical_failures", []),
        },
        "artifacts": _artifact_paths(artifact_result["paths"]),
        "replay": _replay_reference(artifact_result["paths"]["lineage"], artifact_result["lineage"]),
    }
    _write_json(run_dir / "harness_result.json", result)
    return result


def _mock_observer_rows(scenario: dict[str, Any], manifest: dict[str, Any]) -> list[dict[str, Any]]:
    session_id = f"hfr-harness-{_safe_run_id(str(scenario['id']))}"
    model = str(manifest.get("model", {}).get("id") or "hfr-mock")
    rows: list[dict[str, Any]] = [
        {
            "hook": "pre_llm_call",
            "payload": {
                "session_id": session_id,
                "model": model,
                "user_message": str(scenario.get("prompt") or ""),
            },
        }
    ]
    seen: set[str] = set()
    for item in _mock_event_assertions(scenario):
        key = _assertion_event_key(item)
        if key in seen:
            continue
        seen.add(key)
        rows.extend(_observer_rows_for_assertion(item, session_id=session_id, model=model))
    rows.append(
        {
            "hook": "post_llm_call",
            "payload": {
                "session_id": session_id,
                "model": model,
                "assistant_response": _mock_final_answer(scenario, manifest),
            },
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
    for sequence in assertions.get("required_action_sequences") or []:
        if isinstance(sequence, dict):
            items.extend(step for step in sequence.get("steps") or [] if isinstance(step, dict))
    for count in assertions.get("required_event_counts") or []:
        if not isinstance(count, dict):
            continue
        minimum = int(count.get("exact_count", count.get("min_count", 0)) or 0)
        for _index in range(max(0, minimum)):
            items.append(count)
    return items


def _observer_rows_for_assertion(item: dict[str, Any], *, session_id: str, model: str) -> list[dict[str, Any]]:
    event_type = str(item.get("event_type") or "tool_result")
    if event_type not in {"tool_call", "tool_result"}:
        return []
    tool_name = str(item.get("tool_name") or "mock_tool")
    status = str(item.get("status") or ("requested" if event_type == "tool_call" else "ok"))
    args: dict[str, Any] = {}
    result: dict[str, Any] = {}
    text = _text_for_assertion(item)
    _apply_assertion_fields(item, args=args, result=result, text=text)
    tool_call_id = f"call-{_safe_run_id(tool_name)}-{_sha256(json.dumps(item, sort_keys=True))[:8]}"
    if event_type == "tool_call":
        if text and not args:
            args["evidence"] = text
        return [
            {
                "hook": "pre_tool_call",
                "payload": {
                    "session_id": session_id,
                    "model": model,
                    "tool_call_id": tool_call_id,
                    "tool_name": tool_name,
                    "args": args,
                },
            }
        ]
    if text and not result:
        result["content"] = text
    return [
        {
            "hook": "post_tool_call",
            "payload": {
                "session_id": session_id,
                "model": model,
                "tool_call_id": tool_call_id,
                "tool_name": tool_name,
                "status": status,
                "result": result,
            },
        }
    ]


def _assertion_event_key(item: dict[str, Any]) -> str:
    return json.dumps(
        {
            "event_type": item.get("event_type") or "tool_result",
            "tool_name": item.get("tool_name") or "mock_tool",
            "status": item.get("status") or ("requested" if item.get("event_type") == "tool_call" else "ok"),
            "where": item.get("where"),
            "field": item.get("field"),
            "contains": item.get("contains"),
            "matches": item.get("matches"),
            "pattern": item.get("pattern"),
        },
        sort_keys=True,
    )


def _apply_assertion_fields(item: dict[str, Any], *, args: dict[str, Any], result: dict[str, Any], text: str) -> None:
    where = item.get("where")
    if isinstance(where, dict):
        for field, constraint in where.items():
            value = _value_for_constraint(constraint)
            if value is _SKIP:
                continue
            _assign_event_field(str(field), value, args=args, result=result)
    field = item.get("field")
    if field is not None:
        _assign_event_field(str(field), text or _value_for_constraint(item), args=args, result=result)


def _assign_event_field(field: str, value: Any, *, args: dict[str, Any], result: dict[str, Any]) -> None:
    if field == "args":
        args["evidence"] = value
    elif field == "result":
        result["evidence"] = value
    elif field.startswith("args."):
        _set_nested(args, field.removeprefix("args."), value)
    elif field.startswith("result."):
        _set_nested(result, field.removeprefix("result."), value)
    elif field in {"all", "text"}:
        result["content"] = value
    else:
        _set_nested(result, field, value)


def _text_for_assertion(item: dict[str, Any]) -> str:
    value = _value_for_constraint(item)
    if value is _SKIP:
        return ""
    return str(value)


_SKIP = object()


def _value_for_constraint(constraint: Any) -> Any:
    if isinstance(constraint, dict):
        if "equals" in constraint:
            return constraint["equals"]
        if "contains" in constraint:
            return f"mock evidence: {constraint['contains']}"
        if "matches" in constraint:
            return _example_for_regex(str(constraint["matches"]))
        if constraint.get("present") is True:
            return "mock-present"
        if constraint.get("present") is False:
            return _SKIP
    if isinstance(constraint, dict):
        for key in ("contains", "matches", "pattern", "equals"):
            if key in constraint:
                return _value_for_constraint({key: constraint[key]})
    if isinstance(constraint, str):
        return constraint
    if constraint is None or isinstance(constraint, (int, float, bool)):
        return constraint
    return "mock-evidence"


def _example_for_regex(pattern: str) -> str:
    text = pattern
    if text.startswith("^"):
        text = text[1:]
    if text.endswith("$"):
        text = text[:-1]
    text = text.replace(".*", "mock")
    text = text.replace(".+", "mock")
    text = text.replace("\\d+", "123")
    text = text.replace("\\w+", "mock")
    text = text.replace("\\.", ".")
    text = text.replace("\\-", "-")
    if "[" in text or "(" in text or "{" in text:
        return "mock-match"
    return text or "mock-match"


def _mock_final_answer(scenario: dict[str, Any], manifest: dict[str, Any]) -> str:
    configured = (manifest.get("mock") or {}).get("response")
    if configured:
        return str(configured)
    final_contains = scenario.get("assertions", {}).get("final_contains") or []
    required_text = " ".join(str(item) for item in final_contains if item)
    if required_text:
        return f"Mock completed {scenario['id']} with auditable evidence. {required_text}"
    return f"Mock completed {scenario['id']} with auditable evidence."


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


def _write_fake_secret_canaries(home_dir: Path) -> None:
    secret_path = home_dir / ".hermes" / ".env"
    secret_path.parent.mkdir(parents=True, exist_ok=True)
    secret_path.write_text(
        "\n".join(f"{name}={value}" for name, value in sorted(DEFAULT_FAKE_SECRET_CANARIES.items())) + "\n",
        encoding="utf-8",
    )


def write_fake_secret_canaries(home_dir: str | Path) -> list[str]:
    """Write deterministic fake-secret canaries and return the created files."""
    resolved = Path(home_dir).expanduser().resolve()
    _write_fake_secret_canaries(resolved)
    return [str(resolved / ".hermes" / ".env")]


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


def _load_manifest(manifest: dict[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(manifest, dict):
        return dict(manifest)
    return json.loads(Path(manifest).read_text(encoding="utf-8"))


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
    if not sandbox.get("fake_secret_canaries"):
        sandbox["fake_secret_canaries"] = defaults["fake_secret_canaries"]
    return sandbox


def _resolve_output_path(path: str | Path, base_dir: Path) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = base_dir / resolved
    return resolved.resolve()


def _set_nested(target: dict[str, Any], dotted_path: str, value: Any) -> None:
    parts = [part for part in dotted_path.split(".") if part]
    if not parts:
        return
    cursor = target
    for part in parts[:-1]:
        next_value = cursor.get(part)
        if not isinstance(next_value, dict):
            next_value = {}
            cursor[part] = next_value
        cursor = next_value
    cursor[parts[-1]] = value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def hermes_git_info(root: Path) -> dict[str, Any]:
    commit = _git_output(root, "rev-parse", "--short=12", "HEAD")
    status = _git_output(root, "status", "--porcelain")
    branch = _git_output(root, "rev-parse", "--abbrev-ref", "HEAD") or "unknown"
    upstream = _git_output(root, "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}") or ""
    ahead = behind = 0
    if upstream:
        counts = _git_output(root, "rev-list", "--left-right", "--count", f"HEAD...{upstream}") or ""
        parts = counts.split()
        if len(parts) == 2:
            try:
                ahead, behind = int(parts[0]), int(parts[1])
            except ValueError:
                ahead = behind = 0
    return {
        "commit": commit or "unknown",
        "dirty": bool(status) if status is not None else None,
        "branch": branch,
        "upstream": upstream,
        "ahead": ahead,
        "behind": behind,
    }


def _git_output(root: Path, *args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), *args],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def _yaml_string(value: str) -> str:
    return json.dumps(str(value))


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run or inspect Flight Recorder harness manifests")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run-scenario", help="Run one scenario through a harness runner")
    run.add_argument("--scenario", required=True)
    run.add_argument("--out", required=True)
    run.add_argument("--runner", default="mock")
    run.add_argument("--provider", default="mock")
    run.add_argument("--model", default="hfr-mock")
    run.add_argument("--base-url")
    run.add_argument("--mock-response")
    run.add_argument("--force", action="store_true")

    suite = subparsers.add_parser("run-suite", help="Run multiple scenarios through a harness runner")
    suite.add_argument("--scenario", action="append", required=True)
    suite.add_argument("--out", required=True)
    suite.add_argument("--runner", default="mock")
    suite.add_argument("--provider", default="mock")
    suite.add_argument("--model", default="hfr-mock")
    suite.add_argument("--base-url")
    suite.add_argument("--mock-response")
    suite.add_argument("--force", action="store_true")

    probe = subparsers.add_parser("probe-model", help="Write model compatibility metadata for harness use")
    probe.add_argument("--provider", default="mock")
    probe.add_argument("--model", default="hfr-mock")
    probe.add_argument("--base-url")
    probe.add_argument("--out")

    replay = subparsers.add_parser("replay-trace", help="Replay a run from artifact lineage")
    replay.add_argument("--lineage", required=True)
    replay.add_argument("--out", required=True)
    replay.add_argument("--base-dir")
    replay.add_argument("--format", default="auto")
    replay.add_argument("--allow-non-self-contained", action="store_true")
    replay.add_argument("--preserve-paths", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    exit_code = 0
    if args.command == "run-scenario":
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
        exit_code = 0 if result.get("scorecard", {}).get("passed") is True else 1
    elif args.command == "run-suite":
        result = run_suite(
            scenario_paths=args.scenario,
            out_dir=args.out,
            provider=args.provider,
            model=args.model,
            runner=args.runner,
            base_url=args.base_url,
            mock_response=args.mock_response,
            force=args.force,
        )
        exit_code = 0 if int(result.get("failed") or 0) == 0 and int(result.get("error_count") or 0) == 0 else 1
    elif args.command == "probe-model":
        result = probe_model(provider=args.provider, model=args.model, base_url=args.base_url)
        if args.out:
            _write_json(Path(args.out), result)
    elif args.command == "replay-trace":
        result = replay_trace(
            args.lineage,
            args.out,
            base_dir=args.base_dir,
            trace_format=args.format,
            allow_non_self_contained=args.allow_non_self_contained,
            preserve_paths=args.preserve_paths,
        )
        exit_code = 0 if int(result.get("exit_code") or 0) == 0 and result.get("passed") is True else 1
    else:  # pragma: no cover - argparse prevents this branch
        raise ValueError(f"unknown harness command {args.command!r}")
    print(json.dumps(result, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
