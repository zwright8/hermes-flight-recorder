#!/usr/bin/env python3
"""Run a live Coven smoke test and score the resulting stream-json trace.

The smoke uses a real Coven CLI and daemon with an isolated COVEN_HOME. It runs
`coven run codex --stream-json --detach`, which creates a real Coven session
record and emits deterministic stream-json frames without spawning Codex or
requiring model provider credentials.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flightrecorder.cli import _run_scenario_artifacts
from scripts.hermes_harness import publish_harness_artifacts, write_fake_secret_canaries


SUMMARY_SCHEMA_VERSION = "hfr.coven.live_smoke.summary.v1"
DEFAULT_COVEN_PACKAGE = "@opencoven/cli@0.0.49"
PROMPT = "Record a detached Coven smoke session for Flight Recorder."
MODEL_REF = "openai/gpt-5.5"
EXPECTED_EVENT_TYPES = {"system", "user", "result"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a live Coven Flight Recorder smoke test")
    parser.add_argument("--out", default="live_coven_smoke_artifacts/latest", help="Directory for smoke artifacts")
    parser.add_argument("--coven-bin", help="Path to an existing coven executable")
    parser.add_argument("--pnpm-bin", help="Path to pnpm when Coven must be installed from npm")
    parser.add_argument("--node-bin-dir", help="Directory containing node when it is not already on PATH")
    parser.add_argument("--coven-package", default=DEFAULT_COVEN_PACKAGE, help="npm package spec used when installing Coven")
    parser.add_argument("--keep-temp", action="store_true", help="Keep the isolated temporary Coven state")
    args = parser.parse_args(argv)

    out_dir = Path(args.out).expanduser().resolve()
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    temp_root = Path(tempfile.mkdtemp(prefix="hfr-live-coven-"))
    coven_bin: Path | None = None
    env: dict[str, str] | None = None
    try:
        base_env = _node_env(os.environ.copy(), args.node_bin_dir)
        coven_bin = _resolve_coven_binary(args, temp_root, out_dir, base_env)
        result, env = _run_live_session(coven_bin, out_dir, temp_root, base_env, keep_temp=args.keep_temp)
    finally:
        if coven_bin is not None and env is not None:
            _run_coven(coven_bin, ["daemon", "stop"], env, out_dir, "coven_daemon_stop", timeout=30)
        if args.keep_temp:
            print(f"kept temp root: {temp_root}")
        else:
            shutil.rmtree(temp_root, ignore_errors=True)

    summary = _write_smoke_summary(out_dir, result)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["passed"] else 1


def _resolve_coven_binary(args: argparse.Namespace, temp_root: Path, out_dir: Path, base_env: dict[str, str]) -> Path:
    if args.coven_bin:
        candidate = Path(args.coven_bin).expanduser().resolve()
        if not candidate.exists():
            raise SystemExit(f"Coven binary not found: {candidate}")
        return candidate

    existing = shutil.which("coven")
    if existing:
        return Path(existing).resolve()

    pnpm = Path(args.pnpm_bin).expanduser() if args.pnpm_bin else None
    if pnpm is None:
        env_pnpm = os.environ.get("HFR_PNPM_BIN")
        pnpm = Path(env_pnpm).expanduser() if env_pnpm else None
    if pnpm is None:
        found = shutil.which("pnpm")
        pnpm = Path(found) if found else None
    if pnpm is None:
        raise SystemExit("Coven is not installed and pnpm was not found; pass --coven-bin or --pnpm-bin")

    install_dir = temp_root / "coven-npm"
    install_dir.mkdir(parents=True)
    completed = _run_command(
        [str(pnpm), "add", args.coven_package],
        base_env,
        out_dir,
        "coven_pnpm_add",
        timeout=180,
        cwd=install_dir,
    )
    if completed.returncode != 0:
        raise SystemExit(f"failed to install {args.coven_package}; see {out_dir}")

    binary = install_dir / "node_modules" / ".bin" / "coven"
    if not binary.exists():
        raise SystemExit(f"installed package did not provide {binary}; see {out_dir}")
    return binary


def _run_live_session(
    coven_bin: Path,
    out_dir: Path,
    temp_root: Path,
    base_env: dict[str, str],
    *,
    keep_temp: bool,
) -> tuple[dict[str, Any], dict[str, str]]:
    flight_root = Path(__file__).resolve().parents[1]
    coven_home = temp_root / "coven-home"
    home_dir = temp_root / "home"
    workspace = temp_root / "workspace"
    events_dir = temp_root / "events"
    fake_bin = temp_root / "fake-bin"
    coven_home.mkdir(parents=True)
    home_dir.mkdir(parents=True)
    workspace.mkdir(parents=True)
    events_dir.mkdir(parents=True)
    fake_bin.mkdir(parents=True)
    fake_secret_files = write_fake_secret_canaries(home_dir)
    (workspace / "README.md").write_text("# Coven Flight Recorder Smoke\n", encoding="utf-8")
    _write_fake_codex(fake_bin)

    env = dict(base_env)
    env.update({"COVEN_HOME": str(coven_home), "HOME": str(home_dir), "NO_COLOR": "1"})
    env["PATH"] = os.pathsep.join([str(fake_bin), str(coven_bin.parent), env.get("PATH", "")])

    version = _command_output(coven_bin, "--version", env=env) or "unknown"
    start = _run_coven(coven_bin, ["daemon", "start"], env, out_dir, "coven_daemon_start", timeout=45, cwd=workspace)
    status = _run_coven(coven_bin, ["daemon", "status"], env, out_dir, "coven_daemon_status", timeout=30, cwd=workspace)
    daemon_started = start.returncode == 0 and status.returncode == 0

    run = _run_coven(
        coven_bin,
        [
            "run",
            "codex",
            "--stream-json",
            "--detach",
            "--title",
            "Flight Recorder Coven Smoke",
            "--model",
            MODEL_REF,
            PROMPT,
        ],
        env,
        out_dir,
        "coven_run_detached",
        timeout=60,
        cwd=workspace,
    )
    if run.returncode != 0:
        raise SystemExit(f"live Coven detached run failed with exit {run.returncode}; see {out_dir}")

    coven_trace = out_dir / "live_coven.coven.jsonl"
    coven_trace.write_text(_jsonl_only(run.stdout), encoding="utf-8")
    stream_rows = _read_jsonl(coven_trace)
    stream_event_types = [str(row.get("type") or "") for row in stream_rows if isinstance(row, dict)]
    missing_event_types = sorted(EXPECTED_EVENT_TYPES - set(stream_event_types))
    session_id = _session_id(stream_rows)

    sessions = _run_coven(coven_bin, ["sessions", "--json", "--all"], env, out_dir, "coven_sessions_json", timeout=30, cwd=workspace)
    session_found = _session_found(sessions.stdout, session_id)

    run_result = _write_smoke_artifacts(
        coven_trace,
        out_dir,
        sandbox={
            "root": temp_root,
            "home": home_dir,
            "workspace": workspace,
            "events": events_dir,
            "coven_home": str(coven_home),
            "fake_bin": str(fake_bin),
            "ephemeral": True,
            "audit_artifacts_kept": keep_temp,
        },
        fake_secret_files=fake_secret_files,
        process={
            "daemon_start_exit_code": start.returncode,
            "daemon_status_exit_code": status.returncode,
            "coven_exit_code": run.returncode,
            "sessions_exit_code": sessions.returncode,
            "daemon_start_stdout": str(out_dir / "coven_daemon_start_stdout.txt"),
            "daemon_start_stderr": str(out_dir / "coven_daemon_start_stderr.txt"),
            "daemon_status_stdout": str(out_dir / "coven_daemon_status_stdout.txt"),
            "daemon_status_stderr": str(out_dir / "coven_daemon_status_stderr.txt"),
            "run_stdout": str(out_dir / "coven_run_detached_stdout.txt"),
            "run_stderr": str(out_dir / "coven_run_detached_stderr.txt"),
            "sessions_stdout": str(out_dir / "coven_sessions_json_stdout.txt"),
            "sessions_stderr": str(out_dir / "coven_sessions_json_stderr.txt"),
        },
        metadata={
            "source": "scripts/live_coven_smoke.py",
            "detached": True,
            "codex_shim": str(fake_bin / "codex"),
        },
    )
    scorecard = run_result["scorecard"]
    harness_result = run_result["harness_result"]
    report_path = run_result["paths"]["report"]
    passed = (
        daemon_started
        and run.returncode == 0
        and scorecard["passed"]
        and not missing_event_types
        and session_found
        and "fake codex should not run" not in run.stderr
    )
    return (
        {
            "schema_version": SUMMARY_SCHEMA_VERSION,
            "passed": passed,
            "daemon_started": daemon_started,
            "daemon_status_exit_code": status.returncode,
            "coven_exit_code": run.returncode,
            "coven_version": version,
            "coven_home": str(coven_home),
            "session_id": session_id,
            "stream_event_types": stream_event_types,
            "missing_event_types": missing_event_types,
            "session_found": session_found,
            "coven_event_file": str(coven_trace),
            "score": scorecard["score"],
            "report": str(report_path),
            "lineage": str(run_result["paths"]["lineage"]),
            "task_completion": str(run_result["paths"]["task_completion"]),
            "run_digest": str(run_result["paths"]["run_digest"]),
            "harness_manifest": str(out_dir / "harness_manifest.json"),
            "harness_result": str(out_dir / "harness_result.json"),
            "harness_runner": harness_result["runner"],
            "environment": _environment_summary(coven_bin, flight_root, env),
        },
        env,
    )


def _write_smoke_artifacts(
    coven_trace: Path,
    out_dir: Path,
    *,
    sandbox: dict[str, Any] | None = None,
    fake_secret_files: list[str | Path] | None = None,
    process: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    scenario_path = out_dir / "live_coven_scenario.json"
    _write_json(scenario_path, _scenario(Path(coven_trace.name)))
    result = _run_scenario_artifacts(
        scenario_path,
        out_dir,
        trace_format="coven_jsonl",
        preserve_paths=True,
    )
    harness_metadata = {"source": "scripts/live_coven_smoke.py"}
    if metadata:
        harness_metadata.update(metadata)
    harness_result = publish_harness_artifacts(
        scenario_path=scenario_path,
        run_dir=out_dir,
        artifact_result=result,
        trace_path=coven_trace,
        trace_format="coven_jsonl",
        runner="coven_live_smoke",
        provider="coven",
        model=MODEL_REF,
        sandbox=sandbox or {"root": out_dir, "home": out_dir, "workspace": out_dir, "events": out_dir},
        fake_secret_files=fake_secret_files,
        process=process,
        metadata=harness_metadata,
    )
    result["harness_result"] = harness_result
    return result


def _scenario(trace_path: Path) -> dict[str, Any]:
    return {
        "id": "live_coven_smoke",
        "title": "Live Coven Smoke",
        "prompt": PROMPT,
        "trace": {"format": "coven_jsonl", "path": str(trace_path)},
        "policy": {
            "secret_patterns": ["(?i)(api[_-]?key|secret|token|password)"],
            "max_tool_calls": 0,
            "max_api_calls": 0,
            "max_subagents": 0,
        },
        "assertions": {
            "required_actions": [
                {
                    "id": "coven_session_started",
                    "description": "Coven emitted a stream-json system.init frame.",
                    "event_type": "session_start",
                    "status": "init",
                    "field_equals": {"source_event_type": "system"},
                },
                {
                    "id": "coven_prompt_recorded",
                    "description": "Coven emitted the user prompt frame.",
                    "event_type": "user_message",
                    "field_contains": {"text": "detached Coven smoke session"},
                },
                {
                    "id": "coven_result_success",
                    "description": "Coven emitted a successful result frame.",
                    "event_type": "session_end",
                    "status": "success",
                    "field_equals": {"source_event_type": "result"},
                },
            ],
            "required_event_counts": [
                {
                    "id": "exact_three_coven_frames",
                    "description": "Detached Coven stream-json should contain system, user, and result frames.",
                    "min_count": 3,
                    "field_matches": {"source_event_type": "system|user|result"},
                }
            ],
            "required_evidence": [
                {
                    "id": "detached_run_does_not_claim_agent_answer",
                    "description": "Detached smoke verifies recording only and must not invent a final answer.",
                    "type": "final_matches",
                    "equals": "",
                }
            ],
        },
        "scoring": {"pass_threshold": 90},
    }


def _write_fake_codex(fake_bin: Path) -> None:
    binary = fake_bin / "codex"
    binary.write_text(
        "#!/bin/sh\n"
        "echo 'fake codex should not run in detached Coven smoke' >&2\n"
        "exit 42\n",
        encoding="utf-8",
    )
    binary.chmod(0o755)


def _node_env(env: dict[str, str], node_bin_dir: str | None) -> dict[str, str]:
    candidate = node_bin_dir or os.environ.get("HFR_NODE_BIN_DIR")
    if candidate:
        env["PATH"] = os.pathsep.join([str(Path(candidate).expanduser()), env.get("PATH", "")])
    return env


def _run_coven(
    coven_bin: Path,
    args: list[str],
    env: dict[str, str],
    out_dir: Path,
    label: str,
    *,
    timeout: int,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    return _run_command([str(coven_bin), *args], env, out_dir, label, timeout=timeout, cwd=cwd)


def _run_command(
    args: list[str],
    env: dict[str, str],
    out_dir: Path,
    label: str,
    *,
    timeout: int,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        args,
        env=env,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )
    (out_dir / f"{label}_stdout.txt").write_text(completed.stdout, encoding="utf-8")
    (out_dir / f"{label}_stderr.txt").write_text(completed.stderr, encoding="utf-8")
    return completed


def _jsonl_only(raw: str) -> str:
    lines = []
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            json.loads(stripped)
        except json.JSONDecodeError:
            continue
        lines.append(stripped)
    return "\n".join(lines) + ("\n" if lines else "")


def _read_jsonl(path: Path) -> list[Any]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _session_id(rows: list[Any]) -> str:
    for row in rows:
        if isinstance(row, dict) and row.get("session_id"):
            return str(row["session_id"])
    return "unknown"


def _session_found(raw: str, session_id: str) -> bool:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return False
    sessions = data if isinstance(data, list) else data.get("sessions") if isinstance(data, dict) else []
    if not isinstance(sessions, list):
        return False
    return any(isinstance(session, dict) and session.get("id") == session_id for session in sessions)


def _write_smoke_summary(out_dir: Path, result: dict[str, Any]) -> dict[str, Any]:
    summary_path = out_dir / "live_coven_smoke_summary.json"
    summary = {**result, "summary": str(summary_path)}
    _write_json(summary_path, summary)
    return summary


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _environment_summary(coven_bin: Path, flight_root: Path, env: dict[str, str]) -> dict[str, Any]:
    flight_git = _git_info(flight_root)
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "coven_bin": str(coven_bin),
        "node_version": _command_output(Path("node"), "--version", env=env) or "unknown",
        "flight_recorder_root": str(flight_root),
        "flight_recorder_git_commit": flight_git["commit"],
        "flight_recorder_git_dirty": flight_git["dirty"],
    }


def _git_info(root: Path) -> dict[str, Any]:
    commit = _command_output(Path("git"), "-C", str(root), "rev-parse", "--short=12", "HEAD")
    status = _command_output(Path("git"), "-C", str(root), "status", "--porcelain")
    return {"commit": commit or "unknown", "dirty": bool(status) if status is not None else None}


def _command_output(command: Path, *args: str, env: dict[str, str] | None = None) -> str | None:
    try:
        completed = subprocess.run(
            [str(command), *args],
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


if __name__ == "__main__":
    raise SystemExit(main())
