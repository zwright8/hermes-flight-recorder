#!/usr/bin/env python3
"""Preflight local serving runtime dependencies and artifacts without launching a server."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "hfr.serving_runtime_preflight.v1"
DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
DEFAULT_DEPENDENCIES = ("torch", "transformers", "peft", "accelerate")
ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors", "tokenizer_config.json", "chat_template.jinja")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--model-cache", type=Path, help="Local model cache path; defaults to Hugging Face hub cache for --model.")
    parser.add_argument("--adapter", action="append", default=[], metavar="ARM=PATH", help="Adapter directory to preflight. Repeat for trace_only, flightrecorder, etc.")
    parser.add_argument("--baseline-arm", default="baseline", help="Baseline arm name to include in generated command refs.")
    parser.add_argument("--server-script", type=Path, default=Path("scripts/serve_transformers_openai.py"))
    parser.add_argument(
        "--runtime-python",
        type=Path,
        default=Path(sys.executable),
        help="Python interpreter that will run managed serving and eval commands.",
    )
    parser.add_argument("--required-dependency", action="append", default=[], help="Override required runtime dependency list. Repeatable.")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path, help="Optional Markdown report path.")
    parser.add_argument("--allow-blocked", action="store_true", help="Write artifacts and exit zero even when readiness is blocked.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    artifact = build_preflight(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    _write_json(args.out, artifact)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(render_report(artifact), encoding="utf-8")
    print(json.dumps({"passed": artifact["passed"], "readiness": artifact["readiness"], "blocked_checks": artifact["blocked_checks"], "out": str(args.out)}, indent=2))
    return 0 if artifact["passed"] or args.allow_blocked else 1


def build_preflight(args: argparse.Namespace) -> dict[str, Any]:
    model_cache = (args.model_cache or _default_model_cache(args.model)).expanduser().resolve()
    server_script = args.server_script.expanduser().resolve()
    runtime = _runtime_record(args.runtime_python)
    dependencies = tuple(args.required_dependency or DEFAULT_DEPENDENCIES)
    dependency_records = {name: _dependency_record(name, runtime_python=Path(runtime["path"])) for name in dependencies}
    adapters = _adapter_records(_parse_arm_paths(args.adapter))
    server_record = _server_script_record(server_script)
    model_cache_record = {"path": str(model_cache), "exists": model_cache.exists(), "bytes": _path_size(model_cache) if model_cache.exists() else 0}
    command_refs = _command_refs(args=args, server_script=server_script, runtime_python=Path(runtime["path"]), adapters=adapters)
    blocked_checks = _blocked_checks(
        dependency_records=dependency_records,
        model_cache=model_cache_record,
        adapters=adapters,
        server_script=server_record,
        runtime=runtime,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "passed": not blocked_checks,
        "readiness": "ready" if not blocked_checks else "blocked",
        "blocked_checks": blocked_checks,
        "model": args.model,
        "runtime": runtime,
        "model_cache": model_cache_record,
        "server_script": server_record,
        "dependencies": dependency_records,
        "adapters": adapters,
        "command_refs": command_refs,
        "environment": {
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "serving_runtime_python": runtime["path"],
            "uv_executable": shutil.which("uv"),
            "vllm_executable": shutil.which("vllm"),
            "sglang_executable": shutil.which("sglang"),
        },
    }


def render_report(artifact: dict[str, Any]) -> str:
    lines = [
        "# Serving Runtime Preflight",
        "",
        f"- Passed: {artifact['passed']}",
        f"- Readiness: `{artifact['readiness']}`",
        f"- Model: `{artifact['model']}`",
        f"- Runtime Python: `{artifact['runtime']['path']}` ({artifact['runtime']['exists']})",
        f"- Model cache: `{artifact['model_cache']['path']}` ({artifact['model_cache']['exists']})",
        f"- Blocked checks: {', '.join(artifact['blocked_checks']) if artifact['blocked_checks'] else 'none'}",
        "",
        "## Dependencies",
        "",
        "| Dependency | Available | Origin |",
        "| --- | ---: | --- |",
    ]
    for name, record in artifact["dependencies"].items():
        lines.append(f"| `{name}` | {record['available']} | `{record.get('origin') or ''}` |")
    lines.extend(["", "## Adapters", "", "| Arm | Exists | Adapter Config | Adapter Model | Path |", "| --- | ---: | ---: | ---: | --- |"])
    for arm, record in artifact["adapters"].items():
        files = record.get("files") or {}
        lines.append(
            "| {arm} | {exists} | {config} | {model} | `{path}` |".format(
                arm=arm,
                exists=record["exists"],
                config=bool(files.get("adapter_config.json", {}).get("exists")),
                model=bool(files.get("adapter_model.safetensors", {}).get("exists")),
                path=record["path"],
            )
        )
    lines.extend(["", "## Command Refs", ""])
    for ref in artifact["command_refs"]:
        lines.append(f"- `{ref['arm']}` server: `{ref['server_command']}`")
        lines.append(f"- `{ref['arm']}` managed eval: `{ref['managed_eval_command']}`")
    return "\n".join(lines) + "\n"


def _runtime_record(path: Path) -> dict[str, Any]:
    runtime_path = _absolute_without_resolving_symlinks(path)
    exists = runtime_path.exists() and runtime_path.is_file()
    record: dict[str, Any] = {"path": str(runtime_path), "exists": exists}
    if not exists:
        record["python_version"] = ""
        return record
    command = [str(runtime_path), "-c", "import platform; print(platform.python_version())"]
    completed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    record["python_version"] = completed.stdout.strip()
    if completed.returncode != 0:
        record["error"] = completed.stderr.strip()
    return record


def _absolute_without_resolving_symlinks(path: Path) -> Path:
    expanded = path.expanduser()
    if expanded.is_absolute():
        return expanded
    return Path.cwd() / expanded


def _dependency_record(name: str, *, runtime_python: Path) -> dict[str, Any]:
    if not runtime_python.exists():
        return {"available": False, "origin": "", "runtime_python": str(runtime_python)}
    code = (
        "import importlib.util, json\n"
        f"spec = importlib.util.find_spec({name!r})\n"
        "print(json.dumps({'available': spec is not None, 'origin': getattr(spec, 'origin', '') if spec else ''}, sort_keys=True))\n"
    )
    completed = subprocess.run(
        [str(runtime_python), "-c", code],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        return {"available": False, "origin": "", "runtime_python": str(runtime_python), "error": completed.stderr.strip()}
    try:
        data = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return {"available": False, "origin": "", "runtime_python": str(runtime_python), "error": completed.stdout.strip()}
    return {**data, "runtime_python": str(runtime_python)}


def _server_script_record(path: Path) -> dict[str, Any]:
    exists = path.exists()
    return {
        "path": str(path),
        "exists": exists,
        "sha256": _sha256_file(path) if exists and path.is_file() else "",
        "declared_dependencies": _script_dependencies(path) if exists else [],
    }


def _adapter_records(specs: dict[str, Path]) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for arm, path in specs.items():
        resolved = path.expanduser().resolve()
        files = {}
        for name in ADAPTER_FILES:
            candidate = resolved / name
            files[name] = {
                "path": str(candidate),
                "exists": candidate.exists() and candidate.is_file(),
                "sha256": _sha256_file(candidate) if candidate.exists() and candidate.is_file() else "",
                "bytes": candidate.stat().st_size if candidate.exists() and candidate.is_file() else 0,
            }
        records[arm] = {
            "path": str(resolved),
            "exists": resolved.exists() and resolved.is_dir(),
            "files": files,
        }
    return records


def _blocked_checks(
    *,
    dependency_records: dict[str, dict[str, Any]],
    model_cache: dict[str, Any],
    adapters: dict[str, dict[str, Any]],
    server_script: dict[str, Any],
    runtime: dict[str, Any],
) -> list[str]:
    blocked = [f"missing_dependency:{name}" for name, record in dependency_records.items() if not record["available"]]
    if not runtime["exists"]:
        blocked.append("missing_runtime_python")
    if not model_cache["exists"]:
        blocked.append("missing_model_cache")
    if not server_script["exists"]:
        blocked.append("missing_server_script")
    for arm, record in adapters.items():
        if not record["exists"]:
            blocked.append(f"missing_adapter:{arm}")
            continue
        for required in ("adapter_config.json", "adapter_model.safetensors"):
            if not record["files"].get(required, {}).get("exists"):
                blocked.append(f"missing_adapter_file:{arm}:{required}")
    return blocked


def _command_refs(
    *,
    args: argparse.Namespace,
    server_script: Path,
    runtime_python: Path,
    adapters: dict[str, dict[str, Any]],
) -> list[dict[str, str]]:
    refs = [_command_ref(args=args, server_script=server_script, runtime_python=runtime_python, arm=args.baseline_arm, adapter="")]
    for arm, record in adapters.items():
        refs.append(_command_ref(args=args, server_script=server_script, runtime_python=runtime_python, arm=arm, adapter=record["path"]))
    return refs


def _command_ref(*, args: argparse.Namespace, server_script: Path, runtime_python: Path, arm: str, adapter: str) -> dict[str, str]:
    port = "{port}"
    out_dir = f"experiments/qwen3_4b_flightrecorder/serving/{arm}"
    server_parts = [
        str(runtime_python),
        str(server_script),
        "--model",
        args.model,
        "--host",
        "127.0.0.1",
        "--port",
        port,
        "--max-new-tokens",
        str(args.max_new_tokens),
    ]
    managed_parts = [
        str(runtime_python),
        "scripts/run_managed_serving_eval.py",
        "--server-command",
        shlex.join(server_parts + (["--adapter", adapter] if adapter else [])),
        "--base-url",
        f"http://127.0.0.1:{port}/v1",
        "--model",
        args.model,
        "--arm",
        arm,
        "--out",
        out_dir,
        "--require-structured-output",
    ]
    if adapter:
        managed_parts.extend(["--adapter", adapter])
    managed_parts.extend(
        [
            "--eval-command",
            (
                f"{shlex.quote(str(runtime_python))} scripts/evaluate_hermes_heldout.py "
                f"--arm {arm} --model {shlex.quote(args.model)} --base-url {{base_url}} "
                f"--serving-profile {{serving_profile}} --out experiments/qwen3_4b_flightrecorder/evaluations/{arm} --force"
            ),
        ]
    )
    return {
        "arm": arm,
        "server_command": shlex.join(server_parts + (["--adapter", adapter] if adapter else [])),
        "managed_eval_command": shlex.join(managed_parts),
    }


def _parse_arm_paths(specs: list[str]) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for spec in specs:
        if "=" not in spec:
            raise SystemExit(f"--adapter must use ARM=PATH: {spec}")
        arm, value = spec.split("=", 1)
        if not arm or not value:
            raise SystemExit(f"--adapter must use ARM=PATH: {spec}")
        parsed[arm] = Path(value)
    return parsed


def _script_dependencies(path: Path) -> list[str]:
    inside = False
    dependencies: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        stripped = raw.strip()
        if stripped == "# /// script":
            inside = True
            continue
        if inside and stripped == "# ///":
            break
        if not inside or not stripped.startswith("#"):
            continue
        text = stripped[1:].strip()
        if text.startswith(("\"", "'")):
            dependency = text.rstrip(",").strip().strip("\"'")
            if dependency:
                dependencies.append(dependency)
    return dependencies


def _default_model_cache(model: str) -> Path:
    return Path.home() / ".cache" / "huggingface" / "hub" / f"models--{model.replace('/', '--')}"


def _path_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            total += child.stat().st_size
    return total


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


if __name__ == "__main__":
    raise SystemExit(main())
