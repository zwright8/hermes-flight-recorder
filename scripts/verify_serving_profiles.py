#!/usr/bin/env python3
"""Verify a suite of serving profiles before Eval or demo replay."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "hfr.serving_endpoint_suite.v1"
CORE_CAPABILITIES = ("health", "models", "model_metadata", "chat_completions")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        action="append",
        default=[],
        metavar="ARM=PATH",
        help="Serving profile to verify. Repeat for baseline, trace_only, flightrecorder, etc.",
    )
    parser.add_argument(
        "--lifecycle",
        action="append",
        default=[],
        metavar="ARM=PATH",
        help="Optional serving_lifecycle_run.json for an arm.",
    )
    parser.add_argument("--required-arm", action="append", default=[], help="Arm that must be present in the suite.")
    parser.add_argument("--expect-model", action="append", default=[], metavar="ARM=MODEL", help="Expected model identity for an arm.")
    parser.add_argument("--expect-adapter", action="append", default=[], metavar="ARM=ADAPTER", help="Expected adapter path/id for an arm.")
    parser.add_argument("--require-tool-call", action="store_true", help="Require tool-call smoke support for every profile.")
    parser.add_argument("--require-structured-output", action="store_true", help="Require structured-output smoke support for every profile.")
    parser.add_argument("--strict-profile-arm", action="store_true", help="Require profile.arm to match the supplied ARM label.")
    parser.add_argument("--out", type=Path, required=True, help="Serving endpoint suite JSON output path.")
    parser.add_argument("--report", type=Path, help="Optional Markdown readiness report path.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    suite = build_suite(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    _write_json(args.out, suite)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(render_report(suite), encoding="utf-8")
    print(json.dumps({"passed": suite["passed"], "failed_checks": suite["failed_checks"], "out": str(args.out)}, indent=2))
    return 0 if suite["passed"] else 1


def build_suite(args: argparse.Namespace) -> dict[str, Any]:
    profile_specs = _parse_key_paths(args.profile, label="--profile")
    lifecycle_specs = _parse_key_paths(args.lifecycle, label="--lifecycle")
    expected_models = _parse_key_values(args.expect_model, label="--expect-model")
    expected_adapters = _parse_key_values(args.expect_adapter, label="--expect-adapter")

    duplicate_arms = sorted({arm for arm in profile_specs if list(profile_specs).count(arm) > 1})
    missing_required = [arm for arm in args.required_arm if arm not in profile_specs]
    arms = []
    failed_checks = [f"duplicate_arm:{arm}" for arm in duplicate_arms] + [f"missing_required_arm:{arm}" for arm in missing_required]

    for arm, profile_path in profile_specs.items():
        profile = _load_json(profile_path)
        lifecycle_path = lifecycle_specs.get(arm)
        lifecycle = _load_json(lifecycle_path) if lifecycle_path else None
        arm_record = _verify_arm(
            arm=arm,
            profile_path=profile_path,
            profile=profile,
            lifecycle_path=lifecycle_path,
            lifecycle=lifecycle,
            expected_model=expected_models.get(arm, ""),
            expected_adapter=expected_adapters.get(arm, ""),
            require_tool_call=bool(args.require_tool_call),
            require_structured_output=bool(args.require_structured_output),
            strict_profile_arm=bool(args.strict_profile_arm),
        )
        arms.append(arm_record)
        failed_checks.extend(f"{arm}:{check_id}" for check_id in arm_record["failed_checks"])

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "passed": not failed_checks,
        "failed_checks": failed_checks,
        "requirements": {
            "required_arms": list(args.required_arm),
            "require_tool_call": bool(args.require_tool_call),
            "require_structured_output": bool(args.require_structured_output),
            "strict_profile_arm": bool(args.strict_profile_arm),
            "expected_models": expected_models,
            "expected_adapters": expected_adapters,
        },
        "arms": arms,
    }


def _verify_arm(
    *,
    arm: str,
    profile_path: Path,
    profile: dict[str, Any],
    lifecycle_path: Path | None,
    lifecycle: dict[str, Any] | None,
    expected_model: str,
    expected_adapter: str,
    require_tool_call: bool,
    require_structured_output: bool,
    strict_profile_arm: bool,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    capabilities = profile.get("capabilities") if isinstance(profile.get("capabilities"), dict) else {}
    identity = profile.get("model_identity") if isinstance(profile.get("model_identity"), dict) else {}
    adapter = identity.get("adapter") if isinstance(identity.get("adapter"), dict) else {}
    endpoint = profile.get("endpoint") if isinstance(profile.get("endpoint"), dict) else {}
    eval_preflight = profile.get("eval_preflight") if isinstance(profile.get("eval_preflight"), dict) else {}

    checks.append(_check("profile_schema", profile.get("schema_version") == "hfr.serving_profile.v1", {"schema_version": profile.get("schema_version")}))
    checks.append(_check("endpoint_base_url", bool(endpoint.get("base_url")), {"base_url": endpoint.get("base_url")}))
    if strict_profile_arm:
        checks.append(_check("profile_arm", profile.get("arm") == arm, {"expected": arm, "actual": profile.get("arm")}))
    checks.append(_check("eval_preflight_ready", bool(eval_preflight.get("ready")), {"readiness": eval_preflight.get("readiness"), "failed_checks": eval_preflight.get("failed_checks") or []}))
    for capability in CORE_CAPABILITIES:
        checks.append(_check(f"capability_{capability}", capabilities.get(capability) is True, {"actual": capabilities.get(capability)}))
    if require_tool_call:
        checks.append(_check("capability_tool_calls", capabilities.get("tool_calls") == "supported", {"actual": capabilities.get("tool_calls")}))
    if require_structured_output:
        checks.append(
            _check(
                "capability_structured_outputs",
                capabilities.get("structured_outputs") == "supported",
                {"actual": capabilities.get("structured_outputs")},
            )
        )
    if expected_model:
        checks.append(_check("model_identity", _model_matches(expected_model, identity), {"expected": expected_model, "identity": identity}))
    if expected_adapter:
        checks.append(_check("adapter_identity", _adapter_matches(expected_adapter, adapter), {"expected": expected_adapter, "adapter": adapter}))
    if lifecycle_path:
        checks.extend(_lifecycle_checks(lifecycle_path, lifecycle or {}))

    failed = [item["id"] for item in checks if not item["passed"]]
    return {
        "arm": arm,
        "profile_path": str(profile_path),
        "profile_id": profile.get("profile_id"),
        "profile_arm": profile.get("arm"),
        "ready_for_eval": not failed,
        "failed_checks": failed,
        "checks": checks,
        "endpoint": {"base_url": endpoint.get("base_url")},
        "model_identity": {
            "requested_model": identity.get("requested_model"),
            "served_model_id": identity.get("served_model_id"),
            "observed_model_ids": identity.get("observed_model_ids") or [],
            "adapter": adapter,
        },
        "capabilities": {
            "health": capabilities.get("health"),
            "models": capabilities.get("models"),
            "model_metadata": capabilities.get("model_metadata"),
            "chat_completions": capabilities.get("chat_completions"),
            "tool_calls": capabilities.get("tool_calls"),
            "structured_outputs": capabilities.get("structured_outputs"),
        },
        "lifecycle": {"path": str(lifecycle_path), "present": True} if lifecycle_path else {"path": "", "present": False},
    }


def render_report(suite: dict[str, Any]) -> str:
    lines = [
        "# Serving Endpoint Suite",
        "",
        f"- Passed: {suite['passed']}",
        f"- Failed checks: {', '.join(suite['failed_checks']) if suite['failed_checks'] else 'none'}",
        "",
        "## Arms",
        "",
        "| Arm | Ready | Requested Model | Served Model | Endpoint | Tool Calls | Structured Outputs | Lifecycle | Failed Checks |",
        "| --- | ---: | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for arm in suite["arms"]:
        identity = arm["model_identity"]
        capabilities = arm["capabilities"]
        lifecycle = arm.get("lifecycle") or {}
        lines.append(
            "| {arm} | {ready} | `{requested}` | `{served}` | {endpoint} | {tools} | {structured} | {lifecycle} | {failed} |".format(
                arm=arm["arm"],
                ready=arm["ready_for_eval"],
                requested=identity.get("requested_model") or "",
                served=identity.get("served_model_id") or "",
                endpoint=arm["endpoint"].get("base_url") or "",
                tools=capabilities.get("tool_calls"),
                structured=capabilities.get("structured_outputs"),
                lifecycle=_md_link("lifecycle", lifecycle.get("path")) if lifecycle.get("path") else "",
                failed=", ".join(arm["failed_checks"]) if arm["failed_checks"] else "none",
            )
        )
    lines.extend(["", "## Profile Links", ""])
    for arm in suite["arms"]:
        lines.append(f"- `{arm['arm']}`: {_md_link('serving_profile', arm['profile_path'])}")
    return "\n".join(lines) + "\n"


def _lifecycle_checks(path: Path, lifecycle: dict[str, Any]) -> list[dict[str, Any]]:
    cleanup = lifecycle.get("cleanup") if isinstance(lifecycle.get("cleanup"), dict) else {}
    preflight = lifecycle.get("preflight") if isinstance(lifecycle.get("preflight"), dict) else {}
    return [
        _check("lifecycle_schema", lifecycle.get("schema_version") == "hfr.serving_lifecycle_run.v1", {"schema_version": lifecycle.get("schema_version"), "path": str(path)}),
        _check("lifecycle_passed", lifecycle.get("passed") is True, {"passed": lifecycle.get("passed")}),
        _check("lifecycle_preflight", preflight.get("passed") is True, {"readiness": preflight.get("readiness"), "failed_checks": preflight.get("failed_checks") or []}),
        _check(
            "lifecycle_cleanup",
            bool(cleanup.get("attempted") and (cleanup.get("terminated") or cleanup.get("killed") or cleanup.get("exit_code_after_cleanup") is not None)),
            {"cleanup": cleanup},
        ),
    ]


def _model_matches(expected: str, identity: dict[str, Any]) -> bool:
    observed = [
        identity.get("requested_model"),
        identity.get("served_model_id"),
        identity.get("metadata_model"),
        identity.get("chat_response_model"),
        *(identity.get("observed_model_ids") or []),
    ]
    values = [str(value) for value in observed if value]
    return any(value == expected or value.startswith(f"{expected}+") for value in values)


def _adapter_matches(expected: str, adapter: dict[str, Any]) -> bool:
    expected_path = Path(expected).expanduser()
    expected_name = expected_path.name
    observed = [str(adapter.get("id") or ""), str(adapter.get("path") or "")]
    return bool(adapter.get("present")) and any(value == expected or value == expected_name or Path(value).name == expected_name for value in observed if value)


def _parse_key_paths(specs: list[str], *, label: str) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for spec in specs:
        key, value = _split_key_value(spec, label=label)
        parsed[key] = Path(value).expanduser().resolve()
    return parsed


def _parse_key_values(specs: list[str], *, label: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for spec in specs:
        key, value = _split_key_value(spec, label=label)
        parsed[key] = value
    return parsed


def _split_key_value(spec: str, *, label: str) -> tuple[str, str]:
    if "=" not in spec:
        raise SystemExit(f"{label} must use ARM=VALUE: {spec}")
    key, value = spec.split("=", 1)
    if not key or not value:
        raise SystemExit(f"{label} must use ARM=VALUE: {spec}")
    return key, value


def _load_json(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _check(check_id: str, passed: bool, details: dict[str, Any]) -> dict[str, Any]:
    return {"id": check_id, "passed": bool(passed), "details": details}


def _md_link(label: str, path: str | None) -> str:
    if not path:
        return ""
    return f"[{label}]({path})"


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


if __name__ == "__main__":
    raise SystemExit(main())
