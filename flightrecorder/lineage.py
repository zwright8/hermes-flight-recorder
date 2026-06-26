"""Artifact-lineage manifests for Flight Recorder runs."""

from __future__ import annotations

import hashlib
import json
import shlex
from pathlib import Path
from typing import Any

LINEAGE_SCHEMA_VERSION = "hfr.lineage.v1"
REPLAY_BUNDLE_SCHEMA_VERSION = "hfr.replay_bundle.v1"


def write_run_lineage(
    *,
    scenario: dict[str, Any],
    trace: dict[str, Any],
    scorecard: dict[str, Any],
    source_trace_path: str | Path,
    source_state_snapshot_path: str | Path | None = None,
    artifacts: dict[str, str | Path | None],
    out_path: str | Path,
    preserve_paths: bool = False,
) -> dict[str, Any]:
    """Write a per-run lineage manifest and return its JSON payload."""
    lineage = build_run_lineage(
        scenario=scenario,
        trace=trace,
        scorecard=scorecard,
        source_trace_path=source_trace_path,
        source_state_snapshot_path=source_state_snapshot_path,
        artifacts=artifacts,
        preserve_paths=preserve_paths,
    )
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(lineage, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return lineage


def build_run_lineage(
    *,
    scenario: dict[str, Any],
    trace: dict[str, Any],
    scorecard: dict[str, Any],
    source_trace_path: str | Path,
    source_state_snapshot_path: str | Path | None = None,
    artifacts: dict[str, str | Path | None],
    preserve_paths: bool = False,
) -> dict[str, Any]:
    """Build a deterministic manifest connecting inputs, outputs, and evidence refs."""
    scenario_path = scenario.get("_path")
    inputs = [
        _file_record("scenario", scenario_path, "input", preserve_paths=preserve_paths),
        _file_record("source_trace", source_trace_path, "input", preserve_paths=preserve_paths),
    ]
    if source_state_snapshot_path is not None:
        inputs.append(_file_record("source_state_snapshot", source_state_snapshot_path, "input", preserve_paths=preserve_paths))
    outputs = [
        _file_record(name, path, "output", preserve_paths=preserve_paths, sensitive=name.endswith("_sensitive"))
        for name, path in artifacts.items()
        if path is not None
    ]
    evidence_links = _evidence_links(scorecard)
    replay = _replay_contract(
        inputs=inputs,
        artifacts=artifacts,
        scenario_path=scenario_path,
        source_trace_path=source_trace_path,
        source_state_snapshot_path=source_state_snapshot_path,
        preserve_paths=preserve_paths,
    )
    return {
        "schema_version": LINEAGE_SCHEMA_VERSION,
        "scenario": {
            "id": scenario.get("id"),
            "title": scenario.get("title"),
        },
        "trace": {
            "schema_version": trace.get("schema_version"),
            "session_id": trace.get("session", {}).get("id"),
            "source_format": trace.get("session", {}).get("source_format"),
            "model": trace.get("session", {}).get("model"),
            "event_count": len(trace.get("events", [])) if isinstance(trace.get("events"), list) else 0,
        },
        "scorecard": {
            "schema_version": scorecard.get("schema_version"),
            "score": scorecard.get("score"),
            "passed": scorecard.get("passed"),
            "critical_failures": scorecard.get("critical_failures", []),
        },
        "inputs": inputs,
        "outputs": outputs,
        "graph": _artifact_graph(artifacts),
        "evidence_links": evidence_links,
        "replay": replay,
        "summary": {
            "input_count": len(inputs),
            "output_count": len(outputs),
            "evidence_link_count": len(evidence_links),
            "self_contained_replay": replay["self_contained"],
        },
    }


def _artifact_graph(artifacts: dict[str, str | Path | None]) -> list[dict[str, Any]]:
    present = {name for name, path in artifacts.items() if path is not None}
    edges = [
        _edge(["scenario", "source_trace"], "normalized_trace", "normalize"),
        _edge(_score_inputs(present), "scorecard", "score"),
        _edge(["scorecard"], "task_completion", "summarize_task_completion"),
        _edge(["scenario", "normalized_trace", "scorecard"], "report", "render"),
    ]
    if "junit" in present:
        edges.append(_edge(["scorecard"], "junit", "ci_report"))
    if "markdown" in present:
        edges.append(_edge(["scorecard"], "markdown", "markdown_summary"))
    if "regression_scenario" in present:
        edges.append(_edge(["scenario", "source_trace", "scorecard"], "regression_scenario", "regression_capture"))
    if "raw_trace_sensitive" in present:
        edges.append(_edge(["source_trace"], "raw_trace_sensitive", "sensitive_trace_export"))
    return edges


def _score_inputs(present: set[str]) -> list[str]:
    inputs = ["scenario", "normalized_trace"]
    if "state_snapshot" in present:
        inputs.append("state_snapshot")
    return inputs


def _edge(from_nodes: list[str], to_node: str, operation: str) -> dict[str, Any]:
    return {"from": from_nodes, "to": to_node, "operation": operation}


def _evidence_links(scorecard: dict[str, Any]) -> list[dict[str, Any]]:
    links: list[dict[str, Any]] = []
    for rule_index, rule in enumerate(scorecard.get("rules", [])):
        if not isinstance(rule, dict):
            continue
        refs = rule.get("evidence_refs")
        if not isinstance(refs, list):
            continue
        rule_id = str(rule.get("id") or f"rule_{rule_index}")
        for ref_index, ref in enumerate(refs):
            if not isinstance(ref, dict):
                continue
            target = ref.get("target")
            link = {
                "rule_id": rule_id,
                "rule_name": str(rule.get("name") or rule_id),
                "rule_passed": bool(rule.get("passed")),
                "scorecard_pointer": f"/rules/{rule_index}/evidence_refs/{ref_index}",
                "target": target,
                "reason": ref.get("reason"),
            }
            if "passed" in ref:
                link["ref_passed"] = ref.get("passed")
            if target == "event":
                event_index = ref.get("event_index")
                link["event_index"] = event_index
                link["trace_pointer"] = f"/events/{event_index}"
                for field in ("event_type", "tool_name", "status"):
                    if field in ref:
                        link[field] = ref.get(field)
            elif target == "final_answer":
                link["trace_pointer"] = "/final_answer"
            elif target == "episode":
                link["trace_pointer"] = "/"
            elif target == "state_snapshot":
                link["state_pointer"] = "/"
                if "field" in ref:
                    link["state_field"] = ref.get("field")
            links.append(link)
    return links


def _replay_contract(
    *,
    inputs: list[dict[str, Any]],
    artifacts: dict[str, str | Path | None],
    scenario_path: str | Path | None,
    source_trace_path: str | Path,
    source_state_snapshot_path: str | Path | None,
    preserve_paths: bool,
) -> dict[str, Any]:
    out_dir = _run_dir_from_artifacts(artifacts)
    argv = [
        "python",
        "-m",
        "flightrecorder",
        "run",
        "--scenario",
        _display_path(Path(scenario_path), preserve_paths) if scenario_path is not None else "<missing-scenario-path>",
        "--trace",
        _display_path(Path(source_trace_path), preserve_paths),
        "--out",
        _display_path(out_dir, preserve_paths) if out_dir is not None else "<missing-output-dir>",
    ]
    if source_state_snapshot_path is not None:
        argv.extend(["--state", _display_path(Path(source_state_snapshot_path), preserve_paths)])
    replay_paths = [arg for index, arg in enumerate(argv) if index > 0 and argv[index - 1] in {"--scenario", "--trace", "--state", "--out"}]
    input_fingerprints = {
        str(record.get("name")): {
            "path": record.get("path"),
            "sha256": record.get("sha256"),
            "exists": record.get("exists"),
        }
        for record in inputs
        if isinstance(record.get("name"), str)
    }
    required_inputs = ("scenario", "source_trace", *(() if source_state_snapshot_path is None else ("source_state_snapshot",)))
    has_required_hashes = all(isinstance(input_fingerprints.get(name, {}).get("sha256"), str) for name in required_inputs)
    has_required_paths = all(isinstance(input_fingerprints.get(name, {}).get("path"), str) for name in required_inputs)
    paths_are_public = all(not _is_redacted_path(path) for path in replay_paths)
    self_contained = has_required_hashes and has_required_paths and paths_are_public and out_dir is not None
    return {
        "tool": "flightrecorder",
        "argv": argv,
        "command": " ".join(shlex.quote(arg) for arg in argv),
        "input_fingerprints": input_fingerprints,
        "self_contained": self_contained,
        "notes": [
            "Replay reruns normalization, scoring, report generation, and lineage from the recorded scenario and trace inputs.",
            "self_contained is false when paths were redacted or required input fingerprints are missing.",
        ],
    }


def _run_dir_from_artifacts(artifacts: dict[str, str | Path | None]) -> Path | None:
    for path_value in artifacts.values():
        if path_value is not None:
            return Path(path_value).parent
    return None


def _is_redacted_path(value: str) -> bool:
    return value.startswith("<redacted:") or value.startswith("<missing-")


def _file_record(
    name: str,
    path_value: str | Path | None,
    role: str,
    *,
    preserve_paths: bool,
    sensitive: bool = False,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "name": name,
        "role": role,
        "path": None,
        "exists": False,
        "sensitive": sensitive,
    }
    if path_value is None:
        return record
    path = Path(path_value)
    record["path"] = _display_path(path, preserve_paths)
    record["exists"] = path.exists()
    if path.exists() and path.is_file():
        stat = path.stat()
        record["size_bytes"] = stat.st_size
        record["sha256"] = _sha256(path)
    return record


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
