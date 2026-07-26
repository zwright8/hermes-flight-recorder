from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def write_process_segments_fixture(run: Path) -> dict[str, Any]:
    process_root = run / "process_segments"
    segment_root = process_root / "segments" / "segment-0001"
    adapter_root = segment_root / "adapter"
    adapter_root.mkdir(parents=True)
    plan = process_root / "plan.json"
    manifest = process_root / "manifest.json"
    telemetry = segment_root / "telemetry.jsonl"
    optimizer = segment_root / "optimizer_state.safetensors"
    adapter = adapter_root / "adapters.safetensors"
    record_path = segment_root / "segment_record.json"
    _write_json(plan, {"schema_version": "fixture"})
    _write_json(manifest, {"schema_version": "fixture"})
    telemetry.write_text('{"event":"segment"}\n', encoding="utf-8")
    optimizer.write_bytes(b"optimizer-state")
    adapter.write_bytes(b"adapter-state")
    _write_json(record_path, {"terminal_status": "success"})

    adapter_tree = _tree_record(run, adapter_root)
    artifact_tree = _tree_record(run, process_root / "segments")
    policy = {
        "schema_version": "hfr.tau3_mlx_process_segments.v1",
        "total_iters": 400,
        "process_segment_iters": 400,
        "gradient_accumulation": 4,
        "report_every": 20,
        "dropout": 0,
    }
    record = {
        "index": 0,
        "segment_id": "segment-0001",
        "start_iter": 0,
        "end_iter": 400,
        "iteration_count": 400,
        "previous_segment_record_sha256": None,
        "telemetry": _file_record(run, telemetry),
        "adapter_output": _file_record(run, adapter),
        "adapter_tree": adapter_tree,
        "optimizer_state_output": _file_record(run, optimizer),
        "terminal_status": "success",
        "segment_record_sha256": "3" * 64,
    }
    return {
        "schema_version": "hfr.tau3_mlx_process_segments.v1",
        "policy": policy,
        "policy_sha256": "0" * 64,
        "plan": _file_record(run, plan),
        "plan_sha256": "1" * 64,
        "segments": [
            {
                "record": record,
                "record_file": _file_record(run, record_path),
            }
        ],
        "aggregate_telemetry": _file_record(run, telemetry),
        "artifact_tree": artifact_tree,
        "final_adapter": adapter_tree,
        "terminal_status": "success",
        "completed_segment_count": 1,
        "planned_segment_count": 1,
        "recovery": {
            "resumed": False,
            "accepted_segment_count": 0,
            "preserved_partial_artifact_trees": [],
            "preserved_failed_artifact_tree": None,
        },
        "manifest_sha256": "2" * 64,
        "manifest": _file_record(run, manifest),
        "validation": {"passed": True, "errors": []},
    }


def _file_record(run: Path, path: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(run).as_posix(),
        "sha256": _sha256(path),
        "read_only": True,
    }


def _tree_record(run: Path, root: Path) -> dict[str, Any]:
    files = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        files.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": _sha256(path),
                "size": path.stat().st_size,
            }
        )
    encoded = json.dumps(
        files,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "path": root.relative_to(run).as_posix(),
        "file_count": len(files),
        "files": files,
        "tree_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
