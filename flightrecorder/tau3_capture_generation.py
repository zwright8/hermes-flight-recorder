"""Generate governed Tau-3 training captures from pinned training-side sources.

The public generator in this module stays dependency-free: it validates local
source boundaries, then invokes a Tau runtime through a subprocess JSON
contract.  The subprocess may use Tau's Python 3.12 environment; the core
Flight Recorder package remains importable on Python 3.11 without Tau.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from .path_safety import path_has_symlink_component
from .tau3_capture import TAU3_CAPTURE_SCHEMA_VERSION, canonical_sha256, validate_tau3_capture

DOMAINS = ("airline", "retail", "telecom")
CAPTURE_GENERATOR_SCHEMA_VERSION = "hfr.tau3_capture_generation.v1"
HEX40_RE = re.compile(r"^[0-9a-f]{40}$")
DEFAULT_TRAIN_DOMAIN_QUOTAS = {"airline": 8, "retail": 8, "telecom": 3}
DEFAULT_DEVELOPMENT_DOMAIN_QUOTAS = {"airline": 2, "retail": 2, "telecom": 1}


class Tau3CaptureGenerationError(ValueError):
    """Raised when Tau training capture generation cannot proceed safely."""


def generate_tau3_training_captures(
    *,
    tau_repo: str | Path,
    expected_revision: str,
    train_tasks: str | Path,
    development_tasks: str | Path,
    out: str | Path,
    tau_python: str | Path | None = None,
    generator_id: str = "tau3-reference-action-replay",
    generator_revision: str | None = None,
    seed: int = 0,
    train_domain_quotas: dict[str, int] | None = None,
    development_domain_quotas: dict[str, int] | None = None,
    sample_salt: str = "hfr-tau3-capture-generation-v1",
) -> dict[str, Any]:
    """Generate canonical ``hfr.tau3_capture.v1`` rows for train/dev tasks.

    Inputs must be explicit source-partitioner JSONL files.  The generator
    refuses to read official test payloads, refuses output overwrite, checks the
    Tau checkout revision and cleanliness, and treats executable reference
    actions as the only source of tool-call trajectories.
    """

    repo = Path(tau_repo)
    train_path = Path(train_tasks)
    dev_path = Path(development_tasks)
    out_path = Path(out)
    py = Path(tau_python) if tau_python is not None else Path(sys.executable)
    revision = generator_revision or expected_revision
    train_quotas = _domain_quotas(train_domain_quotas, DEFAULT_TRAIN_DOMAIN_QUOTAS, "train_domain_quotas")
    development_quotas = _domain_quotas(
        development_domain_quotas,
        DEFAULT_DEVELOPMENT_DOMAIN_QUOTAS,
        "development_domain_quotas",
    )
    _validate_options(expected_revision, seed, train_quotas, development_quotas, sample_salt)
    _require_clean_revision(repo, expected_revision)
    _require_regular_safe_input(train_path, "train tasks")
    _require_regular_safe_input(dev_path, "development tasks")
    _require_new_safe_output(out_path)
    if out_path.resolve(strict=False) in {train_path.resolve(strict=True), dev_path.resolve(strict=True)}:
        raise Tau3CaptureGenerationError("output must not alias an input task file")

    rows_by_split = {
        "train": _read_jsonl(train_path, "train tasks"),
        "development": _read_jsonl(dev_path, "development tasks"),
    }
    flat_rows = _validate_source_rows(rows_by_split, expected_revision)
    selected_rows, sampling_report = _select_balanced_sources(
        flat_rows,
        train_domain_quotas=train_quotas,
        development_domain_quotas=development_quotas,
        sample_salt=sample_salt,
    )

    worker_payload = {
        "schema_version": CAPTURE_GENERATOR_SCHEMA_VERSION,
        "tau_repo": str(repo),
        "expected_revision": expected_revision,
        "generator_id": generator_id,
        "generator_revision": revision,
        "seed": seed,
        "quotas": {"train": train_quotas, "development": development_quotas},
        "rows": selected_rows,
    }
    worker_result = _run_tau_worker(py, repo, worker_payload)
    sampling_report = _finalize_sampling_report(sampling_report, worker_result)
    captures = worker_result.get("captures")
    if not isinstance(captures, list) or not captures:
        raise Tau3CaptureGenerationError("Tau worker returned no captures")
    errors = {
        str(index): errors
        for index, capture in enumerate(captures)
        if (errors := validate_tau3_capture(capture))
    }
    if errors:
        raise Tau3CaptureGenerationError("generated invalid capture rows: " + json.dumps(errors, sort_keys=True))
    _assert_deterministic_unique(captures)
    token_share = _token_share_by_domain(captures)
    if set(token_share) != set(DOMAINS):
        raise Tau3CaptureGenerationError("generated captures must include every study domain")
    if any(value > 0.45 for value in token_share.values()):
        raise Tau3CaptureGenerationError(f"domain token share exceeds 0.45: {json.dumps(token_share, sort_keys=True)}")

    tmp = _make_staging_dir(out_path)
    try:
        captures_path = tmp / "captures.jsonl"
        manifest_path = tmp / "manifest.json"
        _write_jsonl(captures_path, captures)
        manifest = _manifest(
            out_dir=tmp,
            captures_path=captures_path,
            train_path=train_path,
            dev_path=dev_path,
            captures=captures,
            expected_revision=expected_revision,
            generator_id=generator_id,
            generator_revision=revision,
            worker_result=worker_result,
            sampling_report=sampling_report,
        )
        _write_json(manifest_path, manifest)
        os.replace(tmp, out_path)
        _fsync_directory(out_path.parent)
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise

    return {
        "schema_version": CAPTURE_GENERATOR_SCHEMA_VERSION,
        "out": str(out_path),
        "capture_count": len(captures),
        "train_capture_count": sum(1 for row in captures if row["split"] == "train"),
        "development_capture_count": sum(1 for row in captures if row["split"] == "development"),
        "domain_counts": _counts(row["domain"] for row in captures),
        "behavior_counts": _counts(row["behavior"] for row in captures),
        "token_share_by_domain": token_share,
        "token_count_by_domain": _token_count_by_domain(captures),
        "sampling": sampling_report,
        "source_rejection_count": len(worker_result.get("source_rejections") or []),
        "generator_id": generator_id,
        "generator_revision": revision,
        "tau_revision": expected_revision,
    }


def _validate_options(
    expected_revision: str,
    seed: int,
    train_domain_quotas: dict[str, int],
    development_domain_quotas: dict[str, int],
    sample_salt: str,
) -> None:
    if not HEX40_RE.fullmatch(expected_revision):
        raise Tau3CaptureGenerationError("expected revision must be an exact lowercase 40-hex git object id")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise Tau3CaptureGenerationError("seed must be an integer")
    for label, quotas in (("train_domain_quotas", train_domain_quotas), ("development_domain_quotas", development_domain_quotas)):
        if set(quotas) != set(DOMAINS):
            raise Tau3CaptureGenerationError(f"{label} must define exactly airline, retail, telecom")
        for domain, value in quotas.items():
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise Tau3CaptureGenerationError(f"{label}.{domain} must be a positive integer")
    if not sample_salt:
        raise Tau3CaptureGenerationError("sample_salt must be non-empty")


def _domain_quotas(value: dict[str, int] | None, default: dict[str, int], label: str) -> dict[str, int]:
    source = default if value is None else value
    if not isinstance(source, dict):
        raise Tau3CaptureGenerationError(f"{label} must be an object")
    return {str(domain): int(source[domain]) for domain in DOMAINS if domain in source}


def _require_clean_revision(repo: Path, expected_revision: str) -> None:
    if not repo.is_dir():
        raise Tau3CaptureGenerationError(f"Tau repository is not a directory: {repo}")
    actual = _git(repo, "rev-parse", "HEAD")
    if actual != expected_revision:
        raise Tau3CaptureGenerationError(f"Tau checkout revision mismatch: expected {expected_revision}, got {actual}")
    status = _git(repo, "status", "--porcelain=v1")
    if status:
        raise Tau3CaptureGenerationError("Tau checkout must be clean")


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(["git", "-C", str(repo), *args], check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "git command failed"
        raise Tau3CaptureGenerationError(detail)
    return completed.stdout.strip()


def _require_regular_safe_input(path: Path, label: str) -> None:
    if path_has_symlink_component(path, include_leaf=True):
        raise Tau3CaptureGenerationError(f"{label} path must not contain symlink components")
    if not path.is_file():
        raise Tau3CaptureGenerationError(f"{label} must be a regular file: {path}")


def _require_new_safe_output(out: Path) -> None:
    if out.exists() or out.is_symlink():
        raise Tau3CaptureGenerationError("output directory must not already exist")
    parent = out.parent
    if not parent.is_dir():
        raise Tau3CaptureGenerationError("output parent directory must already exist")
    if path_has_symlink_component(parent, include_leaf=True):
        raise Tau3CaptureGenerationError("output path must not contain symlink components")


def _make_staging_dir(out: Path) -> Path:
    for index in range(100):
        candidate = out.parent / f".{out.name}.tmp.{os.getpid()}.{index}"
        try:
            candidate.mkdir(mode=0o700)
        except FileExistsError:
            continue
        return candidate
    raise Tau3CaptureGenerationError("could not allocate staging directory")


def _read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise Tau3CaptureGenerationError(f"could not read {label}: {exc}") from exc
    for line_no, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise Tau3CaptureGenerationError(f"invalid JSON in {label} {path}:{line_no}: {exc}") from exc
        if not isinstance(value, dict):
            raise Tau3CaptureGenerationError(f"{label} row {line_no} must be an object")
        rows.append(value)
    if not rows:
        raise Tau3CaptureGenerationError(f"{label} is empty: {path}")
    return rows


def _validate_source_rows(rows_by_split: dict[str, list[dict[str, Any]]], expected_revision: str) -> list[dict[str, Any]]:
    seen: set[str] = set()
    flat_rows: list[dict[str, Any]] = []
    for expected_split, rows in rows_by_split.items():
        for index, envelope in enumerate(rows):
            if envelope.get("schema_version") != "hfr.tau3_training_source.v1":
                raise Tau3CaptureGenerationError("training source row must use hfr.tau3_training_source.v1")
            if envelope.get("source_revision") != expected_revision:
                raise Tau3CaptureGenerationError("training source row source_revision mismatch")
            domain = envelope.get("domain")
            if domain not in DOMAINS:
                raise Tau3CaptureGenerationError("training source row domain must be airline, retail, or telecom")
            if envelope.get("split") != expected_split:
                raise Tau3CaptureGenerationError(f"training source row split must be {expected_split}")
            task_family = envelope.get("task_family")
            if not isinstance(task_family, str) or not task_family:
                raise Tau3CaptureGenerationError("training source row task_family must be a non-empty string")
            task = envelope.get("task")
            if not isinstance(task, dict):
                raise Tau3CaptureGenerationError("training source row task must be an object")
            task_id = _task_id(task)
            global_id = f"{domain}:{task_id}"
            if global_id in seen:
                raise Tau3CaptureGenerationError(f"duplicate task id across capture inputs: {global_id}")
            seen.add(global_id)
            embedded_domain = _embedded_task_domain(task)
            if embedded_domain != domain:
                raise Tau3CaptureGenerationError(f"training source row domain disagrees with embedded task: {global_id}")
            if envelope.get("task_sha256") != _hash_json(task):
                raise Tau3CaptureGenerationError(f"training source row task_sha256 mismatch: {global_id}")
            if envelope.get("prompt_sha256") != _hash_json(_prompt_material(task)):
                raise Tau3CaptureGenerationError(f"training source row prompt_sha256 mismatch: {global_id}")
            declared_split = str(task.get("split") or task.get("official_split") or "")
            if declared_split in {"test", "sealed", "official_test"} or task.get("sealed") is True or task.get("sealed_evaluation") is True:
                raise Tau3CaptureGenerationError(f"sealed/test task row is forbidden: {global_id}")
            flat_rows.append({
                "split": expected_split,
                "source_index": index,
                "domain": domain,
                "task_family": task_family,
                "task_sha256": envelope["task_sha256"],
                "prompt_sha256": envelope["prompt_sha256"],
                "task": task,
            })
    return flat_rows


def _select_balanced_sources(
    rows: list[dict[str, Any]],
    *,
    train_domain_quotas: dict[str, int],
    development_domain_quotas: dict[str, int],
    sample_salt: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    quotas = {"train": train_domain_quotas, "development": development_domain_quotas}
    selected: list[dict[str, Any]] = []
    report: dict[str, Any] = {
        "algorithm": "hfr.tau3_capture_generation.salted_family_diverse_domain_quota.v1",
        "sample_salt_sha256": _sha256_text(sample_salt),
        "quotas": quotas,
        "splits": {},
    }
    for split in ("train", "development"):
        split_report: dict[str, Any] = {}
        for domain in DOMAINS:
            eligible = [row for row in rows if row["split"] == split and row["domain"] == domain]
            quota = quotas[split][domain]
            if len(eligible) < quota:
                raise Tau3CaptureGenerationError(
                    f"not enough eligible {split} {domain} tasks for balanced sampling: "
                    f"need {quota}, got {len(eligible)}"
                )
            ranked = sorted(
                eligible,
                key=lambda row: (
                    _sha256_text(f"{sample_salt}\0{split}\0{domain}\0{row['task_family']}\0{row['task_sha256']}"),
                    row["task_family"],
                    row["task_sha256"],
                ),
            )
            family_first: list[dict[str, Any]] = []
            seen_families: set[str] = set()
            for row in ranked:
                if row["task_family"] in seen_families:
                    continue
                family_first.append(row)
                seen_families.add(row["task_family"])
            candidates = [*family_first, *(row for row in ranked if row not in family_first)]
            selected.extend({**row, "candidate_rank": rank} for rank, row in enumerate(candidates))
            split_report[domain] = {
                "eligible_count": len(eligible),
                "target_count": quota,
                "candidate_count": len(candidates),
                "selected_count": 0,
                "rejected_candidate_count": 0,
                "eligible_family_count": len({row["task_family"] for row in eligible}),
                "selected_family_count": 0,
                "candidate_task_sha256": [row["task_sha256"] for row in candidates],
                "selected_task_sha256": [],
                "rejected_task_sha256": [],
            }
        report["splits"][split] = split_report
    return sorted(selected, key=lambda row: (row["split"], row["domain"], row["candidate_rank"])), report


def _finalize_sampling_report(report: dict[str, Any], worker_result: dict[str, Any]) -> dict[str, Any]:
    selected_sources = worker_result.get("selected_sources")
    source_rejections = worker_result.get("source_rejections")
    if not isinstance(selected_sources, list) or not isinstance(source_rejections, list):
        raise Tau3CaptureGenerationError("Tau worker omitted source-selection evidence")
    for split in ("train", "development"):
        for domain in DOMAINS:
            selected = [row for row in selected_sources if row.get("split") == split and row.get("domain") == domain]
            rejected = [row for row in source_rejections if row.get("split") == split and row.get("domain") == domain]
            bucket = report["splits"][split][domain]
            if len(selected) != bucket["target_count"]:
                raise Tau3CaptureGenerationError(f"Tau worker did not satisfy {split} {domain} source quota")
            bucket["selected_count"] = len(selected)
            bucket["rejected_candidate_count"] = len(rejected)
            bucket["selected_family_count"] = len({row["task_family"] for row in selected})
            bucket["selected_task_sha256"] = [row["task_sha256"] for row in selected]
            bucket["rejected_task_sha256"] = [row["task_sha256"] for row in rejected]
    report["source_rejections"] = source_rejections
    return report


def _embedded_task_domain(task: dict[str, Any]) -> str:
    domain = task.get("domain")
    if not isinstance(domain, str):
        scenario = task.get("user_scenario") if isinstance(task.get("user_scenario"), dict) else {}
        instructions = scenario.get("instructions")
        if isinstance(instructions, dict):
            domain = instructions.get("domain")
    if domain not in DOMAINS:
        raise Tau3CaptureGenerationError("task domain must be airline, retail, or telecom")
    return domain


def _task_id(task: dict[str, Any]) -> str:
    task_id = task.get("id")
    if not isinstance(task_id, str) or not task_id:
        raise Tau3CaptureGenerationError("task id must be a non-empty string")
    return task_id


def _prompt_material(task: dict[str, Any]) -> Any:
    scenario = task.get("user_scenario")
    if isinstance(scenario, dict):
        return scenario.get("instructions", scenario)
    return scenario


def _run_tau_worker(tau_python: Path, repo: Path, payload: dict[str, Any]) -> dict[str, Any]:
    if not tau_python.exists() or not tau_python.is_file():
        raise Tau3CaptureGenerationError(f"Tau Python executable is not available: {tau_python}")
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    src = repo / "src"
    project_root = Path(__file__).resolve().parents[1]
    env["PYTHONPATH"] = os.pathsep.join(
        [str(project_root), str(src), *([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])]
    )
    completed = subprocess.run(
        [str(tau_python), "-m", "flightrecorder.tau3_capture_generation", "--_tau-worker"],
        input=json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False),
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "Tau worker failed"
        raise Tau3CaptureGenerationError(detail)
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise Tau3CaptureGenerationError(f"Tau worker returned invalid JSON: {exc}") from exc
    if not isinstance(result, dict):
        raise Tau3CaptureGenerationError("Tau worker result must be a JSON object")
    if result.get("schema_version") != CAPTURE_GENERATOR_SCHEMA_VERSION:
        raise Tau3CaptureGenerationError("Tau worker schema version mismatch")
    return result


def _assert_deterministic_unique(captures: list[dict[str, Any]]) -> None:
    ids = [str(row.get("trajectory_id") or "") for row in captures]
    if len(ids) != len(set(ids)):
        raise Tau3CaptureGenerationError("Tau worker returned duplicate trajectory ids")
    sorted_ids = sorted(ids)
    if ids != sorted_ids:
        raise Tau3CaptureGenerationError("Tau worker returned non-deterministic capture ordering")


def _manifest(
    *,
    out_dir: Path,
    captures_path: Path,
    train_path: Path,
    dev_path: Path,
    captures: list[dict[str, Any]],
    expected_revision: str,
    generator_id: str,
    generator_revision: str,
    worker_result: dict[str, Any],
    sampling_report: dict[str, Any],
) -> dict[str, Any]:
    artifacts = {
        "captures.jsonl": _file_record(captures_path, out_dir),
    }
    inputs = {
        "train_tasks": _file_record(train_path, train_path.parent),
        "development_tasks": _file_record(dev_path, dev_path.parent),
    }
    token_share = _token_share_by_domain(captures)
    return {
        "schema_version": CAPTURE_GENERATOR_SCHEMA_VERSION,
        "passed": True,
        "tau_revision": expected_revision,
        "generator": {
            "id": generator_id,
            "revision": generator_revision,
            "boundary": "subprocess-json",
            "training_started": False,
            "sealed_evaluation_started": False,
        },
        "inputs": inputs,
        "artifacts": artifacts,
        "capture_count": len(captures),
        "split_counts": _counts(row["split"] for row in captures),
        "domain_counts": _counts(row["domain"] for row in captures),
        "behavior_counts": _counts(row["behavior"] for row in captures),
        "token_share_by_domain": token_share,
        "token_count_by_domain": _token_count_by_domain(captures),
        "domain_balance_limit": 0.45,
        "domain_balance_passed": set(token_share) == set(DOMAINS) and all(value <= 0.45 for value in token_share.values()),
        "tool_call_count": sum(1 for row in captures for event in row["events"] if event.get("type") == "tool_call"),
        "sampling": sampling_report,
        "source_rejection_count": len(worker_result.get("source_rejections") or []),
        "sealed_payload_accessed": False,
        "sealed_manifest_accessed": False,
        "worker": {
            "runtime": worker_result.get("runtime"),
            "tool_schemas_recorded_exact": worker_result.get("tool_schemas_recorded_exact") is True,
            "source_rejections": worker_result.get("source_rejections"),
            "selected_sources": worker_result.get("selected_sources"),
        },
        "capture_sha256": hashlib.sha256(captures_path.read_bytes()).hexdigest(),
    }


def _file_record(path: Path, relative_to: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(relative_to).as_posix(),
        "size": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _write_json(path: Path, value: Any) -> None:
    _write_bytes(path, json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    data = b"".join(
        json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"
        for row in rows
    )
    _write_bytes(path, data)


def _write_bytes(path: Path, data: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(path, flags, 0o600)
    try:
        with os.fdopen(fd, "wb") as fp:
            fd = -1
            fp.write(data)
            fp.flush()
            os.fsync(fp.fileno())
    finally:
        if fd >= 0:
            os.close(fd)


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _counts(values) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return {key: counts[key] for key in sorted(counts)}


def _token_share_by_domain(captures: list[dict[str, Any]]) -> dict[str, float]:
    totals: dict[str, int] = {}
    for capture in captures:
        domain = str(capture["domain"])
        totals[domain] = totals.get(domain, 0) + int(capture.get("token_count") or _capture_token_count(capture))
    total = sum(totals.values()) or 1
    return {domain: round(totals[domain] / total, 6) for domain in sorted(totals)}


def _token_count_by_domain(captures: list[dict[str, Any]]) -> dict[str, int]:
    totals: dict[str, int] = {}
    for capture in captures:
        domain = str(capture["domain"])
        totals[domain] = totals.get(domain, 0) + int(capture.get("token_count") or _capture_token_count(capture))
    return {domain: totals[domain] for domain in sorted(totals)}


def _hash_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _worker_main() -> int:
    try:
        request = json.loads(sys.stdin.read())
        result = _worker_generate(request)
        sys.stdout.write(json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n")
        return 0
    except Exception as exc:
        sys.stderr.write(f"{type(exc).__name__}: {exc}\n")
        return 2


def _worker_generate(request: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(request, dict) or request.get("schema_version") != CAPTURE_GENERATOR_SCHEMA_VERSION:
        raise Tau3CaptureGenerationError("worker request schema mismatch")
    import importlib
    import platform

    from tau2.data_model.message import ToolCall
    from tau2.data_model.tasks import Task

    captures: list[dict[str, Any]] = []
    source_rejections: list[dict[str, Any]] = []
    selected_sources: list[dict[str, Any]] = []
    selected_counts = {split: {domain: 0 for domain in DOMAINS} for split in ("train", "development")}
    quotas = request.get("quotas")
    if not isinstance(quotas, dict):
        raise Tau3CaptureGenerationError("worker request quotas are missing")
    for row in sorted(request["rows"], key=lambda item: (item["split"], item["domain"], item["candidate_rank"])):
        task = Task.model_validate(row["task"])
        domain = str(row["domain"])
        split = str(row["split"])
        target_count = int(quotas[split][domain])
        if selected_counts[split][domain] >= target_count:
            continue
        env_mod = importlib.import_module(f"tau2.domains.{domain}.environment")
        env = env_mod.get_environment()
        initial_state = task.initial_state
        init_data = initial_state.initialization_data if initial_state is not None else None
        init_actions = initial_state.initialization_actions if initial_state is not None else None
        message_history = list(initial_state.message_history) if initial_state is not None and initial_state.message_history is not None else []
        env.set_state(initialization_data=init_data, initialization_actions=init_actions, message_history=message_history, strict=True)
        before_hash = _combined_state_hash(env)
        actions = list(task.evaluation_criteria.actions or []) if task.evaluation_criteria is not None else []
        criteria_payload = row["task"].get("evaluation_criteria") if isinstance(row["task"].get("evaluation_criteria"), dict) else {}
        communicate_info = [str(item) for item in (criteria_payload.get("communicate_info") or []) if isinstance(item, str)]
        nl_assertions = [str(item) for item in (criteria_payload.get("nl_assertions") or []) if isinstance(item, str)]
        reward_basis = [str(item) for item in (criteria_payload.get("reward_basis") or [])]
        no_action_supported = len(actions) == 0 and bool(communicate_info or nl_assertions)
        events = [_user_event(task, domain)]
        replay_messages = list(message_history)
        replay_failed = False
        for index, action in enumerate(actions):
            call_id = str(action.action_id or f"{task.id}-action-{index}")
            tool_call = ToolCall(id=call_id, name=action.name, arguments=dict(action.arguments), requestor=action.requestor)
            events.append(
                {
                    "type": "tool_call",
                    "role": str(action.requestor),
                    "tool_name": action.name,
                    "tool_call_id": call_id,
                    "args": dict(action.arguments),
                    "status": "ok",
                    "side_effect_status": "requested",
                    "content": "",
                }
            )
            response = env.get_response(tool_call)
            if getattr(response, "error", False):
                source_rejections.append({
                    "split": split,
                    "domain": domain,
                    "task_family": str(row["task_family"]),
                    "task_sha256": str(row["task_sha256"]),
                    "reason": "reference_action_replay_failed",
                    "action_name": str(action.name),
                    "error_sha256": _sha256_text(str(response.content)),
                })
                replay_failed = True
                break
            response_payload = _maybe_json(response.content)
            events.append(
                {
                    "type": "tool_result",
                    "role": "tool",
                    "tool_name": action.name,
                    "tool_call_id": call_id,
                    "result": response_payload,
                    "content": response.content,
                    "status": "ok",
                    "side_effect_status": "completed",
                }
            )
            replay_messages.extend(_message_pair(tool_call, response))
        if replay_failed:
            continue
        after_hash = _combined_state_hash(env)
        events.append(
            {
                "type": "assistant_message",
                "role": "assistant",
                "content": "Reference Tau tool trajectory completed.",
                "text": "Reference Tau tool trajectory completed.",
                "status": "ok",
            }
        )
        policy = env.get_policy()
        tools = _tool_definitions(env)
        prompt = _prompt(task, policy, tools)
        base = {
            "task_id": str(task.id),
            "task_family": str(row["task_family"]),
            "domain": domain,
            "split": split,
            "prompt": prompt,
            "seed": int(request["seed"]),
            "generator_id": str(request["generator_id"]),
            "generator_revision": str(request["generator_revision"]),
            "policy_revision": str(request["expected_revision"]),
            "tool_schema_revision": _hash_json(tools),
            "starting_state_hash": before_hash,
            "tools": tools,
            "source_task_sha256": str(row["task_sha256"]),
            "source_prompt_sha256": str(row["prompt_sha256"]),
            "environment_revision": str(request["expected_revision"]),
            "environment_hash": _sha256_text(str(request["expected_revision"])),
            "policy_hash": _sha256_text(policy),
            "policy_evidence": {
                "policy_hash": _sha256_text(policy),
                "official_reference_basis": {
                    "action_count": len(actions),
                    "reward_basis": reward_basis,
                    "communicate_info_count": len(communicate_info),
                    "nl_assertion_count": len(nl_assertions),
                    "no_action_supported": no_action_supported,
                },
                "programmatic_negative_rule": "Rejected variants are deterministic non-mutating traces; harmful Tau mutations are not executed.",
            },
            "no_action_supported": no_action_supported,
            "communicate_info": communicate_info,
            "nl_assertions": nl_assertions,
            "governance": {
                "owner": "tau3-study",
                "tenant": "local-research",
                "legal_basis": "research",
                "allowed_purposes": ["agent_training", "evaluation"],
                "sensitivity": "synthetic_benchmark",
                "jurisdiction": "local",
                "retention_expires_at": "2030-01-01T00:00:00+00:00",
                "license": "Tau benchmark training-side source; publication review required",
                "provenance": {
                    "source": "tau3_reference_action_replay",
                    "source_revision": str(request["expected_revision"]),
                    "official_split": split,
                },
                "deletion_subject_ids": [f"{domain}:{task.id}"],
            },
        }
        changes = [] if before_hash == after_hash else [
            {
                "path": "$.tau_environment",
                "before": before_hash,
                "after": after_hash,
                "kind": "changed",
            }
        ]
        variants = _behavior_variants(
            base=base,
            replay_events=events,
            before_hash=before_hash,
            after_hash=after_hash,
            changes=changes,
            action_count=len(actions),
            revision=str(request["expected_revision"]),
        )
        captures.extend(variants)
        selected_counts[split][domain] += 1
        selected_sources.append({
            "split": split,
            "domain": domain,
            "task_family": str(row["task_family"]),
            "task_sha256": str(row["task_sha256"]),
        })
    missing = {
        f"{split}:{domain}": int(quotas[split][domain]) - selected_counts[split][domain]
        for split in ("train", "development")
        for domain in DOMAINS
        if selected_counts[split][domain] < int(quotas[split][domain])
    }
    if missing:
        raise Tau3CaptureGenerationError(
            "tool execution failed; insufficient replayable sources for quotas: " + json.dumps(missing, sort_keys=True)
        )
    return {
        "schema_version": CAPTURE_GENERATOR_SCHEMA_VERSION,
        "captures": sorted(captures, key=lambda item: item["trajectory_id"]),
        "runtime": {"python": sys.version.split()[0], "platform": platform.platform()},
        "tool_schemas_recorded_exact": True,
        "source_rejections": source_rejections,
        "selected_sources": selected_sources,
    }


def _message_pair(tool_call: Any, response: Any) -> list[Any]:
    from tau2.data_model.message import AssistantMessage

    return [AssistantMessage(role="assistant", content=None, tool_calls=[tool_call]), response]


def _behavior_variants(
    *,
    base: dict[str, Any],
    replay_events: list[dict[str, Any]],
    before_hash: str,
    after_hash: str,
    changes: list[dict[str, Any]],
    action_count: int,
    revision: str,
) -> list[dict[str, Any]]:
    variants: list[dict[str, Any]] = []
    positive_reason = (
        "Official training-side reference actions replayed without tool errors."
        if action_count
        else "Official training-side task has no reference tool action; recorded as executable clarification/refusal evidence."
    )
    variants.append(
        _capture_variant(
            base,
            behavior="success",
            events=replay_events,
            before_hash=before_hash,
            after_hash=after_hash,
            changes=changes,
            success=True,
            policy_violation=False,
            harmful_mutation=False,
            disposition="admit",
            reason=positive_reason,
            executable_label="reference_actions_replayed_without_tool_error" if action_count else "no_reference_action_required",
            evidence_refs=_base_evidence_refs(base, revision, [f"tool_call_count:{action_count}", "generation_method:reference_action_replay"]),
        )
    )
    clarification_success = bool(base.get("no_action_supported"))
    clarification_text = _grounded_clarification_text(base) if clarification_success else "Rejected: this task has official reference actions, so generic clarification/refusal is not sufficient."
    variants.append(
        _capture_variant(
            base,
            behavior="clarification_refusal",
            events=[_user_event_from_base(base), _assistant_text_event(clarification_text)],
            before_hash=before_hash,
            after_hash=before_hash,
            changes=[],
            success=clarification_success,
            policy_violation=False,
            harmful_mutation=False,
            disposition="admit" if clarification_success else "reject",
            reason=(
                "Clarification/refusal is grounded in official no-action communicate/NL criteria."
                if clarification_success
                else "Generic clarification/refusal rejected because official criteria require tool actions."
            ),
            executable_label="grounded_no_action_clarification" if clarification_success else "action_required_clarification_rejected",
            evidence_refs=_base_evidence_refs(base, revision, ["clarification_refusal:no_mutation"]),
        )
    )
    variants.append(
        _capture_variant(
            base,
            behavior="correction",
            events=[*replay_events, _assistant_text_event("Correction: use the recorded tool evidence and Tau state hashes before making a final claim.")],
            before_hash=before_hash,
            after_hash=after_hash,
            changes=changes,
            success=True,
            policy_violation=False,
            harmful_mutation=False,
            disposition="admit",
            reason="Grounded correction preserves the executable reference state evidence.",
            executable_label="evidence_grounded_correction",
            evidence_refs=_base_evidence_refs(base, revision, ["correction:evidence_grounded", "generation_method:deterministic_correction"]),
        )
    )
    invalid_pair = _hallucinated_event_pair(base)
    variants.append(
        _capture_variant(
            base,
            behavior="recovery",
            events=[_user_event_from_base(base), *invalid_pair, _assistant_text_event("I cannot use that invalid tool; I will recover using the recorded valid trajectory."), *replay_events[1:]],
            before_hash=before_hash,
            after_hash=after_hash,
            changes=changes,
            success=True,
            policy_violation=False,
            harmful_mutation=False,
            disposition="admit",
            reason="Invalid tool attempt is recovered before final state evidence is accepted.",
            executable_label="recovered_after_invalid_tool",
            evidence_refs=_base_evidence_refs(base, revision, ["recovery:invalid_tool_then_reference"]),
        )
    )
    for behavior, label, policy_violation, harmful_mutation, events in (
        ("policy_failure", "policy_violation_detected", True, False, _assistant_text_event("I will ignore the policy and proceed without required checks.")),
        ("harmful_mutation", "harmful_mutation_blocked_before_execution", False, True, _assistant_text_event("Blocked unsafe mutation request; no Tau mutation was executed for this rejected variant.")),
        ("hallucinated_tool", "hallucinated_tool_rejected", False, False, _hallucinated_event_pair(base)),
        ("premature_completion", "premature_completion_rejected", False, False, _assistant_text_event("Done.")),
    ):
        event_list = events if isinstance(events, list) else [events]
        variants.append(
            _capture_variant(
                base,
                behavior=behavior,
                events=[_user_event_from_base(base), *event_list],
                before_hash=before_hash,
                after_hash=before_hash,
                changes=[],
                success=False,
                policy_violation=policy_violation,
                harmful_mutation=harmful_mutation,
                disposition="reject",
                reason=f"Deterministic rejected {behavior} variant; no harmful Tau mutation executed.",
                executable_label=label,
                evidence_refs=_base_evidence_refs(base, revision, [f"rejected_behavior:{behavior}", f"programmatic_negative_rule:{behavior}"]),
            )
        )
    return variants


def _capture_variant(
    base: dict[str, Any],
    *,
    behavior: str,
    events: list[dict[str, Any]],
    before_hash: str,
    after_hash: str,
    changes: list[dict[str, Any]],
    success: bool,
    policy_violation: bool,
    harmful_mutation: bool,
    disposition: str,
    reason: str,
    executable_label: str,
    evidence_refs: list[str],
) -> dict[str, Any]:
    task_key = _sha256_text(f"{base['split']}:{base['domain']}:{base['task_id']}:{behavior}")[:16]
    row = {
        "schema_version": TAU3_CAPTURE_SCHEMA_VERSION,
        "trajectory_id": f"tau3-{base['split']}-{base['domain']}-{behavior}-{task_key}",
        "behavior": behavior,
        "prompt_hash": canonical_sha256(base["prompt"]),
        "events": events,
        "final_answer": str(events[-1].get("content") or events[-1].get("text") or ""),
        "state_transition": {
            "before_hash": before_hash,
            "after_hash": after_hash,
            "changes": changes,
            "executable": True,
        },
        "outcome": {
            "success": success,
            "executable_label": executable_label,
            "policy_violation": policy_violation,
            "harmful_mutation": harmful_mutation,
            "evidence_refs": [*evidence_refs, f"before_hash:{before_hash}", f"after_hash:{after_hash}"],
        },
        "review": {
            "reviewer": "hfr-tau3-capture-generator",
            "verifier": "tau-runtime-reference-action-replay",
            "disposition": disposition,
            "reason": reason,
        },
        **base,
    }
    row["token_count"] = _capture_token_count(row)
    return row


def _assistant_text_event(text: str) -> dict[str, Any]:
    return {"type": "assistant_message", "role": "assistant", "content": text, "text": text, "status": "ok"}


def _capture_token_count(capture: dict[str, Any]) -> int:
    text = str(capture.get("prompt") or "")
    for event in capture.get("events", []):
        if isinstance(event, dict):
            text += " " + str(event.get("content") or event.get("text") or "")
            if isinstance(event.get("args"), dict):
                text += " " + json.dumps(event["args"], sort_keys=True)
            if "result" in event:
                text += " " + json.dumps(event["result"], sort_keys=True, default=str)
    return max(1, len(text.split()))


def _base_evidence_refs(base: dict[str, Any], revision: str, extra: list[str]) -> list[str]:
    policy = base.get("policy_evidence") if isinstance(base.get("policy_evidence"), dict) else {}
    basis = policy.get("official_reference_basis") if isinstance(policy.get("official_reference_basis"), dict) else {}
    return [
        f"tau_revision:{revision}",
        f"task:{base['domain']}:{base['task_id']}",
        f"policy_hash:{base['policy_hash']}",
        f"official_action_count:{basis.get('action_count', 'unknown')}",
        f"official_no_action_supported:{basis.get('no_action_supported', False)}",
        *extra,
    ]


def _grounded_clarification_text(base: dict[str, Any]) -> str:
    communicate = [str(item) for item in base.get("communicate_info", []) if str(item)]
    assertions = [str(item) for item in base.get("nl_assertions", []) if str(item)]
    material = communicate or assertions
    if not material:
        return "I cannot proceed with a tool action because the official task criteria require communication only."
    return " ".join(material)


def _user_event_from_base(base: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "user_message",
        "role": "user",
        "content": base["prompt"],
        "text": base["prompt"],
        "status": "ok",
        "attributes": {"domain": base["domain"], "source": "tau_prompt"},
    }


def _hallucinated_event_pair(base: dict[str, Any]) -> list[dict[str, Any]]:
    call_id = f"invalid-{_sha256_text(base['task_id'])[:12]}"
    return [
        {
            "type": "tool_call",
            "role": "assistant",
            "tool_name": "invented_tau_tool",
            "tool_call_id": call_id,
            "args": {},
            "content": "",
            "status": "error",
            "side_effect_status": "requested",
        },
        {
            "type": "tool_result",
            "role": "tool",
            "tool_name": "invented_tau_tool",
            "tool_call_id": call_id,
            "args": {},
            "result": "invalid tool rejected before Tau mutation",
            "content": "invalid tool rejected before Tau mutation",
            "status": "error",
            "side_effect_status": "failed",
        },
    ]


def _combined_state_hash(env: Any) -> str:
    value = {
        "agent_db_hash": env.get_db_hash(),
        "user_db_hash": env.get_user_db_hash(),
    }
    return _hash_json(value)


def _tool_definitions(env: Any) -> list[dict[str, Any]]:
    info = env.get_info(include_tool_info=True)
    dumped = info.model_dump(mode="json") if hasattr(info, "model_dump") else dict(info)
    raw_defs = dumped.get("tool_defs") or {}
    tools = []
    for name in sorted(raw_defs):
        raw_definition = raw_defs[name]
        if isinstance(raw_definition, dict):
            parameters = raw_definition.get("parameters") or raw_definition.get("params") or {"type": "object"}
            description = raw_definition.get("doc") or raw_definition.get("description") or ""
        else:
            parameters = {"type": "object"}
            description = ""
        definition = {
            "type": "function",
            "function": {
                "name": name,
                "version": "tau3-recorded-tool-schema-v1",
                "description": str(description),
                "parameters": parameters if isinstance(parameters, dict) else {"type": "object"},
            },
        }
        tools.append(
            {
                **definition,
                "name": name,
                "version": "tau3-recorded-tool-schema-v1",
            }
        )
    if not tools:
        raise Tau3CaptureGenerationError("Tau environment exposed no assistant tools")
    return tools


def _user_event(task: Any, domain: str) -> dict[str, Any]:
    content = str(task.user_scenario)
    return {
        "type": "user_message",
        "role": "user",
        "content": content,
        "text": content,
        "status": "ok",
        "attributes": {"domain": domain, "source": "tau_user_scenario"},
    }


def _prompt(task: Any, policy: str, tools: list[dict[str, Any]]) -> str:
    tool_names = ", ".join(tool["name"] for tool in tools)
    return (
        "You are the Tau text-mode customer support agent.\n\n"
        f"Policy:\n{policy}\n\n"
        f"Available tools: {tool_names}\n\n"
        f"User scenario:\n{task.user_scenario}\n"
    )


def _task_family(domain: str, task: dict[str, Any]) -> str:
    if domain == "telecom":
        task_id = str(task.get("id", ""))
        match = re.match(r"^\[([^\]]+)\]", task_id)
        if match:
            return f"telecom:{match.group(1)}"
    criteria = task.get("evaluation_criteria") if isinstance(task.get("evaluation_criteria"), dict) else {}
    actions = criteria.get("actions") if isinstance(criteria.get("actions"), list) else []
    action_names = [str(action.get("name")) for action in actions if isinstance(action, dict) and action.get("name")]
    return f"{domain}:{_hash_json(action_names)[:16]}"


def _maybe_json(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _parse_domain_quotas(value: str) -> dict[str, int]:
    quotas: dict[str, int] = {}
    for part in value.split(","):
        if not part.strip():
            continue
        if "=" not in part:
            raise Tau3CaptureGenerationError("domain quotas must use domain=count entries")
        domain, raw_count = part.split("=", 1)
        domain = domain.strip()
        try:
            count = int(raw_count.strip())
        except ValueError as exc:
            raise Tau3CaptureGenerationError("domain quota counts must be integers") from exc
        quotas[domain] = count
    return quotas


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate canonical Tau-3 Flight Recorder training captures")
    parser.add_argument("--_tau-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--tau-repo", type=Path, required=False)
    parser.add_argument("--expected-revision", required=False)
    parser.add_argument("--train-tasks", type=Path, required=False)
    parser.add_argument("--development-tasks", type=Path, required=False)
    parser.add_argument("--out", type=Path, required=False)
    parser.add_argument("--tau-python", type=Path)
    parser.add_argument("--generator-id", default="tau3-reference-action-replay")
    parser.add_argument("--generator-revision")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-domain-quotas", default="airline=8,retail=8,telecom=3")
    parser.add_argument("--development-domain-quotas", default="airline=2,retail=2,telecom=1")
    parser.add_argument("--sample-salt", default="hfr-tau3-capture-generation-v1")
    args = parser.parse_args(argv)
    if args._tau_worker:
        return _worker_main()
    try:
        missing = [
            name
            for name in ("tau_repo", "expected_revision", "train_tasks", "development_tasks", "out")
            if getattr(args, name) is None
        ]
        if missing:
            raise Tau3CaptureGenerationError("missing required argument(s): " + ", ".join("--" + item.replace("_", "-") for item in missing))
        summary = generate_tau3_training_captures(
            tau_repo=args.tau_repo,
            expected_revision=args.expected_revision,
            train_tasks=args.train_tasks,
            development_tasks=args.development_tasks,
            out=args.out,
            tau_python=args.tau_python,
            generator_id=args.generator_id,
            generator_revision=args.generator_revision,
            seed=args.seed,
            train_domain_quotas=_parse_domain_quotas(args.train_domain_quotas),
            development_domain_quotas=_parse_domain_quotas(args.development_domain_quotas),
            sample_salt=args.sample_salt,
        )
    except (OSError, Tau3CaptureGenerationError, ValueError) as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 2
    sys.stdout.write(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through CLI tests.
    raise SystemExit(_main())
