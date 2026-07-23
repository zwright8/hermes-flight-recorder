"""Build source-bound Tau-3 retry inputs from governed generation receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from .path_safety import path_has_symlink_component
from .schema_registry import check_schema_contract
from .tau3_capture import canonical_sha256
from .tau3_conversation_ingest import (
    Tau3ConversationIngestError,
    _reject_error_markers,
    _training_exclusion_for_result,
)

TAU3_GENERATION_RETRY_SOURCE_SCHEMA_VERSION = "hfr.tau3_generation_retry_source.v1"
TAU3_TEACHER_GENERATION_RUN_SCHEMA_VERSION = "hfr.tau3_teacher_generation_run.v1"
TRAINING_SOURCE_SCHEMA_VERSION = "hfr.tau3_training_source.v1"
ALLOWED_DOMAINS = {"airline", "retail", "telecom"}
TRAINING_SPLITS = {"train", "development"}
NORMAL_TERMINATIONS = {"agent_stop", "user_stop"}


class Tau3GenerationRetryError(ValueError):
    """Raised when retry-source selection would trust unsafe evidence."""


@dataclass(frozen=True)
class RetrySourceSummary:
    """Paths and manifest for retry-source callers."""

    manifest: dict[str, Any]
    jsonl_path: Path
    manifest_path: Path


@dataclass(frozen=True)
class SourceRow:
    row: dict[str, Any]
    path: Path
    line_number: int
    domain: str
    split: str
    task_id: str
    task_sha256: str
    prompt_sha256: str
    source_revision: str
    row_sha256: str
    task_identity_sha256: str


def build_tau3_generation_retry_source(
    *,
    source_jsonl_paths: Sequence[str | Path],
    generation_manifest_paths: Sequence[str | Path],
    out_jsonl: str | Path,
    manifest_path: str | Path | None = None,
    domains: Sequence[str] | None = None,
    expected_tau_revision: str | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Emit canonical source rows that still lack reward-1 normal Tau results."""

    if not source_jsonl_paths:
        raise Tau3GenerationRetryError("at least one --source-jsonl is required")
    if not generation_manifest_paths:
        raise Tau3GenerationRetryError("at least one --generation-manifest is required")
    requested_domains = _normalize_domains(domains)
    out_path = Path(out_jsonl)
    manifest_out = Path(manifest_path) if manifest_path is not None else out_path.with_suffix(out_path.suffix + ".manifest.json")
    _require_new_file(out_path)
    _require_new_file(manifest_out)
    source_rows = _load_source_rows([Path(path) for path in source_jsonl_paths])
    source_revisions = {row.source_revision for row in source_rows}
    if len(source_revisions) != 1:
        raise Tau3GenerationRetryError("source JSONL rows must bind exactly one Tau revision")
    source_revision = next(iter(source_revisions))
    if expected_tau_revision is not None and source_revision != expected_tau_revision:
        raise Tau3GenerationRetryError("expected Tau revision does not match source JSONL")

    source_by_identity = {(row.domain, row.task_id): row for row in source_rows}
    manifest_records, successes_by_identity = _replay_generation_manifests(
        [Path(path) for path in generation_manifest_paths],
        source_by_identity=source_by_identity,
        expected_tau_revision=source_revision,
    )
    eligible_rows = [
        row for row in source_rows if requested_domains is None or row.domain in requested_domains
    ]
    selected = [row for row in eligible_rows if (row.domain, row.task_id) not in successes_by_identity]
    covered_success_count = sum(
        1 for row in eligible_rows if (row.domain, row.task_id) in successes_by_identity
    )

    _write_jsonl_private(out_path, [row.row for row in selected])
    source_records = [_file_record(path) for path in [Path(path) for path in source_jsonl_paths]]
    selected_identities = [
        {
            "domain": row.domain,
            "split": row.split,
            "task_id_sha256": canonical_sha256(row.task_id),
            "task_identity_sha256": row.task_identity_sha256,
            "task_sha256": row.task_sha256,
            "prompt_sha256": row.prompt_sha256,
            "row_sha256": row.row_sha256,
        }
        for row in selected
    ]
    manifest = {
        "schema_version": TAU3_GENERATION_RETRY_SOURCE_SCHEMA_VERSION,
        "created_at": created_at or _now_utc(),
        "tau_revision": source_revision,
        "source_jsonl_files": source_records,
        "generation_manifests": manifest_records,
        "domain_filter": sorted(requested_domains) if requested_domains is not None else None,
        "source_task_count": len(eligible_rows),
        "covered_success_task_count": covered_success_count,
        "selected_task_count": len(selected),
        "selected_tasks": selected_identities,
        "selected_task_identity_sha256": canonical_sha256(selected_identities),
        "output_jsonl": {
            "path": out_path.name,
            "size_bytes": out_path.stat().st_size,
            "sha256": _sha256(out_path),
        },
        "sealed_rows": 0,
        "test_rows": 0,
        "sealed_payload_accessed": False,
        "training_started": False,
    }
    check = check_schema_contract(manifest, name_or_id="tau3_generation_retry_source")
    if check.get("passed") is not True:
        raise Tau3GenerationRetryError("retry manifest schema failed: " + "; ".join(check.get("errors", [])))
    _write_json_private(manifest_out, manifest)
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-jsonl", action="append", type=Path, required=True)
    parser.add_argument("--generation-manifest", action="append", type=Path, required=True)
    parser.add_argument("--out-jsonl", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--domain", action="append", choices=sorted(ALLOWED_DOMAINS))
    parser.add_argument("--expected-tau-revision")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        manifest = build_tau3_generation_retry_source(
            source_jsonl_paths=args.source_jsonl,
            generation_manifest_paths=args.generation_manifest,
            out_jsonl=args.out_jsonl,
            manifest_path=args.manifest,
            domains=args.domain,
            expected_tau_revision=args.expected_tau_revision,
        )
    except (OSError, Tau3GenerationRetryError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps({"manifest": str(args.manifest or args.out_jsonl.with_suffix(args.out_jsonl.suffix + ".manifest.json")), "selected_task_count": manifest["selected_task_count"]}, sort_keys=True))
    return 0


def _load_source_rows(paths: list[Path]) -> list[SourceRow]:
    rows: list[SourceRow] = []
    seen: set[tuple[str, str]] = set()
    for path in paths:
        _reject_symlink(path, f"{path}: source JSONL")
        if not path.is_file():
            raise Tau3GenerationRetryError(f"source JSONL does not exist: {path}")
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            row = _object(json.loads(line), f"{path}:{line_number}")
            if row.get("schema_version") != TRAINING_SOURCE_SCHEMA_VERSION:
                raise Tau3GenerationRetryError(f"{path}:{line_number}: unexpected schema_version")
            domain = _nonempty_str(row.get("domain"), f"{path}:{line_number}.domain")
            if domain not in ALLOWED_DOMAINS:
                raise Tau3GenerationRetryError(f"{path}:{line_number}: unsupported domain {domain!r}")
            split = _nonempty_str(row.get("split"), f"{path}:{line_number}.split")
            if split not in TRAINING_SPLITS:
                raise Tau3GenerationRetryError(f"{path}:{line_number}: sealed/test split rejected")
            task = _object(row.get("task"), f"{path}:{line_number}.task")
            task_id = _nonempty_str(task.get("id"), f"{path}:{line_number}.task.id")
            key = (domain, task_id)
            if key in seen:
                raise Tau3GenerationRetryError(f"duplicate source task: {domain}/{task_id}")
            seen.add(key)
            task_sha256 = _sha256_string(row.get("task_sha256"), f"{path}:{line_number}.task_sha256")
            if canonical_sha256(task) != task_sha256:
                raise Tau3GenerationRetryError(f"{path}:{line_number}: task_sha256 does not replay task payload")
            rows.append(
                SourceRow(
                    row=row,
                    path=path,
                    line_number=line_number,
                    domain=domain,
                    split=split,
                    task_id=task_id,
                    task_sha256=task_sha256,
                    prompt_sha256=_sha256_string(row.get("prompt_sha256"), f"{path}:{line_number}.prompt_sha256"),
                    source_revision=_sha256_string(row.get("source_revision"), f"{path}:{line_number}.source_revision", length=40),
                    row_sha256=canonical_sha256(row),
                    task_identity_sha256=canonical_sha256({"domain": domain, "task_id": task_id}),
                )
            )
    if not rows:
        raise Tau3GenerationRetryError("source JSONL contains no eligible train/development tasks")
    return rows


def _replay_generation_manifests(
    paths: list[Path],
    *,
    source_by_identity: dict[tuple[str, str], SourceRow],
    expected_tau_revision: str,
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    successes: dict[tuple[str, str], dict[str, Any]] = {}
    for manifest_path in paths:
        _reject_symlink(manifest_path, f"{manifest_path}: generation manifest")
        manifest = _object(_read_json(manifest_path), f"{manifest_path}: generation manifest")
        check = check_schema_contract(manifest, name_or_id="tau3_teacher_generation_run")
        if check.get("passed") is not True:
            raise Tau3GenerationRetryError(
                f"{manifest_path}: generation manifest schema failed: " + "; ".join(check.get("errors", []))
            )
        if manifest.get("phase") != "final":
            raise Tau3GenerationRetryError(f"{manifest_path}: generation manifest phase must be final")
        if manifest.get("tau_revision") != expected_tau_revision:
            raise Tau3GenerationRetryError(f"{manifest_path}: Tau revision mismatch")
        if manifest.get("sealed_rows") != 0 or manifest.get("test_rows") != 0 or manifest.get("sealed_payload_accessed") is not False:
            raise Tau3GenerationRetryError(f"{manifest_path}: sealed/test generation evidence rejected")
        manifest_dir = manifest_path.parent
        _replay_manifest_file_record(manifest_dir, manifest.get("source"), f"{manifest_path}: source")
        _replay_manifest_file_record(manifest_dir, manifest.get("protocol"), f"{manifest_path}: protocol")
        _replay_manifest_file_record(
            manifest_dir,
            manifest.get("prelaunch_receipt"),
            f"{manifest_path}: prelaunch_receipt",
        )
        receipt_refs = _list(manifest.get("task_receipts"), f"{manifest_path}: task_receipts")
        if manifest.get("task_count") != len(receipt_refs):
            raise Tau3GenerationRetryError(f"{manifest_path}: task_count does not match task receipts")
        success_count = 0
        failure_count = 0
        result_counts = {
            "reward_1_normal": 0,
            "training_admitted": 0,
            "training_excluded": 0,
            "non_success": 0,
        }
        for index, raw_ref in enumerate(receipt_refs):
            ref = _object(raw_ref, f"{manifest_path}: task_receipts[{index}]")
            receipt_path = _resolve_child_ref(manifest_dir, ref.get("path"), f"{manifest_path}: task_receipts[{index}].path")
            receipt = _object(_read_json(receipt_path), f"{receipt_path}: receipt")
            if receipt.get("phase") != "task":
                raise Tau3GenerationRetryError(f"{receipt_path}: receipt phase must be task")
            status = _nonempty_str(ref.get("terminal_status"), f"{manifest_path}: task_receipts[{index}].terminal_status")
            if receipt.get("terminal_status") != status:
                raise Tau3GenerationRetryError(f"{receipt_path}: task receipt status mismatch")
            if receipt.get("result_sha256") != ref.get("result_sha256"):
                raise Tau3GenerationRetryError(f"{receipt_path}: task receipt result hash mismatch")
            if status == "success":
                success_count += 1
            else:
                failure_count += 1
            task = _object(receipt.get("task"), f"{receipt_path}: task")
            domain = _nonempty_str(task.get("domain"), f"{receipt_path}: task.domain")
            task_id = _nonempty_str(task.get("task_id"), f"{receipt_path}: task.task_id")
            identity = (domain, task_id)
            source = source_by_identity.get(identity)
            if source is None:
                raise Tau3GenerationRetryError(f"{receipt_path}: task is not in canonical source JSONL")
            _require_receipt_matches_source(receipt_path, task, source)
            result_info = _replay_result(
                receipt,
                receipt_path,
                expected_tau_revision=expected_tau_revision,
                expected_domain=domain,
                expected_task_id=task_id,
            )
            expected_status = _expected_terminal_status(receipt, result_info["reward"], receipt_path)
            if status != expected_status:
                raise Tau3GenerationRetryError(
                    f"{receipt_path}: terminal status does not replay generator contract"
                )
            normal_success = result_info["reward"] == 1.0 and result_info["termination_reason"] in NORMAL_TERMINATIONS
            if normal_success:
                result_counts["reward_1_normal"] += 1
                if status != "success":
                    raise Tau3GenerationRetryError(f"{receipt_path}: reward-1 normal result contradicts non-success status")
            else:
                result_counts["non_success"] += 1
            if status == "success" and not normal_success:
                raise Tau3GenerationRetryError(f"{receipt_path}: successful receipt lacks reward-1 normal result evidence")
            training_exclusion = result_info["training_exclusion"] if normal_success else None
            training_admitted = normal_success and training_exclusion is None
            if normal_success and training_exclusion is not None:
                result_counts["training_excluded"] += 1
            if training_admitted:
                result_counts["training_admitted"] += 1
                existing = successes.get(identity)
                if existing is not None and existing["result_sha256"] != result_info["sha256"]:
                    raise Tau3GenerationRetryError(f"{receipt_path}: conflicting successful result for {domain}/{task_id}")
                successes[identity] = {
                    "manifest_path": str(manifest_path),
                    "receipt_path": str(receipt_path),
                    "result_sha256": result_info["sha256"],
                    "termination_reason": result_info["termination_reason"],
                    "reward": result_info["reward"],
                }
        if manifest.get("success_count") != success_count or manifest.get("failure_count") != failure_count:
            raise Tau3GenerationRetryError(f"{manifest_path}: success/failure counts do not replay")
        records.append(
            {
                "path": str(manifest_path),
                "size_bytes": manifest_path.stat().st_size,
                "sha256": _sha256(manifest_path),
                "task_receipt_count": len(receipt_refs),
                "success_count": success_count,
                "failure_count": failure_count,
                "reward_1_normal_result_count": result_counts["reward_1_normal"],
                "training_admitted_success_count": result_counts["training_admitted"],
                "training_excluded_success_count": result_counts["training_excluded"],
                "non_success_result_count": result_counts["non_success"],
            }
        )
    return records, successes


def _require_receipt_matches_source(receipt_path: Path, task: dict[str, Any], source: SourceRow) -> None:
    checks = {
        "domain": source.domain,
        "split": source.split,
        "task_id": source.task_id,
        "task_family": str(source.row.get("task_family") or ""),
        "task_sha256": source.task_sha256,
        "prompt_sha256": source.prompt_sha256,
    }
    for key, expected in checks.items():
        if task.get(key) != expected:
            raise Tau3GenerationRetryError(f"{receipt_path}: task {key} does not match source row")


def _replay_result(
    receipt: dict[str, Any],
    receipt_path: Path,
    *,
    expected_tau_revision: str,
    expected_domain: str,
    expected_task_id: str,
) -> dict[str, Any]:
    result_sha = receipt.get("result_sha256")
    if result_sha is None:
        if receipt.get("reward") is not None:
            raise Tau3GenerationRetryError(f"{receipt_path}: reward must be null when result is absent")
        return {
            "sha256": None,
            "reward": None,
            "termination_reason": None,
            "training_exclusion": None,
        }
    result_sha = _sha256_string(result_sha, f"{receipt_path}: result_sha256")
    result_path = Path(_nonempty_str(receipt.get("result_path"), f"{receipt_path}: result_path"))
    _reject_symlink(result_path, f"{receipt_path}: result_path")
    if not result_path.is_file() or _sha256(result_path) != result_sha:
        raise Tau3GenerationRetryError(f"{receipt_path}: generated result hash mismatch")
    result = _object(_read_json(result_path), f"{result_path}: result")
    info = _object(result.get("info"), f"{result_path}: info")
    if info.get("git_commit") != expected_tau_revision:
        raise Tau3GenerationRetryError(f"{result_path}: Tau revision does not match receipt/source")
    environment = _object(info.get("environment_info"), f"{result_path}: info.environment_info")
    if environment.get("domain_name") != expected_domain:
        raise Tau3GenerationRetryError(f"{result_path}: result domain does not match receipt/source")
    simulations = _list(result.get("simulations"), f"{result_path}: simulations")
    if len(simulations) != 1:
        raise Tau3GenerationRetryError(f"{result_path}: generated result must contain exactly one simulation")
    simulation = _object(simulations[0], f"{result_path}: simulations[0]")
    if simulation.get("task_id") != expected_task_id:
        raise Tau3GenerationRetryError(f"{result_path}: simulation task_id does not match receipt/source")
    reward_info = _object(simulation.get("reward_info"), f"{result_path}: simulations[0].reward_info")
    reward = reward_info.get("reward")
    if isinstance(reward, bool) or not isinstance(reward, (int, float)):
        raise Tau3GenerationRetryError(f"{result_path}: reward must be numeric")
    receipt_reward = receipt.get("reward")
    if isinstance(receipt_reward, bool) or not isinstance(receipt_reward, (int, float)):
        raise Tau3GenerationRetryError(f"{receipt_path}: receipt reward must be numeric")
    if float(receipt_reward) != float(reward):
        raise Tau3GenerationRetryError(f"{receipt_path}: receipt reward does not match result reward")
    termination = str(simulation.get("termination_reason") or "")
    exclusion = _training_exclusion(simulation, result_path)
    return {
        "sha256": result_sha,
        "reward": float(reward),
        "termination_reason": termination,
        "training_exclusion": exclusion,
    }


def _training_exclusion(simulation: dict[str, Any], result_path: Path) -> dict[str, str] | None:
    simulation_id = str(simulation.get("id") or "simulation-0")
    try:
        _reject_error_markers(simulation, result_path, simulation_id)
    except Tau3ConversationIngestError as exc:
        return {"code": "simulation_error_marker", "reason": str(exc)}
    try:
        exclusion = _training_exclusion_for_result(simulation, result_path)
    except Tau3ConversationIngestError as exc:
        raise Tau3GenerationRetryError(f"{result_path}: could not replay training admission: {exc}") from exc
    if exclusion is None:
        return None
    return {"code": exclusion[0], "reason": exclusion[1]}


def _expected_terminal_status(receipt: dict[str, Any], reward: Any, receipt_path: Path) -> str:
    timed_out = receipt.get("timed_out")
    if not isinstance(timed_out, bool):
        raise Tau3GenerationRetryError(f"{receipt_path}: timed_out must be boolean")
    exit_code = receipt.get("exit_code")
    if exit_code is not None and (isinstance(exit_code, bool) or not isinstance(exit_code, int)):
        raise Tau3GenerationRetryError(f"{receipt_path}: exit_code must be integer or null")
    if timed_out:
        if exit_code is not None:
            raise Tau3GenerationRetryError(f"{receipt_path}: timed-out receipt must have null exit_code")
        return "timeout"
    if exit_code == 0 and reward == 1.0:
        return "success"
    return "failed"


def _replay_manifest_file_record(base: Path, raw_record: Any, where: str) -> None:
    record = _object(raw_record, where)
    raw_path = _nonempty_str(record.get("path"), f"{where}.path")
    path = Path(raw_path)
    if any(part == ".." for part in path.parts):
        raise Tau3GenerationRetryError(f"{where}.path must not traverse parent directories")
    if not path.is_absolute():
        cwd_path = Path.cwd() / path
        path = cwd_path if cwd_path.is_file() else base / path
    _reject_symlink(path, f"{where}.path")
    if not path.is_file():
        raise Tau3GenerationRetryError(f"{where}: referenced file is missing")
    expected_sha = _sha256_string(record.get("sha256"), f"{where}.sha256")
    expected_size = record.get("size")
    if isinstance(expected_size, bool) or not isinstance(expected_size, int) or expected_size < 0:
        raise Tau3GenerationRetryError(f"{where}.size must be a non-negative integer")
    if path.stat().st_size != expected_size or _sha256(path) != expected_sha:
        raise Tau3GenerationRetryError(f"{where}: referenced file hash/size mismatch")


def _normalize_domains(domains: Sequence[str] | None) -> set[str] | None:
    if not domains:
        return None
    selected = {str(domain) for domain in domains}
    unsupported = selected - ALLOWED_DOMAINS
    if unsupported:
        raise Tau3GenerationRetryError(f"unsupported domain filter: {', '.join(sorted(unsupported))}")
    return selected


def _resolve_child_ref(base: Path, value: Any, where: str) -> Path:
    text = _nonempty_str(value, where)
    path = Path(text)
    if path.is_absolute() or any(part == ".." for part in path.parts):
        raise Tau3GenerationRetryError(f"{where} must be a relative path inside the generation directory")
    candidate = base / path
    _reject_symlink(candidate, where)
    resolved_base = base.resolve(strict=True)
    resolved_path = candidate.resolve(strict=False)
    if resolved_base not in resolved_path.parents:
        raise Tau3GenerationRetryError(f"{where} must not escape the generation directory")
    return resolved_path


def _require_new_file(path: Path) -> None:
    if path_has_symlink_component(path.parent, include_leaf=True):
        raise Tau3GenerationRetryError(f"output parent must not contain symlink components: {path.parent}")
    if path.exists():
        raise Tau3GenerationRetryError(f"refusing to overwrite existing file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)


def _write_jsonl_private(path: Path, rows: list[dict[str, Any]]) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(path, flags, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n")
    os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)


def _write_json_private(path: Path, payload: dict[str, Any]) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(path, flags, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2, ensure_ascii=False)
        handle.write("\n")
    os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)


def _file_record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": _sha256(path)}


def _reject_symlink(path: Path, where: str) -> None:
    if path_has_symlink_component(path, include_leaf=True):
        raise Tau3GenerationRetryError(f"{where}: symlink component rejected")


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_string(value: Any, where: str, *, length: int = 64) -> str:
    text = _nonempty_str(value, where)
    if len(text) != length or any(char not in "0123456789abcdef" for char in text):
        raise Tau3GenerationRetryError(f"{where} must be a {length}-character lowercase hex string")
    return text


def _object(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise Tau3GenerationRetryError(f"{where} must be an object")
    return value


def _list(value: Any, where: str) -> list[Any]:
    if not isinstance(value, list):
        raise Tau3GenerationRetryError(f"{where} must be a list")
    return value


def _nonempty_str(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise Tau3GenerationRetryError(f"{where} must be a non-empty string")
    return value


def _now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
