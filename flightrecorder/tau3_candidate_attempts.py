"""Governed Tau-3 candidate attempt wrapper and public-safe ledger."""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import math
import os
import re
import secrets
import signal
import stat
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .path_safety import path_has_symlink_component
from .repeated_eval import canonical_sha256
from .schema_registry import check_schema_contract

try:
    import fcntl
except ImportError:  # pragma: no cover - governed MLX attempts are POSIX-only
    fcntl = None  # type: ignore[assignment]

TAU3_CANDIDATE_ATTEMPT_LEDGER_SCHEMA_VERSION = "hfr.tau3_candidate_attempt_ledger.v1"
TAU3_CANDIDATE_ATTEMPT_OUTCOME_SCHEMA_VERSION = "hfr.tau3_candidate_attempt_outcome.v1"
CAMPAIGN_MARKER = ".hfr_tau3_candidate_attempt_campaign"
SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$")
RECOVERY_OUTCOME_RE = re.compile(r"^attempt_outcome\.recovery-(\d{4})\.json$")
SEALED_TEST_RE = re.compile(r"(?:^|[/_.-])(?:sealed|test)(?:$|[/_.-])", re.IGNORECASE)
PATH_ARG_NAMES = {
    "--bundle",
    "--mixture-dir",
    "--protocol",
    "--model-identity",
    "--model-path",
    "--resume-receipt",
    "--resume-adapter-file",
}
ATTEMPT_STATUSES = (
    "completed",
    "failed",
    "timeout",
    "interrupted",
    "missing-receipt",
    "malformed-receipt",
)
FAILURE_REASONS = {
    "malformed_intent",
    "malformed_outcome",
    "malformed_receipt",
    "missing_intent",
    "missing_outcome",
    "missing_receipt",
    "receipt_not_successful",
    "receipt_parse_error",
    "receipt_reference_error",
    "receipt_schema_invalid",
    "receipt_unsafe_symlink",
}
FINAL_RECEIPT_FIELDS = {
    "adapter",
    "adapter_weight_file_count",
    "elapsed_seconds",
    "exit_code",
    "interrupted",
    "losses",
    "peak_child_rss_kb",
    "timed_out",
}


class Tau3CandidateAttemptError(ValueError):
    """Raised when a candidate attempt or ledger cannot be proven safely."""


def run_candidate_attempt(
    *,
    campaign_root: str | Path,
    training_args: list[str],
    attempt_id: str | None = None,
    created_at: str | None = None,
    workspace_root: str | Path | None = None,
    resume_existing_attempt: bool = False,
) -> dict[str, Any]:
    """Write intent, run the existing Tau-3 MLX script, and always write outcome."""

    root = _workspace_root(workspace_root)
    campaign = _prepare_campaign_root(Path(campaign_root), root)
    _reject_forwarded_args(training_args, root)
    if resume_existing_attempt and attempt_id is None:
        raise Tau3CandidateAttemptError("resume_existing_attempt requires an explicit attempt_id")
    safe_id = _new_attempt_id(attempt_id)
    attempt_dir = campaign / safe_id
    run_dir = attempt_dir / "run"

    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "run_tau3_mlx_training.py"
    if not script.is_file() or path_has_symlink_component(script, include_leaf=True):
        raise Tau3CandidateAttemptError("run_tau3_mlx_training.py must be a regular local script")
    if resume_existing_attempt:
        intent = _load_resume_intent(attempt_dir=attempt_dir, campaign=campaign)
        created = str(intent.get("created_at") or "")
    else:
        created = created_at or _now_utc()
        try:
            attempt_dir.mkdir(mode=0o755)
        except FileExistsError as exc:
            raise Tau3CandidateAttemptError(f"attempt directory already exists: {safe_id}") from exc
    expected_intent = {
        "schema_version": "hfr.tau3_candidate_attempt_intent.v1",
        "created_at": created,
        "attempt_id": safe_id,
        "attempt_dir": ".",
        "run_dir": "run",
        "training_script_sha256": _sha256_file(script),
        "protocol_sha256": _arg_file_sha256(training_args, "--protocol", root),
        "source_bindings": _source_bindings(training_args, root),
        "training_args_sha256": canonical_sha256(_public_training_args(training_args, root)),
        "command_sha256": canonical_sha256(
            ["python", "scripts/run_tau3_mlx_training.py", *_public_training_args(training_args, root), "--out", "run"]
        ),
    }
    intent_path = attempt_dir / "attempt_intent.json"
    if resume_existing_attempt:
        if intent != expected_intent:
            raise Tau3CandidateAttemptError(
                "existing attempt intent does not match supplied training args "
                "and current immutable source bindings"
            )
        if not run_dir.is_dir() or path_has_symlink_component(run_dir, include_leaf=True):
            raise Tau3CandidateAttemptError(
                "resumed attempt run directory must be an existing regular non-symlink directory"
            )
    else:
        _write_new_json(intent_path, expected_intent)
    attempt_lease_fd = _acquire_attempt_lease(attempt_dir)
    try:
        prior_outcome_refs, outcome_path = _prepare_outcome_publication(
            attempt_dir=attempt_dir,
            attempt_id=safe_id,
            resume_existing_attempt=resume_existing_attempt,
        )
        outcome_partial_refs = _freeze_existing_outcome_partials(attempt_dir)
        if resume_existing_attempt:
            prior_log_refs = _snapshot_existing_attempt_logs(attempt_dir)
            stdout_path, stderr_path = _next_recovery_log_paths(attempt_dir)
            child_training_args = [*training_args, "--resume-process-segments"]
        else:
            prior_log_refs = []
            stdout_path = attempt_dir / "child.stdout.log"
            stderr_path = attempt_dir / "child.stderr.log"
            child_training_args = training_args
        command = [sys.executable, str(script), *child_training_args, "--out", str(run_dir)]
        return _execute_candidate_attempt(
            root=root,
            attempt_dir=attempt_dir,
            run_dir=run_dir,
            attempt_id=safe_id,
            command=command,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            prior_log_refs=prior_log_refs,
            prior_outcome_refs=prior_outcome_refs,
            outcome_partial_refs=outcome_partial_refs,
            outcome_path=outcome_path,
            resume_existing_attempt=resume_existing_attempt,
            attempt_lease_fd=attempt_lease_fd,
        )
    finally:
        os.close(attempt_lease_fd)


def _execute_candidate_attempt(
    *,
    root: Path,
    attempt_dir: Path,
    run_dir: Path,
    attempt_id: str,
    command: list[str],
    stdout_path: Path,
    stderr_path: Path,
    prior_log_refs: list[dict[str, Any]],
    prior_outcome_refs: list[dict[str, Any]],
    outcome_partial_refs: list[dict[str, Any]],
    outcome_path: Path,
    resume_existing_attempt: bool,
    attempt_lease_fd: int,
) -> dict[str, Any]:
    child: subprocess.Popen[str] | None = None
    status = "failed"
    exit_code: int | None = None
    interrupted = False
    started = time.monotonic()
    previous_handlers: dict[int, Any] = {}

    def _handle_signal(signum: int, _frame: Any) -> None:
        nonlocal interrupted, status
        interrupted = True
        status = "interrupted"
        if child is not None and child.poll() is None:
            child.terminate()

    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, _handle_signal)
        with stdout_path.open("x", encoding="utf-8") as stdout_handle, stderr_path.open(
            "x",
            encoding="utf-8",
        ) as stderr_handle:
            child = subprocess.Popen(
                command,
                cwd=root,
                text=True,
                stdout=stdout_handle,
                stderr=stderr_handle,
                pass_fds=(attempt_lease_fd,),
            )
            exit_code = child.wait()
        if interrupted or (exit_code is not None and exit_code < 0):
            interrupted = True
            status = "interrupted"
        else:
            status = "completed" if exit_code == 0 else "failed"
    finally:
        _restore_signal_handlers(previous_handlers)
        stdout_record = _freeze_regular_file_record(stdout_path, attempt_dir)
        stderr_record = _freeze_regular_file_record(stderr_path, attempt_dir)
        receipt_path = run_dir / "training_receipt.json"
        receipt, receipt_ref, receipt_reason = _inspect_training_receipt(receipt_path, attempt_dir)
        failure_reasons = [receipt_reason] if receipt_reason is not None else []
        if interrupted or (exit_code is not None and exit_code < 0):
            interrupted = True
            status = "interrupted"
        elif receipt_reason == "missing_receipt":
            if status == "completed":
                status = "missing-receipt"
        elif receipt_reason is not None:
            status = "malformed-receipt"
        elif receipt is None:
            status = "malformed-receipt"
            failure_reasons.append("receipt_reference_error")
        elif receipt.get("timed_out") is True or receipt.get("terminal_status") == "timeout":
            status = "timeout"
        elif receipt.get("interrupted") is True or receipt.get("terminal_status") == "interrupted":
            interrupted = True
            status = "interrupted"
        elif receipt.get("weights_updated") is True and receipt.get("terminal_status") == "success" and exit_code == 0:
            status = "completed"
        elif status == "completed":
            status = "failed"
            failure_reasons.append("receipt_not_successful")
        log_records: dict[str, Any] = {
            "stdout": stdout_record,
            "stderr": stderr_record,
        }
        if resume_existing_attempt:
            log_records["prior"] = prior_log_refs
        outcome = {
            "schema_version": TAU3_CANDIDATE_ATTEMPT_OUTCOME_SCHEMA_VERSION,
            "created_at": _now_utc(),
            "attempt_id": attempt_id,
            "status": status,
            "exit_code": exit_code,
            "interrupted": interrupted,
            "elapsed_seconds": round(time.monotonic() - started, 6),
            "failure_reasons": failure_reasons,
            "training_receipt": receipt_ref,
            "logs": log_records,
            "prior_outcomes": prior_outcome_refs,
            "outcome_partials": outcome_partial_refs,
            "resume_process_segments": resume_existing_attempt,
        }
        schema = check_schema_contract(
            outcome,
            name_or_id="tau3_candidate_attempt_outcome",
        )
        if schema["passed"] is not True:
            raise Tau3CandidateAttemptError(
                "candidate attempt outcome violates schema: "
                + "; ".join(schema["errors"])
            )
        _publish_new_json_readonly(outcome_path, outcome)
    return outcome


def build_candidate_attempt_ledger(
    *,
    campaign_root: str | Path,
    out_path: str | Path,
    created_at: str | None = None,
    lock_created_at: str | None = None,
    lock_sha256: str | None = None,
    workspace_root: str | Path | None = None,
) -> dict[str, Any]:
    """Census every immediate attempt directory and write a public-safe ledger."""

    root = _workspace_root(workspace_root)
    campaign = _require_campaign_root(Path(campaign_root), root)
    output = _resolve_under_root(Path(out_path), root, "ledger output", must_exist=False)
    if output.exists():
        raise Tau3CandidateAttemptError(f"ledger output already exists: {out_path}")
    if output.is_relative_to(campaign):
        raise Tau3CandidateAttemptError("ledger output must not be inside the attempt campaign root")
    lock_dt = _parse_utc(lock_created_at) if lock_created_at else None
    attempts: list[dict[str, Any]] = []
    for child in sorted(campaign.iterdir(), key=lambda path: path.name):
        if child.name == CAMPAIGN_MARKER or not child.is_dir():
            continue
        if path_has_symlink_component(child, include_leaf=True):
            raise Tau3CandidateAttemptError(f"attempt directory must not contain symlink components: {child.name}")
        if lock_dt is not None and _latest_mtime(child) > lock_dt.timestamp():
            raise Tau3CandidateAttemptError(f"attempt {child.name} was modified after candidate lock timestamp")
        attempts.append(_attempt_record(child, campaign))
    ids = [attempt["attempt_id"] for attempt in attempts]
    duplicate_ids = sorted({item for item in ids if ids.count(item) > 1})
    if duplicate_ids:
        raise Tau3CandidateAttemptError("duplicate attempt id(s): " + ", ".join(duplicate_ids))
    counts = {status: 0 for status in ATTEMPT_STATUSES}
    for attempt in attempts:
        counts[str(attempt["status"])] += 1
    ledger = {
        "schema_version": TAU3_CANDIDATE_ATTEMPT_LEDGER_SCHEMA_VERSION,
        "schema_checked": True,
        "created_at": created_at or _now_utc(),
        "campaign": {
            "root_ref": _safe_rel(campaign, root),
            "campaign_marker_sha256": _sha256_file(campaign / CAMPAIGN_MARKER),
        },
        "lock": {"created_at": lock_created_at, "sha256": lock_sha256} if lock_created_at or lock_sha256 else None,
        "attempt_count": len(attempts),
        "status_counts": counts,
        "successful_attempt_count": counts.get("completed", 0),
        "failed_attempt_count": len(attempts) - counts.get("completed", 0),
        "attempts": attempts,
    }
    _assert_public_safe(ledger)
    schema = check_schema_contract(ledger, name_or_id="tau3_candidate_attempt_ledger")
    if schema["passed"] is not True:
        raise Tau3CandidateAttemptError("candidate attempt ledger violates schema: " + "; ".join(schema["errors"]))
    _write_new_json(output, ledger)
    return ledger


def build_run_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one governed Tau-3 candidate attempt.")
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--attempt-id")
    parser.add_argument(
        "--resume-existing-attempt",
        action="store_true",
        help=(
            "Resume an existing interrupted process-segmented attempt after "
            "replaying its immutable intent and committed training state."
        ),
    )
    parser.add_argument(
        "training_args",
        nargs=argparse.REMAINDER,
        help="Arguments for run_tau3_mlx_training.py after --",
    )
    return parser


def run_main(argv: list[str] | None = None) -> int:
    args = build_run_arg_parser().parse_args(argv)
    training_args = list(args.training_args)
    if training_args and training_args[0] == "--":
        training_args = training_args[1:]
    try:
        outcome = run_candidate_attempt(
            campaign_root=args.campaign_root,
            attempt_id=args.attempt_id,
            training_args=training_args,
            resume_existing_attempt=args.resume_existing_attempt,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(outcome, indent=2, sort_keys=True))
    return 0 if outcome["status"] == "completed" else 1


def build_ledger_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a public-safe Tau-3 candidate attempt ledger.")
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--lock-created-at")
    parser.add_argument("--lock-sha256")
    return parser


def ledger_main(argv: list[str] | None = None) -> int:
    args = build_ledger_arg_parser().parse_args(argv)
    try:
        ledger = build_candidate_attempt_ledger(
            campaign_root=args.campaign_root,
            out_path=args.out,
            lock_created_at=args.lock_created_at,
            lock_sha256=args.lock_sha256,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "ledger": str(args.out),
                "attempt_count": ledger["attempt_count"],
                "status_counts": ledger["status_counts"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _attempt_record(attempt_dir: Path, campaign: Path) -> dict[str, Any]:
    intent_path = attempt_dir / "attempt_intent.json"
    try:
        outcome_paths = _candidate_outcome_paths(attempt_dir)
        outcome_paths_malformed = False
    except Tau3CandidateAttemptError:
        outcome_paths = []
        outcome_paths_malformed = True
    outcome_path = (
        outcome_paths[-1]
        if outcome_paths
        else attempt_dir / "attempt_outcome.json"
    )
    receipt_path = attempt_dir / "run" / "training_receipt.json"
    intent, intent_malformed = _load_attempt_artifact(intent_path, "intent")
    attempt_id = _safe_attempt_id(
        attempt_dir.name
        if intent is None
        else str(intent.get("attempt_id") or attempt_dir.name)
    )
    if outcome_paths_malformed:
        outcome, outcome_malformed = None, True
    else:
        outcome, outcome_malformed = _load_candidate_outcome_chain(
            outcome_paths,
            attempt_dir=attempt_dir,
            attempt_id=attempt_id,
        )
    receipt, receipt_malformed = _load_attempt_artifact(
        receipt_path,
        "receipt",
        schema_name="tau3_mlx_training_run",
    )
    reasons: list[str] = []
    if intent_malformed:
        reasons.append("malformed_intent")
    elif intent is None:
        reasons.append("missing_intent")
    if outcome_malformed:
        reasons.append("malformed_outcome")
    elif outcome is None:
        reasons.append("missing_outcome")
    if isinstance(outcome, dict) and isinstance(outcome.get("failure_reasons"), list):
        reasons.extend(
            str(reason)
            for reason in outcome["failure_reasons"]
            if isinstance(reason, str) and reason in FAILURE_REASONS
        )
    status = "failed"
    if isinstance(outcome, dict) and outcome.get("status") in ATTEMPT_STATUSES:
        status = str(outcome["status"])
    if receipt_malformed:
        reasons.append("malformed_receipt")
        if status != "interrupted":
            status = "malformed-receipt"
    elif receipt is None:
        reasons.append("missing_receipt")
        if status == "completed":
            status = "missing-receipt"
    elif (
        receipt.get("terminal_status") == "success"
        and receipt.get("weights_updated") is True
        and status == "completed"
    ):
        status = "completed"
    elif status == "completed":
        status = "failed"
        reasons.append("receipt_not_successful")
    record = {
        "attempt_id": attempt_id,
        "attempt_ref": _safe_rel(attempt_dir, campaign),
        "status": status,
        "failure_reasons": sorted(set(reasons)),
        "intent": _best_effort_file_ref(intent_path, attempt_dir),
        "outcome": _best_effort_file_ref(outcome_path, attempt_dir),
        "training_receipt": _best_effort_file_ref(receipt_path, attempt_dir),
        "bindings": _binding_record(intent, receipt),
        "metrics": _metric_record(outcome, receipt),
    }
    _assert_public_safe(record)
    return record


def _binding_record(intent: dict[str, Any] | None, receipt: dict[str, Any] | None) -> dict[str, Any]:
    receipt_payload: dict[str, Any] = receipt if receipt is not None else {}
    binding = _dict_or_empty(receipt_payload.get("training_binding"))
    protocol = _dict_or_empty(binding.get("protocol"))
    model = _dict_or_empty(binding.get("model"))
    dataset = _dict_or_empty(binding.get("dataset"))
    recipe = _dict_or_empty(binding.get("recipe"))
    adapter = _dict_or_empty(receipt_payload.get("adapter"))
    return {
        "protocol_sha256": _sha256_or_none(protocol.get("sha256") or (intent or {}).get("protocol_sha256")),
        "protocol_signature": _sha256_or_none(protocol.get("protocol_signature")),
        "model_identity_sha256": _sha256_or_none(model.get("identity_sha256")),
        "dataset_manifest_sha256": _sha256_or_none(dataset.get("manifest_sha256")),
        "dataset_files_sha256": _sha256_or_none(dataset.get("files_sha256")),
        "recipe_sha256": _sha256_or_none(recipe.get("recipe_sha256")),
        "config_sha256": (
            canonical_sha256(receipt_payload.get("config"))
            if isinstance(receipt_payload.get("config"), dict)
            else None
        ),
        "adapter_tree_sha256": _sha256_or_none(adapter.get("tree_sha256")),
    }


def _metric_record(outcome: dict[str, Any] | None, receipt: dict[str, Any] | None) -> dict[str, Any]:
    receipt_payload: dict[str, Any] = receipt if receipt is not None else {}
    losses = _dict_or_empty(receipt_payload.get("losses"))
    elapsed = receipt_payload.get("elapsed_seconds") if receipt is not None else (outcome or {}).get("elapsed_seconds")
    return {
        "elapsed_seconds": _nonnegative_number_or_none(elapsed),
        "peak_child_rss_kb": _nonnegative_integer_or_none(receipt_payload.get("peak_child_rss_kb")),
        "weights_updated": receipt_payload.get("weights_updated") is True,
        "last_train_loss": _number_or_none(losses.get("last_train")),
        "last_validation_loss": _number_or_none(losses.get("last_validation")),
        "train_losses": _number_list(losses.get("train")),
        "validation_losses": _number_list(losses.get("validation")),
    }


def _prepare_campaign_root(path: Path, root: Path) -> Path:
    campaign = _resolve_under_root(path, root, "campaign root", must_exist=False)
    if path_has_symlink_component(campaign, include_leaf=True):
        raise Tau3CandidateAttemptError(f"campaign root must not contain symlink components: {path}")
    if campaign.exists() and not campaign.is_dir():
        raise Tau3CandidateAttemptError(f"campaign root must be a directory: {path}")
    campaign.mkdir(parents=True, exist_ok=True)
    marker = campaign / CAMPAIGN_MARKER
    if marker.exists() and path_has_symlink_component(marker, include_leaf=True):
        raise Tau3CandidateAttemptError("campaign marker must not be symlinked")
    if not marker.exists():
        if any(campaign.iterdir()):
            raise Tau3CandidateAttemptError("campaign root must be new/empty or already marked as owned")
        _write_text_new(marker, "hfr.tau3_candidate_attempt_campaign.v1\n")
    return campaign


def _load_resume_intent(
    *,
    attempt_dir: Path,
    campaign: Path,
) -> dict[str, Any]:
    if not attempt_dir.is_dir() or path_has_symlink_component(
        attempt_dir,
        include_leaf=True,
    ):
        raise Tau3CandidateAttemptError(
            "resumed attempt must be an existing regular non-symlink directory"
        )
    try:
        attempt_dir.resolve(strict=True).relative_to(campaign.resolve(strict=True))
    except (OSError, RuntimeError, ValueError) as exc:
        raise Tau3CandidateAttemptError(
            "resumed attempt must remain inside the candidate campaign"
        ) from exc
    intent_path = attempt_dir / "attempt_intent.json"
    if (
        not intent_path.is_file()
        or path_has_symlink_component(intent_path, include_leaf=True)
        or intent_path.stat().st_mode & 0o222
    ):
        raise Tau3CandidateAttemptError(
            "resumed attempt intent must be an immutable regular non-symlink file"
        )
    try:
        intent = _load_json(intent_path)
    except (OSError, ValueError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise Tau3CandidateAttemptError(
            "resumed attempt intent must be valid JSON"
        ) from exc
    if intent.get("schema_version") != "hfr.tau3_candidate_attempt_intent.v1":
        raise Tau3CandidateAttemptError(
            "resumed attempt intent schema_version is invalid"
        )
    return intent


def _snapshot_existing_attempt_logs(
    attempt_dir: Path,
) -> list[dict[str, Any]]:
    log_paths = sorted(
        (
            child
            for child in attempt_dir.iterdir()
            if child.name.startswith("child.") and child.name.endswith(".log")
        ),
        key=lambda path: path.name,
    )
    return [
        _snapshot_regular_file(path, attempt_dir)
        for path in log_paths
    ]


def _snapshot_regular_file(
    source: Path,
    base: Path,
) -> dict[str, Any]:
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(source, flags)
    except OSError as exc:
        raise Tau3CandidateAttemptError(
            f"log snapshot source is unavailable: {source.name}"
        ) from exc
    try:
        before = os.fstat(fd)
        path_state = os.lstat(source)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or (before.st_dev, before.st_ino)
            != (path_state.st_dev, path_state.st_ino)
        ):
            raise Tau3CandidateAttemptError(
                f"log snapshot source must be a single-link regular file: "
                f"{source.name}"
            )
        chunks: list[bytes] = []
        size = 0
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
            size += len(chunk)
        after = os.fstat(fd)
        final_path_state = os.lstat(source)
        if (
            (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            or (after.st_dev, after.st_ino)
            != (final_path_state.st_dev, final_path_state.st_ino)
            or size != after.st_size
        ):
            raise Tau3CandidateAttemptError(
                f"log snapshot source changed while being copied: {source.name}"
            )
    finally:
        os.close(fd)
    snapshot = _next_log_snapshot_path(base, source.name)
    _publish_new_bytes_readonly(snapshot, b"".join(chunks))
    record = _file_ref(snapshot, base)
    record["source_path"] = _safe_rel(source, base)
    return record


def _next_log_snapshot_path(attempt_dir: Path, source_name: str) -> Path:
    for index in range(1, 10_000):
        candidate = attempt_dir / (
            f"evidence-log-snapshot-{index:04d}-{source_name}"
        )
        if not os.path.lexists(candidate):
            return candidate
    raise Tau3CandidateAttemptError(
        "candidate attempt has exhausted log snapshot names"
    )


def _next_recovery_log_paths(attempt_dir: Path) -> tuple[Path, Path]:
    for index in range(1, 10_000):
        suffix = f"recovery-{index:04d}.log"
        stdout_path = attempt_dir / f"child.stdout.{suffix}"
        stderr_path = attempt_dir / f"child.stderr.{suffix}"
        if not os.path.lexists(stdout_path) and not os.path.lexists(stderr_path):
            return stdout_path, stderr_path
    raise Tau3CandidateAttemptError(
        "candidate attempt has exhausted recovery log names"
    )


def _acquire_attempt_lease(attempt_dir: Path) -> int:
    if fcntl is None:
        raise Tau3CandidateAttemptError(
            "candidate attempt leases require POSIX flock support"
        )
    lease_path = attempt_dir / ".attempt.lock"
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(lease_path, flags, 0o600)
    try:
        descriptor = os.fstat(fd)
        path_state = os.lstat(lease_path)
        if (
            not stat.S_ISREG(descriptor.st_mode)
            or descriptor.st_nlink != 1
            or (descriptor.st_dev, descriptor.st_ino)
            != (path_state.st_dev, path_state.st_ino)
        ):
            raise Tau3CandidateAttemptError(
                "candidate attempt lease must be a single-link regular file"
            )
        os.fchmod(fd, 0o600)
        os.fsync(fd)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno in {errno.EACCES, errno.EAGAIN}:
                raise Tau3CandidateAttemptError(
                    "candidate attempt is already active"
                ) from exc
            raise
        locked_descriptor = os.fstat(fd)
        locked_path_state = os.lstat(lease_path)
        if (
            not stat.S_ISREG(locked_descriptor.st_mode)
            or locked_descriptor.st_nlink != 1
            or (locked_descriptor.st_dev, locked_descriptor.st_ino)
            != (locked_path_state.st_dev, locked_path_state.st_ino)
        ):
            raise Tau3CandidateAttemptError(
                "candidate attempt lease changed during acquisition"
            )
        _fsync_directory(attempt_dir)
        return fd
    except BaseException:
        os.close(fd)
        raise


def _freeze_existing_outcome_partials(
    attempt_dir: Path,
) -> list[dict[str, Any]]:
    return [
        _freeze_regular_file_record(path, attempt_dir)
        for path in sorted(
            attempt_dir.glob(".attempt_outcome*.json.partial-*"),
            key=lambda child: child.name,
        )
    ]


def _candidate_outcome_paths(attempt_dir: Path) -> list[Path]:
    primary = attempt_dir / "attempt_outcome.json"
    recovery_paths: dict[int, Path] = {}
    for path in attempt_dir.glob("attempt_outcome.recovery-*.json"):
        match = RECOVERY_OUTCOME_RE.fullmatch(path.name)
        if match is None:
            raise Tau3CandidateAttemptError(
                f"unrecognized candidate recovery outcome: {path.name}"
            )
        index = int(match.group(1))
        if index < 1 or index in recovery_paths:
            raise Tau3CandidateAttemptError(
                f"invalid candidate recovery outcome index: {path.name}"
            )
        recovery_paths[index] = path
    if recovery_paths and not os.path.lexists(primary):
        raise Tau3CandidateAttemptError(
            "candidate recovery outcomes require the primary attempt outcome"
        )
    if recovery_paths:
        expected = list(range(1, max(recovery_paths) + 1))
        if sorted(recovery_paths) != expected:
            raise Tau3CandidateAttemptError(
                "candidate recovery outcome sequence must be contiguous"
            )
    paths = [primary] if os.path.lexists(primary) else []
    paths.extend(recovery_paths[index] for index in sorted(recovery_paths))
    return paths


def _prepare_outcome_publication(
    *,
    attempt_dir: Path,
    attempt_id: str,
    resume_existing_attempt: bool,
) -> tuple[list[dict[str, Any]], Path]:
    outcome_paths = _candidate_outcome_paths(attempt_dir)
    primary = attempt_dir / "attempt_outcome.json"
    if not resume_existing_attempt:
        if outcome_paths:
            raise Tau3CandidateAttemptError(
                "cannot start an attempt with an existing outcome"
            )
        return [], primary
    if not outcome_paths:
        return [], primary

    prior_refs: list[dict[str, Any]] = []
    latest_payload: dict[str, Any] | None = None
    for index, path in enumerate(outcome_paths):
        record = _freeze_regular_file_record(path, attempt_dir)
        try:
            payload = _load_json(path)
        except (OSError, ValueError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise Tau3CandidateAttemptError(
                f"existing outcome is not valid JSON: {path.name}"
            ) from exc
        if (
            check_schema_contract(
                payload,
                name_or_id="tau3_candidate_attempt_outcome",
            )["passed"]
            is not True
            or payload.get("schema_version")
            != TAU3_CANDIDATE_ATTEMPT_OUTCOME_SCHEMA_VERSION
            or payload.get("attempt_id") != attempt_id
            or payload.get("status") not in ATTEMPT_STATUSES
            or not isinstance(payload.get("interrupted"), bool)
            or payload.get("interrupted")
            is not (payload.get("status") == "interrupted")
        ):
            raise Tau3CandidateAttemptError(
                f"existing outcome is not a valid candidate outcome: {path.name}"
            )
        if (
            payload.get("status") != "interrupted"
            or payload.get("interrupted") is not True
        ):
            raise Tau3CandidateAttemptError(
                "cannot resume an attempt whose existing outcome is not "
                f"interrupted: {path.name}"
            )
        if index == 0 and payload.get("prior_outcomes") not in (None, []):
            raise Tau3CandidateAttemptError(
                f"candidate primary outcome has recovery lineage: {path.name}"
            )
        if index > 0 and payload.get("prior_outcomes") != prior_refs:
            raise Tau3CandidateAttemptError(
                f"candidate recovery outcome chain mismatch: {path.name}"
            )
        if _freeze_regular_file_record(path, attempt_dir) != record:
            raise Tau3CandidateAttemptError(
                f"existing outcome changed during replay: {path.name}"
            )
        prior_refs.append(record)
        latest_payload = payload

    if latest_payload is None:
        raise Tau3CandidateAttemptError(
            "cannot resume an attempt whose existing outcome is not interrupted"
        )
    recovery_index = len(outcome_paths)
    if recovery_index >= 10_000:
        raise Tau3CandidateAttemptError(
            "candidate attempt has exhausted recovery outcome names"
        )
    return (
        prior_refs,
        attempt_dir / f"attempt_outcome.recovery-{recovery_index:04d}.json",
    )


def _load_candidate_outcome_chain(
    outcome_paths: list[Path],
    *,
    attempt_dir: Path,
    attempt_id: str,
) -> tuple[dict[str, Any] | None, bool]:
    if not outcome_paths:
        return None, False
    prior_refs: list[dict[str, Any]] = []
    latest_payload: dict[str, Any] | None = None
    for index, path in enumerate(outcome_paths):
        payload, malformed = _load_attempt_artifact(path, "outcome")
        if malformed or payload is None:
            return None, True
        if (
            check_schema_contract(
                payload,
                name_or_id="tau3_candidate_attempt_outcome",
            )["passed"]
            is not True
            or payload.get("schema_version")
            != TAU3_CANDIDATE_ATTEMPT_OUTCOME_SCHEMA_VERSION
            or payload.get("attempt_id") != attempt_id
            or payload.get("status") not in ATTEMPT_STATUSES
            or not isinstance(payload.get("interrupted"), bool)
            or payload.get("interrupted")
            is not (payload.get("status") == "interrupted")
        ):
            return None, True
        if index == 0:
            if payload.get("prior_outcomes") not in (None, []):
                return None, True
        elif payload.get("prior_outcomes") != prior_refs:
            return None, True
        if index < len(outcome_paths) - 1 and (
            payload.get("status") != "interrupted"
            or payload.get("interrupted") is not True
        ):
            return None, True
        record = _best_effort_file_ref(path, attempt_dir)
        if record is None:
            return None, True
        prior_refs.append(record)
        latest_payload = payload
    return latest_payload, False


def _freeze_regular_file_record(
    path: Path,
    base: Path,
) -> dict[str, Any]:
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise Tau3CandidateAttemptError(
            f"immutable evidence file is unavailable: {path.name}"
        ) from exc
    try:
        before = os.fstat(fd)
        path_state = os.lstat(path)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or (before.st_dev, before.st_ino)
            != (path_state.st_dev, path_state.st_ino)
        ):
            raise Tau3CandidateAttemptError(
                f"immutable evidence must be a single-link regular file: {path.name}"
            )
        os.fchmod(fd, 0o444)
        os.fsync(fd)
        os.lseek(fd, 0, os.SEEK_SET)
        digest = hashlib.sha256()
        size = 0
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
        after = os.fstat(fd)
        final_path_state = os.lstat(path)
        if (
            (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            or (after.st_dev, after.st_ino)
            != (final_path_state.st_dev, final_path_state.st_ino)
            or after.st_mode & 0o222
            or size != after.st_size
        ):
            raise Tau3CandidateAttemptError(
                f"immutable evidence changed while being frozen: {path.name}"
            )
        return {
            "path": _safe_rel(path, base),
            "sha256": digest.hexdigest(),
            "size": size,
        }
    finally:
        os.close(fd)


def _publish_new_json_readonly(path: Path, payload: dict[str, Any]) -> None:
    data = (
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    _publish_new_bytes_readonly(path, data)


def _publish_new_bytes_readonly(path: Path, data: bytes) -> None:
    if os.path.lexists(path):
        raise FileExistsError(path)
    partial = path.parent / (
        f".{path.name}.partial-{os.getpid()}-{secrets.token_hex(8)}"
    )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(partial, flags, 0o600)
    try:
        try:
            view = memoryview(data)
            while view:
                written = os.write(fd, view)
                if written <= 0:
                    raise OSError("immutable publication write made no progress")
                view = view[written:]
            os.fsync(fd)
            os.fchmod(fd, 0o444)
            os.fsync(fd)
        except BaseException:
            try:
                os.fsync(fd)
                os.fchmod(fd, 0o444)
                os.fsync(fd)
            finally:
                os.close(fd)
            raise
        os.close(fd)
        os.link(partial, path, follow_symlinks=False)
        _fsync_directory(path.parent)
        os.unlink(partial)
        _fsync_directory(path.parent)
    except BaseException:
        try:
            os.close(fd)
        except OSError:
            pass
        raise


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    fd = os.open(path, flags)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _require_campaign_root(path: Path, root: Path) -> Path:
    campaign = _resolve_under_root(path, root, "campaign root", must_exist=True)
    if not campaign.is_dir() or path_has_symlink_component(campaign, include_leaf=True):
        raise Tau3CandidateAttemptError("campaign root must be a regular non-symlink directory")
    marker = campaign / CAMPAIGN_MARKER
    if not marker.is_file() or path_has_symlink_component(marker, include_leaf=True):
        raise Tau3CandidateAttemptError("campaign root is not an owned Tau-3 candidate attempt campaign")
    return campaign


def _reject_forwarded_args(args: list[str], root: Path) -> None:
    if not args:
        raise Tau3CandidateAttemptError("training args are required")
    if "--out" in args or any(token.startswith("--out=") for token in args):
        raise Tau3CandidateAttemptError("candidate wrapper owns --out; do not forward --out")
    if "--resume-process-segments" in args:
        raise Tau3CandidateAttemptError(
            "candidate wrapper owns --resume-process-segments; use --resume-existing-attempt"
        )
    index = 0
    while index < len(args):
        token = args[index]
        value = None
        if token in PATH_ARG_NAMES and index + 1 < len(args):
            value = args[index + 1]
            index += 1
        elif any(token.startswith(name + "=") for name in PATH_ARG_NAMES):
            value = token.split("=", 1)[1]
        if value:
            if SEALED_TEST_RE.search(value.replace(os.sep, "/")):
                raise Tau3CandidateAttemptError(
                    f"sealed/test path refs are not allowed for candidate attempts: {token}"
                )
            unresolved = Path(value) if Path(value).is_absolute() else root / value
            if path_has_symlink_component(unresolved, include_leaf=True):
                raise Tau3CandidateAttemptError(f"{token} must not contain symlink components")
            candidate = _resolve_under_root(Path(value), root, token, must_exist=True)
            if path_has_symlink_component(candidate, include_leaf=True):
                raise Tau3CandidateAttemptError(f"{token} must not contain symlink components")
        index += 1


def _source_bindings(args: list[str], root: Path) -> dict[str, Any]:
    bindings: dict[str, Any] = {}
    for name in ("--bundle", "--mixture-dir", "--protocol", "--model-identity"):
        value = _arg_value(args, name)
        if value is None:
            continue
        path = _resolve_under_root(Path(value), root, name, must_exist=True)
        bindings[name.removeprefix("--").replace("-", "_")] = {
            "ref": _safe_rel(path, root),
            "sha256": _sha256_file(path) if path.is_file() else _tree_sha256(path),
        }
    return bindings


def _public_training_args(args: list[str], root: Path) -> list[str]:
    safe: list[str] = []
    index = 0
    while index < len(args):
        token = args[index]
        if token in PATH_ARG_NAMES and index + 1 < len(args):
            safe.append(token)
            safe.append(
                _safe_rel(_resolve_under_root(Path(args[index + 1]), root, token, must_exist=True), root)
            )
            index += 2
            continue
        matched = next((name for name in PATH_ARG_NAMES if token.startswith(name + "=")), None)
        if matched is not None:
            value = token.split("=", 1)[1]
            rel = _safe_rel(_resolve_under_root(Path(value), root, matched, must_exist=True), root)
            safe.append(f"{matched}={rel}")
        else:
            safe.append(token)
        index += 1
    return safe


def _arg_value(args: list[str], name: str) -> str | None:
    for index, token in enumerate(args):
        if token == name and index + 1 < len(args):
            return args[index + 1]
        if token.startswith(name + "="):
            return token.split("=", 1)[1]
    return None


def _arg_file_sha256(args: list[str], name: str, root: Path) -> str | None:
    value = _arg_value(args, name)
    if value is None:
        return None
    path = _resolve_under_root(Path(value), root, name, must_exist=True)
    return _sha256_file(path) if path.is_file() else None


def _workspace_root(root: str | Path | None) -> Path:
    path = Path(root) if root is not None else Path.cwd()
    resolved = path.resolve(strict=True)
    if path_has_symlink_component(resolved, include_leaf=True):
        raise Tau3CandidateAttemptError(f"workspace root must not contain symlink components: {path}")
    return resolved


def _resolve_under_root(path: Path, root: Path, label: str, *, must_exist: bool) -> Path:
    candidate = path if path.is_absolute() else root / path
    try:
        resolved = candidate.resolve(strict=must_exist)
    except FileNotFoundError as exc:
        raise Tau3CandidateAttemptError(f"{label} does not exist: {path}") from exc
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise Tau3CandidateAttemptError(f"{label} must resolve under workspace root: {path}") from exc
    return resolved


def _safe_rel(path: Path, base: Path) -> str:
    rel = path.resolve(strict=path.exists()).relative_to(base.resolve(strict=True)).as_posix()
    if rel in {"", "."} or rel.startswith("../") or Path(rel).is_absolute() or "\x00" in rel:
        raise Tau3CandidateAttemptError(f"unsafe relative reference for {path}")
    return rel


def _new_attempt_id(value: str | None) -> str:
    if value is None:
        value = (
            "attempt-"
            + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            + "-"
            + secrets.token_hex(6)
        )
    return _safe_attempt_id(value)


def _safe_attempt_id(value: str) -> str:
    if SAFE_ID_RE.fullmatch(value) is None or value in {".", ".."}:
        raise Tau3CandidateAttemptError(f"unsafe attempt id: {value!r}")
    return value


def _file_ref(path: Path, base: Path) -> dict[str, Any]:
    return {"path": _safe_rel(path, base), "sha256": _sha256_file(path), "size": path.stat().st_size}


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise Tau3CandidateAttemptError(f"expected JSON object: {path}")
    return payload


def _load_attempt_artifact(
    path: Path,
    label: str,
    *,
    schema_name: str | None = None,
) -> tuple[dict[str, Any] | None, bool]:
    if path_has_symlink_component(path, include_leaf=True):
        raise Tau3CandidateAttemptError(f"attempt {label} must not contain symlink components")
    if not path.exists():
        return None, False
    if not path.is_file():
        return None, True
    try:
        payload = _load_json(path)
    except (OSError, ValueError, json.JSONDecodeError, UnicodeDecodeError):
        return None, True
    if schema_name is not None:
        result = check_schema_contract(payload, name_or_id=schema_name)
        if result["passed"] is not True:
            return None, True
        if schema_name == "tau3_mlx_training_run" and not _is_final_training_receipt(payload):
            return None, True
    return payload, False


def _inspect_training_receipt(
    path: Path,
    attempt_dir: Path,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, str | None]:
    try:
        if path_has_symlink_component(path, include_leaf=True):
            return None, None, "receipt_unsafe_symlink"
        if not path.exists():
            return None, None, "missing_receipt"
        if not path.is_file():
            return None, None, "receipt_schema_invalid"
        receipt_ref = _best_effort_file_ref(path, attempt_dir)
        if receipt_ref is None:
            return None, None, "receipt_reference_error"
        try:
            receipt = _load_json(path)
        except (OSError, ValueError, json.JSONDecodeError, UnicodeDecodeError):
            return None, receipt_ref, "receipt_parse_error"
        result = check_schema_contract(receipt, name_or_id="tau3_mlx_training_run")
        if result["passed"] is not True or not _is_final_training_receipt(receipt):
            return None, receipt_ref, "receipt_schema_invalid"
        return receipt, receipt_ref, None
    except Exception:
        return None, None, "receipt_reference_error"


def _restore_signal_handlers(previous_handlers: dict[int, Any]) -> None:
    for signum, handler in previous_handlers.items():
        try:
            signal.signal(signum, handler)
        except (OSError, ValueError):
            continue


def _best_effort_file_ref(path: Path, base: Path) -> dict[str, Any] | None:
    try:
        if not path.is_file() or path_has_symlink_component(path, include_leaf=True):
            return None
        return _file_ref(path, base)
    except (OSError, ValueError):
        return None


def _write_new_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    path.chmod(0o444)


def _sha256_or_none(value: Any) -> str | None:
    return value if isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) else None


def _dict_or_empty(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _is_final_training_receipt(value: dict[str, Any]) -> bool:
    return (
        value.get("phase") == "final"
        and value.get("schema_checked") is True
        and FINAL_RECEIPT_FIELDS.issubset(value)
    )


def _number_or_none(value: Any) -> int | float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return value


def _nonnegative_number_or_none(value: Any) -> int | float | None:
    number = _number_or_none(value)
    return number if number is not None and number >= 0 else None


def _nonnegative_integer_or_none(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _number_list(value: Any) -> list[int | float]:
    if not isinstance(value, list):
        return []
    numbers = [_number_or_none(item) for item in value]
    return [number for number in numbers if number is not None]


def _write_text_new(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(value)
    path.chmod(0o444)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_sha256(path: Path) -> str:
    records = []
    for child in sorted(p for p in path.rglob("*") if p.is_file()):
        if path_has_symlink_component(child, include_leaf=True):
            raise Tau3CandidateAttemptError(f"tree contains symlink component: {path}")
        records.append(
            {
                "path": child.relative_to(path).as_posix(),
                "sha256": _sha256_file(child),
                "size": child.stat().st_size,
            }
        )
    return canonical_sha256(records)


def _latest_mtime(path: Path) -> float:
    latest = path.stat().st_mtime
    for child in path.rglob("*"):
        latest = max(latest, child.stat().st_mtime)
    return latest


def _parse_utc(value: str) -> datetime:
    text = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        raise Tau3CandidateAttemptError("lock timestamp must include a timezone")
    return parsed.astimezone(timezone.utc)


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _assert_public_safe(value: Any) -> None:
    strings: list[str] = []
    _collect_strings(value, strings)
    home = str(Path.home())
    for item in strings:
        if (
            item.startswith(home)
            or item.startswith("/Users/")
            or item.startswith("/private/")
            or Path(item).is_absolute()
        ):
            raise Tau3CandidateAttemptError(f"public ledger contains private/absolute path: {item}")


def _collect_strings(value: Any, out: list[str]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            out.append(str(key))
            _collect_strings(item, out)
    elif isinstance(value, list):
        for item in value:
            _collect_strings(item, out)
    elif isinstance(value, str):
        out.append(value)
