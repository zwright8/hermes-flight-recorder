"""Governed Tau-3 candidate attempt wrapper and public-safe ledger."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import secrets
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .path_safety import path_has_symlink_component
from .repeated_eval import canonical_sha256
from .schema_registry import check_schema_contract

TAU3_CANDIDATE_ATTEMPT_LEDGER_SCHEMA_VERSION = "hfr.tau3_candidate_attempt_ledger.v1"
CAMPAIGN_MARKER = ".hfr_tau3_candidate_attempt_campaign"
SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$")
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
) -> dict[str, Any]:
    """Write intent, run the existing Tau-3 MLX script, and always write outcome."""

    root = _workspace_root(workspace_root)
    campaign = _prepare_campaign_root(Path(campaign_root), root)
    _reject_forwarded_args(training_args, root)
    created = created_at or _now_utc()
    safe_id = _new_attempt_id(attempt_id)
    attempt_dir = campaign / safe_id
    try:
        attempt_dir.mkdir(mode=0o755)
    except FileExistsError as exc:
        raise Tau3CandidateAttemptError(f"attempt directory already exists: {safe_id}") from exc
    run_dir = attempt_dir / "run"

    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "run_tau3_mlx_training.py"
    if not script.is_file() or path_has_symlink_component(script, include_leaf=True):
        raise Tau3CandidateAttemptError("run_tau3_mlx_training.py must be a regular local script")
    command = [sys.executable, str(script), *training_args, "--out", str(run_dir)]
    intent = {
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
    _write_new_json(attempt_dir / "attempt_intent.json", intent)

    child: subprocess.Popen[str] | None = None
    status = "failed"
    exit_code: int | None = None
    interrupted = False
    started = time.monotonic()
    previous_handlers: dict[int, Any] = {}
    stdout_path = attempt_dir / "child.stdout.log"
    stderr_path = attempt_dir / "child.stderr.log"

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
            child = subprocess.Popen(command, cwd=root, text=True, stdout=stdout_handle, stderr=stderr_handle)
            exit_code = child.wait()
        stdout_path.chmod(0o444)
        stderr_path.chmod(0o444)
        if interrupted or (exit_code is not None and exit_code < 0):
            interrupted = True
            status = "interrupted"
        else:
            status = "completed" if exit_code == 0 else "failed"
    finally:
        _restore_signal_handlers(previous_handlers)
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
        outcome = {
            "schema_version": "hfr.tau3_candidate_attempt_outcome.v1",
            "created_at": _now_utc(),
            "attempt_id": safe_id,
            "status": status,
            "exit_code": exit_code,
            "interrupted": interrupted,
            "elapsed_seconds": round(time.monotonic() - started, 6),
            "failure_reasons": failure_reasons,
            "training_receipt": receipt_ref,
            "logs": {
                "stdout": _best_effort_file_ref(stdout_path, attempt_dir),
                "stderr": _best_effort_file_ref(stderr_path, attempt_dir),
            },
        }
        _best_effort_write_outcome(attempt_dir / "attempt_outcome.json", outcome)
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
    outcome_path = attempt_dir / "attempt_outcome.json"
    receipt_path = attempt_dir / "run" / "training_receipt.json"
    intent, intent_malformed = _load_attempt_artifact(intent_path, "intent")
    outcome, outcome_malformed = _load_attempt_artifact(outcome_path, "outcome")
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
        "attempt_id": _safe_attempt_id(
            attempt_dir.name if intent is None else str(intent.get("attempt_id") or attempt_dir.name)
        ),
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


def _best_effort_write_outcome(path: Path, payload: dict[str, Any]) -> None:
    try:
        _write_new_json(path, payload)
    except FileExistsError:
        return


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
