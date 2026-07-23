"""Canonical Tau-3 text trajectory capture for governed training exports.

This module does not import Tau or run a model.  It accepts an already observed
Tau text-mode episode, validates the training-critical identities, and projects
the observable conversation, tool calls, state transition, and executable
outcome into Flight Recorder artifacts.  That boundary keeps benchmark access
and model inference outside the dependency-free core while making the evidence
needed for admission deterministic and replayable.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .schema_registry import SchemaRegistryError, check_schema_contract
from .trajectory_v2 import check_trajectory_v2, trajectory_v2_from_trace

TAU3_CAPTURE_SCHEMA_VERSION = "hfr.tau3_capture.v1"
TAU3_DOMAINS = {"airline", "retail", "telecom"}


class Tau3CaptureError(ValueError):
    """Raised when a Tau capture is not safe or complete enough to record."""


def validate_tau3_capture(capture: Any) -> list[str]:
    """Return deterministic validation errors for one canonical capture row."""

    if not isinstance(capture, dict):
        return ["capture must be a JSON object"]
    errors: list[str] = []
    try:
        schema = check_schema_contract(capture, name_or_id="tau3_capture")
        errors.extend(f"schema: {error}" for error in schema.get("errors", []))
    except SchemaRegistryError as exc:
        errors.append(f"schema: {exc}")
    required_strings = (
        "trajectory_id",
        "task_id",
        "task_family",
        "behavior",
        "generator_id",
        "generator_revision",
        "policy_revision",
        "tool_schema_revision",
        "starting_state_hash",
        "prompt_hash",
    )
    if capture.get("schema_version") != TAU3_CAPTURE_SCHEMA_VERSION:
        errors.append(f"schema_version must be {TAU3_CAPTURE_SCHEMA_VERSION!r}")
    for field in required_strings:
        if not isinstance(capture.get(field), str) or not capture[field].strip():
            errors.append(f"{field} must be a non-empty string")
    if str(capture.get("domain") or "").lower() not in TAU3_DOMAINS:
        errors.append("domain must be airline, retail, or telecom")
    if capture.get("split") not in {"train", "development"}:
        errors.append("split must be train or development; sealed captures are forbidden")
    if not isinstance(capture.get("seed"), int) or isinstance(capture.get("seed"), bool):
        errors.append("seed must be an integer")
    events = capture.get("events")
    if not isinstance(events, list) or not events:
        errors.append("events must be a non-empty list")
    elif not any(isinstance(item, dict) and item.get("type") == "user_message" for item in events):
        errors.append("events must contain a user_message")
    if not isinstance(capture.get("tools"), list) or not capture["tools"]:
        errors.append("tools must contain exact recorded tool definitions")
    transition = capture.get("state_transition")
    if not isinstance(transition, dict):
        errors.append("state_transition must be an object")
    else:
        for field in ("before_hash", "after_hash", "changes", "executable"):
            if field not in transition:
                errors.append(f"state_transition.{field} is required")
        if transition.get("before_hash") != capture.get("starting_state_hash"):
            errors.append("state_transition.before_hash must match starting_state_hash")
        if transition.get("executable") is not True:
            errors.append("state_transition.executable must be true")
        changes = transition.get("changes")
        if isinstance(changes, list):
            for index, change in enumerate(changes):
                if not isinstance(change, dict):
                    errors.append(f"state_transition.changes[{index}] must be an object")
                    continue
                if change.get("kind") not in {"added", "removed", "changed"}:
                    errors.append(
                        f"state_transition.changes[{index}].kind must be added, removed, or changed"
                    )
                if not isinstance(change.get("path"), str) or not change["path"]:
                    errors.append(f"state_transition.changes[{index}].path must be a non-empty string")
                if "before" not in change or "after" not in change:
                    errors.append(f"state_transition.changes[{index}] must include before and after")
    outcome = capture.get("outcome")
    if not isinstance(outcome, dict):
        errors.append("outcome must be an object")
    else:
        for field in ("success", "executable_label", "policy_violation", "harmful_mutation", "evidence_refs"):
            if field not in outcome:
                errors.append(f"outcome.{field} is required")
        if not isinstance(outcome.get("success"), bool):
            errors.append("outcome.success must be a boolean")
        if not isinstance(outcome.get("evidence_refs"), list) or not outcome["evidence_refs"]:
            errors.append("outcome.evidence_refs must be a non-empty list")
    review = capture.get("review")
    if not isinstance(review, dict):
        errors.append("review must be an object")
    else:
        for field in ("reviewer", "verifier", "disposition", "reason"):
            if not isinstance(review.get(field), str) or not review[field].strip():
                errors.append(f"review.{field} must be a non-empty string")
    if isinstance(capture.get("prompt"), str):
        expected = canonical_sha256(capture["prompt"])
        if capture.get("prompt_hash") != expected:
            errors.append("prompt_hash must replay from prompt")
    else:
        errors.append("prompt must be a string")
    if capture.get("sealed") is True or capture.get("sealed_evaluation") is True:
        errors.append("sealed captures are forbidden from the training capture path")
    return errors


def capture_to_hfr(capture: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Project one validated Tau capture to Flight Recorder run artifacts."""

    errors = validate_tau3_capture(capture)
    if errors:
        raise Tau3CaptureError("; ".join(errors))
    trajectory_id = str(capture["trajectory_id"])
    outcome = capture["outcome"]
    transition = capture["state_transition"]
    review = capture["review"]
    success = outcome["success"] is True
    trace_events = [_trace_event(index, raw, trajectory_id) for index, raw in enumerate(capture["events"])]
    final_answer = str(capture.get("final_answer") or _last_assistant_text(trace_events))
    trace = {
        "schema_version": "hfr.trace.v1",
        "session": {
            "id": trajectory_id,
            "model": str(capture["generator_id"]),
            "source_format": "tau3_text_capture",
        },
        "events": trace_events,
        "final_answer": final_answer,
        "metadata": {
            "completed": True,
            "domain": str(capture["domain"]),
            "task_id": str(capture["task_id"]),
            "split": str(capture["split"]),
            "seed": int(capture["seed"]),
        },
    }
    evidence_refs = [
        {
            "target": "episode",
            "description": str(value),
            "passed": success,
        }
        for value in outcome["evidence_refs"]
    ]
    completion_check = {
        "id": "tau_executable_outcome",
        "rule_id": "required_state_and_outcome",
        "description": "Tau executable outcome and state transition agree.",
        "passed": success,
        "evidence": str(outcome["executable_label"]),
        "evidence_refs": evidence_refs,
    }
    task_completion = {
        "schema_version": "hfr.task_completion.v1",
        "status": "complete" if success else "incomplete",
        "passed": success,
        "task_evidence_configured": True,
        "required_check_count": 1,
        "passed_check_count": 1 if success else 0,
        "failed_check_count": 0 if success else 1,
        "blocking_rule_ids": [] if success else ["required_state_and_outcome"],
        "summary": "Executable Tau task evidence passed." if success else "Executable Tau task evidence failed.",
        "checks": [completion_check],
        "evidence_refs": evidence_refs,
        "missing_evidence_refs": [],
    }
    failed_rule = {
        "id": "required_state_and_outcome",
        "name": "Tau executable state and policy outcome",
        "passed": success,
        "critical": True,
        "penalty": 100,
        "evidence": [str(outcome["executable_label"])],
        "evidence_refs": evidence_refs,
        "items": [completion_check],
    }
    scorecard = {
        "schema_version": "hfr.scorecard.v1",
        "scenario_id": trajectory_id,
        "scenario_title": f"Tau-3 {capture['domain']} {capture['task_id']}",
        "task_family": str(capture["task_family"]),
        "score": 100 if success else 0,
        "pass_threshold": 100,
        "passed": success,
        "critical_failures": [] if success else ["required_state_and_outcome"],
        "summary": "PASS: executable outcome and policy evidence agree." if success else "FAIL: executable outcome or policy evidence rejected the trajectory.",
        "rules": [failed_rule],
        "task_completion": task_completion,
    }
    changes = transition.get("changes") if isinstance(transition.get("changes"), list) else []
    state_diff = {
        "schema_version": "hfr.state_diff.v1",
        "changed": bool(changes),
        "change_count": len(changes),
        "truncated": False,
        "comparison_complete": True,
        "change_status": "changed" if changes else "unchanged",
        "max_changes": max(200, len(changes)),
        "changes": changes,
        "summary": f"{len(changes)} state change(s) captured from the Tau environment.",
    }
    trajectory_v2 = trajectory_v2_from_trace(
        trace,
        source_format="tau3_text_capture",
        context={
            "root_session_id": trajectory_id,
            "model": {
                "provider": "local",
                "name": str(capture["generator_id"]),
                "revision": str(capture["generator_revision"]),
            },
            "tokenizer": {
                "name": str(capture.get("tokenizer_id") or capture["generator_id"]),
                "revision": str(capture.get("tokenizer_revision") or capture["generator_revision"]),
            },
            "chat_template": {
                "name": str(capture.get("chat_template") or "tau3-fixed-chat-template"),
                "revision": str(capture.get("chat_template_revision") or "v1"),
                "sha256": str(capture.get("chat_template_hash") or canonical_sha256("tau3-fixed-chat-template-v1")),
            },
            "policy": {
                "id": f"tau3-{capture['domain']}-policy",
                "version": str(capture["policy_revision"]),
                "sha256": str(capture.get("policy_hash") or canonical_sha256(str(capture["policy_revision"]))),
            },
            "environment": {
                "id": f"tau3-{capture['domain']}-text",
                "version": str(capture.get("environment_revision") or capture["generator_revision"]),
                "sha256": str(capture.get("environment_hash") or canonical_sha256(str(capture.get("environment_revision") or capture["generator_revision"]))),
            },
            "governance": _governance(capture),
            "tools": capture["tools"],
            "metadata": {
                "domain": str(capture["domain"]),
                "reviewer": str(review["reviewer"]),
                "verifier": str(review["verifier"]),
            },
        },
    )
    trajectory_status = check_trajectory_v2(trajectory_v2)
    if trajectory_status.get("errors"):
        raise Tau3CaptureError("trajectory_v2 projection failed: " + "; ".join(trajectory_status["errors"]))
    artifacts = {
        "normalized_trace": trace,
        "scorecard": scorecard,
        "task_completion": task_completion,
        "state_diff": state_diff,
    }
    # Quarantined trajectories remain negative evidence, but export_rl_dataset
    # deliberately rejects a trajectory_v2 artifact that is not action-SFT
    # eligible.  Omit only that trainer-facing projection; the lossless trace,
    # tool result, state delta, scorecard, and rejection ledger remain intact.
    if trajectory_status.get("passed") is True:
        artifacts["trajectory_v2"] = trajectory_v2
    return artifacts


def admission_record(capture: dict[str, Any]) -> dict[str, Any]:
    """Build the explicit admission or rejection ledger row for a capture."""

    outcome = capture["outcome"]
    review = capture["review"]
    safe = outcome.get("policy_violation") is False and outcome.get("harmful_mutation") is False
    admitted = outcome.get("success") is True and safe and review.get("disposition") == "admit"
    return {
        "schema_version": "hfr.tau3_admission.v1",
        "trajectory_id": str(capture["trajectory_id"]),
        "domain": str(capture["domain"]),
        "admitted": admitted,
        "disposition": "admitted" if admitted else "rejected",
        "reason": str(review["reason"]),
        "evidence": {
            "lineage": {
                "source": str(capture["generator_id"]),
                "revision": str(capture["generator_revision"]),
                "prompt_hash": str(capture["prompt_hash"]),
                "seed": int(capture["seed"]),
            },
            "state_transition": capture["state_transition"],
            "executable": {
                "label": str(outcome["executable_label"]),
                "success": outcome["success"] is True,
                "evidence_refs": outcome["evidence_refs"],
            },
            "safety": {
                "policy_violation": outcome["policy_violation"] is True,
                "harmful_mutation": outcome["harmful_mutation"] is True,
            },
            "reviewer": {
                "reviewer": str(review["reviewer"]),
                "verifier": str(review["verifier"]),
                "disposition": str(review["disposition"]),
            },
        },
    }


def canonical_sha256(value: Any) -> str:
    """Hash JSON-compatible content with the project's canonical encoding."""

    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_record(path: Path, *, name: str, relative_to: Path) -> dict[str, Any]:
    """Return a portable lineage file record."""

    return {
        "name": name,
        "path": path.relative_to(relative_to).as_posix(),
        "role": "input",
        "sensitive": False,
        "exists": True,
        "size_bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _trace_event(index: int, raw: Any, session_id: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise Tau3CaptureError(f"events[{index}] must be an object")
    event = dict(raw)
    event.setdefault("order", index)
    event.setdefault("session_id", session_id)
    event.setdefault("parent_session_id", None)
    event.setdefault("timestamp", None)
    event.setdefault("status", None)
    event.setdefault("tool_name", None)
    event.setdefault("args", {})
    event.setdefault("text", str(event.get("content") or ""))
    return event


def _last_assistant_text(events: list[dict[str, Any]]) -> str:
    for event in reversed(events):
        if event.get("type") == "assistant_message" or event.get("role") == "assistant":
            return str(event.get("text") or event.get("content") or "")
    return ""


def _governance(capture: dict[str, Any]) -> dict[str, Any]:
    supplied = capture.get("governance") if isinstance(capture.get("governance"), dict) else {}
    return {
        "owner": str(supplied.get("owner") or "tau3-study"),
        "tenant": str(supplied.get("tenant") or "local-research"),
        "legal_basis": str(supplied.get("legal_basis") or "research"),
        "allowed_purposes": supplied.get("allowed_purposes") or ["agent_training", "evaluation"],
        "sensitivity": str(supplied.get("sensitivity") or "synthetic_benchmark"),
        "jurisdiction": str(supplied.get("jurisdiction") or "local"),
        "retention_expires_at": str(supplied.get("retention_expires_at") or "2030-01-01T00:00:00+00:00"),
        "license": str(supplied.get("license") or "benchmark-license-review-required"),
        "provenance": supplied.get("provenance") or {
            "source": "tau3_text_capture",
            "source_revision": str(capture["generator_revision"]),
        },
        "deletion_subject_ids": supplied.get("deletion_subject_ids") or [str(capture["trajectory_id"])],
    }
