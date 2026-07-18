"""Review-aware native action labels, credit, replay, and deterministic curation."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from typing import Any

from .data_governance import task_contract_fingerprint

REVIEWED_ACTION_SFT_SCHEMA_VERSION = "hfr.reviewed.action_sft.v1"
CONTRACT_PREFERENCE_SCHEMA_VERSION = "hfr.reviewed.contract_preference.v1"
ACTION_CREDIT_SCHEMA_VERSION = "hfr.action_credit.v1"
BRANCH_REPLAY_DATASET_SCHEMA_VERSION = "hfr.branch_replay_dataset.v1"
CURATED_DATASET_SCHEMA_VERSION = "hfr.curated_dataset.v1"

_NEGATIVE_LABELS = {"reject", "unsafe", "incomplete"}
_NEGATIVE_STATUSES = {"failed", "failure", "error", "timed_out", "timeout", "denied", "cancelled", "canceled"}
_NEUTRAL_STATUSES = {"superseded", "retried", "skipped", "noop", "no_op"}


class ReviewSemanticsError(ValueError):
    """Raised when review evidence cannot be made training-safe."""


def build_reviewed_action_rows(
    action_rows: list[dict[str, Any]],
    reviewed_labels: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Join accepted human labels to native action rows without flattening them."""

    labels: dict[str, dict[str, Any]] = {}
    for index, label in enumerate(reviewed_labels):
        episode_id = _required_string(label, "episode_id", f"reviewed label {index}")
        if episode_id in labels:
            raise ReviewSemanticsError(f"duplicate reviewed label for episode {episode_id!r}")
        labels[episode_id] = label

    output: list[dict[str, Any]] = []
    seen_actions: set[str] = set()
    for index, action in enumerate(action_rows):
        episode_id = _required_string(action, "episode_id", f"action row {index}")
        if episode_id in seen_actions:
            raise ReviewSemanticsError(f"duplicate action row for episode {episode_id!r}")
        seen_actions.add(episode_id)
        label = labels.get(episode_id)
        if label is None or label.get("human_label") != "accept":
            continue
        messages = action.get("messages")
        tools = action.get("tools")
        if not isinstance(messages, list) or not messages:
            raise ReviewSemanticsError(f"accepted action row {episode_id!r} has no native messages")
        if not isinstance(tools, list):
            raise ReviewSemanticsError(f"accepted action row {episode_id!r} has no native tools list")
        has_tool_calls = any(
            isinstance(message, dict)
            and isinstance(message.get("tool_calls"), list)
            and bool(message["tool_calls"])
            for message in messages
        )
        if has_tool_calls and str(action.get("tool_schema_provenance") or "recorded").startswith("inferred"):
            continue
        trajectory_v2 = action.get("trajectory_v2")
        if isinstance(trajectory_v2, dict):
            from .trajectory_v2 import check_trajectory_v2

            trajectory_status = check_trajectory_v2(trajectory_v2)
            if trajectory_status.get("action_training_eligible") is not True:
                continue
        contract_fingerprint = task_contract_fingerprint(action)
        output.append(
            {
                **action,
                "schema_version": REVIEWED_ACTION_SFT_SCHEMA_VERSION,
                "task_contract_fingerprint": contract_fingerprint,
                "review_item_id": _required_string(label, "review_item_id", f"reviewed label {episode_id}"),
                "review_item_sha256": _required_sha256(label, "review_item_sha256", f"reviewed label {episode_id}"),
                "human_label": "accept",
                "reviewer_confidence": str(label.get("reviewer_confidence") or "unknown"),
                "quality_gate": "human_reviewed_native_action_accept",
                "source_artifact": "reviewed_labels.jsonl+action_sft.jsonl",
            }
        )
    return sorted(output, key=lambda row: str(row["episode_id"]))


def build_contract_preferences(reviewed_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pair accepted/negative rows only when their complete task contracts match."""

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, row in enumerate(reviewed_rows):
        human_label = str(row.get("human_label") or "")
        if human_label != "accept" and human_label not in _NEGATIVE_LABELS:
            continue
        expected = task_contract_fingerprint(row)
        supplied = str(row.get("task_contract_fingerprint") or expected)
        if supplied != expected:
            raise ReviewSemanticsError(f"reviewed row {index} task contract fingerprint is stale")
        grouped[expected].append({**row, "task_contract_fingerprint": expected})

    preferences: list[dict[str, Any]] = []
    for contract, rows in sorted(grouped.items()):
        positives = sorted((row for row in rows if row.get("human_label") == "accept"), key=_episode_id)
        negatives = sorted((row for row in rows if row.get("human_label") in _NEGATIVE_LABELS), key=_episode_id)
        for chosen in positives:
            for rejected in negatives:
                chosen_id = _episode_id(chosen)
                rejected_id = _episode_id(rejected)
                if chosen_id == rejected_id:
                    raise ReviewSemanticsError(f"preference self-pair for episode {chosen_id!r}")
                chosen_completion = _completion(chosen)
                rejected_completion = _completion(rejected)
                chosen_sha = _canonical_sha256(chosen_completion)
                rejected_sha = _canonical_sha256(rejected_completion)
                if chosen_sha == rejected_sha:
                    raise ReviewSemanticsError(
                        f"preference pair {chosen_id!r}>{rejected_id!r} has identical completions"
                    )
                preferences.append(
                    {
                        "schema_version": CONTRACT_PREFERENCE_SCHEMA_VERSION,
                        "preference_id": f"{contract[:16]}:{chosen_id}>{rejected_id}",
                        "task_contract_fingerprint": contract,
                        "task_family": str(chosen.get("task_family") or rejected.get("task_family") or "unknown"),
                        "prompt": str(chosen.get("prompt") or rejected.get("prompt") or ""),
                        "tools": chosen.get("tools") if isinstance(chosen.get("tools"), list) else [],
                        "chosen_episode_id": chosen_id,
                        "rejected_episode_id": rejected_id,
                        "chosen": chosen_completion,
                        "rejected": rejected_completion,
                        "chosen_completion_sha256": chosen_sha,
                        "rejected_completion_sha256": rejected_sha,
                        "chosen_review_item_sha256": chosen.get("review_item_sha256"),
                        "rejected_review_item_sha256": rejected.get("review_item_sha256"),
                        "chosen_reviewer_confidence": chosen.get("reviewer_confidence"),
                        "rejected_reviewer_confidence": rejected.get("reviewer_confidence"),
                        "reason": "Human-reviewed outcomes under an identical task contract.",
                    }
                )
    return preferences


def build_action_credit(trajectory: dict[str, Any]) -> list[dict[str, Any]]:
    """Label each observed tool action independently of terminal episode success."""

    messages = trajectory.get("messages")
    if not isinstance(messages, list):
        raise ReviewSemanticsError("trajectory messages must be a list")
    results: dict[str, dict[str, Any]] = {}
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "tool":
            continue
        call_id = str(message.get("tool_call_id") or "")
        if not call_id:
            raise ReviewSemanticsError("tool result is missing tool_call_id")
        if call_id in results:
            raise ReviewSemanticsError(f"tool call {call_id!r} has multiple results")
        results[call_id] = message

    credits: list[dict[str, Any]] = []
    seen_calls: set[str] = set()
    for message_index, message in enumerate(messages):
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        calls = message.get("tool_calls")
        if not isinstance(calls, list):
            continue
        for call in calls:
            if not isinstance(call, dict):
                raise ReviewSemanticsError(f"assistant message {message_index} has an invalid tool call")
            call_id = _required_string(call, "id", f"tool call at message {message_index}")
            if call_id in seen_calls:
                raise ReviewSemanticsError(f"duplicate tool call id {call_id!r}")
            seen_calls.add(call_id)
            result = results.get(call_id)
            status = _tool_status(result)
            if result is None or status in _NEGATIVE_STATUSES:
                label, reward, confidence = "negative", -1.0, 1.0
            elif status in _NEUTRAL_STATUSES:
                label, reward, confidence = "neutral", 0.0, 1.0
            elif status in {"ok", "success", "passed", "complete", "completed"}:
                label, reward, confidence = "positive", 1.0, 1.0
            else:
                label, reward, confidence = "neutral", 0.0, 0.5
            function = call.get("function") if isinstance(call.get("function"), dict) else {}
            credits.append(
                {
                    "schema_version": ACTION_CREDIT_SCHEMA_VERSION,
                    "episode_id": str(trajectory.get("episode_id") or ""),
                    "message_index": message_index,
                    "tool_call_id": call_id,
                    "tool_name": str(function.get("name") or ""),
                    "label": label,
                    "reward": reward,
                    "status": status or "missing_result",
                    "confidence": confidence,
                    "source": "deterministic_tool_result",
                    "episode_outcome": str(trajectory.get("episode_outcome") or "unknown"),
                    "observation": result if result is not None else None,
                    "action": call,
                }
            )
    unmatched = sorted(set(results) - seen_calls)
    if unmatched:
        raise ReviewSemanticsError(f"tool results have no matching calls: {unmatched!r}")
    return credits


def branch_replay_state_fingerprint(
    source_trajectory: dict[str, Any],
    event_index: int,
) -> str:
    """Fingerprint the exact source prefix and state used to branch a replay."""

    messages = source_trajectory.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ReviewSemanticsError("branch replay source trajectory must contain messages")
    if (
        not isinstance(event_index, int)
        or isinstance(event_index, bool)
        or event_index <= 0
        or event_index > len(messages)
    ):
        raise ReviewSemanticsError(
            f"replay point event_index must be between 1 and {len(messages)}"
        )
    return _canonical_sha256(
        {
            "event_index": event_index,
            "source_prefix_messages": messages[:event_index],
            "tools": source_trajectory.get("tools", []),
            "environment": source_trajectory.get("environment", {}),
            "policy": source_trajectory.get("policy", {}),
            "scenario_contract": source_trajectory.get("scenario_contract", {}),
            "source_state": source_trajectory.get("state", {}),
        }
    )


def build_branch_replay_dataset(
    *,
    source_trajectory: dict[str, Any],
    replay_point: dict[str, Any],
    candidates: list[dict[str, Any]],
    verifier_results: list[dict[str, Any]],
    high_impact: bool,
    novel_behavior: bool,
    grader_disagreement: bool,
    review_confidence_threshold: float = 0.8,
) -> dict[str, Any]:
    """Build deterministic replay preferences from externally generated continuations."""

    event_index = replay_point.get("event_index")
    expected_state_fingerprint = branch_replay_state_fingerprint(
        source_trajectory,
        event_index,
    )
    state_fingerprint = _required_sha256(replay_point, "state_fingerprint", "replay point")
    if state_fingerprint != expected_state_fingerprint:
        raise ReviewSemanticsError(
            "replay point state_fingerprint does not match the exact source prefix and state"
        )
    source_messages = source_trajectory["messages"]
    source_prefix = source_messages[:event_index]
    candidate_by_id: dict[str, dict[str, Any]] = {}
    for index, candidate in enumerate(candidates):
        candidate_id = _required_string(candidate, "candidate_id", f"candidate {index}")
        if candidate_id in candidate_by_id:
            raise ReviewSemanticsError(f"duplicate replay candidate {candidate_id!r}")
        if not isinstance(candidate.get("continuation"), list) or not candidate["continuation"]:
            raise ReviewSemanticsError(f"replay candidate {candidate_id!r} has no continuation")
        candidate_by_id[candidate_id] = candidate
    verifier_by_id: dict[str, dict[str, Any]] = {}
    for result in verifier_results:
        candidate_id = _required_string(result, "candidate_id", "verifier result")
        if candidate_id in verifier_by_id:
            raise ReviewSemanticsError(f"duplicate verifier result for {candidate_id!r}")
        verifier_by_id[candidate_id] = result
    if set(candidate_by_id) != set(verifier_by_id):
        raise ReviewSemanticsError("every replay candidate must have exactly one verifier result")

    ranked = sorted(
        candidate_by_id,
        key=lambda candidate_id: (
            verifier_by_id[candidate_id].get("passed") is True and verifier_by_id[candidate_id].get("safe") is True,
            _number(verifier_by_id[candidate_id].get("score")),
            _number(verifier_by_id[candidate_id].get("confidence")),
            candidate_id,
        ),
        reverse=True,
    )
    eligible = [
        candidate_id
        for candidate_id in ranked
        if verifier_by_id[candidate_id].get("passed") is True and verifier_by_id[candidate_id].get("safe") is True
    ]
    chosen_id = eligible[0] if eligible else ""
    preferences: list[dict[str, Any]] = []
    if chosen_id:
        for rejected_id in ranked:
            if rejected_id == chosen_id:
                continue
            chosen_completion = candidate_by_id[chosen_id]["continuation"]
            rejected_completion = candidate_by_id[rejected_id]["continuation"]
            if _canonical_sha256(chosen_completion) == _canonical_sha256(rejected_completion):
                continue
            preferences.append(
                {
                    "preference_id": f"replay:{state_fingerprint[:12]}:{chosen_id}>{rejected_id}",
                    "chosen_candidate_id": chosen_id,
                    "rejected_candidate_id": rejected_id,
                    "chosen": chosen_completion,
                    "rejected": rejected_completion,
                    "chosen_verifier": verifier_by_id[chosen_id],
                    "rejected_verifier": verifier_by_id[rejected_id],
                }
            )

    low_confidence = any(
        _number(result.get("confidence")) < review_confidence_threshold
        for result in verifier_results
    )
    review_reasons = [
        reason
        for condition, reason in (
            (high_impact, "high_impact"),
            (novel_behavior, "novel_behavior"),
            (grader_disagreement, "grader_disagreement"),
            (low_confidence, "low_confidence"),
            (not chosen_id, "no_verified_candidate"),
        )
        if condition
    ]
    identity = {
        "source": _canonical_sha256(source_trajectory),
        "replay_point": {
            "event_index": event_index,
            "state_fingerprint": state_fingerprint,
        },
        "candidates": candidates,
        "verifiers": verifier_results,
    }
    return {
        "schema_version": BRANCH_REPLAY_DATASET_SCHEMA_VERSION,
        "replay_id": f"replay-{_canonical_sha256(identity)[:16]}",
        "source_episode_id": str(source_trajectory.get("episode_id") or ""),
        "source_trajectory_sha256": _canonical_sha256(source_trajectory),
        "source_prefix_messages": source_prefix,
        "tools": source_trajectory.get("tools", []),
        "task_family": str(source_trajectory.get("task_family") or "unknown"),
        "replay_point": {"event_index": event_index, "state_fingerprint": state_fingerprint},
        "candidate_count": len(candidates),
        "chosen_candidate_id": chosen_id,
        "preference_count": len(preferences),
        "preferences": preferences,
        "verifier_results": [verifier_by_id[candidate_id] for candidate_id in sorted(verifier_by_id)],
        "review_required": bool(review_reasons),
        "review_reasons": review_reasons,
        "generation_boundary": {
            "continuations_generated_by_flight_recorder": False,
            "provider_calls_started": False,
            "source_state_replay_required": True,
        },
    }


def curate_training_rows(
    rows: list[dict[str, Any]],
    *,
    recipe: dict[str, Any],
) -> dict[str, Any]:
    """Select deterministic balanced rows and explain every inclusion/exclusion."""

    seed = str(recipe.get("seed") or "hfr-curation-v1")
    max_rows = _non_negative_int(recipe.get("max_rows")) or len(rows)
    max_per_source = _non_negative_int(recipe.get("max_per_source")) or len(rows)
    minimum_quality = _number(recipe.get("minimum_quality", 0.0))
    allowed_roles = {
        str(value)
        for value in recipe.get("allowed_roles", [])
        if isinstance(value, str) and value
    }
    ranked = sorted(
        (dict(row) for row in rows),
        key=lambda row: (
            -_number(row.get("quality_score")),
            hashlib.sha256(f"{seed}:{_row_id(row)}".encode("utf-8")).hexdigest(),
            _row_id(row),
        ),
    )
    selected: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    source_counts: Counter[str] = Counter()
    for row in ranked:
        row_id = _row_id(row)
        source_id = str(row.get("source_id") or "unknown")
        role = str(row.get("training_role") or "unknown")
        if _number(row.get("quality_score")) < minimum_quality:
            excluded.append({"row_id": row_id, "reason": "minimum_quality", "source_id": source_id})
        elif allowed_roles and role not in allowed_roles:
            excluded.append({"row_id": row_id, "reason": "role_not_allowed", "source_id": source_id})
        elif source_counts[source_id] >= max_per_source:
            excluded.append({"row_id": row_id, "reason": "source_cap", "source_id": source_id})
        elif len(selected) >= max_rows:
            excluded.append({"row_id": row_id, "reason": "row_cap", "source_id": source_id})
        else:
            source_counts[source_id] += 1
            selected.append(
                {
                    **row,
                    "selection_reason": "passed_recipe",
                    "selection_weight": _number(
                        (recipe.get("mixture_weights") or {}).get(role, 1.0)
                        if isinstance(recipe.get("mixture_weights"), dict)
                        else 1.0
                    ),
                }
            )
    weights = [max(0.0, _number(row.get("selection_weight"))) for row in selected]
    weight_sum = sum(weights)
    weight_square_sum = sum(weight * weight for weight in weights)
    effective_sample_size = (weight_sum * weight_sum / weight_square_sum) if weight_square_sum else 0.0
    identity = {
        "recipe": recipe,
        "selected": [_canonical_sha256(row) for row in selected],
        "excluded": excluded,
    }
    return {
        "schema_version": CURATED_DATASET_SCHEMA_VERSION,
        "curation_id": f"hfrcur-{_canonical_sha256(identity)[:16]}",
        "recipe": recipe,
        "recipe_fingerprint": _canonical_sha256(recipe),
        "input_count": len(rows),
        "selected_count": len(selected),
        "excluded_count": len(excluded),
        "selected": selected,
        "excluded": excluded,
        "selected_role_counts": _counts(selected, "training_role"),
        "selected_family_counts": _counts(selected, "task_family"),
        "selected_source_counts": _counts(selected, "source_id"),
        "effective_sample_size": round(effective_sample_size, 6),
        "selection_fingerprint": _canonical_sha256(identity),
    }


def _completion(row: dict[str, Any]) -> Any:
    if "response" in row:
        return row.get("response")
    messages = row.get("messages")
    if isinstance(messages, list):
        return [message for message in messages if isinstance(message, dict) and message.get("role") == "assistant"]
    return ""


def _tool_status(result: dict[str, Any] | None) -> str:
    if result is None:
        return ""
    status = str(result.get("status") or "").strip().casefold()
    if status:
        return status
    content = result.get("content")
    if isinstance(content, str):
        try:
            decoded = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            lowered = content.casefold()
            if any(token in lowered for token in ("failed", "failure", "error", "timeout", "denied")):
                return "failed"
            return "unknown"
        if isinstance(decoded, dict):
            return str(decoded.get("status") or decoded.get("state") or "unknown").casefold()
    return "unknown"


def _counts(rows: list[dict[str, Any]], field: str) -> list[dict[str, Any]]:
    counter = Counter(str(row.get(field) or "unknown") for row in rows)
    return [{field: key, "count": counter[key]} for key in sorted(counter)]


def _required_string(value: dict[str, Any], field: str, label: str) -> str:
    rendered = value.get(field)
    if not isinstance(rendered, str) or not rendered:
        raise ReviewSemanticsError(f"{label} is missing {field}")
    return rendered


def _required_sha256(value: dict[str, Any], field: str, label: str) -> str:
    rendered = _required_string(value, field, label)
    if len(rendered) != 64 or any(character not in "0123456789abcdef" for character in rendered):
        raise ReviewSemanticsError(f"{label} {field} must be a lowercase SHA-256")
    return rendered


def _episode_id(row: dict[str, Any]) -> str:
    return _required_string(row, "episode_id", "reviewed row")


def _row_id(row: dict[str, Any]) -> str:
    return str(row.get("row_id") or row.get("episode_id") or row.get("review_item_id") or "")


def _number(value: Any) -> float:
    if isinstance(value, bool):
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _non_negative_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
