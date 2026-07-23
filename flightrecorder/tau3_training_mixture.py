"""Governed Tau-3 MLX training-mixture derivation."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .path_safety import path_has_symlink_component
from .schema_registry import check_schema_contract

TAU3_TRAINING_MIXTURE_SCHEMA_VERSION = "hfr.tau3_training_mixture.v1"

INPUT_FIELDS = {"messages", "metadata", "tools"}
VARIANTS = ("full_trajectories", "assistant_turn_targets", "action_upweighted")
TRAIN_SPLITS = {"train"}
VALID_SPLITS = {"development", "validation", "valid"}
FORBIDDEN_EPISODE_MARKERS = ("-test-", "-sealed-", "_test_", "_sealed_")
EVALUATOR_LEAK_PATTERNS = (
    "check that agent ",
    "check that the agent ",
    "check whether agent ",
    "check whether the agent ",
    "does not offer",
    "does not reveal",
    "does not provide",
    "the agent should not",
    "the assistant should not",
)


class Tau3TrainingMixtureError(ValueError):
    """Raised when mixture derivation would weaken source-data governance."""


@dataclass(frozen=True)
class TokenizerStats:
    row_count: int
    min_rendered_tokens: int
    max_rendered_tokens: int
    over_max_seq_length_count: int
    over_context_window_count: int
    longest_row_id: str
    chat_template_sha256: str

    @property
    def passed(self) -> bool:
        return self.over_max_seq_length_count == 0 and self.over_context_window_count == 0


def build_tau3_training_mixtures(
    source_dir: str | Path,
    out_dir: str | Path,
    *,
    tokenizer_path: str | Path,
    max_seq_length: int = 4096,
    context_window: int = 8192,
    max_action_repeat: int = 3,
    max_action_to_non_action_ratio: float = 3.0,
) -> dict[str, Any]:
    """Derive new-only MLX mixture variants from clean train/valid views."""

    source = Path(source_dir)
    out = Path(out_dir)
    tokenizer_root = Path(tokenizer_path)
    _reject_symlink_path(source, "source")
    _reject_symlink_path(out, "output")
    _reject_symlink_path(tokenizer_root, "tokenizer")
    _require_source_file(source / "train.jsonl", "train source")
    _require_source_file(source / "valid.jsonl", "valid source")
    _require_new_output(out)
    if max_seq_length <= 0 or context_window <= 0:
        raise Tau3TrainingMixtureError("token sequence budgets must be positive")
    if max_action_repeat < 1:
        raise Tau3TrainingMixtureError("max_action_repeat must be at least 1")
    if max_action_to_non_action_ratio < 1:
        raise Tau3TrainingMixtureError("max_action_to_non_action_ratio must be at least 1")

    train = _load_split(source / "train.jsonl", split_name="train", allowed_splits=TRAIN_SPLITS)
    valid = _load_split(source / "valid.jsonl", split_name="valid", allowed_splits=VALID_SPLITS)
    overlap = sorted({row.family_id for row in train} & {row.family_id for row in valid})
    if overlap:
        raise Tau3TrainingMixtureError("train/valid task-family overlap: " + overlap[0])

    tokenizer = _load_tokenizer(tokenizer_root)
    variants = {
        "full_trajectories": {
            "train": [_full_row(row) for row in train],
            "valid": [_full_row(row) for row in valid],
        },
        "assistant_turn_targets": {
            "train": _assistant_turn_rows(train),
            "valid": _assistant_turn_rows(valid),
        },
    }
    variants["action_upweighted"] = {
        split: _action_upweighted_rows(
            rows,
            max_action_repeat=max_action_repeat,
            max_action_to_non_action_ratio=max_action_to_non_action_ratio,
        )
        for split, rows in variants["assistant_turn_targets"].items()
    }

    out.mkdir(parents=True)
    source_binding = _source_binding(source)
    variant_manifests = []
    for variant_name in VARIANTS:
        variant_dir = out / variant_name
        variant_dir.mkdir()
        rows_by_split = variants[variant_name]
        for split in ("train", "valid"):
            _write_jsonl(variant_dir / f"{split}.jsonl", rows_by_split[split])
        token_stats = _tokenizer_stats(
            tokenizer,
            [row for split in ("train", "valid") for row in rows_by_split[split]],
            max_seq_length=max_seq_length,
            context_window=context_window,
        )
        if not token_stats.passed:
            raise Tau3TrainingMixtureError(f"{variant_name} exceeds tokenizer sequence/context budget")
        manifest = _variant_manifest(
            variant_dir,
            variant_name,
            rows_by_split,
            source_binding=source_binding,
            tokenizer_stats=token_stats,
            max_seq_length=max_seq_length,
            context_window=context_window,
        )
        check = check_schema_contract(manifest, name_or_id=TAU3_TRAINING_MIXTURE_SCHEMA_VERSION)
        if check["passed"] is not True:
            raise Tau3TrainingMixtureError(f"{variant_name} manifest schema failed: {check['errors']}")
        _write_json(variant_dir / "manifest.json", manifest)
        variant_manifests.append(manifest)

    root_manifest = {
        "schema_version": TAU3_TRAINING_MIXTURE_SCHEMA_VERSION,
        "variant": "mixture_set",
        "format": "mlx-chat-jsonl",
        "passed": True,
        "source_binding": source_binding,
        "variants": [
            {
                "name": manifest["variant"],
                "path": manifest["variant"],
                "train_count": manifest["counts"]["train"],
                "valid_count": manifest["counts"]["valid"],
                "manifest_sha256": _sha256(out / manifest["variant"] / "manifest.json"),
            }
            for manifest in variant_manifests
        ],
        "sealed_rows": 0,
        "test_rows": 0,
        "training_started": False,
    }
    _write_json(out / "manifest.json", root_manifest)
    return root_manifest


@dataclass(frozen=True)
class _SourceRow:
    split_name: str
    line_number: int
    source_row_sha256: str
    episode_id: str
    family_id: str
    row: dict[str, Any]


def _load_split(path: Path, *, split_name: str, allowed_splits: set[str]) -> list[_SourceRow]:
    rows = []
    seen_episodes: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise Tau3TrainingMixtureError(f"{path.name}:{line_number}: invalid JSON: {exc.msg}") from exc
        if not isinstance(row, dict):
            raise Tau3TrainingMixtureError(f"{path.name}:{line_number}: row must be an object")
        unexpected = sorted(set(row) - INPUT_FIELDS)
        if unexpected:
            raise Tau3TrainingMixtureError(f"{path.name}:{line_number}: unexpected field: {unexpected[0]}")
        metadata = _object(row.get("metadata"), f"{path.name}:{line_number}: metadata")
        episode_id = str(metadata.get("episode_id") or "")
        if not episode_id:
            raise Tau3TrainingMixtureError(f"{path.name}:{line_number}: missing episode_id")
        if episode_id in seen_episodes:
            raise Tau3TrainingMixtureError(f"{path.name}:{line_number}: duplicate episode_id: {episode_id}")
        seen_episodes.add(episode_id)
        episode_split = _episode_split(episode_id)
        if episode_split not in allowed_splits:
            raise Tau3TrainingMixtureError(f"{path.name}:{line_number}: split mismatch for {episode_id}")
        if any(marker in episode_id.lower() for marker in FORBIDDEN_EPISODE_MARKERS):
            raise Tau3TrainingMixtureError(f"{path.name}:{line_number}: sealed/test episode rejected: {episode_id}")
        if str(metadata.get("source_fingerprint_status") or "") != "verified":
            raise Tau3TrainingMixtureError(f"{path.name}:{line_number}: unverified source fingerprint")
        messages = _messages(row.get("messages"), f"{path.name}:{line_number}: messages")
        tools = _tools(row.get("tools"), f"{path.name}:{line_number}: tools")
        _reject_evaluator_leak(messages, f"{path.name}:{line_number}")
        _validate_tool_pairing(messages, tools, f"{path.name}:{line_number}")
        rows.append(_SourceRow(
            split_name=split_name,
            line_number=line_number,
            source_row_sha256=_canonical_sha256(row),
            episode_id=episode_id,
            family_id=str(metadata.get("task_family") or metadata.get("family_id") or episode_id),
            row=row,
        ))
    if not rows:
        raise Tau3TrainingMixtureError(f"{path.name}: no rows")
    return rows


def _full_row(source: _SourceRow) -> dict[str, Any]:
    row = source.row
    return _derived_row(
        source,
        messages=[dict(message) for message in row["messages"]],
        tools=row["tools"],
        variant="full_trajectories",
        target_index=None,
        target_kind="trajectory",
        repetition_index=0,
    )


def _assistant_turn_rows(rows: list[_SourceRow]) -> list[dict[str, Any]]:
    derived = []
    for source in rows:
        messages = source.row["messages"]
        for index, message in enumerate(messages):
            if message.get("role") != "assistant":
                continue
            derived.append(_derived_row(
                source,
                messages=[dict(item) for item in messages[: index + 1]],
                tools=source.row["tools"],
                variant="assistant_turn_targets",
                target_index=index,
                target_kind=_assistant_target_kind(message),
                repetition_index=0,
            ))
    if not derived:
        raise Tau3TrainingMixtureError("assistant-turn mixture has no assistant targets")
    return derived


def _action_upweighted_rows(
    rows: list[dict[str, Any]],
    *,
    max_action_repeat: int,
    max_action_to_non_action_ratio: float,
) -> list[dict[str, Any]]:
    base = [_with_repetition(row, 0) for row in rows]
    action = [row for row in rows if row["metadata"]["target_kind"] == "tool_call"]
    non_action = [row for row in rows if row["metadata"]["target_kind"] != "tool_call"]
    if not action or not non_action:
        return base
    allowed_action_total = math.floor(len(non_action) * max_action_to_non_action_ratio)
    target_action_total = min(len(action) * max_action_repeat, allowed_action_total)
    extras_needed = max(0, target_action_total - len(action))
    extras = []
    for repetition_index in range(1, max_action_repeat):
        for row in action:
            if len(extras) >= extras_needed:
                break
            extras.append(_with_repetition(row, repetition_index))
        if len(extras) >= extras_needed:
            break
    return base + extras


def _derived_row(
    source: _SourceRow,
    *,
    messages: list[dict[str, Any]],
    tools: Any,
    variant: str,
    target_index: int | None,
    target_kind: str,
    repetition_index: int,
) -> dict[str, Any]:
    source_metadata = _object(source.row.get("metadata"), "metadata")
    source_hashes = {
        "source_row_sha256": source.source_row_sha256,
        "source_messages_sha256": _canonical_sha256(source.row["messages"]),
        "source_metadata_sha256": _canonical_sha256(source_metadata),
        "source_tools_sha256": _canonical_sha256(tools),
    }
    output_metadata = {
        **source_metadata,
        "mixture_schema_version": TAU3_TRAINING_MIXTURE_SCHEMA_VERSION,
        "mixture_variant": variant,
        "source_split": source.split_name,
        "source_line_number": source.line_number,
        "source_episode_id": source.episode_id,
        "source_family_id": source.family_id,
        "target_message_index": target_index,
        "target_kind": target_kind,
        "repetition_index": repetition_index,
        "provenance_hashes": source_hashes,
    }
    derived = {"messages": messages, "tools": tools, "metadata": output_metadata}
    output_metadata["derived_row_sha256"] = _canonical_sha256({
        "messages": messages,
        "tools": tools,
        "metadata_without_derived_hash": {k: v for k, v in output_metadata.items() if k != "derived_row_sha256"},
    })
    return derived


def _with_repetition(row: dict[str, Any], repetition_index: int) -> dict[str, Any]:
    copied = {
        "messages": [dict(message) for message in row["messages"]],
        "tools": row["tools"],
        "metadata": dict(row["metadata"]),
    }
    copied["metadata"]["mixture_variant"] = "action_upweighted"
    copied["metadata"]["repetition_index"] = repetition_index
    copied["metadata"]["derived_row_sha256"] = _canonical_sha256({
        "messages": copied["messages"],
        "tools": copied["tools"],
        "metadata_without_derived_hash": {k: v for k, v in copied["metadata"].items() if k != "derived_row_sha256"},
    })
    return copied


def _variant_manifest(
    variant_dir: Path,
    variant_name: str,
    rows_by_split: dict[str, list[dict[str, Any]]],
    *,
    source_binding: dict[str, Any],
    tokenizer_stats: TokenizerStats,
    max_seq_length: int,
    context_window: int,
) -> dict[str, Any]:
    counts = {split: len(rows_by_split[split]) for split in ("train", "valid")}
    target_counts: dict[str, int] = {}
    for row in rows_by_split["train"] + rows_by_split["valid"]:
        kind = str(row["metadata"]["target_kind"])
        target_counts[kind] = target_counts.get(kind, 0) + 1
    return {
        "schema_version": TAU3_TRAINING_MIXTURE_SCHEMA_VERSION,
        "variant": variant_name,
        "format": "mlx-chat-jsonl",
        "passed": True,
        "source_binding": source_binding,
        "counts": counts,
        "target_counts": target_counts,
        "files": {
            split: {
                "path": f"{split}.jsonl",
                "size": (variant_dir / f"{split}.jsonl").stat().st_size,
                "sha256": _sha256(variant_dir / f"{split}.jsonl"),
            }
            for split in ("train", "valid")
        },
        "tokenizer": {
            "checked": True,
            "method": "pinned_base_apply_chat_template",
            "chat_template_sha256": tokenizer_stats.chat_template_sha256,
            "row_count": tokenizer_stats.row_count,
            "min_rendered_tokens": tokenizer_stats.min_rendered_tokens,
            "max_rendered_tokens": tokenizer_stats.max_rendered_tokens,
            "max_seq_length": max_seq_length,
            "harness_context_window": context_window,
            "over_max_seq_length_count": tokenizer_stats.over_max_seq_length_count,
            "over_context_window_count": tokenizer_stats.over_context_window_count,
            "longest_row_id": tokenizer_stats.longest_row_id,
        },
        "sealed_rows": 0,
        "test_rows": 0,
        "training_started": False,
    }


def _tokenizer_stats(
    tokenizer: Any,
    rows: list[dict[str, Any]],
    *,
    max_seq_length: int,
    context_window: int,
) -> TokenizerStats:
    lengths: list[tuple[int, str]] = []
    for row in rows:
        encoded = tokenizer.apply_chat_template(
            row["messages"],
            tools=row.get("tools"),
            tokenize=True,
            add_generation_prompt=False,
        )
        input_ids = encoded.get("input_ids") if hasattr(encoded, "get") else encoded
        if not isinstance(input_ids, list) or not input_ids:
            raise Tau3TrainingMixtureError("pinned tokenizer returned no input_ids")
        lengths.append((len(input_ids), str(row["metadata"].get("derived_row_sha256") or "")))
    longest_length, longest_row_id = max(lengths, default=(0, ""))
    return TokenizerStats(
        row_count=len(lengths),
        min_rendered_tokens=min((length for length, _ in lengths), default=0),
        max_rendered_tokens=longest_length,
        over_max_seq_length_count=sum(length > max_seq_length for length, _ in lengths),
        over_context_window_count=sum(length > context_window for length, _ in lengths),
        longest_row_id=longest_row_id,
        chat_template_sha256=_canonical_sha256(str(getattr(tokenizer, "chat_template", "") or "")),
    )


def _load_tokenizer(path: Path) -> Any:
    try:
        from transformers import AutoTokenizer
    except Exception as exc:  # pragma: no cover - exercised by integration environments.
        raise Tau3TrainingMixtureError("transformers is required to render with the pinned tokenizer") from exc
    try:
        return AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    except Exception as exc:
        raise Tau3TrainingMixtureError(f"could not load pinned local tokenizer: {type(exc).__name__}") from exc


def _validate_tool_pairing(messages: list[dict[str, Any]], tools: list[dict[str, Any]], label: str) -> None:
    tool_names = {_tool_name(tool) for tool in tools}
    pending: dict[str, str] = {}
    for index, message in enumerate(messages):
        role = message.get("role")
        if role == "assistant":
            for call in message.get("tool_calls") or []:
                if not isinstance(call, dict):
                    raise Tau3TrainingMixtureError(f"{label}: tool call at message {index} must be an object")
                call_id = str(call.get("id") or "")
                function = _object(call.get("function"), f"{label}: tool call function")
                name = str(function.get("name") or "")
                if not call_id or not name:
                    raise Tau3TrainingMixtureError(f"{label}: incomplete tool call at message {index}")
                if name not in tool_names:
                    raise Tau3TrainingMixtureError(f"{label}: missing tool schema for {name}")
                if call_id in pending:
                    raise Tau3TrainingMixtureError(f"{label}: duplicate tool_call_id {call_id}")
                pending[call_id] = name
        elif role == "tool":
            call_id = str(message.get("tool_call_id") or "")
            if call_id not in pending:
                raise Tau3TrainingMixtureError(f"{label}: unpaired tool result {call_id}")
            del pending[call_id]
        elif pending:
            raise Tau3TrainingMixtureError(f"{label}: tool call not immediately paired before message {index}")
    if pending:
        raise Tau3TrainingMixtureError(f"{label}: missing tool result for {sorted(pending)[0]}")


def _reject_evaluator_leak(messages: list[dict[str, Any]], label: str) -> None:
    for index, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        content = str(message.get("content") or "").strip().lower()
        if any(pattern in content for pattern in EVALUATOR_LEAK_PATTERNS):
            raise Tau3TrainingMixtureError(f"{label}: evaluator-criteria leakage in assistant target at message {index}")


def _assistant_target_kind(message: dict[str, Any]) -> str:
    if message.get("tool_calls"):
        return "tool_call"
    content = str(message.get("content") or "").lower()
    if "transfer" in content or "cannot" in content or "can't" in content or "not possible" in content:
        return "refusal_or_clarification"
    return "final_answer"


def _episode_split(episode_id: str) -> str:
    parts = episode_id.split("-")
    return parts[1] if len(parts) > 2 and parts[0] == "tau3" else ""


def _source_binding(source: Path) -> dict[str, Any]:
    return {
        "source_dir": str(source),
        "train": {"path": "train.jsonl", "sha256": _sha256(source / "train.jsonl")},
        "valid": {"path": "valid.jsonl", "sha256": _sha256(source / "valid.jsonl")},
    }


def _messages(value: Any, label: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise Tau3TrainingMixtureError(f"{label} must be a non-empty list")
    messages = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise Tau3TrainingMixtureError(f"{label}[{index}] must be an object")
        role = item.get("role")
        if role not in {"user", "assistant", "tool", "system"}:
            raise Tau3TrainingMixtureError(f"{label}[{index}] has invalid role")
        messages.append(item)
    if not any(message.get("role") == "assistant" for message in messages):
        raise Tau3TrainingMixtureError(f"{label} has no assistant target")
    return messages


def _tools(value: Any, label: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise Tau3TrainingMixtureError(f"{label} must be a non-empty list")
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise Tau3TrainingMixtureError(f"{label}[{index}] must be an object")
        name = _tool_name(item)
        function = item.get("function")
        if not name or not isinstance(function, dict) or not isinstance(function.get("parameters"), dict):
            raise Tau3TrainingMixtureError(f"{label}[{index}] is missing exact function schema")
    return value


def _tool_name(tool: dict[str, Any]) -> str:
    function = tool.get("function") if isinstance(tool.get("function"), dict) else {}
    return str(tool.get("name") or function.get("name") or "")


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise Tau3TrainingMixtureError(f"{label} must be an object")
    return value


def _require_source_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise Tau3TrainingMixtureError(f"{label} is missing: {path}")
    _reject_symlink_path(path, label)


def _reject_symlink_path(path: Path, label: str) -> None:
    if path_has_symlink_component(path, include_leaf=True):
        raise Tau3TrainingMixtureError(f"{label} path must not contain symlink components: {path}")


def _require_new_output(path: Path) -> None:
    if path.exists():
        if path.is_dir() and not any(path.iterdir()):
            return
        raise Tau3TrainingMixtureError(f"output directory must be new or empty: {path}")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def remove_output_for_tests(path: Path) -> None:
    """Test helper for cleaning temporary mixture output directories."""

    shutil.rmtree(path)
