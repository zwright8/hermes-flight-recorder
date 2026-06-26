"""Human-review exports for Flight Recorder evidence runs."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .training import RunRecord, TrainingExportError, load_run_records

REVIEW_MANIFEST_SCHEMA_VERSION = "hfr.review.manifest.v1"
REVIEW_ITEM_SCHEMA_VERSION = "hfr.review.item.v1"
REVIEW_LABEL_SCHEMA_VERSION = "hfr.review.label.v1"
REVIEWED_MANIFEST_SCHEMA_VERSION = "hfr.reviewed.manifest.v1"
REVIEWED_LABEL_SCHEMA_VERSION = "hfr.reviewed.label.v1"
REVIEWED_SFT_SCHEMA_VERSION = "hfr.reviewed.sft.v1"
REVIEWED_REWARD_MODEL_SCHEMA_VERSION = "hfr.reviewed.reward_model.v1"
REVIEWED_PREFERENCE_SCHEMA_VERSION = "hfr.reviewed.preference.v1"
REVIEWED_DPO_SCHEMA_VERSION = "hfr.reviewed.dpo.v1"
REVIEW_LABELS = ("accept", "reject", "needs_review", "unsafe", "incomplete")
TRAINING_NEGATIVE_LABELS = {"reject", "unsafe", "incomplete"}
FAMILY_SUFFIX_RE = re.compile(r"([_-](good|bad|pass|fail|passing|failing|chosen|rejected))+$", re.IGNORECASE)


class ReviewExportError(ValueError):
    """Raised when a human-review export cannot be produced."""


def export_review_queue(
    runs_dir: str | Path,
    out_dir: str | Path,
    *,
    only_failed: bool = False,
    preserve_paths: bool = False,
) -> dict[str, Any]:
    """Export run evidence as a queue for human labeling and curation."""
    source = Path(runs_dir)
    target = Path(out_dir)
    try:
        records = load_run_records(source)
    except TrainingExportError as exc:
        raise ReviewExportError(str(exc)) from exc
    if only_failed:
        records = [record for record in records if not bool(record.scorecard.get("passed"))]
    if not records:
        raise ReviewExportError("No runs matched the requested review export filters")

    target.mkdir(parents=True, exist_ok=True)
    items = [_review_item(record, preserve_paths) for record in records]
    labels = [_label_template(item) for item in items]
    paths = {
        "review_items": target / "review_items.jsonl",
        "label_template": target / "label_template.jsonl",
        "instructions": target / "REVIEW_INSTRUCTIONS.md",
        "manifest": target / "manifest.json",
    }
    _write_jsonl(paths["review_items"], items)
    _write_jsonl(paths["label_template"], labels)

    manifest = {
        "schema_version": REVIEW_MANIFEST_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_runs_dir": _display_path(source, preserve_paths),
        "output_dir": _display_path(target, preserve_paths),
        "only_failed": only_failed,
        "item_count": len(items),
        "passed_count": sum(1 for item in items if item["scorecard"]["passed"] is True),
        "failed_count": sum(1 for item in items if item["scorecard"]["passed"] is False),
        "label_options": list(REVIEW_LABELS),
        "task_families": sorted({str(item["task_family"]) for item in items}),
        "outputs": {name: _display_path(path, preserve_paths) for name, path in paths.items()},
        "notes": [
            "Review items are derived from normalized_trace.json, scorecard.json, report.html, and optional artifact_lineage.json.",
            "Use label_template.jsonl as a starting point for human labels; do not treat suggested labels as ground truth.",
            "Human labels should be grounded in observable events, scorecard evidence, reports, and lineage.",
        ],
    }
    _write_text(paths["instructions"], _instructions(manifest))
    _write_json(paths["manifest"], manifest)
    return manifest


def apply_review_labels(
    review_export_dir: str | Path,
    out_dir: str | Path,
    *,
    labels_path: str | Path | None = None,
    max_pairs_per_family: int = 0,
    preserve_paths: bool = False,
) -> dict[str, Any]:
    """Turn completed human labels into reviewed trainer-ready evidence views."""
    if max_pairs_per_family < 0:
        raise ReviewExportError("max_pairs_per_family must be non-negative")
    source = Path(review_export_dir)
    target = Path(out_dir)
    label_file = Path(labels_path) if labels_path is not None else source / "label_template.jsonl"
    items = _review_items_by_id(source / "review_items.jsonl")
    labels = _read_jsonl(label_file, "review labels")
    reviewed_labels = _reviewed_labels(items, labels, label_file, preserve_paths)
    if not reviewed_labels:
        raise ReviewExportError("No completed human labels found; set human_label in the labels JSONL first")

    target.mkdir(parents=True, exist_ok=True)
    sft = _reviewed_sft(reviewed_labels)
    reward_model = _reviewed_reward_model(reviewed_labels)
    preferences = _reviewed_preferences(reviewed_labels, max_pairs_per_family=max_pairs_per_family)
    dpo = _reviewed_dpo(preferences)
    paths = {
        "reviewed_labels": target / "reviewed_labels.jsonl",
        "reviewed_sft": target / "reviewed_sft.jsonl",
        "reviewed_reward_model": target / "reviewed_reward_model.jsonl",
        "reviewed_preferences": target / "reviewed_preferences.jsonl",
        "reviewed_dpo": target / "reviewed_dpo.jsonl",
        "manifest": target / "manifest.json",
    }
    _write_jsonl(paths["reviewed_labels"], reviewed_labels)
    _write_jsonl(paths["reviewed_sft"], sft)
    _write_jsonl(paths["reviewed_reward_model"], reward_model)
    _write_jsonl(paths["reviewed_preferences"], preferences)
    _write_jsonl(paths["reviewed_dpo"], dpo)

    manifest = {
        "schema_version": REVIEWED_MANIFEST_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_review_export": _display_path(source, preserve_paths),
        "labels_path": _display_path(label_file, preserve_paths),
        "output_dir": _display_path(target, preserve_paths),
        "max_pairs_per_family": max_pairs_per_family,
        "reviewed_label_count": len(reviewed_labels),
        "sft_count": len(sft),
        "reward_model_count": len(reward_model),
        "preference_count": len(preferences),
        "dpo_count": len(dpo),
        "label_counts": _label_counts(reviewed_labels),
        "task_families": sorted({str(row["task_family"]) for row in reviewed_labels}),
        "outputs": {name: _display_path(path, preserve_paths) for name, path in paths.items()},
        "notes": [
            "Reviewed exports are derived from review_items.jsonl plus completed human labels.",
            "Rows with human_label='needs_review' are kept in reviewed_labels.jsonl but excluded from trainer-ready views.",
            "Reviewed SFT rows include only human_label='accept'.",
            "Reviewed reward-model rows include accept/reject/unsafe/incomplete labels.",
            "Reviewed preferences pair accepted rows against rejected/unsafe/incomplete rows in the same task family.",
        ],
    }
    _write_json(paths["manifest"], manifest)
    return manifest


def _review_item(record: RunRecord, preserve_paths: bool) -> dict[str, Any]:
    scorecard = record.scorecard
    trace = record.trace
    scenario_id = str(scorecard.get("scenario_id") or record.run_id)
    passed = bool(scorecard.get("passed"))
    return {
        "schema_version": REVIEW_ITEM_SCHEMA_VERSION,
        "review_item_id": record.run_id,
        "episode_id": record.run_id,
        "scenario_id": scenario_id,
        "scenario_title": str(scorecard.get("scenario_title") or scenario_id),
        "task_family": _task_family(scenario_id),
        "source_artifacts": _source_artifacts(record, preserve_paths),
        "prompt": _prompt_from_trace(trace),
        "final_answer": str(trace.get("final_answer") or ""),
        "event_count": len(trace.get("events", [])) if isinstance(trace.get("events"), list) else 0,
        "scorecard": {
            "passed": passed,
            "score": _score(scorecard),
            "pass_threshold": scorecard.get("pass_threshold"),
            "summary": str(scorecard.get("summary") or ""),
            "critical_failures": scorecard.get("critical_failures", []),
            "failed_rules": _failed_rule_ids(scorecard),
        },
        "rule_summaries": [_rule_summary(rule) for rule in scorecard.get("rules", []) if isinstance(rule, dict)],
        "task_evidence": _task_evidence(scorecard),
        "evidence_target_counts": _evidence_target_counts(scorecard),
        "suggested_human_label": "accept" if passed else "reject",
        "label_options": list(REVIEW_LABELS),
    }


def _source_artifacts(record: RunRecord, preserve_paths: bool) -> dict[str, str]:
    artifacts = {
        "run_dir": record.run_dir,
        "normalized_trace": record.run_dir / "normalized_trace.json",
        "scorecard": record.run_dir / "scorecard.json",
        "report": record.run_dir / "report.html",
    }
    if record.lineage_path is not None:
        artifacts["lineage"] = record.lineage_path
    regression = record.run_dir / "regression_scenario.json"
    if regression.exists():
        artifacts["regression_scenario"] = regression
    return {name: _display_path(path, preserve_paths) for name, path in artifacts.items()}


def _label_template(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": REVIEW_LABEL_SCHEMA_VERSION,
        "review_item_id": item["review_item_id"],
        "episode_id": item["episode_id"],
        "scenario_id": item["scenario_id"],
        "suggested_human_label": item["suggested_human_label"],
        "human_label": None,
        "corrected_score": None,
        "reviewer": None,
        "reviewed_at": None,
        "notes": "",
        "accepted_evidence_refs": [],
        "rejected_evidence_refs": [],
    }


def _review_items_by_id(path: Path) -> dict[str, dict[str, Any]]:
    rows = _read_jsonl(path, "review_items.jsonl")
    items: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        item_id = row.get("review_item_id")
        if not isinstance(item_id, str) or not item_id:
            raise ReviewExportError(f"review_items.jsonl row {index + 1} missing review_item_id")
        if item_id in items:
            raise ReviewExportError(f"review_items.jsonl duplicates review_item_id {item_id!r}")
        items[item_id] = row
    return items


def _reviewed_labels(
    items: dict[str, dict[str, Any]],
    labels: list[dict[str, Any]],
    labels_path: Path,
    preserve_paths: bool,
) -> list[dict[str, Any]]:
    reviewed: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, label in enumerate(labels):
        item_id = label.get("review_item_id")
        if not isinstance(item_id, str) or not item_id:
            raise ReviewExportError(f"label row {index + 1} missing review_item_id")
        if item_id in seen:
            raise ReviewExportError(f"label row {index + 1} duplicates review_item_id {item_id!r}")
        seen.add(item_id)
        item = items.get(item_id)
        if item is None:
            raise ReviewExportError(f"label row {index + 1} references missing review item {item_id!r}")
        human_label = label.get("human_label")
        if human_label is None:
            continue
        if human_label not in REVIEW_LABELS:
            raise ReviewExportError(f"label row {index + 1} has unsupported human_label {human_label!r}")
        corrected_score = label.get("corrected_score")
        if corrected_score is not None and (
            not isinstance(corrected_score, int) or isinstance(corrected_score, bool) or corrected_score < 0 or corrected_score > 100
        ):
            raise ReviewExportError(f"label row {index + 1} corrected_score must be null or an integer from 0 to 100")
        reviewed.append(_reviewed_label_row(item, label, labels_path, preserve_paths))
    return reviewed


def _reviewed_label_row(
    item: dict[str, Any],
    label: dict[str, Any],
    labels_path: Path,
    preserve_paths: bool,
) -> dict[str, Any]:
    scorecard = item.get("scorecard") if isinstance(item.get("scorecard"), dict) else {}
    human_label = str(label["human_label"])
    corrected_score = label.get("corrected_score")
    score = int(corrected_score) if corrected_score is not None else _label_score(human_label, _score(scorecard))
    return {
        "schema_version": REVIEWED_LABEL_SCHEMA_VERSION,
        "review_item_id": item["review_item_id"],
        "episode_id": item.get("episode_id"),
        "scenario_id": item.get("scenario_id"),
        "scenario_title": item.get("scenario_title"),
        "task_family": item.get("task_family"),
        "prompt": item.get("prompt", ""),
        "response": item.get("final_answer", ""),
        "human_label": human_label,
        "suggested_human_label": label.get("suggested_human_label"),
        "corrected_score": corrected_score,
        "score": score,
        "reward": round(score / 100.0, 6),
        "reviewer": label.get("reviewer"),
        "reviewed_at": label.get("reviewed_at"),
        "notes": str(label.get("notes") or ""),
        "accepted_evidence_refs": label.get("accepted_evidence_refs", []),
        "rejected_evidence_refs": label.get("rejected_evidence_refs", []),
        "source_label_file": _display_path(labels_path, preserve_paths),
        "source_artifacts": item.get("source_artifacts", {}),
        "scorecard": {
            "passed": bool(scorecard.get("passed")),
            "score": _score(scorecard),
            "failed_rules": scorecard.get("failed_rules", []),
            "critical_failures": scorecard.get("critical_failures", []),
            "summary": str(scorecard.get("summary") or ""),
        },
    }


def _reviewed_sft(reviewed_labels: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "schema_version": REVIEWED_SFT_SCHEMA_VERSION,
            "review_item_id": row["review_item_id"],
            "episode_id": row["episode_id"],
            "scenario_id": row["scenario_id"],
            "task_family": row["task_family"],
            "prompt": row["prompt"],
            "response": row["response"],
            "human_label": row["human_label"],
            "quality_gate": "human_reviewed_accept",
            "source_artifact": "reviewed_labels.jsonl",
        }
        for row in reviewed_labels
        if row["human_label"] == "accept"
    ]


def _reviewed_reward_model(reviewed_labels: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in reviewed_labels:
        if row["human_label"] == "needs_review":
            continue
        rows.append(
            {
                "schema_version": REVIEWED_REWARD_MODEL_SCHEMA_VERSION,
                "review_item_id": row["review_item_id"],
                "episode_id": row["episode_id"],
                "scenario_id": row["scenario_id"],
                "task_family": row["task_family"],
                "prompt": row["prompt"],
                "response": row["response"],
                "human_label": row["human_label"],
                "score": row["score"],
                "reward": row["reward"],
                "source_artifact": "reviewed_labels.jsonl",
            }
        )
    return rows


def _reviewed_preferences(
    reviewed_labels: list[dict[str, Any]],
    *,
    max_pairs_per_family: int,
) -> list[dict[str, Any]]:
    by_family: dict[str, list[dict[str, Any]]] = {}
    for row in reviewed_labels:
        if row["human_label"] == "accept" or row["human_label"] in TRAINING_NEGATIVE_LABELS:
            by_family.setdefault(str(row["task_family"] or "unknown"), []).append(row)

    preferences: list[dict[str, Any]] = []
    for family, rows in sorted(by_family.items()):
        positives = sorted([row for row in rows if row["human_label"] == "accept"], key=lambda row: str(row["episode_id"]))
        negatives = sorted([row for row in rows if row["human_label"] in TRAINING_NEGATIVE_LABELS], key=lambda row: str(row["episode_id"]))
        pair_count = 0
        for chosen in positives:
            for rejected in negatives:
                preferences.append(_reviewed_preference(family, chosen, rejected))
                pair_count += 1
                if max_pairs_per_family and pair_count >= max_pairs_per_family:
                    break
            if max_pairs_per_family and pair_count >= max_pairs_per_family:
                break
    return preferences


def _reviewed_preference(family: str, chosen: dict[str, Any], rejected: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": REVIEWED_PREFERENCE_SCHEMA_VERSION,
        "preference_id": f"{family}:{chosen['episode_id']}>{rejected['episode_id']}",
        "task_family": family,
        "prompt": chosen.get("prompt") or rejected.get("prompt") or "",
        "chosen_episode_id": chosen["episode_id"],
        "rejected_episode_id": rejected["episode_id"],
        "chosen_label": chosen["human_label"],
        "rejected_label": rejected["human_label"],
        "chosen_score": chosen["score"],
        "rejected_score": rejected["score"],
        "reason": "Human review accepted the chosen episode and rejected the comparison episode.",
        "chosen": {
            "episode_id": chosen["episode_id"],
            "scenario_id": chosen["scenario_id"],
            "response": chosen["response"],
            "score": chosen["score"],
            "human_label": chosen["human_label"],
        },
        "rejected": {
            "episode_id": rejected["episode_id"],
            "scenario_id": rejected["scenario_id"],
            "response": rejected["response"],
            "score": rejected["score"],
            "human_label": rejected["human_label"],
        },
        "source_artifact": "reviewed_labels.jsonl",
    }


def _reviewed_dpo(preferences: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "schema_version": REVIEWED_DPO_SCHEMA_VERSION,
            "preference_id": preference["preference_id"],
            "task_family": preference["task_family"],
            "prompt": preference["prompt"],
            "chosen": preference["chosen"]["response"],
            "rejected": preference["rejected"]["response"],
            "chosen_episode_id": preference["chosen_episode_id"],
            "rejected_episode_id": preference["rejected_episode_id"],
            "reason": preference["reason"],
            "source_artifact": "reviewed_preferences.jsonl",
        }
        for preference in preferences
    ]


def _label_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        label = str(row.get("human_label") or "unknown")
        counts[label] = counts.get(label, 0) + 1
    return counts


def _label_score(label: str, fallback_score: int) -> int:
    if label == "accept":
        return 100
    if label in TRAINING_NEGATIVE_LABELS:
        return 0
    return fallback_score


def _rule_summary(rule: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(rule.get("id") or ""),
        "name": str(rule.get("name") or rule.get("id") or ""),
        "passed": bool(rule.get("passed")),
        "critical": bool(rule.get("critical")),
        "penalty": int(rule.get("penalty", 0) or 0),
        "evidence": [str(item) for item in rule.get("evidence", [])],
        "evidence_ref_count": len(rule.get("evidence_refs", [])) if isinstance(rule.get("evidence_refs"), list) else 0,
    }


def _task_evidence(scorecard: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rule in scorecard.get("rules", []):
        if not isinstance(rule, dict) or rule.get("id") not in {"required_actions", "required_action_sequences", "required_event_counts"}:
            continue
        for item in rule.get("items", []):
            if not isinstance(item, dict):
                continue
            rows.append(
                {
                    "rule_id": str(rule.get("id") or ""),
                    "id": str(item.get("id") or ""),
                    "description": str(item.get("description") or item.get("id") or ""),
                    "passed": bool(item.get("passed")),
                    "evidence": str(item.get("evidence") or ""),
                    "event_indices": item.get("event_indices", []),
                }
            )
    return rows


def _evidence_target_counts(scorecard: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for rule in scorecard.get("rules", []):
        if not isinstance(rule, dict):
            continue
        refs = rule.get("evidence_refs")
        if not isinstance(refs, list):
            continue
        for ref in refs:
            if not isinstance(ref, dict):
                continue
            target = str(ref.get("target") or "unknown")
            counts[target] = counts.get(target, 0) + 1
    return counts


def _instructions(manifest: dict[str, Any]) -> str:
    labels = ", ".join(f"`{label}`" for label in manifest["label_options"])
    return "\n".join(
        [
            "# Flight Recorder Review Queue",
            "",
            "This export is for human curation before evidence becomes training data.",
            "",
            f"- Items: `{manifest['item_count']}`",
            f"- Passed: `{manifest['passed_count']}`",
            f"- Failed: `{manifest['failed_count']}`",
            f"- Label options: {labels}",
            "",
            "Review `review_items.jsonl` alongside each item report and lineage file.",
            "Fill `label_template.jsonl` with `human_label`, `reviewer`, `reviewed_at`, and notes.",
            "Human labels should be grounded in observable trace events, scorecard evidence, reports, and lineage.",
            "A suggested label is only a starting point; prefer observable trace evidence over final-answer claims.",
            "",
        ]
    )


def _prompt_from_trace(trace: dict[str, Any]) -> str:
    for event in trace.get("events", []):
        if isinstance(event, dict) and event.get("type") == "user_message" and event.get("text"):
            return str(event["text"])
    return ""


def _failed_rule_ids(scorecard: dict[str, Any]) -> list[str]:
    return [
        str(rule.get("id"))
        for rule in scorecard.get("rules", [])
        if isinstance(rule, dict) and rule.get("id") and not rule.get("passed")
    ]


def _score(scorecard: dict[str, Any]) -> int:
    try:
        return max(0, min(100, int(scorecard.get("score", 0))))
    except (TypeError, ValueError):
        return 0


def _task_family(scenario_id: str) -> str:
    return FAMILY_SUFFIX_RE.sub("", scenario_id) or scenario_id


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    if not path.exists():
        raise ReviewExportError(f"{label} not found: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ReviewExportError(f"{label}:{line_number} contains invalid JSON: {exc}") from exc
        if not isinstance(value, dict):
            raise ReviewExportError(f"{label}:{line_number} must contain a JSON object")
        rows.append(value)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


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
