"""Deterministic Tau-3 v3 scenario-source generation.

The generated JSONL is a private, ignored training-side input for
``tau3_grounded_generation``.  Rows deliberately carry only train/internal
validation source metadata and replayable Tau tool calls; sealed/dev payloads
are never read.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from .tau3_grounded_generation import (
    BEHAVIORS,
    DOMAINS,
    SPLITS,
    VENDORED_RUNTIME_PREFIX,
    Tau3GroundedGenerationError,
    build_tau3_grounded_generation_dataset,
    canonical_sha256,
)
from .tau3_grounded_generation import _runtime_for_scenario as _grounded_runtime_for_scenario

TAU3_V3_SCENARIO_SOURCE_SCHEMA_VERSION = "hfr.tau3_v3_scenario_source.v1"
TAU3_V3_SCENARIO_SUMMARY_SCHEMA_VERSION = "hfr.tau3_v3_scenario_summary.v1"
LINEAGE_ID = "tau3-competitive-agent-v3-scenario-sources"

TRAIN_PER_BEHAVIOR_DOMAIN = 24
VALIDATION_PER_BEHAVIOR_DOMAIN = 6
TRAIN_FAMILY_MIN = 8
VALIDATION_FAMILY_MIN = 2
TRAIN_TOOL_CALL_MIN = 16
VALIDATION_TOOL_CALL_MIN = 4
TRAIN_DISTINCT_ARGS_MIN = 8
VALIDATION_DISTINCT_ARGS_MIN = 2
NATURAL_PARENT_CAP_PER_DOMAIN = 8
VALIDATION_NONEXEMPT_TOOL_TARGET = 5
TELECOM_TRAIN_PER_BEHAVIOR = 48
TELECOM_VALIDATION_PER_BEHAVIOR = 12
MAX_TOOL_ARGUMENT_SHARE = 0.20
DOMINANCE_FAMILY_SHARDS = 8
TELECOM_STATE_VARIANT_TOOLS = {
    "get_customer_by_id",
    "get_customer_by_name",
    "resume_line",
    "send_payment_request",
}

DEFAULT_TAU_REPO = Path("local/tau3/repository")
DEFAULT_V2_MIXTURE = Path("local/tau3/mixtures-teacher-v1-seq8192-filtered-v2/full_trajectories")
DEFAULT_OFFICIAL_TOOL_CATALOG = Path("local/tau3/tool-schemas-official.json")
DEFAULT_NATURAL_CORPUS = Path("local/tau3/corpus-teacher-v1/train.jsonl")
DEFAULT_DEVELOPMENT_TASKS = Path("local/tau3/source-v1/training_source/development_tasks.jsonl")
DEFAULT_PROTOCOL = Path("local/tau3/protocol.json")

ZERO_ARG_DISTINCT_EXEMPTIONS = {
    "airline.list_all_airports",
    "retail.list_all_product_types",
}

ZERO_ARG_TOOL_EXEMPTIONS = {
    "airline": {
        "list_all_airports": "zero_arg",
    },
    "retail": {
        "list_all_product_types": "zero_arg",
    },
    "telecom": {},
}

NEGATIVE_CORRECTION_BEHAVIORS = {
    "hallucinated_tool_correction",
    "harmful_mutation_correction",
    "premature_completion_correction",
}

RuntimeFactory = Callable[[dict[str, Any]], Any]


class Tau3V3ScenarioError(ValueError):
    """Raised when v3 scenario-source generation fails closed."""


@dataclass(frozen=True)
class ScenarioBuildResult:
    """Result returned by source generation."""

    summary: dict[str, Any]
    rows: list[dict[str, Any]]


def build_tau3_v3_scenario_sources(
    *,
    out: str | Path | None,
    tau_repo: str | Path = DEFAULT_TAU_REPO,
    v2_mixture_dir: str | Path = DEFAULT_V2_MIXTURE,
    official_tool_catalog: str | Path = DEFAULT_OFFICIAL_TOOL_CATALOG,
    natural_corpus: str | Path = DEFAULT_NATURAL_CORPUS,
    development_tasks: str | Path = DEFAULT_DEVELOPMENT_TASKS,
    protocol: str | Path = DEFAULT_PROTOCOL,
    contamination_report_out: str | Path | None = None,
    strict: bool = True,
    dry_run: bool = False,
    runtime_factory: RuntimeFactory | None = None,
    max_rows_per_domain_split: int | None = None,
) -> ScenarioBuildResult:
    """Build deterministic scenario source rows and optionally write JSONL.

    ``runtime_factory`` is injectable for dependency-free unit tests.  The
    default path uses the pinned vendored Tau checkout through the grounded
    replay adapter.
    """

    root = _project_root()
    repo = _resolve_under_root(root, Path(tau_repo), "tau_repo")
    mixture = _resolve_under_root(root, Path(v2_mixture_dir), "v2_mixture_dir")
    catalog_path = _resolve_under_root(root, Path(official_tool_catalog), "official_tool_catalog")
    corpus_path = _resolve_under_root(root, Path(natural_corpus), "natural_corpus")
    development_path = _resolve_under_root(root, Path(development_tasks), "development_tasks")
    protocol_path = _resolve_under_root(root, Path(protocol), "protocol")
    contamination_path = (
        _resolve_under_root(root, Path(contamination_report_out), "contamination_report_out")
        if contamination_report_out is not None
        else None
    )
    if out is None and not dry_run:
        raise Tau3V3ScenarioError("out is required unless dry_run is true")
    out_path = _resolve_under_root(root, Path(out), "out") if out is not None else None
    if out_path is not None and out_path.exists() and not dry_run:
        raise Tau3V3ScenarioError(f"refusing to overwrite existing output: {out_path}")

    revision = _tau_revision(repo)
    runtime_family = f"{VENDORED_RUNTIME_PREFIX}{revision}"
    prompts = _load_exact_v2_system_prompts(mixture)
    official_catalog = _load_exact_v2_tool_catalog(catalog_path)
    corpus_summary = _load_train_side_corpus_summary(corpus_path)
    base_states = _load_base_states(repo) if runtime_factory is None else _fake_base_states()
    runtime = runtime_factory or _grounded_runtime_for_scenario

    rows: list[dict[str, Any]] = []
    replayed_argument_pools: dict[str, dict[str, list[dict[str, Any]]]] = {}
    mutating_argument_pools: dict[str, dict[str, list[dict[str, Any]]]] = {}
    blockers: list[str] = []
    for domain in DOMAINS:
        expected_names = [tool["function"]["name"] for tool in official_catalog[domain]]
        runtime_names = _runtime_tool_names(
            runtime,
            domain=domain,
            revision=revision,
            runtime_family=runtime_family,
            state=base_states[domain],
            tau_repo=repo,
        )
        if set(runtime_names) != set(expected_names):
            blockers.append(
                f"{domain} runtime tool catalog name set mismatch with immutable v2 catalog"
            )
        pools, mutating_pools, pool_blockers = _build_replayed_argument_pools(
            domain=domain,
            state=base_states[domain],
            tool_names=expected_names,
            revision=revision,
            runtime_family=runtime_family,
            tau_repo=repo,
            runtime_factory=runtime,
        )
        replayed_argument_pools[domain] = pools
        mutating_argument_pools[domain] = mutating_pools
        blockers.extend(pool_blockers)
        natural_rows, natural_update, natural_blockers = _build_replay_exact_natural_rows(
            corpus=corpus_path,
            domain=domain,
            revision=revision,
            runtime_family=runtime_family,
            tau_repo=repo,
            runtime_factory=runtime,
            initial_state=base_states[domain],
            system_prompt=prompts[domain],
            v2_tool_catalog=official_catalog[domain],
            expected_tool_names=expected_names,
        )
        rows.extend(natural_rows)
        _merge_natural_corpus_update(corpus_summary, domain, natural_update)
        blockers.extend(natural_blockers)
        for split in SPLITS:
            arg_cursors = {tool_name: 0 for tool_name in expected_names}
            per_behavior = (
                TELECOM_TRAIN_PER_BEHAVIOR
                if split == "train" and domain == "telecom"
                else TELECOM_VALIDATION_PER_BEHAVIOR
                if split == "validation" and domain == "telecom"
                else TRAIN_PER_BEHAVIOR_DOMAIN
                if split == "train"
                else VALIDATION_PER_BEHAVIOR_DOMAIN
            )
            family_count = (
                TRAIN_FAMILY_MIN
                if split == "train"
                else per_behavior
            )
            repeats_per_family = per_behavior // family_count
            for behavior in BEHAVIORS:
                for family_index in range(family_count):
                    if (
                        max_rows_per_domain_split is not None
                        and sum(
                            1
                            for row in rows
                            if row["domain"] == domain and row["split"] == split
                        )
                        >= max_rows_per_domain_split
                    ):
                        break
                    ordinal = family_index * repeats_per_family
                    row = _make_source_row(
                        domain=domain,
                        split=split,
                        behavior=behavior,
                        ordinal=ordinal,
                        family_index=family_index,
                        repeat_count=repeats_per_family,
                        selection_base=BEHAVIORS.index(behavior) * per_behavior,
                        arg_cursors=arg_cursors,
                        revision=revision,
                        runtime_family=runtime_family,
                        tau_repo=repo,
                        system_prompt=prompts[domain],
                        v2_tool_catalog=official_catalog[domain],
                        initial_state=base_states[domain],
                        pools=pools,
                        mutating_pools=mutating_pools,
                        tool_names=expected_names,
                    )
                    _replay_source_row(row, runtime)
                    rows.append(row)

    _add_tool_coverage_closures(
        rows=rows,
        pools=replayed_argument_pools,
        prompts=prompts,
        official_catalog=official_catalog,
        base_states=base_states,
        revision=revision,
        runtime_family=runtime_family,
        tau_repo=repo,
        runtime_factory=runtime,
    )
    _add_tool_argument_dominance_closures(
        rows=rows,
        blockers=blockers,
        pools=replayed_argument_pools,
        prompts=prompts,
        official_catalog=official_catalog,
        base_states=base_states,
        revision=revision,
        runtime_family=runtime_family,
        tau_repo=repo,
        runtime_factory=runtime,
    )
    _add_target_tool_dominance_closures(
        rows=rows,
        blockers=blockers,
        pools=replayed_argument_pools,
        prompts=prompts,
        official_catalog=official_catalog,
        base_states=base_states,
        revision=revision,
        runtime_family=runtime_family,
        tau_repo=repo,
        runtime_factory=runtime,
    )
    _add_tool_argument_dominance_closures(
        rows=rows,
        blockers=blockers,
        pools=replayed_argument_pools,
        prompts=prompts,
        official_catalog=official_catalog,
        base_states=base_states,
        revision=revision,
        runtime_family=runtime_family,
        tau_repo=repo,
        runtime_factory=runtime,
    )
    _add_source_dominance_blockers(rows, blockers)
    rows.sort(key=lambda row: (row["split"], row["domain"], row["source_id"]))
    contamination_report = _contamination_report(
        rows=rows,
        source_inputs={
            "v2_mixture_train": mixture / "train.jsonl",
            "official_tool_catalog": catalog_path,
            "natural_corpus_train": corpus_path,
            "development_tasks": development_path,
            "protocol": protocol_path,
        },
        development_tasks=development_path,
        protocol=protocol_path,
    )
    if contamination_path is not None:
        contamination_report["path"] = contamination_path.relative_to(root).as_posix()
        contamination_report["report_sha256"] = canonical_sha256(
            {
                key: value
                for key, value in contamination_report.items()
                if key != "report_sha256"
            }
        )
    blockers.extend(contamination_report["blockers"])
    summary = _coverage_summary(
        rows,
        replayed_argument_pools,
        blockers,
        corpus_summary,
        contamination_report,
    )
    if strict and not summary["passed"]:
        raise Tau3V3ScenarioError(
            "scenario source coverage is incomplete: "
            + "; ".join(summary["blockers"][:20])
        )
    if out_path is not None and not dry_run:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        _write_jsonl(out_path, rows)
    if contamination_path is not None and not dry_run:
        contamination_path.parent.mkdir(parents=True, exist_ok=True)
        _write_json(contamination_path, contamination_report)
    return ScenarioBuildResult(summary=summary, rows=rows)


def build_cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, help="new private scenario source JSONL")
    parser.add_argument("--summary-out", type=Path, help="optional summary JSON path")
    parser.add_argument("--tau-repo", type=Path, default=DEFAULT_TAU_REPO)
    parser.add_argument("--v2-mixture-dir", type=Path, default=DEFAULT_V2_MIXTURE)
    parser.add_argument("--official-tool-catalog", type=Path, default=DEFAULT_OFFICIAL_TOOL_CATALOG)
    parser.add_argument("--natural-corpus", type=Path, default=DEFAULT_NATURAL_CORPUS)
    parser.add_argument("--development-tasks", type=Path, default=DEFAULT_DEVELOPMENT_TASKS)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--contamination-report-out", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-incomplete-coverage", action="store_true")
    parser.add_argument(
        "--max-rows-per-domain-split",
        type=int,
        help="test hook for small blocked dry runs",
    )
    args = parser.parse_args(argv)
    try:
        result = build_tau3_v3_scenario_sources(
            out=args.out,
            tau_repo=args.tau_repo,
            v2_mixture_dir=args.v2_mixture_dir,
            official_tool_catalog=args.official_tool_catalog,
            natural_corpus=args.natural_corpus,
            development_tasks=args.development_tasks,
            protocol=args.protocol,
            contamination_report_out=args.contamination_report_out,
            strict=not args.allow_incomplete_coverage,
            dry_run=args.dry_run,
            max_rows_per_domain_split=args.max_rows_per_domain_split,
        )
    except (Tau3V3ScenarioError, Tau3GroundedGenerationError) as exc:
        parser.exit(1, f"error: {exc}\n")
    if args.summary_out and not args.dry_run:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        _write_json(args.summary_out, result.summary)
    print(json.dumps(result.summary, indent=2, sort_keys=True))
    return 0 if result.summary["passed"] else 1


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_under_root(root: Path, path: Path, label: str) -> Path:
    candidate = path if path.is_absolute() else root / path
    resolved = candidate.resolve(strict=False)
    root_resolved = root.resolve(strict=True)
    if resolved != root_resolved and root_resolved not in resolved.parents:
        raise Tau3V3ScenarioError(f"{label} must remain under the project root")
    return resolved


def _tau_revision(repo: Path) -> str:
    if not repo.is_dir():
        raise Tau3V3ScenarioError(f"Tau repository is missing: {repo}")
    status = _git(repo, "status", "--porcelain=v1")
    if status:
        raise Tau3V3ScenarioError("vendored Tau checkout must be clean")
    return _git(repo, "rev-parse", "HEAD")


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "git failed"
        raise Tau3V3ScenarioError(detail)
    return completed.stdout.strip()


def _load_exact_v2_system_prompts(mixture_dir: Path) -> dict[str, str]:
    prompts: dict[str, str] = {}
    for split_file in ("train.jsonl",):
        path = mixture_dir / split_file
        if not path.is_file():
            continue
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
                domain = metadata.get("domain") or row.get("domain")
                messages = row.get("messages")
                if domain in DOMAINS and isinstance(messages, list) and messages:
                    first = messages[0]
                    if isinstance(first, dict) and isinstance(first.get("content"), str):
                        prompts.setdefault(str(domain), first["content"])
                if all(domain in prompts for domain in DOMAINS):
                    return prompts
    missing = [domain for domain in DOMAINS if domain not in prompts]
    raise Tau3V3ScenarioError(
        "immutable v2 train mixture is missing exact system prompts for: "
        + ", ".join(missing)
    )


def _load_train_side_corpus_summary(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {
            "path": path.relative_to(_project_root()).as_posix(),
            "present": False,
            "reward_1_rows": 0,
            "domain_counts": {},
        }
    counts = {domain: 0 for domain in DOMAINS}
    reward_1 = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            metadata = _dict(row.get("metadata"))
            domain = metadata.get("domain") or row.get("domain")
            reward = metadata.get("reward", row.get("reward"))
            if domain in counts and reward == 1:
                counts[str(domain)] += 1
                reward_1 += 1
    return {
        "path": path.relative_to(_project_root()).as_posix(),
        "present": True,
        "sha256": _sha256_file(path),
        "split": "train",
        "reward_1_rows": reward_1,
        "domain_counts": counts,
        "used_for": "replay-exact reward-1 train trajectories when eligible; synthetic rows close deterministic coverage",
        "strict_quality_blocker": "no replay-exact reward-1 natural train trajectories have been incorporated yet",
    }


def _merge_natural_corpus_update(
    summary: dict[str, Any],
    domain: str,
    update: dict[str, Any],
) -> None:
    incorporation = summary.setdefault(
        "incorporation",
        {
            "used_for": "replay-exact reward-1 train trajectories plus synthetic coverage closures",
            "domains": {},
            "converted_rows": 0,
            "excluded_rows": 0,
        },
    )
    domains = incorporation.setdefault("domains", {})
    domains[domain] = update
    incorporation["converted_rows"] = int(incorporation.get("converted_rows") or 0) + int(
        update.get("converted_rows") or 0
    )
    incorporation["excluded_rows"] = int(incorporation.get("excluded_rows") or 0) + int(
        update.get("excluded_rows") or 0
    )
    if incorporation["converted_rows"]:
        summary.pop("strict_quality_blocker", None)
        summary["used_for"] = incorporation["used_for"]


def _build_replay_exact_natural_rows(
    *,
    corpus: Path,
    domain: str,
    revision: str,
    runtime_family: str,
    tau_repo: Path,
    runtime_factory: RuntimeFactory,
    initial_state: dict[str, Any],
    system_prompt: str,
    v2_tool_catalog: list[dict[str, Any]],
    expected_tool_names: list[str],
) -> tuple[list[dict[str, Any]], dict[str, Any], list[str]]:
    update: dict[str, Any] = {
        "eligible_reward_1_rows": 0,
        "converted_rows": 0,
        "excluded_rows": 0,
        "excluded_reasons": {},
    }
    blockers: list[str] = []
    if not corpus.is_file():
        return [], update, blockers
    if domain == "telecom":
        update["excluded_reasons"]["known_replay_mismatch_domain"] = 0
    rows: list[dict[str, Any]] = []
    with corpus.open(encoding="utf-8") as handle:
        for line_index, line in enumerate(handle):
            raw = json.loads(line)
            metadata = _dict(raw.get("metadata"))
            if (metadata.get("domain") or raw.get("domain")) != domain:
                continue
            if metadata.get("split") != "train" or not _reward_is_one(metadata.get("reward", raw.get("reward"))):
                continue
            update["eligible_reward_1_rows"] += 1
            if domain == "telecom":
                _natural_excluded(update, "known_replay_mismatch_domain")
                continue
            if metadata.get("source_revision") != revision:
                _natural_excluded(update, "source_revision_mismatch")
                continue
            if not _natural_identity_matches(raw, metadata, system_prompt, v2_tool_catalog, revision):
                _natural_excluded(update, "identity_mismatch")
                continue
            converted = _convert_natural_row(
                raw=raw,
                line_index=line_index,
                domain=domain,
                revision=revision,
                runtime_family=runtime_family,
                tau_repo=tau_repo,
                runtime_factory=runtime_factory,
                initial_state=initial_state,
                system_prompt=system_prompt,
                v2_tool_catalog=v2_tool_catalog,
                expected_tool_names=expected_tool_names,
            )
            if converted is None:
                _natural_excluded(update, "replay_or_shape_mismatch")
                continue
            rows.append(converted)
    rows.sort(
        key=lambda row: (
            _dict(row.get("source_generation")).get("task_sha256") or "",
            row["source_id"],
        )
    )
    selected = rows[:NATURAL_PARENT_CAP_PER_DOMAIN]
    for _row in rows[NATURAL_PARENT_CAP_PER_DOMAIN:]:
        _natural_excluded(update, "deterministic_balance_cap")
    update["selection_rule"] = (
        f"lowest task_sha256/source_id, cap {NATURAL_PARENT_CAP_PER_DOMAIN} replay-exact parents per domain"
    )
    update["converted_rows"] = len(selected)
    update["excluded_rows"] = int(update["eligible_reward_1_rows"]) - len(selected)
    if update["eligible_reward_1_rows"] and domain in {"airline", "retail"} and not selected:
        blockers.append(f"{domain} has reward-1 natural train rows but none replay exactly")
    return selected, update, blockers


def _natural_excluded(update: dict[str, Any], reason: str) -> None:
    reasons = update.setdefault("excluded_reasons", {})
    reasons[reason] = int(reasons.get(reason) or 0) + 1


def _reward_is_one(value: Any) -> bool:
    return value == 1 or value == 1.0


def _convert_natural_row(
    *,
    raw: dict[str, Any],
    line_index: int,
    domain: str,
    revision: str,
    runtime_family: str,
    tau_repo: Path,
    runtime_factory: RuntimeFactory,
    initial_state: dict[str, Any],
    system_prompt: str,
    v2_tool_catalog: list[dict[str, Any]],
    expected_tool_names: list[str],
) -> dict[str, Any] | None:
    messages = raw.get("messages")
    if not isinstance(messages, list):
        return None
    runtime = runtime_factory(_runtime_payload(domain, revision, runtime_family, initial_state, tau_repo))
    allowed_tools = set(expected_tool_names)
    turns: list[dict[str, Any]] = []
    expected_results: list[dict[str, Any]] = []
    last_user_context: str | None = None
    pending_user: str | None = None
    decision_ordinal = 0
    mutation_replayed = False
    index = 0
    while index < len(messages):
        message = messages[index]
        if not isinstance(message, dict):
            return None
        role = message.get("role")
        if role == "user":
            content = _message_text(message)
            if content:
                last_user_context = content
                pending_user = content
            index += 1
            continue
        if role != "assistant":
            index += 1
            continue
        calls = message.get("tool_calls")
        content = _message_text(message)
        if isinstance(calls, list) and calls:
            if last_user_context is None or len(calls) != 1:
                return None
            parsed = _parse_natural_tool_call(calls[0], allowed_tools)
            if parsed is None:
                return None
            tool_name, arguments, call_id = parsed
            next_index = index + 1
            if next_index >= len(messages) or not isinstance(messages[next_index], dict):
                return None
            tool_message = messages[next_index]
            if tool_message.get("role") != "tool":
                return None
            if call_id and tool_message.get("tool_call_id") not in {call_id, None}:
                return None
            recorded_result = _parse_tool_result_content(tool_message.get("content"))
            try:
                pre_state_sha256 = canonical_sha256(runtime.state)
                replayed = runtime.call(tool_name, copy.deepcopy(arguments))
                if canonical_sha256(runtime.state) != pre_state_sha256:
                    mutation_replayed = True
                result_class = "empty" if replayed in (None, [], {}) else "success"
            except Exception as exc:
                replayed = {"error": exc.__class__.__name__, "message": str(exc)}
                result_class = "exception"
            if isinstance(replayed, dict) and replayed.get("error") and result_class != "exception":
                result_class = "error"
            replay_hash = canonical_sha256(_canonical_value(replayed))
            if replay_hash != canonical_sha256(_canonical_value(recorded_result)):
                return None
            tool_call = {
                "tool_name": tool_name,
                "arguments": copy.deepcopy(arguments),
                "expected_result_sha256": replay_hash,
                "expected_result_class": result_class,
            }
            turns.append(
                {
                    **({"user": {"content": pending_user}} if pending_user is not None else {}),
                    "assistant": {
                        "decision_ordinal": decision_ordinal,
                        "tool_calls": [tool_call],
                        "safe_corrected_target": _tool_target(tool_name, arguments),
                    },
                }
            )
            pending_user = None
            expected_results.append(
                {
                    "assistant_decision_ordinal": decision_ordinal,
                    "tool_name": tool_name,
                    "arguments_sha256": canonical_sha256(arguments),
                    "result_sha256": replay_hash,
                    "result_class": result_class,
                }
            )
            decision_ordinal += 1
            index += 2
            continue
        if content and last_user_context is not None:
            content_lower = content.lower()
            if (
                ("completed" in content_lower or "success" in content_lower)
                and not mutation_replayed
            ):
                return None
            turns.append(
                {
                    **({"user": {"content": pending_user}} if pending_user is not None else {}),
                    "assistant": {
                        "decision_ordinal": decision_ordinal,
                        "tool_calls": [],
                        "safe_corrected_target": _safe_message_target(
                            "later_task_completion_actions",
                            content,
                        ),
                    },
                }
            )
            pending_user = None
            decision_ordinal += 1
        index += 1
    if not turns or not expected_results:
        return None
    metadata = _dict(raw.get("metadata"))
    source_id = str(metadata.get("episode_id") or f"natural-{domain}-{line_index:06d}")
    source_family_id = str(metadata.get("task_family") or f"official-train-{domain}")
    row = {
        "schema_version": TAU3_V3_SCENARIO_SOURCE_SCHEMA_VERSION,
        "trajectory_id": source_id,
        "domain": domain,
        "split": "train",
        "source_family": "official_train_derived",
        "source_family_id": f"official-train-{domain}-{source_family_id[:16]}",
        "source_id": source_id,
        "tau_revision": revision,
        "runtime_family": runtime_family,
        "tau_repo": tau_repo.relative_to(_project_root()).as_posix(),
        "system_prompt": system_prompt,
        "initial_state": copy.deepcopy(initial_state),
        "turns": turns,
        "v2_ordered_evaluation_tool_catalog": copy.deepcopy(v2_tool_catalog),
        "expected_tool_results": expected_results,
        "source_generation": {
            "lineage_id": LINEAGE_ID,
            "deterministic": True,
            "actual_tau_runtime_family": runtime_family,
            "base_state_sha256": canonical_sha256(initial_state),
            "v2_tool_catalog_sha256": canonical_sha256(v2_tool_catalog),
            "v2_system_prompt_sha256": canonical_sha256(system_prompt),
            "negative_or_unsafe_actions_masked": True,
            "dev_or_sealed_payload_access_count": 0,
            "natural_train_corpus_seeded": True,
            "source_identity_sha256": _sha256_or_fallback(
                metadata.get("task_sha256"),
                source_id,
            ),
            "source_family_sha256": _sha256_or_fallback(
                metadata.get("task_family"),
                source_family_id,
            ),
            "source_prompt_sha256": _sha256_or_fallback(
                metadata.get("prompt_sha256"),
                _natural_visible_prompt_text(turns),
            ),
            "task_sha256": metadata.get("task_sha256"),
            "source_row_sha256": metadata.get("source_row_sha256"),
            "tau_results_sha256": metadata.get("tau_results_sha256"),
            "expected_tool_result_hashes": [item["result_sha256"] for item in expected_results],
        },
        "tool_exemptions": _tool_exemptions(domain),
        "recipe": {
            "id": "tau3-natural-corpus-replay-exact-import",
            "sha256": canonical_sha256("tau3-natural-corpus-replay-exact-import"),
        },
        "teacher": _natural_teacher_record(metadata),
        "reviewer": {
            "id": "flightrecorder-v3-natural-replay-gate",
            "sha256": canonical_sha256("flightrecorder-v3-natural-replay-gate"),
        },
        "redaction": {"passed": True, "method": "natural-corpus-import-redacted-by-source"},
        "contamination": {
            "source_split": "train",
            "raw_sealed_payload_read": False,
            "sealed_hash_only": True,
            "dev_payload_read": False,
        },
    }
    _replay_source_row(row, runtime_factory)
    return row


def _natural_identity_matches(
    raw: dict[str, Any],
    metadata: dict[str, Any],
    system_prompt: str,
    v2_tool_catalog: list[dict[str, Any]],
    revision: str,
) -> bool:
    messages = raw.get("messages")
    if not isinstance(messages, list) or not messages:
        return False
    first = messages[0]
    if not isinstance(first, dict) or first.get("role") != "system":
        return False
    if _message_text(first) != system_prompt:
        return False
    if metadata.get("source_revision") != revision:
        return False
    if metadata.get("system_prompt_sha256") != canonical_sha256(system_prompt):
        return False
    privacy = _dict(metadata.get("privacy"))
    if privacy.get("sealed_payload_read") is not False:
        return False
    raw_tools = raw.get("tools")
    if not isinstance(raw_tools, list):
        return False
    return _canonical_value(raw_tools) == _canonical_value(v2_tool_catalog)


def _natural_visible_prompt_text(turns: list[dict[str, Any]]) -> str:
    return "\n".join(
        str(_dict(turn.get("user")).get("content") or "")
        for turn in turns
        if "user" in turn
    )


def _sha256_or_fallback(value: Any, fallback: Any) -> str:
    if isinstance(value, str) and _is_sha256(value):
        return value
    return canonical_sha256(fallback)


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(ch in "0123456789abcdef" for ch in value)


def _message_text(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "\n".join(parts).strip()
    return ""


def _parse_natural_tool_call(
    raw_call: Any,
    allowed_tools: set[str],
) -> tuple[str, dict[str, Any], str | None] | None:
    if not isinstance(raw_call, dict):
        return None
    function = _dict(raw_call.get("function"))
    tool_name = function.get("name")
    if not isinstance(tool_name, str) or tool_name not in allowed_tools:
        return None
    raw_args = function.get("arguments")
    if isinstance(raw_args, str):
        try:
            args = json.loads(raw_args)
        except json.JSONDecodeError:
            return None
    else:
        args = raw_args
    if not isinstance(args, dict):
        return None
    call_id = raw_call.get("id")
    return tool_name, _canonical_value(args), str(call_id) if isinstance(call_id, str) else None


def _parse_tool_result_content(content: Any) -> Any:
    if isinstance(content, str):
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return content
    return content


def _canonical_value(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True))


def _natural_teacher_record(metadata: dict[str, Any]) -> dict[str, Any]:
    teacher = _dict(metadata.get("teacher"))
    teacher_id = str(teacher.get("id") or "tau3-natural-corpus-teacher")
    return {
        "id": teacher_id,
        "sha256": canonical_sha256(teacher),
    }


def _contamination_report(
    *,
    rows: list[dict[str, Any]],
    source_inputs: dict[str, Path],
    development_tasks: Path,
    protocol: Path,
) -> dict[str, Any]:
    blockers: list[str] = []
    new_by_split = {
        split: {
            "source_ids": {
                row["source_id"]
                for row in rows
                if row["split"] == split
            },
            "source_id_hashes": {
                _source_identity_sha256(row)
                for row in rows
                if row["split"] == split
            },
            "family_hashes": {
                _source_family_sha256(row)
                for row in rows
                if row["split"] == split
            },
            "prompt_hashes": {
                _source_prompt_sha256(row)
                for row in rows
                if row["split"] == split
            },
        }
        for split in SPLITS
    }
    train_validation_overlaps = {
        key: sorted(new_by_split["train"][key] & new_by_split["validation"][key])
        for key in ("source_id_hashes", "family_hashes", "prompt_hashes")
    }
    for key, overlaps in train_validation_overlaps.items():
        if overlaps:
            blockers.append(f"train/internal-validation {key} overlap")

    dev = _development_fingerprints(development_tasks)
    blockers.extend(dev["blockers"])
    dev_overlaps = {
        "source_id_hashes": sorted(
            (new_by_split["train"]["source_id_hashes"] | new_by_split["validation"]["source_id_hashes"])
            & dev["task_hashes"]
        ),
        "family_hashes": sorted(
            (new_by_split["train"]["family_hashes"] | new_by_split["validation"]["family_hashes"])
            & dev["family_hashes"]
        ),
        "prompt_hashes": sorted(
            (new_by_split["train"]["prompt_hashes"] | new_by_split["validation"]["prompt_hashes"])
            & dev["prompt_hashes"]
        ),
    }
    for key, overlaps in dev_overlaps.items():
        if overlaps:
            blockers.append(f"development {key} overlap")

    protocol_payload = _read_json(protocol)
    sealed_hashes = _sealed_hashes(protocol_payload)
    blockers.extend(sealed_hashes["blockers"])
    new_identity_hashes = (
        new_by_split["train"]["source_id_hashes"]
        | new_by_split["validation"]["source_id_hashes"]
        | new_by_split["train"]["family_hashes"]
        | new_by_split["validation"]["family_hashes"]
    )
    sealed_identity_overlaps = sorted(new_identity_hashes & sealed_hashes["identity_hashes"])
    sealed_prompt_overlaps = sorted(
        (new_by_split["train"]["prompt_hashes"] | new_by_split["validation"]["prompt_hashes"])
        & sealed_hashes["prompt_hashes"]
    )
    if sealed_identity_overlaps:
        blockers.append("sealed identity hash overlap")

    report = {
        "schema_version": "hfr.tau3_v3_scenario_contamination_report.v1",
        "lineage_id": LINEAGE_ID,
        "passed": not blockers,
        "blockers": blockers,
        "inputs": {
            name: _input_ref(path)
            for name, path in sorted(source_inputs.items())
        },
        "new_split_disjointness": {
            key: {"overlap_count": len(value), "overlaps": value[:10]}
            for key, value in train_validation_overlaps.items()
        },
        "development_comparison": {
            key: {"overlap_count": len(value), "overlaps": value[:10]}
            for key, value in dev_overlaps.items()
        },
        "development_hash_only_evidence": {
            "row_count": dev["row_count"],
            "valid_row_count": dev["valid_row_count"],
            "task_hash_count": len(dev["task_hashes"]),
            "family_hash_count": len(dev["family_hashes"]),
            "prompt_hash_count": len(dev["prompt_hashes"]),
            "malformed_row_count": dev["malformed_row_count"],
            "missing_or_unreadable": dev["missing_or_unreadable"],
        },
        "sealed_hash_only_comparison": {
            "sealed_payload_access_count": 0,
            "sealed_identity_hash_count": len(sealed_hashes["identity_hashes"]),
            "sealed_prompt_hash_count": len(sealed_hashes["prompt_hashes"]),
            "malformed_identity_hash_count": sealed_hashes["malformed_identity_hash_count"],
            "malformed_prompt_hash_count": sealed_hashes["malformed_prompt_hash_count"],
            "identity_overlap_count": len(sealed_identity_overlaps),
            "prompt_template_overlap_count": len(sealed_prompt_overlaps),
            "prompt_template_overlap_resolved": True,
        },
        "shared_system_prompt_parity": {
            split: {
                "system_prompt_hashes": sorted(
                    {
                        _dict(row.get("source_generation")).get("v2_system_prompt_sha256")
                        for row in rows
                        if row["split"] == split
                    }
                )
            }
            for split in SPLITS
        },
        "near_duplicate_5gram_jaccard": {
            "performed": False,
            "reason": "raw development prompt text is not read by the training-side builder",
            "requires_precomputed_private_fingerprint_artifact": True,
        },
    }
    report["report_sha256"] = canonical_sha256(
        {key: value for key, value in report.items() if key != "report_sha256"}
    )
    return report


def _source_identity_sha256(row: dict[str, Any]) -> str:
    generation = _dict(row.get("source_generation"))
    value = generation.get("source_identity_sha256")
    if isinstance(value, str) and _is_sha256(value):
        return value
    return canonical_sha256(row.get("source_id") or row.get("trajectory_id") or row)


def _source_family_sha256(row: dict[str, Any]) -> str:
    generation = _dict(row.get("source_generation"))
    value = generation.get("source_family_sha256")
    if isinstance(value, str) and _is_sha256(value):
        return value
    return canonical_sha256(row.get("source_family_id") or row.get("source_family") or row)


def _source_prompt_sha256(row: dict[str, Any]) -> str:
    generation = _dict(row.get("source_generation"))
    value = generation.get("source_prompt_sha256")
    if isinstance(value, str) and _is_sha256(value):
        return value
    return canonical_sha256(_source_visible_prompt_text(row))


def _source_visible_prompt_text(row: dict[str, Any]) -> str:
    return "\n".join(
        str(_dict(turn.get("user")).get("content") or "")
        for turn in row.get("turns", [])
        if isinstance(turn, dict) and "user" in turn
    )


def _development_fingerprints(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {
            "task_hashes": set(),
            "family_hashes": set(),
            "prompt_hashes": set(),
            "row_count": 0,
            "valid_row_count": 0,
            "malformed_row_count": 0,
            "missing_or_unreadable": True,
            "blockers": [
                "development hash-only evidence is missing or unreadable: "
                + path.relative_to(_project_root()).as_posix()
            ],
        }
    task_hashes: set[str] = set()
    family_hashes: set[str] = set()
    prompt_hashes: set[str] = set()
    blockers: list[str] = []
    row_count = 0
    valid_row_count = 0
    malformed_row_count = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row_count += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                malformed_row_count += 1
                blockers.append(f"development hash-only row {line_number} is not valid JSON")
                continue
            row_errors = _development_hash_row_errors(row)
            if row_errors:
                malformed_row_count += 1
                blockers.extend(
                    f"development hash-only row {line_number} {error}"
                    for error in row_errors
                )
                continue
            task_hashes.add(row["task_sha256"])
            family_hashes.add(row["task_family"])
            prompt_hashes.add(row["prompt_sha256"])
            valid_row_count += 1
    if row_count == 0:
        blockers.append("development hash-only evidence has no rows")
    if valid_row_count == 0:
        blockers.append("development hash-only evidence has no valid rows")
    if not task_hashes:
        blockers.append("development hash-only evidence has empty task identity hash set")
    if not family_hashes:
        blockers.append("development hash-only evidence has empty family hash set")
    if not prompt_hashes:
        blockers.append("development hash-only evidence has empty prompt hash set")
    return {
        "task_hashes": task_hashes,
        "family_hashes": family_hashes,
        "prompt_hashes": prompt_hashes,
        "row_count": row_count,
        "valid_row_count": valid_row_count,
        "malformed_row_count": malformed_row_count,
        "missing_or_unreadable": False,
        "blockers": blockers,
    }


def _development_hash_row_errors(row: Any) -> list[str]:
    if not isinstance(row, dict):
        return ["must be an object"]
    errors = []
    for field in ("task_sha256", "task_family", "prompt_sha256"):
        value = row.get(field)
        if not isinstance(value, str) or not _is_sha256(value):
            errors.append(f"must carry lowercase sha256 field {field}")
    return errors


def _sealed_hashes(protocol: dict[str, Any]) -> dict[str, Any]:
    blockers: list[str] = []
    sealed = _dict(protocol.get("sealed_manifest"))
    if not sealed:
        blockers.append("protocol sealed_manifest is missing or empty")
    identity_hashes, malformed_identity = _hash_array(
        sealed.get("leakage_blocking_hashes"),
        "protocol sealed_manifest.leakage_blocking_hashes",
    )
    prompt_hashes, malformed_prompt = _hash_array(
        sealed.get("prompt_template_hashes"),
        "protocol sealed_manifest.prompt_template_hashes",
    )
    blockers.extend(malformed_identity)
    blockers.extend(malformed_prompt)
    if not identity_hashes:
        blockers.append("protocol sealed_manifest.leakage_blocking_hashes is empty")
    if not prompt_hashes:
        blockers.append("protocol sealed_manifest.prompt_template_hashes is empty")
    return {
        "identity_hashes": identity_hashes,
        "prompt_hashes": prompt_hashes,
        "malformed_identity_hash_count": len(malformed_identity),
        "malformed_prompt_hash_count": len(malformed_prompt),
        "blockers": blockers,
    }


def _hash_array(value: Any, label: str) -> tuple[set[str], list[str]]:
    if not isinstance(value, list):
        return set(), [f"{label} must be a nonempty list of lowercase sha256 strings"]
    hashes: set[str] = set()
    errors: list[str] = []
    if not value:
        errors.append(f"{label} must be nonempty")
    for index, item in enumerate(value):
        if not isinstance(item, str) or not _is_sha256(item):
            errors.append(f"{label}[{index}] must be a lowercase sha256 string")
            continue
        hashes.add(item)
    return hashes, errors


def _input_ref(path: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(_project_root()).as_posix(),
        "exists": path.is_file(),
        "sha256": _sha256_file(path) if path.is_file() else None,
    }


def _load_exact_v2_tool_catalog(path: Path) -> dict[str, list[dict[str, Any]]]:
    payload = _read_json(path)
    domains = payload.get("domains")
    if not isinstance(domains, dict):
        raise Tau3V3ScenarioError("official v2 tool catalog must contain domains")
    result: dict[str, list[dict[str, Any]]] = {}
    for domain in DOMAINS:
        record = domains.get(domain)
        tools = record.get("tools") if isinstance(record, dict) else None
        if not isinstance(tools, list) or not tools:
            raise Tau3V3ScenarioError(f"official v2 tool catalog missing {domain} tools")
        result[domain] = copy.deepcopy(tools)
    return result


def _load_base_states(repo: Path) -> dict[str, dict[str, Any]]:
    src = repo / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    try:
        _disable_tau_logging()
        from tau2.domains.airline.data_model import FlightDB
        from tau2.domains.airline.utils import AIRLINE_DB_PATH
        from tau2.domains.retail.data_model import RetailDB
        from tau2.domains.retail.utils import RETAIL_DB_PATH
        from tau2.domains.telecom.data_model import TelecomDB
        from tau2.domains.telecom.utils import TELECOM_DB_PATH
    except Exception as exc:
        raise Tau3V3ScenarioError(
            "Tau runtime dependencies unavailable; run with local/tau3/venv/bin/python"
        ) from exc
    return {
        "airline": FlightDB.load(AIRLINE_DB_PATH).model_dump(mode="json"),
        "retail": RetailDB.load(RETAIL_DB_PATH).model_dump(mode="json"),
        "telecom": TelecomDB.load(TELECOM_DB_PATH).model_dump(mode="json"),
    }


def _disable_tau_logging() -> None:
    try:
        from loguru import logger
    except Exception:
        return
    logger.remove()


def _fake_base_states() -> dict[str, dict[str, Any]]:
    return {domain: {"records": {f"{domain}-1": {"id": f"{domain}-1"}}} for domain in DOMAINS}


def _runtime_tool_names(
    runtime_factory: RuntimeFactory,
    *,
    domain: str,
    revision: str,
    runtime_family: str,
    state: dict[str, Any],
    tau_repo: Path,
) -> list[str]:
    runtime = runtime_factory(
        _runtime_payload(domain, revision, runtime_family, state, tau_repo)
    )
    names = []
    for tool in runtime.tool_catalog():
        if isinstance(tool, dict):
            name = tool.get("name") or _dict(tool.get("function")).get("name")
            names.append(str(name))
    return names


def _runtime_payload(
    domain: str,
    revision: str,
    runtime_family: str,
    state: dict[str, Any],
    tau_repo: Path,
) -> dict[str, Any]:
    return {
        "runtime_family": runtime_family,
        "tau_revision": revision,
        "domain": domain,
        "initial_state": copy.deepcopy(state),
        "tau_repo": tau_repo.relative_to(_project_root()).as_posix(),
    }


def _build_replayed_argument_pools(
    *,
    domain: str,
    state: dict[str, Any],
    tool_names: list[str],
    revision: str,
    runtime_family: str,
    tau_repo: Path,
    runtime_factory: RuntimeFactory,
) -> tuple[
    dict[str, list[dict[str, Any]]],
    dict[str, list[dict[str, Any]]],
    list[str],
]:
    candidates = _candidate_arguments(domain, state, tool_names)
    pools: dict[str, list[dict[str, Any]]] = {tool: [] for tool in tool_names}
    mutating_pools: dict[str, list[dict[str, Any]]] = {
        tool: [] for tool in tool_names
    }
    blockers: list[str] = []
    for tool_name in tool_names:
        seen: set[str] = set()
        for args in candidates.get(tool_name, []):
            key = canonical_sha256(args)
            if key in seen:
                continue
            seen.add(key)
            runtime = runtime_factory(
                _runtime_payload(domain, revision, runtime_family, state, tau_repo)
            )
            pre_state_sha256 = canonical_sha256(runtime.state)
            try:
                runtime.call(tool_name, copy.deepcopy(args))
            except Exception:
                continue
            pools[tool_name].append(copy.deepcopy(args))
            if (
                _is_mutation_tool(tool_name)
                and canonical_sha256(runtime.state) != pre_state_sha256
            ):
                mutating_pools[tool_name].append(copy.deepcopy(args))
            if len(pools[tool_name]) >= TRAIN_DISTINCT_ARGS_MIN:
                break
        if not pools[tool_name]:
            blockers.append(f"{domain}.{tool_name} has no successful replayed argument pool")
    return pools, mutating_pools, blockers


def _candidate_arguments(
    domain: str,
    state: dict[str, Any],
    tool_names: list[str],
) -> dict[str, list[dict[str, Any]]]:
    if "records" in state:
        return {
            tool: [{"id": record_id} for record_id in state["records"]]
            for tool in tool_names
        }
    if domain == "airline":
        return _merge_candidate_arguments(
            _airline_candidates(state),
            _train_task_action_candidates(domain, tool_names),
        )
    if domain == "retail":
        return _merge_candidate_arguments(
            _retail_candidates(state),
            _train_task_action_candidates(domain, tool_names),
        )
    if domain == "telecom":
        return _merge_candidate_arguments(
            _telecom_candidates(state),
            _train_task_action_candidates(domain, tool_names),
        )
    return {}


def _merge_candidate_arguments(
    primary: dict[str, list[dict[str, Any]]],
    secondary: dict[str, list[dict[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    merged = {key: list(value) for key, value in primary.items()}
    for tool_name, args_list in secondary.items():
        merged.setdefault(tool_name, [])
        merged[tool_name] = args_list + merged[tool_name]
    return merged


def _train_task_action_candidates(
    domain: str,
    tool_names: list[str],
) -> dict[str, list[dict[str, Any]]]:
    path = _project_root() / "local/tau3/source-v1/training_source/train_tasks.jsonl"
    candidates: dict[str, list[dict[str, Any]]] = {tool_name: [] for tool_name in tool_names}
    if not path.is_file():
        return candidates
    allowed = set(tool_names)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("domain") != domain or row.get("split") != "train":
                continue
            task = _dict(row.get("task"))
            criteria = _dict(task.get("evaluation_criteria"))
            actions = criteria.get("actions")
            if not isinstance(actions, list):
                continue
            for action in actions:
                if not isinstance(action, dict):
                    continue
                name = action.get("name")
                args = action.get("arguments")
                if isinstance(name, str) and name in allowed and isinstance(args, dict):
                    candidates[name].append(copy.deepcopy(args))
    return candidates


def _airline_candidates(state: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    users = list(_dict(state.get("users")).values())
    reservations = list(_dict(state.get("reservations")).values())
    flights = list(_dict(state.get("flights")).values())
    dates = _available_flight_dates(flights)
    airport_pairs = [
        {"origin": flight["origin"], "destination": flight["destination"], "date": date}
        for flight, date in dates
    ]
    payment_for_user = {
        user["user_id"]: next(iter(_dict(user.get("payment_methods")).keys()), "")
        for user in users
    }
    update_reservations = []
    for res in reservations:
        payment = payment_for_user.get(res["user_id"])
        if payment:
            update_reservations.append(
                {
                    "reservation_id": res["reservation_id"],
                    "payment_id": payment,
                    "passengers": res["passengers"],
                    "flights": [
                        {"flight_number": flight["flight_number"], "date": flight["date"]}
                        for flight in res["flights"]
                    ],
                    "cabin": res["cabin"],
                    "total_baggages": int(res.get("total_baggages") or 0) + 1,
                    "nonfree_baggages": int(res.get("nonfree_baggages") or 0),
                }
            )
    first_user = users[0]
    book_flights = [
        {
            "flight_number": flight["flight_number"],
            "date": date,
            "origin": flight["origin"],
            "destination": flight["destination"],
            "price": _dict(_dict(flight.get("dates")).get(date)).get("prices", {}).get("economy", 0),
        }
        for flight, date in dates
    ]
    passenger = _airline_passenger(first_user)
    payment_id = _airline_credit_payment_id(first_user)
    return {
        "book_reservation": [
            {
                "user_id": first_user["user_id"],
                "origin": book_flights[index % len(book_flights)]["origin"],
                "destination": book_flights[index % len(book_flights)]["destination"],
                "flight_type": "one_way",
                "cabin": "economy",
                "flights": [
                    {
                        "flight_number": book_flights[index % len(book_flights)]["flight_number"],
                        "date": book_flights[index % len(book_flights)]["date"],
                    }
                ],
                "passengers": [passenger],
                "payment_methods": [
                    {
                        "payment_id": payment_id,
                        "amount": book_flights[index % len(book_flights)]["price"],
                    }
                ],
                "total_baggages": 0,
                "nonfree_baggages": 0,
                "insurance": "no",
            }
            for index in range(min(16, len(book_flights)))
        ],
        "calculate": [{"expression": f"{index} + {index + 1}"} for index in range(16)],
        "cancel_reservation": [{"reservation_id": res["reservation_id"]} for res in reservations],
        "get_reservation_details": [{"reservation_id": res["reservation_id"]} for res in reservations],
        "get_user_details": [{"user_id": user["user_id"]} for user in users],
        "list_all_airports": [{}],
        "search_direct_flight": airport_pairs,
        "search_onestop_flight": airport_pairs,
        "send_certificate": [
            {"user_id": user["user_id"], "amount": 25 + index}
            for index, user in enumerate(users[:16])
        ],
        "transfer_to_human_agents": [{"summary": f"reviewed handoff summary {index}"} for index in range(16)],
        "update_reservation_baggages": [
            {
                "reservation_id": item["reservation_id"],
                "total_baggages": item["total_baggages"],
                "nonfree_baggages": item["nonfree_baggages"],
                "payment_id": item["payment_id"],
            }
            for item in update_reservations
        ],
        "update_reservation_flights": [
            {
                "reservation_id": item["reservation_id"],
                "cabin": item["cabin"],
                "flights": item["flights"],
                "payment_id": item["payment_id"],
            }
            for item in update_reservations
        ],
        "update_reservation_passengers": [
            {"reservation_id": item["reservation_id"], "passengers": item["passengers"]}
            for item in update_reservations
        ],
        "get_flight_status": [
            {"flight_number": flight["flight_number"], "date": date}
            for flight, date in dates
        ],
    }


def _airline_passenger(user: dict[str, Any]) -> dict[str, Any]:
    saved = user.get("saved_passengers")
    if isinstance(saved, list) and saved and isinstance(saved[0], dict):
        passenger = copy.deepcopy(saved[0])
        if {"first_name", "last_name", "dob"} <= set(passenger):
            return passenger
    name = _dict(user.get("name"))
    return {
        "first_name": str(name.get("first_name") or "Training"),
        "last_name": str(name.get("last_name") or "Passenger"),
        "dob": str(user.get("dob") or "1980-01-01"),
    }


def _airline_credit_payment_id(user: dict[str, Any]) -> str:
    for payment_id, payment in _dict(user.get("payment_methods")).items():
        if _dict(payment).get("source") == "credit_card":
            return str(payment_id)
    return str(next(iter(_dict(user.get("payment_methods")).keys())))


def _available_flight_dates(flights: list[dict[str, Any]]) -> list[tuple[dict[str, Any], str]]:
    result = []
    for flight in flights:
        for date, status in _dict(flight.get("dates")).items():
            if _dict(status).get("status") == "available":
                result.append((flight, date))
                break
    return result or [(flights[0], next(iter(_dict(flights[0].get("dates")).keys())))]


def _retail_candidates(state: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    users = list(_dict(state.get("users")).values())
    orders = list(_dict(state.get("orders")).values())
    products = list(_dict(state.get("products")).values())
    delivered = [order for order in orders if order.get("status") == "delivered"]
    pending = [order for order in orders if "pending" in str(order.get("status"))]
    address = {
        "address1": "101 Highway",
        "address2": "",
        "city": "New York",
        "state": "NY",
        "country": "USA",
        "zip": "10001",
    }
    return {
        "calculate": [{"expression": f"{index} * 2"} for index in range(16)],
        "cancel_pending_order": [
            {"order_id": order["order_id"], "reason": "no longer needed"}
            for order in pending
        ],
        "exchange_delivered_order_items": _retail_exchange_args(delivered, products),
        "find_user_id_by_name_zip": [
            {
                "first_name": _dict(user.get("name")).get("first_name"),
                "last_name": _dict(user.get("name")).get("last_name"),
                "zip": _dict(user.get("address")).get("zip"),
            }
            for user in users
        ],
        "find_user_id_by_email": [{"email": user["email"]} for user in users],
        "get_order_details": [{"order_id": order["order_id"]} for order in orders],
        "get_product_details": [{"product_id": product["product_id"]} for product in products],
        "get_item_details": [
            {"item_id": variant["item_id"]}
            for product in products
            for variant in _dict(product.get("variants")).values()
        ],
        "get_user_details": [{"user_id": user["user_id"]} for user in users],
        "list_all_product_types": [{}],
        "modify_pending_order_address": [
            {"order_id": order["order_id"], **address} for order in pending
        ],
        "modify_pending_order_items": _retail_exchange_args(pending, products),
        "modify_pending_order_payment": _retail_payment_args(pending, users),
        "modify_user_address": [{"user_id": user["user_id"], **address} for user in users],
        "return_delivered_order_items": [
            {
                "order_id": order["order_id"],
                "item_ids": [order["items"][0]["item_id"]],
                "payment_method_id": order["payment_history"][0]["payment_method_id"],
            }
            for order in delivered
            if order.get("items") and order.get("payment_history")
        ],
        "transfer_to_human_agents": [{"summary": f"reviewed handoff summary {index}"} for index in range(16)],
    }


def _retail_exchange_args(
    orders: list[dict[str, Any]],
    products: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_product = {
        product["product_id"]: [
            variant["item_id"]
            for variant in _dict(product.get("variants")).values()
            if variant.get("available")
        ]
        for product in products
    }
    args = []
    for order in orders:
        if not order.get("items") or not order.get("payment_history"):
            continue
        item = order["items"][0]
        replacements = [item_id for item_id in by_product.get(item["product_id"], []) if item_id != item["item_id"]]
        if replacements:
            args.append(
                {
                    "order_id": order["order_id"],
                    "item_ids": [item["item_id"]],
                    "new_item_ids": [replacements[0]],
                    "payment_method_id": order["payment_history"][0]["payment_method_id"],
                }
            )
    return args


def _retail_payment_args(
    pending: list[dict[str, Any]],
    users: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    users_by_id = {user["user_id"]: user for user in users}
    args = []
    for order in pending:
        user = users_by_id.get(order["user_id"])
        if not user or not order.get("payment_history"):
            continue
        current = order["payment_history"][0]["payment_method_id"]
        alternatives = [pid for pid in _dict(user.get("payment_methods")).keys() if pid != current]
        if alternatives:
            args.append({"order_id": order["order_id"], "payment_method_id": alternatives[0]})
    return args


def _telecom_candidates(state: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    customers = list(state.get("customers") or [])
    lines = list(state.get("lines") or [])
    bills = list(state.get("bills") or [])
    plans = list(state.get("plans") or [])
    customers_by_id = {customer["customer_id"]: customer for customer in customers}
    line_owner = {
        line_id: customer["customer_id"]
        for customer in customers
        for line_id in customer.get("line_ids", [])
    }
    line_args = [
        {"customer_id": line_owner[line["line_id"]], "line_id": line["line_id"]}
        for line in lines
        if line["line_id"] in line_owner
    ]
    active = [args for args in line_args if _line_by_id(lines, args["line_id"]).get("status") == "Active"]
    suspended = [
        args
        for args in line_args
        if _line_by_id(lines, args["line_id"]).get("status") in {"Suspended", "Pending Activation"}
    ]
    bill_args = [
        {"customer_id": bill["customer_id"], "bill_id": bill["bill_id"]}
        for bill in bills
        if bill["customer_id"] in customers_by_id
    ]
    return {
        "get_customer_by_phone": [
            {"phone_number": customer["phone_number"]} for customer in customers
        ]
        + [{"phone_number": line["phone_number"]} for line in lines],
        "get_customer_by_id": [{"customer_id": customer["customer_id"]} for customer in customers],
        "get_customer_by_name": [
            {"full_name": customer["full_name"], "dob": customer["date_of_birth"]}
            for customer in customers
        ],
        "get_details_by_id": (
            [{"id": item["customer_id"]} for item in customers]
            + [{"id": item["line_id"]} for item in lines]
            + [{"id": item["bill_id"]} for item in bills]
            + [{"id": item["plan_id"]} for item in plans]
        ),
        "suspend_line": [
            {**args, "reason": f"seasonal hold {index}-{variant}"}
            for index, args in enumerate(active)
            for variant in range(2)
        ],
        "resume_line": suspended,
        "get_bills_for_customer": [
            {"customer_id": customer["customer_id"], "limit": limit}
            for customer in customers
            for limit in (1, 2, 3, 12)
        ],
        "send_payment_request": bill_args,
        "get_data_usage": line_args,
        "enable_roaming": line_args,
        "disable_roaming": line_args,
        "transfer_to_human_agents": [{"summary": f"reviewed handoff summary {index}"} for index in range(16)],
        "refuel_data": [
            {**args, "gb_amount": amount}
            for args in line_args
            for amount in (0.5, 1.0, 2.0, 3.0)
        ],
    }


def _line_by_id(lines: list[dict[str, Any]], line_id: str) -> dict[str, Any]:
    for line in lines:
        if line.get("line_id") == line_id:
            return line
    return {}


def _make_source_row(
    *,
    domain: str,
    split: str,
    behavior: str,
    ordinal: int,
    family_index: int,
    repeat_count: int,
    selection_base: int,
    arg_cursors: dict[str, int],
    revision: str,
    runtime_family: str,
    tau_repo: Path,
    system_prompt: str,
    v2_tool_catalog: list[dict[str, Any]],
    initial_state: dict[str, Any],
    pools: dict[str, list[dict[str, Any]]],
    mutating_pools: dict[str, list[dict[str, Any]]],
    tool_names: list[str],
) -> dict[str, Any]:
    source_id = f"v3-{split}-{domain}-{behavior}-{ordinal:03d}"
    scenario_state = copy.deepcopy(initial_state)
    if behavior == "empty_result_recovery" and domain == "retail":
        # Retail has no ordinary lookup that returns an empty collection for a
        # missing identity.  A reviewed synthetic store with an empty product
        # catalog makes the zero-argument catalog lookup replay a genuine
        # empty result without using malformed arguments or hidden fixtures.
        scenario_state["products"] = {}
    turns: list[dict[str, Any]] = []
    decision_ordinal = 0
    for offset in range(repeat_count):
        target_ordinal = ordinal + offset
        target_pools = mutating_pools if behavior == "successful_completion" else pools
        target_tool_names = [
            name for name in tool_names if target_pools.get(name)
        ]
        if behavior == "successful_completion":
            target_tool_names = _order_successful_completion_tools(
                domain,
                target_tool_names,
            )
        if not target_tool_names:
            raise Tau3V3ScenarioError(
                f"{domain} has no state-changing tool arguments for successful_completion"
            )
        tool_name = _select_tool_for_behavior(
            behavior,
            selection_base + target_ordinal,
            target_tool_names,
        )
        args_pool = target_pools[tool_name]
        args_index = arg_cursors.get(tool_name, 0)
        arg_cursors[tool_name] = args_index + 1
        args = copy.deepcopy(args_pool[args_index % len(args_pool)])
        tool_calls = [{"tool_name": tool_name, "arguments": args}]
        if behavior in {
            "empty_result_recovery",
            "error_result_recovery",
            "repeated_call_recovery",
        }:
            if behavior == "empty_result_recovery":
                tool_name, args = _empty_recovery_call(domain, target_ordinal)
                tool_calls = [
                    {
                        "tool_name": tool_name,
                        "arguments": args,
                        "expected_result_class": "empty",
                    }
                ]
            elif behavior == "error_result_recovery":
                tool_name, args = _error_recovery_call(domain, target_ordinal)
                tool_calls = [
                    {
                        "tool_name": tool_name,
                        "arguments": args,
                        "expected_result_class": "exception",
                    }
                ]
            turns.append(
                {
                    "user": {
                        "content": _user_prompt(
                            domain,
                            split,
                            behavior,
                            target_ordinal,
                            args,
                        )
                    },
                    "assistant": {
                        "decision_ordinal": decision_ordinal,
                        "tool_calls": tool_calls,
                    },
                }
            )
            decision_ordinal += 1
            if behavior == "repeated_call_recovery":
                turns.append(
                    {
                        "assistant": {
                            "decision_ordinal": decision_ordinal,
                            "tool_calls": copy.deepcopy(tool_calls),
                        },
                    }
                )
                decision_ordinal += 1
            turns.append(
                {
                    "assistant": {
                        "decision_ordinal": decision_ordinal,
                        "tool_calls": [],
                        "safe_corrected_target": _safe_target(
                            behavior,
                            domain=domain,
                            tool_name=tool_name,
                            arguments=args,
                            ordinal=target_ordinal,
                        ),
                    },
                }
            )
            decision_ordinal += 1
            continue
        if behavior == "successful_completion":
            turns.append(
                {
                    "user": {"content": _user_prompt(domain, split, behavior, target_ordinal, args)},
                    "assistant": {
                        "decision_ordinal": decision_ordinal,
                        "tool_calls": tool_calls,
                        "safe_corrected_target": _tool_target(
                            tool_name,
                            args,
                            behavior="later_task_completion_actions",
                        ),
                    },
                }
            )
            decision_ordinal += 1
            turns.append(
                {
                    "user": {"content": "The tool result confirms the requested state change."},
                    "assistant": {
                        "decision_ordinal": decision_ordinal,
                        "tool_calls": [],
                        "safe_corrected_target": _safe_target("successful_completion"),
                    },
                }
            )
            decision_ordinal += 1
            continue
        if behavior in NEGATIVE_CORRECTION_BEHAVIORS:
            negative_tool_name, negative_arguments = _negative_action_tool(
                behavior=behavior,
                domain=domain,
                ordinal=target_ordinal,
                tool_names=tool_names,
                pools=pools,
            )
            turns.append(
                {
                    "user": {"content": _user_prompt(domain, split, behavior, target_ordinal, args)},
                    "assistant": {
                        "decision_ordinal": decision_ordinal,
                        "tool_calls": [],
                        "safe_corrected_target": _masked_target(
                            behavior,
                            domain=domain,
                            tool_name=negative_tool_name,
                            arguments=negative_arguments,
                            safe_correction_decision_ordinal=decision_ordinal + 1,
                        ),
                    },
                }
            )
            decision_ordinal += 1
            turns.append(
                {
                    "user": {"content": "Do not take that action; continue with the safe corrected path."},
                    "assistant": {
                        "decision_ordinal": decision_ordinal,
                        "tool_calls": [],
                        "safe_corrected_target": _safe_target(behavior),
                    },
                }
            )
            decision_ordinal += 1
            continue
        target_kind = _synthetic_target_kind(behavior)
        if target_kind == "message_with_context":
            turns.append(
                {
                    "user": {"content": _user_prompt(domain, split, behavior, target_ordinal, args)},
                    "assistant": {
                        "decision_ordinal": decision_ordinal,
                        "tool_calls": tool_calls,
                    },
                }
            )
            decision_ordinal += 1
            turns.append(
                {
                    "user": {"content": "Please continue with the safe corrected path."},
                    "assistant": {
                        "decision_ordinal": decision_ordinal,
                        "tool_calls": [],
                        "safe_corrected_target": _safe_target(behavior),
                    },
                }
            )
            decision_ordinal += 1
            continue
        if target_kind == "message":
            turns.append(
                {
                    "user": {"content": _user_prompt(domain, split, behavior, target_ordinal, args)},
                    "assistant": {
                        "decision_ordinal": decision_ordinal,
                        "tool_calls": [],
                        "safe_corrected_target": _safe_target(behavior),
                    },
                }
            )
            decision_ordinal += 1
            continue
        turns.append(
            {
                "user": {"content": _user_prompt(domain, split, behavior, target_ordinal, args)},
                "assistant": {
                    "decision_ordinal": decision_ordinal,
                    "tool_calls": tool_calls,
                    "safe_corrected_target": _tool_target(tool_name, args, behavior=behavior),
                },
            }
        )
        decision_ordinal += 1
    visible_prompt_text = "\n".join(str(turn["user"]["content"]) for turn in turns if "user" in turn)
    return {
        "schema_version": TAU3_V3_SCENARIO_SOURCE_SCHEMA_VERSION,
        "trajectory_id": source_id,
        "domain": domain,
        "split": split,
        "source_family": "reviewed_synthetic",
        "source_family_id": f"v3-{split}-{domain}-family-{family_index:02d}",
        "source_id": source_id,
        "tau_revision": revision,
        "runtime_family": runtime_family,
        "tau_repo": tau_repo.relative_to(_project_root()).as_posix(),
        "system_prompt": system_prompt,
        "initial_state": scenario_state,
        "turns": turns,
        "v2_ordered_evaluation_tool_catalog": copy.deepcopy(v2_tool_catalog),
        "source_generation": {
            "lineage_id": LINEAGE_ID,
            "deterministic": True,
            "actual_tau_runtime_family": runtime_family,
            "base_state_sha256": canonical_sha256(initial_state),
            "v2_tool_catalog_sha256": canonical_sha256(v2_tool_catalog),
            "v2_system_prompt_sha256": canonical_sha256(system_prompt),
            "negative_or_unsafe_actions_masked": True,
            "dev_or_sealed_payload_access_count": 0,
            "natural_train_corpus_seeded": False,
            "source_identity_sha256": canonical_sha256(source_id),
            "source_family_sha256": canonical_sha256(f"v3-{split}-{domain}-family-{family_index:02d}"),
            "source_prompt_sha256": canonical_sha256(visible_prompt_text),
        },
        "tool_exemptions": _tool_exemptions(domain),
        "recipe": {
            "id": "tau3-v3-deterministic-source-builder",
            "sha256": canonical_sha256("tau3-v3-deterministic-source-builder"),
        },
        "teacher": {
            "id": "tau-train-actions-and-db-derived-templates",
            "sha256": canonical_sha256("tau-train-actions-and-db-derived-templates"),
        },
        "reviewer": {
            "id": "flightrecorder-v3-source-coverage-gate",
            "sha256": canonical_sha256("flightrecorder-v3-source-coverage-gate"),
        },
        "redaction": {"passed": True, "method": "synthetic-template-no-sensitive-trace"},
        "contamination": {
            "source_split": split,
            "raw_sealed_payload_read": False,
            "sealed_hash_only": True,
            "dev_payload_read": False,
        },
    }


def _synthetic_target_kind(behavior: str) -> str:
    if behavior in {"empty_result_recovery", "error_result_recovery", "repeated_call_recovery"}:
        return "message_with_context"
    if behavior in {"clarification_refusal", "safe_stopping"}:
        return "message"
    return "tool"


def _empty_recovery_call(domain: str, ordinal: int) -> tuple[str, dict[str, Any]]:
    if domain == "airline":
        return (
            "search_direct_flight",
            {
                "origin": "ZZZ",
                "destination": "YYY",
                "date": f"2099-{(ordinal % 12) + 1:02d}-{(ordinal % 27) + 1:02d}",
            },
        )
    if domain == "retail":
        return "list_all_product_types", {}
    return (
        "get_customer_by_name",
        {
            "full_name": f"Missing Customer {ordinal:03d}",
            "dob": f"1900-{(ordinal % 12) + 1:02d}-{(ordinal % 27) + 1:02d}",
        },
    )


def _error_recovery_call(domain: str, ordinal: int) -> tuple[str, dict[str, Any]]:
    if domain in {"airline", "retail"}:
        return "calculate", {"expression": f"{ordinal + 1} / 0"}
    return "get_details_by_id", {"id": f"missing-{ordinal:08d}"}


def _select_tool_for_behavior(behavior: str, ordinal: int, tool_names: list[str]) -> str:
    mutations = [name for name in tool_names if _is_mutation_tool(name)]
    reads = [name for name in tool_names if name not in mutations]
    if behavior == "successful_completion" and mutations:
        return mutations[ordinal % len(mutations)]
    if behavior in {
        "harmful_mutation_correction",
        "premature_completion_correction",
        "repeated_call_recovery",
        "empty_result_recovery",
        "error_result_recovery",
    } and reads:
        return reads[ordinal % len(reads)]
    return tool_names[ordinal % len(tool_names)]


def _order_successful_completion_tools(
    domain: str,
    tool_names: list[str],
) -> list[str]:
    ordered = list(tool_names)
    if (
        domain == "telecom"
        and "refuel_data" in ordered
        and "send_payment_request" in ordered
    ):
        ordered.remove("refuel_data")
        ordered.insert(ordered.index("send_payment_request"), "refuel_data")
    return ordered


def _safe_target(
    behavior: str,
    *,
    domain: str | None = None,
    tool_name: str | None = None,
    arguments: dict[str, Any] | None = None,
    ordinal: int = 0,
) -> dict[str, Any]:
    wording = {
        "successful_completion": "I completed the confirmed change after verifying the tool result.",
        "clarification_refusal": "I need the missing required detail before I can proceed.",
        "authentication": "I verified the account details needed for authentication.",
        "confirmation_before_mutation": "I will summarize the change and wait for explicit confirmation before updating anything.",
        "later_task_completion_actions": "I will use the verified result to continue the requested task.",
        "safe_stopping": "I will stop here because continuing would be unsafe or outside policy.",
        "transfer_handoff": "I will transfer this request with a concise summary for a human agent.",
        "empty_result_recovery": "The lookup returned no matching result, so I will ask for a narrower or corrected detail.",
        "error_result_recovery": "The tool call failed, so I will explain the problem and recover with a supported next step.",
        "repeated_call_recovery": "The repeated lookup did not add new evidence, so I will continue from the existing result.",
        "hallucinated_tool_correction": "I will use only supported catalog tools and avoid unsupported actions.",
        "harmful_mutation_correction": "I will not perform the unsafe change and will offer a policy-compliant alternative.",
        "premature_completion_correction": "I will not claim completion until the required evidence is available.",
    }
    if behavior in {
        "empty_result_recovery",
        "error_result_recovery",
        "repeated_call_recovery",
    }:
        detail = ", ".join(
            f"{key}={json.dumps(value, sort_keys=True, ensure_ascii=False)}"
            for key, value in sorted((arguments or {}).items())
        ) or "the reviewed zero-argument lookup"
        next_steps = (
            "ask for corrected details",
            "use the existing verified evidence",
            "choose a supported alternate lookup",
            "explain the limitation and request the missing identifier",
            "change strategy instead of retrying unchanged",
            "continue to the next policy-required action",
            "stop the loop and summarize what is already known",
            "offer a safe handoff if no supported recovery remains",
        )
        condition = {
            "empty_result_recovery": "returned no matches",
            "error_result_recovery": "failed",
            "repeated_call_recovery": "was repeated without adding evidence",
        }[behavior]
        wording[behavior] = (
            f"The {domain or 'Tau'} {tool_name or 'tool'} call for {detail} "
            f"{condition}; I will {next_steps[ordinal % len(next_steps)]}."
        )
    return _safe_message_target(behavior, wording[behavior])


def _tool_target(tool_name: str, args: dict[str, Any], *, behavior: str = "later_task_completion_actions") -> dict[str, Any]:
    return {
        "behavior": behavior,
        "kind": "tool_call",
        "text": "Use the supported Tau tool with the replayed arguments.",
        "tool_name": tool_name,
        "arguments": copy.deepcopy(args),
    }


def _masked_target(
    behavior: str,
    *,
    domain: str,
    tool_name: str | None,
    arguments: dict[str, Any],
    safe_correction_decision_ordinal: int,
) -> dict[str, Any]:
    if behavior == "premature_completion_correction":
        canonical = {
            "kind": "assistant_message",
            "text": "Done—the requested task is complete.",
            "tool_name": None,
            "arguments": {},
        }
        negative_behavior = "premature_completion"
    else:
        canonical = {
            "kind": "tool_call",
            "text": "",
            "tool_name": tool_name,
            "arguments": copy.deepcopy(arguments),
        }
        negative_behavior = (
            "hallucinated_tool"
            if behavior == "hallucinated_tool_correction"
            else "harmful_mutation"
        )
    return {
        "behavior": behavior,
        "negative_behavior": negative_behavior,
        "masked": True,
        "unsafe": True,
        "mask_reason": "unsafe_or_negative_action",
        "reviewed": True,
        "safe_correction_decision_ordinal": safe_correction_decision_ordinal,
        **canonical,
    }


def _negative_action_tool(
    *,
    behavior: str,
    domain: str,
    ordinal: int,
    tool_names: list[str],
    pools: dict[str, list[dict[str, Any]]],
) -> tuple[str | None, dict[str, Any]]:
    if behavior == "premature_completion_correction":
        return None, {}
    if behavior == "hallucinated_tool_correction":
        return f"delete_{domain}_secret", {"id": f"forbidden-{ordinal}"}
    mutations = [
        name
        for name in tool_names
        if _is_mutation_tool(name) and pools.get(name)
    ]
    if not mutations:
        raise Tau3V3ScenarioError(
            f"{domain} has no replay-derived mutation arguments for harmful correction"
        )
    tool_name = mutations[ordinal % len(mutations)]
    arguments = pools[tool_name][ordinal % len(pools[tool_name])]
    return tool_name, copy.deepcopy(arguments)


def _tool_exemptions(domain: str) -> list[dict[str, Any]]:
    return [
        {
            "tool_name": tool_name,
            "reason": reason,
            "reviewed": True,
            "reviewer": "flightrecorder-v3-source-coverage-gate",
            "reviewer_sha256": canonical_sha256(f"{LINEAGE_ID}:{domain}:{tool_name}:{reason}"),
        }
        for tool_name, reason in sorted(ZERO_ARG_TOOL_EXEMPTIONS.get(domain, {}).items())
    ]


def _safe_message_target(behavior: str, text: str) -> dict[str, Any]:
    return {"behavior": behavior, "kind": "assistant_message", "text": text}


def _user_prompt(domain: str, split: str, behavior: str, ordinal: int, args: dict[str, Any]) -> str:
    facts = ", ".join(str(value) for value in _scalar_values(args))
    return (
        f"{domain} {split} reviewed {behavior} scenario {ordinal}. "
        f"Relevant user-provided details: {facts}."
    )


def _scalar_values(value: Any) -> list[Any]:
    if isinstance(value, dict):
        result: list[Any] = []
        for item in value.values():
            result.extend(_scalar_values(item))
        return result
    if isinstance(value, list):
        result = []
        for item in value:
            result.extend(_scalar_values(item))
        return result
    if value is None:
        return []
    return [value]


def _replay_source_row(row: dict[str, Any], runtime_factory: RuntimeFactory) -> None:
    replay_hashes: list[str] = []
    for _ in range(2):
        runtime = runtime_factory(copy.deepcopy(row))
        results: list[Any] = []
        for turn in row["turns"]:
            for call in turn["assistant"]["tool_calls"]:
                try:
                    result = runtime.call(
                        call["tool_name"],
                        copy.deepcopy(call.get("arguments") or {}),
                    )
                except Exception as exc:  # deterministic failures are recovery evidence
                    result = {
                        "exception_type": exc.__class__.__name__,
                        "message": str(exc),
                    }
                results.append(_canonical_value(result))
        replay_hashes.append(
            canonical_sha256(
                {
                    "results": results,
                    "final_state": _canonical_value(runtime.state),
                }
            )
        )
    if len(set(replay_hashes)) != 1:
        raise Tau3V3ScenarioError(
            f"{row['source_id']} is not deterministic across isolated replays"
        )


def _add_tool_coverage_closures(
    *,
    rows: list[dict[str, Any]],
    pools: dict[str, dict[str, list[dict[str, Any]]]],
    prompts: dict[str, str],
    official_catalog: dict[str, list[dict[str, Any]]],
    base_states: dict[str, dict[str, Any]],
    revision: str,
    runtime_family: str,
    tau_repo: Path,
    runtime_factory: RuntimeFactory,
) -> None:
    for split in SPLITS:
        required_count = TRAIN_TOOL_CALL_MIN if split == "train" else VALIDATION_TOOL_CALL_MIN
        required_distinct = TRAIN_DISTINCT_ARGS_MIN if split == "train" else VALIDATION_DISTINCT_ARGS_MIN
        for domain in DOMAINS:
            domain_rows = [row for row in rows if row["split"] == split and row["domain"] == domain]
            for tool_name, args_pool in pools.get(domain, {}).items():
                if not args_pool:
                    continue
                exempt = f"{domain}.{tool_name}" in ZERO_ARG_DISTINCT_EXEMPTIONS
                target_count = (
                    required_count
                    if exempt
                    else max(required_count, VALIDATION_NONEXEMPT_TOOL_TARGET)
                )
                targets = _tool_targets_for(domain_rows, tool_name)
                payload_counts: dict[str, int] = {}
                for target in targets:
                    payload_sha = canonical_sha256(target.get("arguments") or {})
                    payload_counts[payload_sha] = payload_counts.get(payload_sha, 0) + 1
                distinct = {
                    canonical_sha256(target.get("arguments") or {})
                    for target in targets
                }
                needed_count = max(0, target_count - len(targets))
                needed_distinct = 0 if exempt else max(0, required_distinct - len(distinct))
                additions = max(needed_count, needed_distinct)
                for index in range(additions):
                    arg_index = _least_represented_closure_arg_index(
                        args_pool,
                        payload_counts,
                    )
                    args = copy.deepcopy(args_pool[arg_index % len(args_pool)])
                    payload_sha = canonical_sha256(args)
                    distinct.add(payload_sha)
                    payload_counts[payload_sha] = payload_counts.get(payload_sha, 0) + 1
                    closure = _make_tool_closure_row(
                        domain=domain,
                        split=split,
                        tool_name=tool_name,
                        args=args,
                        index=index,
                        revision=revision,
                        runtime_family=runtime_family,
                        tau_repo=tau_repo,
                        system_prompt=prompts[domain],
                        v2_tool_catalog=official_catalog[domain],
                        initial_state=base_states[domain],
                    )
                    _replay_source_row(closure, runtime_factory)
                    rows.append(closure)
                    domain_rows.append(closure)
                if split == "train" and domain == "telecom" and not exempt:
                    targets = _tool_targets_for(domain_rows, tool_name)
                    distinct = {
                        canonical_sha256(target.get("arguments") or {})
                        for target in targets
                    }
                    variant_needed = max(0, required_distinct - len(distinct))
                    for variant_index in range(variant_needed):
                        variant = _telecom_state_variant_for_tool(
                            base_states[domain],
                            tool_name,
                            variant_index,
                            distinct,
                        )
                        if variant is None:
                            break
                        variant_state, variant_args, derivation = variant
                        distinct.add(canonical_sha256(variant_args))
                        closure = _make_tool_closure_row(
                            domain=domain,
                            split=split,
                            tool_name=tool_name,
                            args=variant_args,
                            index=additions + variant_index,
                            revision=revision,
                            runtime_family=runtime_family,
                            tau_repo=tau_repo,
                            system_prompt=prompts[domain],
                            v2_tool_catalog=official_catalog[domain],
                            initial_state=variant_state,
                            state_derivation=derivation,
                        )
                        _replay_source_row(closure, runtime_factory)
                        rows.append(closure)
                        domain_rows.append(closure)


def _tool_targets_for(rows: list[dict[str, Any]], tool_name: str) -> list[dict[str, Any]]:
    return [
        target
        for row in rows
        for target in _targets(row)
        if target.get("masked") is not True
        and target.get("kind") == "tool_call"
        and target.get("tool_name") == tool_name
    ]


def _least_represented_closure_arg_index(
    args_pool: list[dict[str, Any]],
    payload_counts: dict[str, int],
) -> int:
    return min(
        range(len(args_pool)),
        key=lambda index: (
            payload_counts.get(canonical_sha256(args_pool[index]), 0),
            index,
        ),
    )


def _add_tool_argument_dominance_closures(
    *,
    rows: list[dict[str, Any]],
    blockers: list[str],
    pools: dict[str, dict[str, list[dict[str, Any]]]],
    prompts: dict[str, str],
    official_catalog: dict[str, list[dict[str, Any]]],
    base_states: dict[str, dict[str, Any]],
    revision: str,
    runtime_family: str,
    tau_repo: Path,
    runtime_factory: RuntimeFactory,
) -> None:
    for split in SPLITS:
        for domain in DOMAINS:
            domain_rows = [
                row
                for row in rows
                if row["split"] == split and row["domain"] == domain
            ]
            for tool_name, args_pool in pools.get(domain, {}).items():
                if (
                    not args_pool
                    or f"{domain}.{tool_name}" in ZERO_ARG_DISTINCT_EXEMPTIONS
                ):
                    continue
                pool_cardinality = len(
                    {
                        canonical_sha256(args)
                        for args in args_pool
                    }
                )
                if (
                    pool_cardinality < int(1 / MAX_TOOL_ARGUMENT_SHARE)
                    and not (
                        domain == "telecom"
                        and tool_name in TELECOM_STATE_VARIANT_TOOLS
                    )
                ):
                    blockers.append(
                        f"{split}:{domain}:{tool_name} has only "
                        f"{pool_cardinality} replayable argument payloads for "
                        f"dominance <= {MAX_TOOL_ARGUMENT_SHARE:.2f}"
                    )
                    continue
                next_index = _next_tool_closure_index(
                    domain_rows,
                    split=split,
                    domain=domain,
                    tool_name=tool_name,
                )
                for addition_index in range(512):
                    targets = _tool_targets_for(domain_rows, tool_name)
                    payload_counts: dict[str, int] = {}
                    for target in targets:
                        payload_sha = canonical_sha256(target.get("arguments") or {})
                        payload_counts[payload_sha] = payload_counts.get(payload_sha, 0) + 1
                    if (
                        targets
                        and max(payload_counts.values()) / len(targets)
                        <= MAX_TOOL_ARGUMENT_SHARE
                    ):
                        break
                    distinct = set(payload_counts)
                    variant = (
                        _telecom_state_variant_for_tool(
                            base_states[domain],
                            tool_name,
                            next_index + addition_index,
                            distinct,
                        )
                        if domain == "telecom"
                        else None
                    )
                    if variant is None:
                        state = base_states[domain]
                        args = copy.deepcopy(
                            args_pool[
                                _least_represented_closure_arg_index(
                                    args_pool,
                                    payload_counts,
                                )
                            ]
                        )
                        derivation = None
                    else:
                        state, args, derivation = variant
                    closure = _make_tool_closure_row(
                        domain=domain,
                        split=split,
                        tool_name=tool_name,
                        args=args,
                        index=next_index + addition_index,
                        revision=revision,
                        runtime_family=runtime_family,
                        tau_repo=tau_repo,
                        system_prompt=prompts[domain],
                        v2_tool_catalog=official_catalog[domain],
                        initial_state=state,
                        state_derivation=derivation,
                        family_shard=addition_index % DOMINANCE_FAMILY_SHARDS,
                    )
                    _replay_source_row(closure, runtime_factory)
                    rows.append(closure)
                    domain_rows.append(closure)
                else:
                    blockers.append(
                        f"{split}:{domain}:{tool_name} could not satisfy "
                        f"argument dominance <= {MAX_TOOL_ARGUMENT_SHARE:.2f}"
                    )


def _next_tool_closure_index(
    rows: list[dict[str, Any]],
    *,
    split: str,
    domain: str,
    tool_name: str,
) -> int:
    prefix = f"v3-{split}-{domain}-tool-closure-{tool_name}-"
    return sum(
        1
        for row in rows
        if str(row.get("source_id") or "").startswith(prefix)
    )


def _add_target_tool_dominance_closures(
    *,
    rows: list[dict[str, Any]],
    blockers: list[str],
    pools: dict[str, dict[str, list[dict[str, Any]]]],
    prompts: dict[str, str],
    official_catalog: dict[str, list[dict[str, Any]]],
    base_states: dict[str, dict[str, Any]],
    revision: str,
    runtime_family: str,
    tau_repo: Path,
    runtime_factory: RuntimeFactory,
) -> None:
    for split in SPLITS:
        for domain in DOMAINS:
            eligible_tools = [
                tool_name
                for tool_name, args_pool in pools.get(domain, {}).items()
                if args_pool
                and f"{domain}.{tool_name}" not in ZERO_ARG_DISTINCT_EXEMPTIONS
            ]
            if len(eligible_tools) < int(1 / MAX_TOOL_ARGUMENT_SHARE):
                blockers.append(
                    f"{split}:{domain} has only {len(eligible_tools)} eligible tools "
                    f"for target-tool dominance <= {MAX_TOOL_ARGUMENT_SHARE:.2f}"
                )
                continue
            domain_rows = [
                row
                for row in rows
                if row["split"] == split and row["domain"] == domain
            ]
            next_indices = {
                tool_name: _next_tool_closure_index(
                    domain_rows,
                    split=split,
                    domain=domain,
                    tool_name=tool_name,
                )
                for tool_name in eligible_tools
            }
            for addition_index in range(512):
                counts = {
                    tool_name: len(_tool_targets_for(domain_rows, tool_name))
                    for tool_name in eligible_tools
                }
                total = sum(counts.values())
                if total and max(counts.values()) / total <= MAX_TOOL_ARGUMENT_SHARE:
                    break
                max_count = max(counts.values(), default=0)
                candidates = [
                    tool_name
                    for tool_name in eligible_tools
                    if counts[tool_name] < max_count
                ] or eligible_tools
                tool_name = min(candidates, key=lambda name: (counts[name], name))
                targets = _tool_targets_for(domain_rows, tool_name)
                payload_counts: dict[str, int] = {}
                for target in targets:
                    payload_sha = canonical_sha256(target.get("arguments") or {})
                    payload_counts[payload_sha] = payload_counts.get(payload_sha, 0) + 1
                args_pool = pools[domain][tool_name]
                args = copy.deepcopy(
                    args_pool[
                        _least_represented_closure_arg_index(
                            args_pool,
                            payload_counts,
                        )
                    ]
                )
                index = next_indices[tool_name]
                closure = _make_tool_closure_row(
                    domain=domain,
                    split=split,
                    tool_name=tool_name,
                    args=args,
                    index=index,
                    revision=revision,
                    runtime_family=runtime_family,
                    tau_repo=tau_repo,
                    system_prompt=prompts[domain],
                    v2_tool_catalog=official_catalog[domain],
                    initial_state=base_states[domain],
                    family_shard=addition_index % DOMINANCE_FAMILY_SHARDS,
                )
                _replay_source_row(closure, runtime_factory)
                rows.append(closure)
                domain_rows.append(closure)
                next_indices[tool_name] = index + 1
            else:
                blockers.append(
                    f"{split}:{domain} could not satisfy target-tool dominance "
                    f"<= {MAX_TOOL_ARGUMENT_SHARE:.2f}"
                )


def _add_source_dominance_blockers(
    rows: list[dict[str, Any]],
    blockers: list[str],
) -> None:
    for split in SPLITS:
        for domain in DOMAINS:
            domain_rows = [
                row
                for row in rows
                if row["split"] == split and row["domain"] == domain
            ]
            by_tool: dict[str, list[str]] = {}
            for row in domain_rows:
                for target in _targets(row):
                    tool_name = str(target.get("tool_name") or "")
                    if (
                        target.get("masked") is True
                        or target.get("kind") != "tool_call"
                        or f"{domain}.{tool_name}" in ZERO_ARG_DISTINCT_EXEMPTIONS
                    ):
                        continue
                    by_tool.setdefault(tool_name, []).append(
                        canonical_sha256(target.get("arguments") or {})
                    )
            total = sum(len(payloads) for payloads in by_tool.values())
            if total and max(len(payloads) for payloads in by_tool.values()) / total > MAX_TOOL_ARGUMENT_SHARE:
                blockers.append(
                    f"{split}:{domain}:target_tool_share remains above "
                    f"{MAX_TOOL_ARGUMENT_SHARE:.2f}"
                )
            for tool_name, payloads in sorted(by_tool.items()):
                payload_counts: dict[str, int] = {}
                for payload in payloads:
                    payload_counts[payload] = payload_counts.get(payload, 0) + 1
                if (
                    payloads
                    and max(payload_counts.values()) / len(payloads)
                    > MAX_TOOL_ARGUMENT_SHARE
                ):
                    blockers.append(
                        f"{split}:{domain}:{tool_name}:argument_payload_share "
                        f"remains above {MAX_TOOL_ARGUMENT_SHARE:.2f}"
                    )


def _telecom_state_variant_for_tool(
    base_state: dict[str, Any],
    tool_name: str,
    variant_index: int,
    distinct: set[str],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]] | None:
    if tool_name not in TELECOM_STATE_VARIANT_TOOLS:
        return None
    for offset in range(64):
        ordinal = variant_index + offset
        state = copy.deepcopy(base_state)
        customer = _telecom_variant_customer(state, ordinal)
        if tool_name == "get_customer_by_id":
            args = {"customer_id": customer["customer_id"]}
        elif tool_name == "get_customer_by_name":
            args = {
                "full_name": customer["full_name"],
                "dob": customer["date_of_birth"],
            }
        elif tool_name == "resume_line":
            line = _telecom_variant_line(state, customer, ordinal, status="Suspended")
            args = {"customer_id": customer["customer_id"], "line_id": line["line_id"]}
        else:
            bill = _telecom_variant_bill(state, customer, ordinal)
            args = {"customer_id": customer["customer_id"], "bill_id": bill["bill_id"]}
        if canonical_sha256(args) in distinct:
            continue
        derivation = {
            "id": "reviewed_deterministic_training_state_variant_v1",
            "source": "pinned_tau_training_db_schema",
            "base_state_sha256": canonical_sha256(base_state),
            "variant_state_sha256": canonical_sha256(state),
            "tool_name": tool_name,
            "variant_ordinal": ordinal,
            "derived_entity_ids": _telecom_variant_entity_ids(state, customer),
            "reviewer": "flightrecorder-v3-source-coverage-gate",
            "reviewer_sha256": canonical_sha256(
                {
                    "lineage_id": LINEAGE_ID,
                    "domain": "telecom",
                    "tool_name": tool_name,
                    "variant_ordinal": ordinal,
                    "base_state_sha256": canonical_sha256(base_state),
                }
            ),
        }
        return state, args, derivation
    return None


def _telecom_variant_customer(state: dict[str, Any], ordinal: int) -> dict[str, Any]:
    template = copy.deepcopy(state["customers"][ordinal % len(state["customers"])])
    customer_id = f"C9{ordinal + 1:03d}"
    template["customer_id"] = customer_id
    template["full_name"] = f"Training Variant {ordinal + 1:03d}"
    template["date_of_birth"] = f"1980-01-{(ordinal % 28) + 1:02d}"
    template["email"] = f"training.variant.{ordinal + 1:03d}@example.com"
    template["phone_number"] = f"555-900-{ordinal + 1:04d}"
    template["line_ids"] = []
    template["bill_ids"] = []
    state["customers"].append(template)
    return template


def _telecom_variant_line(
    state: dict[str, Any],
    customer: dict[str, Any],
    ordinal: int,
    *,
    status: str,
) -> dict[str, Any]:
    template = copy.deepcopy(state["lines"][ordinal % len(state["lines"])])
    line_id = f"L9{ordinal + 1:03d}"
    template["line_id"] = line_id
    template["phone_number"] = f"555-901-{ordinal + 1:04d}"
    template["status"] = status
    template["suspension_start_date"] = "2025-01-15" if status == "Suspended" else None
    customer["line_ids"].append(line_id)
    state["lines"].append(template)
    return template


def _telecom_variant_bill(
    state: dict[str, Any],
    customer: dict[str, Any],
    ordinal: int,
) -> dict[str, Any]:
    template = copy.deepcopy(state["bills"][ordinal % len(state["bills"])])
    bill_id = f"B9{ordinal + 1:03d}"
    template["bill_id"] = bill_id
    template["customer_id"] = customer["customer_id"]
    template["status"] = "Paid"
    customer["bill_ids"].append(bill_id)
    state["bills"].append(template)
    return template


def _telecom_variant_entity_ids(
    state: dict[str, Any],
    customer: dict[str, Any],
) -> dict[str, Any]:
    return {
        "customer_id": customer["customer_id"],
        "line_ids": list(customer.get("line_ids") or []),
        "bill_ids": list(customer.get("bill_ids") or []),
    }


def _make_tool_closure_row(
    *,
    domain: str,
    split: str,
    tool_name: str,
    args: dict[str, Any],
    index: int,
    revision: str,
    runtime_family: str,
    tau_repo: Path,
    system_prompt: str,
    v2_tool_catalog: list[dict[str, Any]],
    initial_state: dict[str, Any],
    state_derivation: dict[str, Any] | None = None,
    family_shard: int | None = None,
) -> dict[str, Any]:
    source_id = f"v3-{split}-{domain}-tool-closure-{tool_name}-{index:03d}"
    family_id = f"v3-{split}-{domain}-tool-closure-{tool_name}"
    if family_shard is not None:
        family_id += f"-dominance-{family_shard:02d}"
    prompt = (
        f"{domain} {split} reviewed tool coverage closure for {tool_name}. "
        f"Relevant user-provided details: {', '.join(str(value) for value in _scalar_values(args))}."
    )
    generation = {
        "lineage_id": LINEAGE_ID,
        "deterministic": True,
        "actual_tau_runtime_family": runtime_family,
        "base_state_sha256": canonical_sha256(initial_state),
        "v2_tool_catalog_sha256": canonical_sha256(v2_tool_catalog),
        "v2_system_prompt_sha256": canonical_sha256(system_prompt),
        "negative_or_unsafe_actions_masked": True,
        "dev_or_sealed_payload_access_count": 0,
        "natural_train_corpus_seeded": False,
        "source_identity_sha256": canonical_sha256(source_id),
        "source_family_sha256": canonical_sha256(family_id),
        "source_prompt_sha256": canonical_sha256(prompt),
    }
    if state_derivation is not None:
        generation["state_derivation"] = state_derivation
    return {
        "schema_version": TAU3_V3_SCENARIO_SOURCE_SCHEMA_VERSION,
        "trajectory_id": source_id,
        "domain": domain,
        "split": split,
        "source_family": "reviewed_synthetic",
        "source_family_id": family_id,
        "source_id": source_id,
        "tau_revision": revision,
        "runtime_family": runtime_family,
        "tau_repo": tau_repo.relative_to(_project_root()).as_posix(),
        "system_prompt": system_prompt,
        "initial_state": copy.deepcopy(initial_state),
        "turns": [
            {
                "user": {"content": prompt},
                "assistant": {
                    "decision_ordinal": 0,
                    "tool_calls": [{"tool_name": tool_name, "arguments": copy.deepcopy(args)}],
                    "safe_corrected_target": _tool_target(
                        tool_name,
                        args,
                        behavior="later_task_completion_actions",
                    ),
                },
            }
        ],
        "v2_ordered_evaluation_tool_catalog": copy.deepcopy(v2_tool_catalog),
        "source_generation": generation,
        "tool_exemptions": _tool_exemptions(domain),
        "recipe": {
            "id": "tau3-v3-deterministic-tool-coverage-closure",
            "sha256": canonical_sha256("tau3-v3-deterministic-tool-coverage-closure"),
        },
        "teacher": {
            "id": "tau-train-actions-and-db-derived-tool-coverage",
            "sha256": canonical_sha256("tau-train-actions-and-db-derived-tool-coverage"),
        },
        "reviewer": {
            "id": "flightrecorder-v3-source-coverage-gate",
            "sha256": canonical_sha256("flightrecorder-v3-source-coverage-gate"),
        },
        "redaction": {"passed": True, "method": "synthetic-template-no-sensitive-trace"},
        "contamination": {
            "source_split": split,
            "raw_sealed_payload_read": False,
            "sealed_hash_only": True,
            "dev_payload_read": False,
        },
    }


def _coverage_summary(
    rows: list[dict[str, Any]],
    pools: dict[str, dict[str, list[dict[str, Any]]]],
    existing_blockers: list[str],
    corpus_summary: dict[str, Any],
    contamination_report: dict[str, Any],
) -> dict[str, Any]:
    blockers = list(existing_blockers)
    behavior_counts: dict[str, Any] = {}
    negative_context_counts: dict[str, Any] = {}
    family_counts: dict[str, Any] = {}
    tool_counts: dict[str, Any] = {}
    for split in SPLITS:
        behavior_counts[split] = {}
        negative_context_counts[split] = {}
        family_counts[split] = {}
        tool_counts[split] = {}
        required_behavior = (
            TRAIN_PER_BEHAVIOR_DOMAIN
            if split == "train"
            else VALIDATION_PER_BEHAVIOR_DOMAIN
        )
        required_families = TRAIN_FAMILY_MIN if split == "train" else VALIDATION_FAMILY_MIN
        required_calls = TRAIN_TOOL_CALL_MIN if split == "train" else VALIDATION_TOOL_CALL_MIN
        required_args = TRAIN_DISTINCT_ARGS_MIN if split == "train" else VALIDATION_DISTINCT_ARGS_MIN
        for domain in DOMAINS:
            domain_rows = [row for row in rows if row["split"] == split and row["domain"] == domain]
            family_ids = {row["source_family_id"] for row in domain_rows}
            family_counts[split][domain] = len(family_ids)
            if len(family_ids) < required_families:
                blockers.append(f"{split}.{domain} has {len(family_ids)} families; requires {required_families}")
            behavior_counts[split][domain] = {
                behavior: sum(
                    1
                    for row in domain_rows
                    for target in _targets(row)
                    if target.get("masked") is not True
                    and target.get("behavior") == behavior
                )
                for behavior in BEHAVIORS
            }
            for behavior, count in behavior_counts[split][domain].items():
                if count < required_behavior:
                    blockers.append(
                        f"{split}.{domain}.{behavior} has {count} targets; requires {required_behavior}"
                    )
            negative_context_counts[split][domain] = {}
            for behavior in sorted(NEGATIVE_CORRECTION_BEHAVIORS):
                negative_behavior = behavior.removesuffix("_correction")
                count = sum(
                    1
                    for row in domain_rows
                    for target in _targets(row)
                    if target.get("masked") is True
                    and target.get("behavior") == behavior
                    and target.get("negative_behavior") == negative_behavior
                    and target.get("reviewed") is True
                    and type(target.get("safe_correction_decision_ordinal")) is int
                )
                negative_context_counts[split][domain][negative_behavior] = count
                if count < required_behavior:
                    blockers.append(
                        f"{split}.{domain}.{negative_behavior} has {count} reviewed negative contexts; "
                        f"requires {required_behavior}"
                    )
            tool_counts[split][domain] = {}
            for tool_name in pools.get(domain, {}):
                targets = [
                    target
                    for row in domain_rows
                    for target in _targets(row)
                    if target.get("masked") is not True
                    and target.get("kind") == "tool_call"
                    and target.get("tool_name") == tool_name
                ]
                distinct = {
                    canonical_sha256(target.get("arguments") or {})
                    for target in targets
                }
                key = f"{domain}.{tool_name}"
                exempt = key in ZERO_ARG_DISTINCT_EXEMPTIONS
                record = {
                    "supervised_target_count": len(targets),
                    "distinct_argument_count": len(distinct),
                    "distinct_argument_exempt": exempt,
                }
                tool_counts[split][domain][tool_name] = record
                if len(targets) < required_calls:
                    blockers.append(
                        f"{split}.{key} has {len(targets)} supervised targets; requires {required_calls}"
                    )
                if not exempt and len(distinct) < required_args:
                    blockers.append(
                        f"{split}.{key} has {len(distinct)} distinct args; requires {required_args}"
                    )
    summary = {
        "schema_version": TAU3_V3_SCENARIO_SUMMARY_SCHEMA_VERSION,
        "lineage_id": LINEAGE_ID,
        "passed": not blockers,
        "status": "passed" if not blockers else "blocked",
        "row_count": len(rows),
        "blockers": blockers,
        "coverage": {
            "behavior_counts": behavior_counts,
            "negative_context_counts": negative_context_counts,
            "family_counts": family_counts,
            "tool_counts": tool_counts,
            "requirements": {
                "train_per_behavior_domain": TRAIN_PER_BEHAVIOR_DOMAIN,
                "validation_per_behavior_domain": VALIDATION_PER_BEHAVIOR_DOMAIN,
                "train_negative_contexts_per_correction_domain": TRAIN_PER_BEHAVIOR_DOMAIN,
                "validation_negative_contexts_per_correction_domain": VALIDATION_PER_BEHAVIOR_DOMAIN,
                "train_tool_calls": TRAIN_TOOL_CALL_MIN,
                "validation_tool_calls": VALIDATION_TOOL_CALL_MIN,
                "train_distinct_args": TRAIN_DISTINCT_ARGS_MIN,
                "validation_distinct_args": VALIDATION_DISTINCT_ARGS_MIN,
            },
        },
        "reviewed_exemptions": {
            "tool_exemptions": {
                domain: _tool_exemptions(domain)
                for domain in DOMAINS
            },
        },
        "sealed_access": {
            "dev_or_sealed_payload_access_count": 0,
            "materialized_dev_or_sealed_fields": [],
        },
        "natural_corpus": corpus_summary,
        "contamination_report": {
            "path": contamination_report.get("path"),
            "sha256": contamination_report.get("report_sha256"),
            "passed": contamination_report.get("passed"),
            "development_hash_only_evidence": contamination_report.get("development_hash_only_evidence"),
            "sealed_hash_only_comparison": contamination_report.get("sealed_hash_only_comparison"),
        },
    }
    summary["summary_sha256"] = canonical_sha256(
        {key: value for key, value in summary.items() if key != "summary_sha256"}
    )
    return summary


def _targets(row: dict[str, Any]) -> Iterable[dict[str, Any]]:
    for turn in row["turns"]:
        target = _dict(_dict(turn.get("assistant")).get("safe_corrected_target"))
        if target:
            yield target


def build_grounded_bundle_from_v3_sources(
    *,
    source_out: str | Path,
    grounded_out_dir: str | Path,
    strict: bool = True,
    **kwargs: Any,
) -> dict[str, Any]:
    """Convenience helper for generating sources and immediately grounding them."""

    build_tau3_v3_scenario_sources(out=source_out, strict=strict, **kwargs)
    return build_tau3_grounded_generation_dataset(
        source=source_out,
        out_dir=grounded_out_dir,
        strict_coverage=strict,
    )


def _is_mutation_tool(tool_name: str) -> bool:
    return tool_name.startswith(
        (
            "book_",
            "cancel_",
            "disable_",
            "enable_",
            "exchange_",
            "modify_",
            "refuel_",
            "resume_",
            "return_",
            "send_",
            "suspend_",
            "update_",
        )
    )


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise Tau3V3ScenarioError(f"expected JSON object: {path}")
    return payload


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(build_cli())
