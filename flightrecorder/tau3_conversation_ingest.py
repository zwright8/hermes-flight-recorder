"""Import official Tau text-mode simulations into governed training rows.

The importer accepts already-produced Tau ``results.json`` files.  It does not
run Tau, does not read sealed task payloads, and does not trust benchmark result
metadata alone: each admitted simulation is re-bound to the hash-addressed
Hermes training-side source files and rejected unless executable reward,
termination, split, source revision, tool schema, and leakage checks all pass.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import subprocess
import sys
from argparse import ArgumentParser
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from .path_safety import path_has_symlink_component
from .schema_registry import check_schema_contract
from .tau3_capture import canonical_sha256

TAU3_CONVERSATION_CORPUS_SCHEMA_VERSION = "hfr.tau3_conversation_import.v1"
TAU3_TOOL_SCHEMA_EXPORT_VERSION = "hfr.tau3_tool_schemas.v1"
ALLOWED_DOMAINS = {"airline", "retail", "telecom"}
TRAINING_SOURCE_SCHEMA_VERSION = "hfr.tau3_training_source.v1"
NORMAL_TERMINATIONS = {"user_stop", "agent_stop"}
TAU_AGENT_INSTRUCTION = """You are a customer service agent that helps the user according to the <policy> provided below.
In each turn you can either:
- Send a message to the user.
- Make a tool call.
You cannot do both at the same time.

Try to be helpful and always follow the policy. Always make sure you generate valid JSON only."""
TAU_SYSTEM_PROMPT = """<instructions>
{agent_instruction}
</instructions>
<policy>
{domain_policy}
</policy>"""
HIDDEN_KEYS = {
    "annotations",
    "audio_content",
    "audio_path",
    "audio_script_gold",
    "evaluation_criteria",
    "initial_state",
    "raw_data",
    "resolution_steps",
    "ticks",
    "usage",
    "user_scenario",
}
FORBIDDEN_TEXT_PATTERNS = (
    "<resolution_steps",
    "</resolution_steps",
    "you are testing that our user simulator is working correctly",
    "user simulator will have an issue for you to solve",
    "evaluation_criteria",
    "user_scenario",
    "task_instructions",
    "reward_basis",
    "<think>",
    "</think>",
    "###stop###",
    "###transfer###",
)
TAU_USER_CONTROL_MARKER_RE = re.compile(r"\s*(###(?:STOP|TRANSFER)###)$")
HIDDEN_PROSE_MIN_CHARS = 24
HIDDEN_PROSE_MIN_WORDS = 4
ERRORED_TOOL_RESULT_REJECTION_CODE = "errored_tool_result"
ERRORED_TOOL_RESULT_REJECTION_REASON = "generated result contains an errored assistant tool result"
SEALED_IDENTITY_OVERLAP_REJECTION_CODE = "sealed_task_identity_overlap"
SEALED_IDENTITY_OVERLAP_REJECTION_REASON = "source task identity intersects the hash-only sealed manifest"
THINKING_TAG_REJECTION_CODE = "thinking_tag"
THINKING_TAG_REJECTION_REASON = "generated result contains a forbidden thinking tag"
GENERATOR_FAILURE_REJECTION_CODE = "generator_terminal_failure"
GENERATOR_FAILURE_REJECTION_REASON = "generator terminal status was not success"


class Tau3ConversationIngestError(ValueError):
    """Raised when importing Tau simulations would produce unsafe training data."""


@dataclass(frozen=True)
class ImportSummary:
    """Small return object for callers that need paths and counts."""

    manifest: dict[str, Any]
    train_path: Path
    valid_path: Path
    manifest_path: Path


@dataclass(frozen=True)
class SourceTask:
    domain: str
    split: str
    task_id: str
    task_family: str
    task_sha256: str
    prompt_sha256: str
    source_revision: str
    row_sha256: str
    task: dict[str, Any]


@dataclass(frozen=True)
class SealedHashIndex:
    identity_hashes: frozenset[str]
    prompt_hashes: frozenset[str]


def import_tau3_conversations(
    results_paths: Sequence[str | Path],
    out_dir: str | Path,
    *,
    source_dir: str | Path,
    tool_schema_path: str | Path,
    generation_manifest_paths: list[str | Path] | None = None,
    expected_tau_revision: str | None = None,
    sealed_manifest_path: str | Path | None = None,
    allow_teacher_protocol_normalization: bool = False,
    teacher_id: str = "tau3-official-simulation",
    license_id: str = "tau3-benchmark-license-review-required",
) -> dict[str, Any]:
    """Import official Tau text-mode results into new-only MLX chat rows.

    Output files are ``train.jsonl``, ``valid.jsonl``, and ``manifest.json``.
    They are written with owner-only permissions.  Any rejection aborts the
    import so partial or mixed-quality corpora cannot silently enter training.
    """

    out = Path(out_dir)
    _require_new_output_dir(out)
    source_root = Path(source_dir)
    tool_schema_file = Path(tool_schema_path)
    tool_schema_artifact = _read_json(tool_schema_file)
    tool_contracts = _load_tool_schemas(tool_schema_file)
    source_tasks = _load_source_tasks(source_root)
    sealed_hashes = (
        _load_sealed_hashes(Path(sealed_manifest_path))
        if sealed_manifest_path
        else SealedHashIndex(identity_hashes=frozenset(), prompt_hashes=frozenset())
    )

    generation_records: list[dict[str, Any]] = []
    generation_protocol: dict[str, Any] | None = None
    if generation_manifest_paths:
        derived_results, generation_records, generation_protocol = _derive_results_from_generation_manifests(
            [Path(path) for path in generation_manifest_paths],
            expected_tau_revision=expected_tau_revision,
            source_tasks=source_tasks,
            sealed_hashes=sealed_hashes,
        )
        if results_paths:
            _require_explicit_results_match_derived(results_paths, derived_results)
        results_paths = derived_results
    if not results_paths:
        raise Tau3ConversationIngestError("at least one Tau results.json path or --generation-manifest is required")
    source_revision_set = {task.source_revision for task in source_tasks.values()}
    if len(source_revision_set) != 1:
        raise Tau3ConversationIngestError("training sources must bind exactly one Tau revision")
    if expected_tau_revision is not None and expected_tau_revision not in source_revision_set:
        raise Tau3ConversationIngestError("expected Tau revision does not match training sources")
    tool_revision = tool_schema_artifact.get("tau_revision") if isinstance(tool_schema_artifact, dict) else None
    if tool_revision is not None and tool_revision not in source_revision_set:
        raise Tau3ConversationIngestError("tool schema artifact Tau revision mismatch")
    if generation_records:
        generation_revisions = {record["tau_revision"] for record in generation_records}
        if generation_revisions != source_revision_set:
            raise Tau3ConversationIngestError("generation manifest Tau revision mismatch")
    rows_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "valid": []}
    seen_episode_ids: set[str] = set()
    seen_simulations: set[str] = set()
    seen_tau_simulation_identities: set[tuple[str, str, str]] = set()
    normalization_count = 0
    tau_control_marker_row_count = 0
    tau_control_marker_count = 0
    tau_control_marker_only_drop_count = 0
    rejection_count = 0
    source_revisions: set[str] = set()
    domains: set[str] = set()
    sealed_prompt_template_overlaps: set[str] = set()
    result_records = []

    for results_path_raw in results_paths:
        results_path = Path(results_path_raw)
        results = _read_json(results_path)
        info = _object(results.get("info"), f"{results_path}: info")
        source_revision = _nonempty_str(info.get("git_commit"), f"{results_path}: info.git_commit")
        source_revisions.add(source_revision)
        domain = _domain_from_results(results, results_path)
        domains.add(domain)
        if domain not in tool_contracts:
            raise Tau3ConversationIngestError(f"{results_path}: missing tool schema for domain {domain!r}")
        policy = _policy_from_results(results, results_path)
        contract = tool_contracts[domain]
        expected_policy_sha256 = contract.get("policy_sha256")
        actual_policy_sha256 = hashlib.sha256(policy.encode("utf-8")).hexdigest()
        if expected_policy_sha256 and expected_policy_sha256 != actual_policy_sha256:
            raise Tau3ConversationIngestError(
                f"{results_path}: policy hash does not match the pinned tool-schema artifact"
            )
        system_prompt = TAU_SYSTEM_PROMPT.format(
            agent_instruction=TAU_AGENT_INSTRUCTION,
            domain_policy=policy,
        )
        simulations = _simulations(results, results_path)
        result_records.append(
            {
                "path": str(results_path),
                "sha256": _file_sha256(results_path),
                "source_revision": source_revision,
                "domain": domain,
                "simulation_count": len(simulations),
            }
        )
        for sim_index, simulation in enumerate(simulations):
            try:
                row, normalization = _convert_simulation(
                    simulation,
                    sim_index=sim_index,
                    results_path=results_path,
                    domain=domain,
                    source_revision=source_revision,
                    source_tasks=source_tasks,
                    sealed_hashes=sealed_hashes,
                    tools=contract["tools"],
                    system_prompt=system_prompt,
                    allow_teacher_protocol_normalization=allow_teacher_protocol_normalization,
                    teacher_id=teacher_id,
                    license_id=license_id,
                )
            except Tau3ConversationIngestError:
                rejection_count += 1
                raise
            episode_id = str(row["metadata"]["episode_id"])
            if episode_id in seen_episode_ids:
                raise Tau3ConversationIngestError(f"duplicate episode_id: {episode_id}")
            tau_identity = (
                str(row["metadata"]["domain"]),
                str(row["metadata"]["task_id"]),
                str(row["metadata"]["source_simulation_id"]),
            )
            if tau_identity in seen_tau_simulation_identities:
                raise Tau3ConversationIngestError(
                    f"duplicate Tau simulation identity: {tau_identity[0]}/{tau_identity[1]}/{tau_identity[2]}"
                )
            sim_key = f"{results_path.resolve()}::{simulation.get('id') or sim_index}"
            if sim_key in seen_simulations:
                raise Tau3ConversationIngestError(f"duplicate simulation: {sim_key}")
            seen_episode_ids.add(episode_id)
            seen_simulations.add(sim_key)
            seen_tau_simulation_identities.add(tau_identity)
            source = source_tasks[(str(row["metadata"]["domain"]), str(row["metadata"]["task_id"]))]
            if source.prompt_sha256 in sealed_hashes.prompt_hashes:
                sealed_prompt_template_overlaps.add(source.prompt_sha256)
            normalization_count += 1 if normalization["mixed_content_tool_call"] else 0
            tau_control_marker_row_count += 1 if normalization["tau_control_markers_stripped"] else 0
            tau_control_marker_count += normalization["tau_control_markers_stripped"]
            tau_control_marker_only_drop_count += normalization["tau_control_marker_only_messages_dropped"]
            rows_by_split["train" if row["metadata"]["split"] == "train" else "valid"].append(row)

    if not rows_by_split["train"]:
        raise Tau3ConversationIngestError("no train rows admitted")
    if not rows_by_split["valid"]:
        raise Tau3ConversationIngestError("no valid rows admitted")
    overlap = sorted(
        {row["metadata"]["task_family"] for row in rows_by_split["train"]}
        & {row["metadata"]["task_family"] for row in rows_by_split["valid"]}
    )
    if overlap:
        raise Tau3ConversationIngestError(f"train/valid family overlap: {overlap[0]}")

    out.mkdir(mode=0o700)
    train_path = out / "train.jsonl"
    valid_path = out / "valid.jsonl"
    _write_jsonl_private(train_path, rows_by_split["train"])
    _write_jsonl_private(valid_path, rows_by_split["valid"])
    manifest = _manifest(
        out,
        rows_by_split=rows_by_split,
        result_records=result_records,
        source_dir=source_root,
        tool_schema_path=Path(tool_schema_path),
        sealed_manifest_path=Path(sealed_manifest_path) if sealed_manifest_path else None,
        source_revisions=source_revisions,
        domains=domains,
        normalization_count=normalization_count,
        tau_control_marker_row_count=tau_control_marker_row_count,
        tau_control_marker_count=tau_control_marker_count,
        tau_control_marker_only_drop_count=tau_control_marker_only_drop_count,
        rejection_count=rejection_count,
        allow_teacher_protocol_normalization=allow_teacher_protocol_normalization,
        teacher_id=teacher_id,
        license_id=license_id,
        generation_records=generation_records,
        generation_protocol=generation_protocol,
        sealed_prompt_template_overlap_count=len(sealed_prompt_template_overlaps),
    )
    check = check_schema_contract(manifest, name_or_id=TAU3_CONVERSATION_CORPUS_SCHEMA_VERSION)
    if check.get("passed") is not True:
        raise Tau3ConversationIngestError("manifest schema failed: " + "; ".join(check.get("errors", [])))
    _write_json_private(out / "manifest.json", manifest)
    return manifest


def export_tau3_tool_schemas(
    *,
    tau_repo: str | Path,
    tau_venv_python: str | Path,
    out_path: str | Path,
    expected_tau_revision: str,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Export exact assistant tool schemas and policy hashes from Tau."""

    repo = Path(tau_repo).resolve(strict=True)
    python = Path(tau_venv_python)
    if not python.is_absolute():
        python = Path.cwd() / python
    out = Path(out_path)
    if out.exists():
        raise Tau3ConversationIngestError(f"output already exists: {out}")
    if not python.is_file() or not os.access(python, os.X_OK):
        raise Tau3ConversationIngestError(f"Tau venv Python is not executable: {python}")
    revision = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout.strip()
    if revision != expected_tau_revision:
        raise Tau3ConversationIngestError(f"Tau repository revision mismatch: {revision!r}")
    code = r'''
import hashlib, json
from tau2.registry import registry
domains = {}
for domain in ("airline", "retail", "telecom"):
    env = registry.get_env_constructor(domain)()
    tools = [tool.openai_schema for tool in env.get_tools()]
    policy = env.get_policy()
    domains[domain] = {
        "tool_count": len(tools),
        "tools": tools,
        "tools_sha256": hashlib.sha256(json.dumps(tools, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest(),
        "policy_sha256": hashlib.sha256(policy.encode("utf-8")).hexdigest(),
    }
print(json.dumps({"domains": domains}, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
'''
    env = {key: value for key, value in os.environ.items() if not key.endswith("_API_KEY") and "TOKEN" not in key.upper()}
    env["PYTHONPATH"] = str(repo / "src")
    proc = subprocess.run(
        [str(python), "-c", code],
        cwd=str(repo),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=120,
    )
    if proc.returncode != 0:
        raise Tau3ConversationIngestError("Tau tool schema export failed: " + _redact(proc.stderr)[-500:])
    try:
        exported = json.loads(proc.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        raise Tau3ConversationIngestError("Tau tool schema export did not produce JSON") from exc
    payload = {
        "schema_version": TAU3_TOOL_SCHEMA_EXPORT_VERSION,
        "created_at": created_at or _now_utc(),
        "tau_revision": revision,
        "domains": exported["domains"],
        "sealed_payload_accessed": False,
        "test_payload_accessed": False,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    _write_json_private(out, payload)
    return payload


def build_arg_parser() -> ArgumentParser:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--result", dest="results", action="append", type=Path)
    parser.add_argument("--generation-manifest", dest="generation_manifests", action="append", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--tool-schemas", type=Path)
    parser.add_argument("--sealed-manifest", type=Path)
    parser.add_argument("--allow-teacher-protocol-normalization", action="store_true")
    parser.add_argument("--teacher-id", default="tau3-official-simulation")
    parser.add_argument("--license-id", default="tau3-benchmark-license-review-required")
    parser.add_argument("--export-tool-schemas", type=Path)
    parser.add_argument("--tau-repo", type=Path)
    parser.add_argument("--tau-venv-python", type=Path)
    parser.add_argument("--expected-tau-revision", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        if args.export_tool_schemas is not None:
            if args.tau_repo is None or args.tau_venv_python is None:
                raise Tau3ConversationIngestError("--export-tool-schemas requires --tau-repo and --tau-venv-python")
            payload = export_tau3_tool_schemas(
                tau_repo=args.tau_repo,
                tau_venv_python=args.tau_venv_python,
                out_path=args.export_tool_schemas,
                expected_tau_revision=args.expected_tau_revision,
            )
            if not args.results and not args.generation_manifests and args.out is None and args.source_dir is None and args.tool_schemas is None:
                print(json.dumps({"tool_schemas": str(args.export_tool_schemas), "domains": sorted(payload["domains"])}, indent=2, sort_keys=True))
                return 0
        if (not args.results and not args.generation_manifests) or args.out is None or args.source_dir is None or args.tool_schemas is None:
            raise Tau3ConversationIngestError("conversation import requires --result or --generation-manifest, plus --out, --source-dir, and --tool-schemas")
        manifest = import_tau3_conversations(
            args.results or [],
            args.out,
            source_dir=args.source_dir,
            tool_schema_path=args.tool_schemas,
            generation_manifest_paths=args.generation_manifests,
            expected_tau_revision=args.expected_tau_revision,
            sealed_manifest_path=args.sealed_manifest,
            allow_teacher_protocol_normalization=args.allow_teacher_protocol_normalization,
            teacher_id=args.teacher_id,
            license_id=args.license_id,
        )
    except (OSError, Tau3ConversationIngestError, ValueError, subprocess.SubprocessError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps({"manifest": str(Path(args.out) / "manifest.json"), "counts": manifest["counts"]}, indent=2, sort_keys=True))
    return 0


def _convert_simulation(
    simulation: dict[str, Any],
    *,
    sim_index: int,
    results_path: Path,
    domain: str,
    source_revision: str,
    source_tasks: dict[tuple[str, str], SourceTask],
    sealed_hashes: SealedHashIndex,
    tools: list[dict[str, Any]],
    system_prompt: str,
    allow_teacher_protocol_normalization: bool,
    teacher_id: str,
    license_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    sim_id = _nonempty_str(simulation.get("id") or f"simulation-{sim_index}", f"{results_path}: simulations[{sim_index}].id")
    task_id = _nonempty_str(simulation.get("task_id"), f"{results_path}: {sim_id}.task_id")
    source = source_tasks.get((domain, task_id))
    if source is None:
        raise Tau3ConversationIngestError(f"{results_path}: {sim_id}: task {domain}/{task_id} is not in train/development source")
    if source.source_revision != source_revision:
        raise Tau3ConversationIngestError(
            f"{results_path}: {sim_id}: source revision mismatch {source_revision!r} != {source.source_revision!r}"
        )
    _reject_if_sealed(source, sealed_hashes, results_path, sim_id)
    if float(_reward(simulation, sim_id)) != 1.0:
        raise Tau3ConversationIngestError(f"{results_path}: {sim_id}: reward must be 1")
    termination = str(simulation.get("termination_reason") or "")
    if termination not in NORMAL_TERMINATIONS:
        raise Tau3ConversationIngestError(f"{results_path}: {sim_id}: non-normal termination {termination!r}")
    _reject_error_markers(simulation, results_path, sim_id)
    messages, normalization = _messages_to_training_rows(
        _list(simulation.get("messages"), f"{results_path}: {sim_id}.messages"),
        tools,
        system_prompt=system_prompt,
        allow_teacher_protocol_normalization=allow_teacher_protocol_normalization,
        where=f"{results_path}: {sim_id}",
    )
    if not any(message.get("role") == "user" for message in messages):
        raise Tau3ConversationIngestError(f"{results_path}: {sim_id}: no visible user message")
    if not any(message.get("role") == "assistant" for message in messages):
        raise Tau3ConversationIngestError(f"{results_path}: {sim_id}: no assistant target")
    _reject_hidden_payloads(messages, source)
    split = "train" if source.split == "train" else "development"
    episode_id = f"tau3-{split}-{domain}-{task_id}-{_short_hash({'source': str(results_path), 'sim': sim_id})}"
    metadata = {
        "schema_version": TAU3_CONVERSATION_CORPUS_SCHEMA_VERSION,
        "episode_id": episode_id,
        "source_simulation_id": sim_id,
        "domain": domain,
        "task_id": task_id,
        "split": split,
        "task_family": source.task_family,
        "source_fingerprint_status": "verified",
        "source_revision": source_revision,
        "source_row_sha256": source.row_sha256,
        "task_sha256": source.task_sha256,
        "prompt_sha256": source.prompt_sha256,
        "tau_results_sha256": _file_sha256(results_path),
        "termination_reason": termination,
        "reward": 1.0,
        "teacher": {"id": teacher_id, "license": license_id, "harness": "official-tau-text-mode"},
        "system_prompt_sha256": canonical_sha256(system_prompt),
        "normalization": {
            "teacher_protocol_normalized": bool(
                normalization["mixed_content_tool_call"]
                or normalization["tau_control_markers_stripped"]
            ),
            "allow_teacher_protocol_normalization": allow_teacher_protocol_normalization,
            "mixed_content_tool_call_normalized": normalization["mixed_content_tool_call"],
            "tau_control_markers_stripped": normalization["tau_control_markers_stripped"],
            "tau_control_marker_only_user_messages_dropped": normalization[
                "tau_control_marker_only_messages_dropped"
            ],
        },
        "privacy": {
            "sealed_payload_read": False,
            "raw_data_stripped": True,
            "usage_stripped": True,
            "hidden_task_fields_stripped": True,
        },
    }
    row = {"messages": messages, "tools": tools, "metadata": metadata}
    metadata["row_sha256"] = canonical_sha256(
        {
            "messages": messages,
            "tools": tools,
            "metadata_without_row_hash": {k: v for k, v in metadata.items() if k != "row_sha256"},
        }
    )
    return row, normalization


def _messages_to_training_rows(
    raw_messages: list[Any],
    tools: list[dict[str, Any]],
    *,
    system_prompt: str,
    allow_teacher_protocol_normalization: bool,
    where: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    tool_names = {_tool_name(tool) for tool in tools}
    converted: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
    pending_tool_call_ids: set[str] = set()
    normalization = {
        "mixed_content_tool_call": False,
        "tau_control_markers_stripped": 0,
        "tau_control_marker_only_messages_dropped": 0,
    }
    for index, raw in enumerate(raw_messages):
        msg = _object(raw, f"{where}.messages[{index}]")
        role = str(msg.get("role") or "")
        if role == "system":
            continue
        if role == "user":
            tool_calls = msg.get("tool_calls") or []
            if tool_calls:
                # User tool calls are simulator/environment plumbing.  They are
                # not assistant targets and should never be learned as actions.
                continue
            content = _clean_text(msg.get("content"), f"{where}.messages[{index}].content")
            content, marker_stripped = _strip_tau_user_control_marker(
                content,
                allow_teacher_protocol_normalization=allow_teacher_protocol_normalization,
                where=f"{where}.messages[{index}].content",
            )
            if marker_stripped:
                normalization["tau_control_markers_stripped"] += 1
                if not content:
                    normalization["tau_control_marker_only_messages_dropped"] += 1
            if content:
                converted.append({"role": "user", "content": content})
            continue
        if role == "assistant":
            content = _clean_text(msg.get("content"), f"{where}.messages[{index}].content", allow_empty=True)
            tool_calls = _assistant_tool_calls(msg.get("tool_calls") or [], tool_names, f"{where}.messages[{index}]")
            if content and tool_calls:
                if not allow_teacher_protocol_normalization:
                    raise Tau3ConversationIngestError(
                        f"{where}.messages[{index}]: assistant mixed content+tool call requires --allow-teacher-protocol-normalization"
                    )
                normalization["mixed_content_tool_call"] = True
                content = ""
            if tool_calls:
                converted.append({"role": "assistant", "tool_calls": tool_calls})
                pending_tool_call_ids.update(str(call["id"]) for call in tool_calls)
            elif content:
                converted.append({"role": "assistant", "content": content})
            else:
                raise Tau3ConversationIngestError(f"{where}.messages[{index}]: empty assistant message")
            continue
        if role == "tool":
            if str(msg.get("requestor") or "assistant") != "assistant":
                continue
            tool_messages = msg.get("tool_messages")
            if tool_messages:
                for sub_index, tool_message in enumerate(_list(tool_messages, f"{where}.messages[{index}].tool_messages")):
                    sub_message = _object(tool_message, f"{where}.messages[{index}].tool_messages[{sub_index}]")
                    if str(sub_message.get("requestor") or "assistant") != "assistant":
                        continue
                    converted.append(_tool_result(sub_message, pending_tool_call_ids, f"{where}.messages[{index}].tool_messages[{sub_index}]"))
                continue
            converted.append(_tool_result(msg, pending_tool_call_ids, f"{where}.messages[{index}]"))
            continue
        raise Tau3ConversationIngestError(f"{where}.messages[{index}]: unsupported role {role!r}")
    if pending_tool_call_ids:
        raise Tau3ConversationIngestError(f"{where}: unpaired tool call id: {sorted(pending_tool_call_ids)[0]}")
    return converted, normalization


def _strip_tau_user_control_marker(
    content: str,
    *,
    allow_teacher_protocol_normalization: bool,
    where: str,
) -> tuple[str, bool]:
    marker = TAU_USER_CONTROL_MARKER_RE.search(content)
    if marker is None:
        return content, False
    if not allow_teacher_protocol_normalization:
        raise Tau3ConversationIngestError(
            f"{where}: trailing Tau control marker requires --allow-teacher-protocol-normalization"
        )
    return content[: marker.start()].rstrip(), True


def _assistant_tool_calls(raw_calls: list[Any], tool_names: set[str], where: str) -> list[dict[str, Any]]:
    calls = []
    for index, raw_call in enumerate(raw_calls):
        call = _object(raw_call, f"{where}.tool_calls[{index}]")
        requestor = str(call.get("requestor") or "assistant")
        if requestor != "assistant":
            continue
        name = call.get("name")
        args = call.get("arguments")
        if name is None and isinstance(call.get("function"), dict):
            name = call["function"].get("name")
            args = call["function"].get("arguments")
        name = _nonempty_str(name, f"{where}.tool_calls[{index}].name")
        if name not in tool_names:
            raise Tau3ConversationIngestError(f"{where}.tool_calls[{index}]: undefined tool {name!r}")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError as exc:
                raise Tau3ConversationIngestError(f"{where}.tool_calls[{index}]: arguments are not JSON: {exc.msg}") from exc
        if not isinstance(args, dict):
            raise Tau3ConversationIngestError(f"{where}.tool_calls[{index}]: arguments must be a JSON object")
        call_id = str(call.get("id") or f"call_{index}")
        calls.append(
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args, sort_keys=True, separators=(",", ":"))},
            }
        )
    return calls


def _tool_result(raw: Any, pending_tool_call_ids: set[str], where: str) -> dict[str, Any]:
    msg = _object(raw, where)
    if msg.get("error") is True:
        raise Tau3ConversationIngestError(f"{where}: errored tool result cannot be a positive target")
    tool_call_id = _nonempty_str(msg.get("tool_call_id") or msg.get("id"), f"{where}.tool_call_id")
    if tool_call_id not in pending_tool_call_ids:
        raise Tau3ConversationIngestError(f"{where}: tool result id {tool_call_id!r} has no assistant call")
    pending_tool_call_ids.remove(tool_call_id)
    content = _clean_text(msg.get("content"), f"{where}.content", allow_empty=True)
    if not content:
        payload = {k: v for k, v in msg.items() if k not in HIDDEN_KEYS and k not in {"role", "tool_call_id", "id", "timestamp"}}
        content = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return {"role": "tool", "tool_call_id": tool_call_id, "content": content}


def _load_source_tasks(source_dir: Path) -> dict[tuple[str, str], SourceTask]:
    source_tasks: dict[tuple[str, str], SourceTask] = {}
    for split, filename in (("train", "train_tasks.jsonl"), ("development", "development_tasks.jsonl")):
        path = source_dir / filename
        if not path.is_file():
            raise Tau3ConversationIngestError(f"missing source task file: {path}")
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("schema_version") != TRAINING_SOURCE_SCHEMA_VERSION:
                raise Tau3ConversationIngestError(f"{path}:{line_number}: unexpected schema_version")
            domain = _nonempty_str(row.get("domain"), f"{path}:{line_number}.domain")
            if domain not in ALLOWED_DOMAINS:
                raise Tau3ConversationIngestError(f"{path}:{line_number}: unsupported domain {domain!r}")
            task = _object(row.get("task"), f"{path}:{line_number}.task")
            task_id = _nonempty_str(task.get("id"), f"{path}:{line_number}.task.id")
            key = (domain, task_id)
            if key in source_tasks:
                raise Tau3ConversationIngestError(f"duplicate source task: {domain}/{task_id}")
            source_tasks[key] = SourceTask(
                domain=domain,
                split=split,
                task_id=task_id,
                task_family=_nonempty_str(row.get("task_family"), f"{path}:{line_number}.task_family"),
                task_sha256=_nonempty_str(row.get("task_sha256"), f"{path}:{line_number}.task_sha256"),
                prompt_sha256=_nonempty_str(row.get("prompt_sha256"), f"{path}:{line_number}.prompt_sha256"),
                source_revision=_nonempty_str(row.get("source_revision"), f"{path}:{line_number}.source_revision"),
                row_sha256=canonical_sha256(row),
                task=task,
            )
    return source_tasks


def _load_tool_schemas(path: Path) -> dict[str, dict[str, Any]]:
    payload = _read_json(path)
    if isinstance(payload, dict) and isinstance(payload.get("domains"), dict):
        payload = payload["domains"]
    if not isinstance(payload, dict):
        raise Tau3ConversationIngestError("tool schema JSON must be an object keyed by domain")
    schemas: dict[str, dict[str, Any]] = {}
    for domain, raw_contract in payload.items():
        domain_name = str(domain)
        if domain_name not in ALLOWED_DOMAINS:
            continue
        if isinstance(raw_contract, dict):
            raw_tools = raw_contract.get("tools")
            policy_sha256 = raw_contract.get("policy_sha256")
            tools_sha256 = raw_contract.get("tools_sha256")
        else:
            raw_tools = raw_contract
            policy_sha256 = None
            tools_sha256 = None
        tools = _list(raw_tools, f"tool schemas {domain_name}")
        names = []
        clean_tools = []
        for index, tool in enumerate(tools):
            tool_obj = _object(tool, f"tool schemas {domain_name}[{index}]")
            name = _tool_name(tool_obj)
            if not name:
                raise Tau3ConversationIngestError(f"tool schemas {domain_name}[{index}]: missing function name")
            if name in names:
                raise Tau3ConversationIngestError(f"tool schemas {domain_name}: duplicate tool {name!r}")
            names.append(name)
            clean_tools.append(tool_obj)
        if not clean_tools:
            raise Tau3ConversationIngestError(f"tool schemas {domain_name}: no tools")
        if policy_sha256 is not None and not _is_sha256(policy_sha256):
            raise Tau3ConversationIngestError(f"tool schemas {domain_name}: invalid policy_sha256")
        actual_tools_sha256 = canonical_sha256(clean_tools)
        if tools_sha256 is not None and tools_sha256 != actual_tools_sha256:
            raise Tau3ConversationIngestError(f"tool schemas {domain_name}: tools_sha256 mismatch")
        schemas[domain_name] = {
            "tools": clean_tools,
            "policy_sha256": policy_sha256,
            "tools_sha256": actual_tools_sha256,
        }
    return schemas


def _derive_results_from_generation_manifests(
    manifest_paths: list[Path],
    *,
    expected_tau_revision: str | None,
    source_tasks: dict[tuple[str, str], SourceTask],
    sealed_hashes: SealedHashIndex,
) -> tuple[list[Path], list[dict[str, Any]], dict[str, Any]]:
    derived_results: list[Path] = []
    generation_records: list[dict[str, Any]] = []
    protocol_records: list[dict[str, Any]] = []
    for manifest_path in manifest_paths:
        manifest = _object(_read_json(manifest_path), f"{manifest_path}: manifest")
        check = check_schema_contract(manifest, name_or_id="tau3_teacher_generation_run")
        if check.get("passed") is not True:
            raise Tau3ConversationIngestError(
                f"{manifest_path}: generation manifest schema failed: " + "; ".join(check.get("errors", []))
            )
        if manifest.get("phase") != "final":
            raise Tau3ConversationIngestError(f"{manifest_path}: generation manifest phase must be final")
        tau_revision = _nonempty_str(manifest.get("tau_revision"), f"{manifest_path}: tau_revision")
        if expected_tau_revision is not None and tau_revision != expected_tau_revision:
            raise Tau3ConversationIngestError(f"{manifest_path}: generation manifest Tau revision mismatch")
        protocol = _object(manifest.get("protocol"), f"{manifest_path}: protocol")
        protocol_sha256 = _nonempty_str(protocol.get("sha256"), f"{manifest_path}: protocol.sha256")
        if not _is_sha256(protocol_sha256):
            raise Tau3ConversationIngestError(f"{manifest_path}: protocol.sha256 must be a SHA-256 string")
        _replay_protocol_record(manifest_path.parent, protocol, tau_revision, f"{manifest_path}: protocol")
        protocol_records.append(protocol)
        prelaunch = _object(manifest.get("prelaunch_receipt"), f"{manifest_path}: prelaunch_receipt")
        _replay_file_record(manifest_path.parent, prelaunch, f"{manifest_path}: prelaunch_receipt")
        receipts = _list(manifest.get("task_receipts"), f"{manifest_path}: task_receipts")
        success_count = 0
        failure_count = 0
        admitted_success_count = 0
        excluded_success_count = 0
        result_refs = []
        for index, receipt_ref_raw in enumerate(receipts):
            receipt_ref = _object(receipt_ref_raw, f"{manifest_path}: task_receipts[{index}]")
            receipt_path = _resolve_generation_ref(manifest_path.parent, receipt_ref.get("path"), f"{manifest_path}: task_receipts[{index}].path")
            receipt = _object(_read_json(receipt_path), f"{receipt_path}: receipt")
            receipt_sha256 = _file_sha256(receipt_path)
            if receipt.get("phase") != "task":
                raise Tau3ConversationIngestError(f"{receipt_path}: generation task receipt phase must be task")
            if receipt.get("terminal_status") != receipt_ref.get("terminal_status"):
                raise Tau3ConversationIngestError(f"{receipt_path}: task receipt status mismatch")
            if receipt.get("result_sha256") != receipt_ref.get("result_sha256"):
                raise Tau3ConversationIngestError(f"{receipt_path}: task receipt result hash mismatch")
            status = receipt_ref.get("terminal_status")
            if status == "success":
                success_count += 1
                if float(_receipt_reward(receipt, receipt_path)) != 1.0:
                    raise Tau3ConversationIngestError(f"{receipt_path}: successful task receipt reward must be 1")
                result_sha256 = _nonempty_str(receipt.get("result_sha256"), f"{receipt_path}: result_sha256")
                if not _is_sha256(result_sha256):
                    raise Tau3ConversationIngestError(f"{receipt_path}: result_sha256 must be a SHA-256 string")
                result_path = _receipt_result_path(receipt, receipt_path)
                if not result_path.is_file() or _file_sha256(result_path) != result_sha256:
                    raise Tau3ConversationIngestError(f"{receipt_path}: generated result hash mismatch")
                simulation = _require_normal_result_evidence(result_path)
                exclusion = _training_exclusion_for_result(simulation, result_path)
                if exclusion is None and sealed_hashes.identity_hashes:
                    results = _object(_read_json(result_path), f"{result_path}: results")
                    domain = _domain_from_results(results, result_path)
                    task_id = _nonempty_str(simulation.get("task_id"), f"{result_path}: simulation.task_id")
                    source = source_tasks.get((domain, task_id))
                    if source is None:
                        raise Tau3ConversationIngestError(
                            f"{result_path}: generated task {domain}/{task_id} is not in train/development source"
                        )
                    if _sealed_identity_overlap(source, sealed_hashes):
                        exclusion = (
                            SEALED_IDENTITY_OVERLAP_REJECTION_CODE,
                            SEALED_IDENTITY_OVERLAP_REJECTION_REASON,
                        )
                result_ref = {
                    "path": str(result_path),
                    "sha256": result_sha256,
                    "terminal_status": status,
                    "task_receipt_path": str(receipt_path),
                    "task_receipt_sha256": receipt_sha256,
                }
                if exclusion is None:
                    admitted_success_count += 1
                    derived_results.append(result_path)
                    result_ref.update(
                        {
                            "training_admitted": True,
                            "training_rejection_code": None,
                            "training_rejection_reason": None,
                        }
                    )
                else:
                    excluded_success_count += 1
                    result_ref.update(
                        {
                            "training_admitted": False,
                            "training_rejection_code": exclusion[0],
                            "training_rejection_reason": exclusion[1],
                        }
                    )
                result_refs.append(result_ref)
            else:
                failure_count += 1
                result_refs.append({
                    "path": str(receipt.get("result_path") or ""),
                    "sha256": receipt.get("result_sha256"),
                    "training_admitted": False,
                    "terminal_status": status,
                    "task_receipt_path": str(receipt_path),
                    "task_receipt_sha256": receipt_sha256,
                    "training_rejection_code": GENERATOR_FAILURE_REJECTION_CODE,
                    "training_rejection_reason": GENERATOR_FAILURE_REJECTION_REASON,
                })
        if manifest.get("success_count") != success_count or manifest.get("failure_count") != failure_count:
            raise Tau3ConversationIngestError(f"{manifest_path}: generation manifest success/failure counts do not replay")
        if admitted_success_count + excluded_success_count != success_count:
            raise Tau3ConversationIngestError(f"{manifest_path}: training admission counts do not replay")
        generation_records.append(
            {
                "path": str(manifest_path),
                "sha256": _file_sha256(manifest_path),
                "tau_revision": tau_revision,
                "protocol_sha256": protocol_sha256,
                "task_receipt_count": len(receipts),
                "success_count": success_count,
                "failure_count": failure_count,
                "admitted_success_count": admitted_success_count,
                "excluded_success_count": excluded_success_count,
                "results": result_refs,
            }
        )
    if not derived_results:
        raise Tau3ConversationIngestError(
            "generation manifests contain no training-admitted successful reward-1 results"
        )
    protocol_hashes = {record["sha256"] for record in protocol_records}
    if len(protocol_hashes) != 1:
        raise Tau3ConversationIngestError("generation manifests must share one protocol SHA")
    return derived_results, generation_records, protocol_records[0]


def _require_explicit_results_match_derived(
    explicit_results: Sequence[str | Path],
    derived_results: list[Path],
) -> None:
    explicit = [str(Path(path).resolve()) for path in explicit_results]
    derived = [str(path.resolve()) for path in derived_results]
    if explicit != derived:
        raise Tau3ConversationIngestError("explicit --result paths do not exactly match generation-derived success results")


def _replay_file_record(base: Path, record: dict[str, Any], where: str) -> None:
    path = _resolve_generation_ref_with_cwd_fallback(base, record.get("path"), f"{where}.path")
    if path_has_symlink_component(path, include_leaf=True):
        raise Tau3ConversationIngestError(f"{where}.path must not contain symlink components")
    resolved_base = base.resolve(strict=True)
    resolved_path = path.resolve(strict=False)
    if resolved_base not in resolved_path.parents:
        raise Tau3ConversationIngestError(f"{where}.path must remain inside the generation directory")
    expected_sha256 = _nonempty_str(record.get("sha256"), f"{where}.sha256")
    if not _is_sha256(expected_sha256):
        raise Tau3ConversationIngestError(f"{where}.sha256 must be a SHA-256 string")
    if not path.is_file() or _file_sha256(path) != expected_sha256:
        raise Tau3ConversationIngestError(f"{where}: file hash mismatch")


def _replay_protocol_record(base: Path, record: dict[str, Any], tau_revision: str, where: str) -> None:
    path = _resolve_generation_ref_with_cwd_fallback(base, record.get("path"), f"{where}.path")
    expected_sha256 = _nonempty_str(record.get("sha256"), f"{where}.sha256")
    if not _is_sha256(expected_sha256):
        raise Tau3ConversationIngestError(f"{where}.sha256 must be a SHA-256 string")
    if not path.is_file() or _file_sha256(path) != expected_sha256:
        raise Tau3ConversationIngestError(f"{where}: file hash mismatch")
    payload = _object(_read_json(path), f"{path}: protocol")
    check = check_schema_contract(payload, name_or_id="tau3_protocol_config")
    if check.get("passed") is not True:
        raise Tau3ConversationIngestError(
            f"{path}: protocol schema failed: " + "; ".join(check.get("errors", []))
        )
    revision_record = _object(payload.get("tau_revision"), f"{path}: tau_revision")
    if revision_record.get("revision") != tau_revision:
        raise Tau3ConversationIngestError(f"{path}: protocol Tau revision mismatch")


def _resolve_generation_ref(base: Path, value: Any, where: str) -> Path:
    text = _nonempty_str(value, where)
    path = Path(text)
    if path.is_absolute():
        return path
    if any(part == ".." for part in path.parts):
        raise Tau3ConversationIngestError(f"{where} must not escape the generation directory")
    return base / path


def _resolve_generation_ref_with_cwd_fallback(base: Path, value: Any, where: str) -> Path:
    text = _nonempty_str(value, where)
    path = Path(text)
    if path.is_absolute():
        return path
    if any(part == ".." for part in path.parts):
        raise Tau3ConversationIngestError(f"{where} must not escape the generation directory")
    base_relative = base / path
    if base_relative.is_file():
        return base_relative
    cwd_relative = Path.cwd() / path
    return cwd_relative if cwd_relative.is_file() else base_relative


def _receipt_reward(receipt: dict[str, Any], receipt_path: Path) -> float:
    reward = receipt.get("reward")
    if isinstance(reward, bool) or not isinstance(reward, (int, float)):
        raise Tau3ConversationIngestError(f"{receipt_path}: reward must be numeric")
    return float(reward)


def _receipt_result_path(receipt: dict[str, Any], receipt_path: Path) -> Path:
    raw_path = _nonempty_str(receipt.get("result_path"), f"{receipt_path}: result_path")
    return Path(raw_path)


def _require_normal_result_evidence(result_path: Path) -> dict[str, Any]:
    results = _object(_read_json(result_path), f"{result_path}: result")
    simulations = _simulations(results, result_path)
    if len(simulations) != 1:
        raise Tau3ConversationIngestError(f"{result_path}: generated result must contain exactly one simulation")
    sim_id = str(simulations[0].get("id") or "simulation-0")
    if float(_reward(simulations[0], sim_id)) != 1.0:
        raise Tau3ConversationIngestError(f"{result_path}: generated result reward must be 1")
    termination = str(simulations[0].get("termination_reason") or "")
    if termination not in NORMAL_TERMINATIONS:
        raise Tau3ConversationIngestError(f"{result_path}: generated result has non-normal termination {termination!r}")
    _reject_error_markers(simulations[0], result_path, sim_id)
    return simulations[0]


def _training_exclusion_for_result(
    simulation: dict[str, Any],
    result_path: Path,
) -> tuple[str, str] | None:
    messages = _list(simulation.get("messages"), f"{result_path}: generated result messages")
    for index, raw_message in enumerate(messages):
        message = _object(raw_message, f"{result_path}: generated result messages[{index}]")
        content = message.get("content")
        if isinstance(content, str) and re.search(r"</?think>", content, flags=re.IGNORECASE):
            return THINKING_TAG_REJECTION_CODE, THINKING_TAG_REJECTION_REASON
        if message.get("role") != "tool" or str(message.get("requestor") or "assistant") != "assistant":
            continue
        tool_messages = message.get("tool_messages")
        if tool_messages:
            nested_messages = _list(
                tool_messages,
                f"{result_path}: generated result messages[{index}].tool_messages",
            )
            for nested_index, raw_nested in enumerate(nested_messages):
                nested = _object(
                    raw_nested,
                    f"{result_path}: generated result messages[{index}].tool_messages[{nested_index}]",
                )
                if (
                    str(nested.get("requestor") or "assistant") == "assistant"
                    and nested.get("error") is True
                ):
                    return ERRORED_TOOL_RESULT_REJECTION_CODE, ERRORED_TOOL_RESULT_REJECTION_REASON
                nested_content = nested.get("content")
                if isinstance(nested_content, str) and re.search(
                    r"</?think>",
                    nested_content,
                    flags=re.IGNORECASE,
                ):
                    return THINKING_TAG_REJECTION_CODE, THINKING_TAG_REJECTION_REASON
        elif message.get("error") is True:
            return ERRORED_TOOL_RESULT_REJECTION_CODE, ERRORED_TOOL_RESULT_REJECTION_REASON
    return None


def _tool_name(tool: dict[str, Any]) -> str:
    if isinstance(tool.get("function"), dict):
        return str(tool["function"].get("name") or "")
    return str(tool.get("name") or "")


def _load_sealed_hashes(path: Path) -> SealedHashIndex:
    manifest = _read_json(path)
    if not isinstance(manifest, dict):
        raise Tau3ConversationIngestError("sealed manifest must be a hash-only object")
    identity_hashes: set[str] = set()
    prompt_hashes: set[str] = set()
    for key in ("task_id_hashes", "task_hashes", "leakage_blocking_hashes", "sealed_task_hashes"):
        value = manifest.get(key)
        if isinstance(value, list):
            identity_hashes.update(str(item) for item in value if isinstance(item, str))
    legacy_prompt_hashes = manifest.get("prompt_hashes")
    if isinstance(legacy_prompt_hashes, list):
        prompt_hashes.update(str(item) for item in legacy_prompt_hashes if isinstance(item, str))
    entries = manifest.get("entries")
    if isinstance(entries, list):
        allowed_entry_keys = {"task_id_sha256", "task_sha256", "prompt_sha256"}
        for index, entry in enumerate(entries):
            if not isinstance(entry, dict) or not set(entry).issubset(allowed_entry_keys):
                raise Tau3ConversationIngestError(
                    f"sealed manifest entry {index} is not hash-only"
                )
            for key, value in entry.items():
                if not _is_sha256(value):
                    continue
                if key == "prompt_sha256":
                    prompt_hashes.add(str(value))
                else:
                    identity_hashes.add(str(value))
    return SealedHashIndex(
        identity_hashes=frozenset(identity_hashes),
        prompt_hashes=frozenset(prompt_hashes),
    )


def _reject_if_sealed(
    source: SourceTask,
    sealed_hashes: SealedHashIndex,
    results_path: Path,
    sim_id: str,
) -> None:
    if _sealed_identity_overlap(source, sealed_hashes):
        raise Tau3ConversationIngestError(f"{results_path}: {sim_id}: source task identity intersects sealed hash manifest")


def _sealed_identity_overlap(source: SourceTask, sealed_hashes: SealedHashIndex) -> bool:
    candidates = {
        source.task_sha256,
        canonical_sha256(source.task_id),
        canonical_sha256({"domain": source.domain, "task_id": source.task_id}),
        hashlib.sha256(f"{source.domain}:{source.task_id}".encode("utf-8")).hexdigest(),
    }
    return bool(candidates & sealed_hashes.identity_hashes)


def _domain_from_results(results: dict[str, Any], path: Path) -> str:
    info = _object(results.get("info"), f"{path}: info")
    env = _object(info.get("environment_info"), f"{path}: info.environment_info")
    domain = _nonempty_str(env.get("domain_name"), f"{path}: info.environment_info.domain_name")
    if domain not in ALLOWED_DOMAINS:
        raise Tau3ConversationIngestError(f"{path}: unsupported domain {domain!r}")
    return domain


def _policy_from_results(results: dict[str, Any], path: Path) -> str:
    info = _object(results.get("info"), f"{path}: info")
    env = _object(info.get("environment_info"), f"{path}: info.environment_info")
    policy = env.get("policy")
    if not isinstance(policy, str) or not policy.strip():
        raise Tau3ConversationIngestError(f"{path}: info.environment_info.policy must be a non-empty string")
    return policy


def _simulations(results: dict[str, Any], path: Path) -> list[dict[str, Any]]:
    simulations = results.get("simulations")
    if simulations is None:
        sims_dir = path.parent / "simulations"
        if sims_dir.is_dir():
            return [_read_json(item) for item in sorted(sims_dir.glob("*.json"))]
    sims = _list(simulations, f"{path}: simulations")
    return [_object(sim, f"{path}: simulations[{index}]") for index, sim in enumerate(sims)]


def _reward(simulation: dict[str, Any], sim_id: str) -> float:
    reward_info = _object(simulation.get("reward_info"), f"{sim_id}.reward_info")
    reward = reward_info.get("reward")
    if isinstance(reward, bool) or not isinstance(reward, (int, float)):
        raise Tau3ConversationIngestError(f"{sim_id}: reward_info.reward must be numeric")
    return float(reward)


def _reject_error_markers(simulation: dict[str, Any], results_path: Path, sim_id: str) -> None:
    info = simulation.get("info")
    if isinstance(info, dict):
        for key, value in info.items():
            if "error" in str(key).lower() and value:
                raise Tau3ConversationIngestError(f"{results_path}: {sim_id}: simulation info contains error marker {key!r}")
    review = simulation.get("review")
    if isinstance(review, dict) and review.get("error"):
        raise Tau3ConversationIngestError(f"{results_path}: {sim_id}: review error marker present")


def _reject_hidden_payloads(messages: list[dict[str, Any]], source: SourceTask) -> None:
    hidden_needles = _source_hidden_needles(source.task)
    serialized = json.dumps(messages, sort_keys=True, ensure_ascii=False)
    lowered = serialized.lower()
    for pattern in FORBIDDEN_TEXT_PATTERNS:
        if pattern in lowered:
            raise Tau3ConversationIngestError(f"hidden/evaluator marker leaked into messages: {pattern}")
    message_strings = _strings(messages)
    for needle in hidden_needles:
        if any(needle in value for value in message_strings):
            raise Tau3ConversationIngestError("source hidden/evaluator text leaked into messages")
    if re.search(r"<think>.*?</think>", serialized, flags=re.IGNORECASE | re.DOTALL):
        raise Tau3ConversationIngestError("thinking tags leaked into messages")


def _source_hidden_needles(task: dict[str, Any]) -> list[str]:
    needles: list[str] = []
    user_scenario = task.get("user_scenario")
    if isinstance(user_scenario, dict):
        instructions = user_scenario.get("instructions")
        if isinstance(instructions, dict):
            for key in ("known_info", "unknown_info", "reason_for_call", "task_instructions"):
                needles.extend(_strings(instructions.get(key)))
        else:
            needles.extend(_strings(instructions))
        needles.extend(_strings(user_scenario.get("persona")))
    evaluation_criteria = task.get("evaluation_criteria")
    if isinstance(evaluation_criteria, dict):
        needles.extend(_strings(evaluation_criteria.get("nl_assertions")))
    return list(dict.fromkeys(needle.strip() for needle in needles if _looks_like_hidden_prose(needle)))


def _looks_like_hidden_prose(value: str) -> bool:
    text = value.strip()
    if len(text) < HIDDEN_PROSE_MIN_CHARS:
        return False
    words = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text)
    return len(words) >= HIDDEN_PROSE_MIN_WORDS


def _strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        list_strings: list[str] = []
        for item in value:
            list_strings.extend(_strings(item))
        return list_strings
    if isinstance(value, dict):
        dict_strings: list[str] = []
        for item in value.values():
            dict_strings.extend(_strings(item))
        return dict_strings
    return []


def _clean_text(value: Any, where: str, *, allow_empty: bool = False) -> str:
    if value is None:
        if allow_empty:
            return ""
        raise Tau3ConversationIngestError(f"{where} must be a string")
    if not isinstance(value, str):
        raise Tau3ConversationIngestError(f"{where} must be a string")
    if re.search(r"</?think>", value, flags=re.IGNORECASE):
        raise Tau3ConversationIngestError(f"{where}: thinking tags are forbidden")
    cleaned = value.strip()
    if not cleaned and not allow_empty:
        raise Tau3ConversationIngestError(f"{where} must be non-empty")
    return cleaned


def _manifest(
    out: Path,
    *,
    rows_by_split: dict[str, list[dict[str, Any]]],
    result_records: list[dict[str, Any]],
    source_dir: Path,
    tool_schema_path: Path,
    sealed_manifest_path: Path | None,
    source_revisions: set[str],
    domains: set[str],
    normalization_count: int,
    tau_control_marker_row_count: int,
    tau_control_marker_count: int,
    tau_control_marker_only_drop_count: int,
    rejection_count: int,
    allow_teacher_protocol_normalization: bool,
    teacher_id: str,
    license_id: str,
    generation_records: list[dict[str, Any]] | None = None,
    generation_protocol: dict[str, Any] | None = None,
    sealed_prompt_template_overlap_count: int = 0,
) -> dict[str, Any]:
    counts = {"train": len(rows_by_split["train"]), "valid": len(rows_by_split["valid"])}
    files = {
        "train": _file_record(out / "train.jsonl"),
        "valid": _file_record(out / "valid.jsonl"),
    }
    row_hashes = {
        split: [row["metadata"]["row_sha256"] for row in rows]
        for split, rows in rows_by_split.items()
    }
    tau_revision = sorted(source_revisions)[0] if len(source_revisions) == 1 else "mixed"
    generation_records = generation_records or []
    generation_protocol_hash = generation_protocol.get("sha256") if generation_protocol else None
    return {
        "schema_version": TAU3_CONVERSATION_CORPUS_SCHEMA_VERSION,
        "tau_revision": tau_revision,
        "format": "mlx-chat-jsonl",
        "passed": True,
        "counts": counts,
        "domains": sorted(domains),
        "source_revisions": sorted(source_revisions),
        "inputs": {
            "results": result_records,
            "generation_manifests": generation_records,
            "generation_protocol": generation_protocol,
            "source_dir": {"path": str(source_dir), "sha256": _dir_listing_hash(source_dir)},
            "tool_schema": {"path": str(tool_schema_path), "sha256": _file_sha256(tool_schema_path)},
            "sealed_manifest": (
                {"path": str(sealed_manifest_path), "sha256": _file_sha256(sealed_manifest_path), "payload_read": False}
                if sealed_manifest_path
                else None
            ),
        },
        "source_results": result_records,
        "source_generation_manifests": generation_records,
        "generation_provenance": {
            "manifest_count": len(generation_records),
            "task_receipt_count": sum(record["task_receipt_count"] for record in generation_records),
            "success_count": sum(record["success_count"] for record in generation_records),
            "failure_count": sum(record["failure_count"] for record in generation_records),
            "admitted_success_count": sum(record["admitted_success_count"] for record in generation_records),
            "excluded_success_count": sum(record["excluded_success_count"] for record in generation_records),
            "protocol_sha256": generation_protocol_hash,
            "protocol": generation_protocol,
        },
        "tool_schema_artifact": {"path": str(tool_schema_path), "sha256": _file_sha256(tool_schema_path)},
        "files": files,
        "row_hashes": row_hashes,
        "sealed_rows": 0,
        "test_rows": 0,
        "hidden_instruction_exposure": False,
        "evaluation_criteria_exposure": False,
        "governance": {
            "sealed_payloads_read": False,
            "sealed_task_identity_overlap_count": 0,
            "sealed_prompt_template_overlap_count": sealed_prompt_template_overlap_count,
            "sealed_prompt_template_overlap_resolved": sealed_prompt_template_overlap_count > 0,
            "sealed_prompt_template_overlap_resolution": (
                "resolved_shared_official_template"
                if sealed_prompt_template_overlap_count > 0
                else "not_observed"
            ),
            "train_dev_only": True,
            "executable_reward_required": 1.0,
            "normal_terminations": sorted(NORMAL_TERMINATIONS),
            "raw_data_stripped": True,
            "hidden_task_fields_stripped": True,
            "user_tool_calls_excluded_from_assistant_targets": True,
            "teacher_id": teacher_id,
            "license": license_id,
        },
        "normalization": {
            "allow_teacher_protocol_normalization": allow_teacher_protocol_normalization,
            "normalized_mixed_content_tool_call_rows": normalization_count,
            "normalized_tau_control_marker_rows": tau_control_marker_row_count,
            "stripped_tau_control_markers": tau_control_marker_count,
            "dropped_tau_control_marker_only_user_messages": tau_control_marker_only_drop_count,
        },
        "rejected_before_abort": rejection_count,
        "training_started": False,
    }


def _file_record(path: Path) -> dict[str, Any]:
    return {"path": path.name, "size_bytes": path.stat().st_size, "sha256": _file_sha256(path)}


def _dir_listing_hash(path: Path) -> str:
    records = []
    for item in sorted(path.glob("*.jsonl")):
        records.append({"name": item.name, "sha256": _file_sha256(item)})
    return canonical_sha256(records)


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


def _require_new_output_dir(out: Path) -> None:
    if out.exists():
        raise Tau3ConversationIngestError(f"output directory already exists: {out}")
    parent = out.parent
    if parent and not parent.exists():
        parent.mkdir(parents=True)


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _short_hash(value: Any) -> str:
    return canonical_sha256(value)[:16]


def _object(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise Tau3ConversationIngestError(f"{where} must be an object")
    return value


def _list(value: Any, where: str) -> list[Any]:
    if not isinstance(value, list):
        raise Tau3ConversationIngestError(f"{where} must be a list")
    return value


def _nonempty_str(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise Tau3ConversationIngestError(f"{where} must be a non-empty string")
    return value.strip()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _redact(text: str) -> str:
    redacted = re.sub(r"\b(?:sk-[A-Za-z0-9_-]{8,}|hf_[A-Za-z0-9]{8,})\b", "[REDACTED]", text)
    return redacted.replace("api_key", "api_key_redacted").replace("API_KEY", "API_KEY_REDACTED")
