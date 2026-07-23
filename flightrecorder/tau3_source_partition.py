"""Deterministic fail-closed Tau training-source partitioning."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .path_safety import path_has_symlink_component

DOMAINS = ("airline", "retail", "telecom")
SOURCE_SCHEMA_VERSION = "hfr.tau3_source_partition.v1"
SPLIT_SCHEMA_VERSION = "hfr.tau3_source_split.v1"
TRAINING_SOURCE_SCHEMA_VERSION = "hfr.tau3_training_source.v1"
SEALED_SCHEMA_VERSION = "hfr.tau3_sealed_source_manifest.v1"
ALGORITHM_ID = "hfr.tau3_source_partition.domain_stratified_family_split.v1"
HEX40_RE = re.compile(r"^[0-9a-f]{40}$")
TELECOM_FAMILY_RE = re.compile(r"^\[([^\]]+)\]")


class Tau3SourcePartitionError(ValueError):
    """Raised when Tau source partitioning cannot be done safely."""


def prepare_tau3_training_sources(
    tau_repo: str | Path,
    expected_revision: str,
    out_dir: str | Path,
    *,
    development_fraction: float = 0.2,
    salt: str = "hfr-tau3-core-v1",
) -> dict[str, Any]:
    """Prepare train/development source manifests from a pinned Tau checkout."""

    repo = Path(tau_repo)
    out = Path(out_dir)
    _validate_options(expected_revision, development_fraction, salt)
    _require_clean_revision(repo, expected_revision)
    _require_new_safe_output(out)

    tasks_by_domain, splits_by_domain = _load_tau_sources(repo)
    official_train, sealed = _validate_sources(tasks_by_domain, splits_by_domain)
    assignments = _partition_official_train(official_train, development_fraction, salt)

    tmp = _make_staging_dir(out)
    try:
        for child in ("training_source",):
            (tmp / child).mkdir(mode=0o700)
        train_rows = [record for record in official_train if assignments[record["global_id"]] == "train"]
        dev_rows = [record for record in official_train if assignments[record["global_id"]] == "development"]

        _write_json(tmp / "train.json", _split_manifest("train", train_rows, expected_revision, salt))
        _write_json(tmp / "development.json", _split_manifest("development", dev_rows, expected_revision, salt))
        _write_jsonl(
            tmp / "training_source" / "train_tasks.jsonl",
            [_training_source_row(record, "train", expected_revision) for record in train_rows],
        )
        _write_jsonl(
            tmp / "training_source" / "development_tasks.jsonl",
            [_training_source_row(record, "development", expected_revision) for record in dev_rows],
        )
        _write_json(tmp / "sealed.json", _sealed_manifest(sealed, expected_revision))
        root_manifest = _root_manifest(tmp, expected_revision, salt, train_rows, dev_rows, sealed)
        _write_json(tmp / "manifest.json", root_manifest)
        os.replace(tmp, out)
        _fsync_directory(out.parent)
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise

    return {
        "schema_version": SOURCE_SCHEMA_VERSION,
        "out": str(out),
        "source_revision": expected_revision,
        "official_train_task_count": len(official_train),
        "train_task_count": len(train_rows),
        "development_task_count": len(dev_rows),
        "sealed_task_count": len(sealed),
        "train_family_count": len({record["family_id"] for record in train_rows}),
        "development_family_count": len({record["family_id"] for record in dev_rows}),
        "sealed_payload_written": False,
    }


def _validate_options(expected_revision: str, development_fraction: float, salt: str) -> None:
    if not HEX40_RE.fullmatch(expected_revision):
        raise Tau3SourcePartitionError("expected revision must be an exact lowercase 40-hex git object id")
    if not 0.0 < development_fraction < 1.0:
        raise Tau3SourcePartitionError("development fraction must be greater than 0 and less than 1")
    if not salt:
        raise Tau3SourcePartitionError("salt must be non-empty")


def _require_clean_revision(repo: Path, expected_revision: str) -> None:
    if not repo.is_dir():
        raise Tau3SourcePartitionError(f"Tau repository is not a directory: {repo}")
    if path_has_symlink_component(repo, include_leaf=True):
        raise Tau3SourcePartitionError("Tau repository path must not contain symlink components")
    actual = _git(repo, "rev-parse", "HEAD")
    if actual != expected_revision:
        raise Tau3SourcePartitionError(f"Tau checkout revision mismatch: expected {expected_revision}, got {actual}")
    status = _git(repo, "status", "--porcelain=v1")
    if status:
        raise Tau3SourcePartitionError("Tau checkout must be clean")


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "git command failed"
        raise Tau3SourcePartitionError(detail)
    return completed.stdout.strip()


def _require_new_safe_output(out: Path) -> None:
    if out.exists() or out.is_symlink():
        raise Tau3SourcePartitionError("output directory must not already exist")
    parent = out.parent
    if not parent.is_dir():
        raise Tau3SourcePartitionError("output parent directory must already exist")
    if path_has_symlink_component(parent, include_leaf=True):
        raise Tau3SourcePartitionError("output path must not contain symlink components")


def _make_staging_dir(out: Path) -> Path:
    base = out.parent
    for index in range(100):
        candidate = base / f".{out.name}.tmp.{os.getpid()}.{index}"
        try:
            candidate.mkdir(mode=0o700)
        except FileExistsError:
            continue
        return candidate
    raise Tau3SourcePartitionError("could not allocate staging directory")


def _load_tau_sources(repo: Path) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]]]:
    domains_root = repo / "data" / "tau2" / "domains"
    if not domains_root.is_dir():
        raise Tau3SourcePartitionError("Tau checkout is missing data/tau2/domains")
    actual_domains = {path.name for path in domains_root.iterdir() if path.is_dir()}
    missing_domains = sorted(set(DOMAINS) - actual_domains)
    if missing_domains:
        raise Tau3SourcePartitionError(
            "Tau checkout is missing required study domain(s): " + ", ".join(missing_domains)
        )

    tasks: dict[str, list[dict[str, Any]]] = {}
    splits: dict[str, dict[str, Any]] = {}
    for domain in DOMAINS:
        domain_root = domains_root / domain
        tasks_payload = _read_json(domain_root / "tasks.json")
        split_payload = _read_json(domain_root / "split_tasks.json")
        if not isinstance(tasks_payload, list) or not all(isinstance(item, dict) for item in tasks_payload):
            raise Tau3SourcePartitionError(f"{domain}/tasks.json must contain a list of objects")
        if not isinstance(split_payload, dict):
            raise Tau3SourcePartitionError(f"{domain}/split_tasks.json must contain an object")
        tasks[domain] = tasks_payload
        splits[domain] = split_payload
    return tasks, splits


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise Tau3SourcePartitionError(f"missing required source file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise Tau3SourcePartitionError(f"invalid JSON in {path}: {exc}") from exc


def _validate_sources(
    tasks_by_domain: dict[str, list[dict[str, Any]]],
    splits_by_domain: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train_records: list[dict[str, Any]] = []
    sealed_records: list[dict[str, Any]] = []
    for domain in DOMAINS:
        tasks = tasks_by_domain[domain]
        splits = splits_by_domain[domain]
        for required in ("train", "test", "base"):
            if not isinstance(splits.get(required), list) or not all(isinstance(item, str) for item in splits[required]):
                raise Tau3SourcePartitionError(f"{domain}/split_tasks.json must define string list {required}")
        train_ids = list(splits["train"])
        test_ids = list(splits["test"])
        base_ids = list(splits["base"])
        if len(set(train_ids)) != len(train_ids) or len(set(test_ids)) != len(test_ids) or len(set(base_ids)) != len(base_ids):
            raise Tau3SourcePartitionError(f"{domain} split IDs must be unique")
        if set(train_ids) & set(test_ids):
            raise Tau3SourcePartitionError(f"{domain} official train/test splits overlap")
        if set(train_ids) | set(test_ids) != set(base_ids):
            raise Tau3SourcePartitionError(f"{domain} official train/test union must equal base")
        by_id: dict[str, dict[str, Any]] = {}
        for task in tasks:
            task_id = task.get("id")
            if not isinstance(task_id, str):
                raise Tau3SourcePartitionError(f"{domain} task has missing or non-string id")
            if task_id in by_id:
                raise Tau3SourcePartitionError(f"{domain} task id resolves more than once: {task_id}")
            by_id[task_id] = task
        for split_name, ids in (("train", train_ids), ("test", test_ids), ("base", base_ids)):
            unresolved = [task_id for task_id in ids if task_id not in by_id]
            if unresolved:
                raise Tau3SourcePartitionError(f"{domain} {split_name} split contains unresolved task id: {unresolved[0]}")
        for task_id in train_ids:
            train_records.append(_task_record(domain, by_id[task_id], "official_train"))
        for task_id in test_ids:
            sealed_records.append(_task_record(domain, by_id[task_id], "official_test"))
    return train_records, sealed_records


def _task_record(domain: str, task: dict[str, Any], official_split: str) -> dict[str, Any]:
    raw_id = str(task["id"])
    family_material = _family_material(domain, task)
    family_id = _hash_json(family_material)
    return {
        "domain": domain,
        "raw_id": raw_id,
        "global_id": f"{domain}:{raw_id}",
        "official_split": official_split,
        "task": task,
        "task_sha256": _hash_json(task),
        "prompt_sha256": _hash_json(_prompt_material(task)),
        "raw_id_sha256": _sha256_text(f"{domain}:{raw_id}"),
        "family_id": family_id,
        "family_sha256": family_id,
        "family_descriptor": family_material,
    }


def _family_material(domain: str, task: dict[str, Any]) -> dict[str, Any]:
    if domain == "telecom":
        match = TELECOM_FAMILY_RE.match(str(task.get("id", "")))
        if not match:
            raise Tau3SourcePartitionError("telecom task id must start with a bracketed issue family")
        return {"domain": domain, "kind": "telecom_issue", "issue_family": match.group(1)}

    criteria = task.get("evaluation_criteria")
    if not isinstance(criteria, dict):
        raise Tau3SourcePartitionError(f"{domain} task evaluation_criteria must be an object")
    actions = criteria.get("actions")
    reward_basis = criteria.get("reward_basis")
    if not isinstance(actions, list) or not isinstance(reward_basis, list):
        raise Tau3SourcePartitionError(f"{domain} evaluation_criteria must include actions and reward_basis lists")
    action_names: list[str] = []
    for action in actions:
        if not isinstance(action, dict) or not isinstance(action.get("name"), str):
            raise Tau3SourcePartitionError(f"{domain} evaluation action must include a string name")
        action_names.append(action["name"])
    basis = [str(item) for item in reward_basis]
    if action_names:
        return {
            "domain": domain,
            "kind": "action_sequence",
            "action_names": action_names,
            "reward_basis": basis,
        }
    communicate_info = criteria.get("communicate_info") if isinstance(criteria.get("communicate_info"), list) else []
    nl_assertions = criteria.get("nl_assertions") if isinstance(criteria.get("nl_assertions"), list) else []
    return {
        "domain": domain,
        "kind": "no_action_refusal",
        "action_names": [],
        "reward_basis": basis,
        "communicate_info_count": len(communicate_info),
        "nl_assertion_count": len(nl_assertions),
    }


def _prompt_material(task: dict[str, Any]) -> Any:
    scenario = task.get("user_scenario")
    if isinstance(scenario, dict):
        return scenario.get("instructions", scenario)
    return scenario


def _partition_official_train(
    train_records: list[dict[str, Any]],
    development_fraction: float,
    salt: str,
) -> dict[str, str]:
    families_by_domain: dict[str, dict[str, list[dict[str, Any]]]] = {
        domain: {} for domain in DOMAINS
    }
    for record in train_records:
        families_by_domain[record["domain"]].setdefault(record["family_id"], []).append(record)
    dev_families: set[str] = set()
    for domain in DOMAINS:
        families = families_by_domain[domain]
        if len(families) < 2:
            raise Tau3SourcePartitionError(
                f"official {domain} train split must contain at least two task families"
            )
        domain_task_count = sum(len(records) for records in families.values())
        target_dev = max(1, round(domain_task_count * development_fraction))
        ranked = sorted(
            families.items(),
            key=lambda item: (_sha256_text(f"{salt}\0{domain}\0{item[0]}"), item[0]),
        )
        domain_dev_count = 0
        domain_dev_families: set[str] = set()
        for family_id, records in ranked:
            if len(domain_dev_families) == len(ranked) - 1:
                break
            domain_dev_families.add(family_id)
            domain_dev_count += len(records)
            if domain_dev_count >= target_dev:
                break
        if not domain_dev_families or domain_dev_families == set(families):
            raise Tau3SourcePartitionError(
                f"family split would produce an empty {domain} train or development split"
            )
        dev_families.update(domain_dev_families)

    assignments: dict[str, str] = {}
    for record in train_records:
        assignments[record["global_id"]] = "development" if record["family_id"] in dev_families else "train"
    train_count = sum(1 for split in assignments.values() if split == "train")
    final_dev_count = sum(1 for split in assignments.values() if split == "development")
    if train_count == 0 or final_dev_count == 0:
        raise Tau3SourcePartitionError("family split produced an empty train or development split")
    return assignments


def _split_manifest(
    split: str,
    records: list[dict[str, Any]],
    expected_revision: str,
    salt: str,
) -> dict[str, Any]:
    families = sorted({record["family_id"] for record in records})
    return {
        "schema_version": SPLIT_SCHEMA_VERSION,
        "split": split,
        "source_revision": expected_revision,
        "task_schema_version": "tau2.tasks.v1",
        "algorithm": ALGORITHM_ID,
        "salt_sha256": _sha256_text(salt),
        "task_count": len(records),
        "family_count": len(families),
        "family_ids": families,
        "tasks": [
            {
                "domain": record["domain"],
                "raw_id": record["raw_id"],
                "raw_id_sha256": record["raw_id_sha256"],
                "prompt_sha256": record["prompt_sha256"],
                "task_sha256": record["task_sha256"],
                "family_id": record["family_id"],
            }
            for record in sorted(records, key=lambda item: (item["domain"], item["raw_id"]))
        ],
    }


def _sealed_manifest(records: list[dict[str, Any]], expected_revision: str) -> dict[str, Any]:
    entries = [
        {
            "task_id_sha256": record["raw_id_sha256"],
            "prompt_sha256": record["prompt_sha256"],
            "task_sha256": record["task_sha256"],
        }
        for record in sorted(records, key=lambda item: (item["domain"], item["raw_id"]))
    ]
    return {
        "schema_version": SEALED_SCHEMA_VERSION,
        "source_revision": expected_revision,
        "hashes_only": True,
        "task_count": len(entries),
        "entries": entries,
    }


def _training_source_row(
    record: dict[str, Any],
    split: str,
    expected_revision: str,
) -> dict[str, Any]:
    """Bind each permitted task payload to its immutable source lineage."""

    return {
        "schema_version": TRAINING_SOURCE_SCHEMA_VERSION,
        "source_revision": expected_revision,
        "domain": record["domain"],
        "split": split,
        "task_family": record["family_id"],
        "task_sha256": record["task_sha256"],
        "prompt_sha256": record["prompt_sha256"],
        "task": record["task"],
    }


def _root_manifest(
    tmp: Path,
    expected_revision: str,
    salt: str,
    train_rows: list[dict[str, Any]],
    dev_rows: list[dict[str, Any]],
    sealed: list[dict[str, Any]],
) -> dict[str, Any]:
    train_families = {record["family_id"] for record in train_rows}
    dev_families = {record["family_id"] for record in dev_rows}
    artifacts = {
        rel: _artifact_record(tmp / rel)
        for rel in (
            "train.json",
            "development.json",
            "training_source/train_tasks.jsonl",
            "training_source/development_tasks.jsonl",
            "sealed.json",
        )
    }
    return {
        "schema_version": SOURCE_SCHEMA_VERSION,
        "source_revision": expected_revision,
        "task_schema_version": "tau2.tasks.v1",
        "algorithm": ALGORITHM_ID,
        "algorithm_sha256": _sha256_text(ALGORITHM_ID),
        "salt_sha256": _sha256_text(salt),
        "development_fraction_algorithm": "domain_stratified_salted_family_order_until_rounded_task_target",
        "counts": {
            "train_tasks": len(train_rows),
            "development_tasks": len(dev_rows),
            "sealed_tasks": len(sealed),
            "train_families": len(train_families),
            "development_families": len(dev_families),
        },
        "proofs": {
            "train_development_family_disjoint": not (train_families & dev_families),
            "sealed_payload_non_materialization": True,
            "sealed_payload_files": [],
            "official_test_sealed": True,
        },
        "artifacts": artifacts,
    }


def _artifact_record(path: Path) -> dict[str, Any]:
    return {"size": path.stat().st_size, "sha256": _sha256_file(path)}


def _write_json(path: Path, value: Any) -> None:
    rendered = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    _write_text_new(path, rendered)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    rendered = "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows)
    _write_text_new(path, rendered)


def _write_text_new(path: Path, rendered: str) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise


def _hash_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":  # pragma: no cover
        return
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
