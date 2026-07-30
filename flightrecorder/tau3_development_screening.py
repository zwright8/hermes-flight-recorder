"""Deterministic, non-qualifying Tau-3 development screening plans."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .schema_registry import check_schema_contract

TAU3_DEVELOPMENT_SCREENING_SCHEMA_VERSION = (
    "hfr.tau3_development_screening.v1"
)
DOMAINS = ("airline", "retail", "telecom")
SEEDS = (101, 202, 303, 404)
ALGORITHM = "domain-minimum-content-hash-v1"


class Tau3DevelopmentScreeningError(ValueError):
    """Raised when a development screening plan is not replayable."""


def build_tau3_development_screening(
    *,
    development_source: str | Path,
    out_path: str | Path,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Write a create-once, one-task-per-domain screening plan."""

    source_path = Path(development_source)
    output = Path(out_path)
    if output.exists():
        raise Tau3DevelopmentScreeningError(
            f"screening plan already exists: {output}"
        )
    source = _load_json_object(source_path)
    payload = _build_payload(
        source=source,
        source_sha256=_sha256_file(source_path),
        source_size=source_path.stat().st_size,
        created_at=created_at or _now_utc(),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return payload


def validate_tau3_development_screening(
    *,
    screening: str | Path,
    development_source: str | Path,
) -> dict[str, Any]:
    """Replay a screening plan against its full development source."""

    screening_path = Path(screening)
    source_path = Path(development_source)
    payload = _load_json_object(screening_path)
    source = _load_json_object(source_path)
    expected = _build_payload(
        source=source,
        source_sha256=_sha256_file(source_path),
        source_size=source_path.stat().st_size,
        created_at=str(payload.get("created_at") or ""),
    )
    errors: list[str] = []
    schema = check_schema_contract(
        payload,
        name_or_id="tau3_development_screening",
    )
    errors.extend(str(error) for error in schema.get("errors") or [])
    if payload != expected:
        errors.append(
            "screening plan does not replay from the bound development source"
        )
    return {
        "schema_version": "hfr.validation.v1",
        "artifact_schema_version": (
            TAU3_DEVELOPMENT_SCREENING_SCHEMA_VERSION
        ),
        "screening_sha256": _sha256_file(screening_path),
        "development_source_sha256": _sha256_file(source_path),
        "task_count": len(payload.get("selected_tasks") or []),
        "expected_run_count": payload.get("expected_run_count"),
        "candidate_eligible": payload.get("candidate_eligible"),
        "passed": not errors,
        "error_count": len(errors),
        "errors": errors,
    }


def selected_tasks_by_domain(
    *,
    screening: str | Path,
    development_source: str | Path,
) -> dict[str, list[str]]:
    """Return selected raw task IDs only after full semantic replay."""

    validation = validate_tau3_development_screening(
        screening=screening,
        development_source=development_source,
    )
    if validation["passed"] is not True:
        raise Tau3DevelopmentScreeningError(
            "; ".join(str(error) for error in validation["errors"])
        )
    payload = _load_json_object(Path(screening))
    selected: dict[str, list[str]] = {domain: [] for domain in DOMAINS}
    for task in payload["selected_tasks"]:
        selected[str(task["domain"])].append(str(task["raw_id"]))
    return selected


def _build_payload(
    *,
    source: dict[str, Any],
    source_sha256: str,
    source_size: int,
    created_at: str,
) -> dict[str, Any]:
    source_schema = check_schema_contract(
        source,
        name_or_id="tau3_source_split",
    )
    if source_schema.get("passed") is not True:
        raise Tau3DevelopmentScreeningError(
            "development source schema failed: "
            + "; ".join(
                str(error) for error in source_schema.get("errors") or []
            )
        )
    if source.get("schema_version") != "hfr.tau3_source_split.v1":
        raise Tau3DevelopmentScreeningError(
            "development source schema_version mismatch"
        )
    if source.get("split") != "development":
        raise Tau3DevelopmentScreeningError(
            "screening source must be the development split"
        )
    revision = source.get("source_revision")
    if (
        not isinstance(revision, str)
        or len(revision) != 40
        or any(character not in "0123456789abcdef" for character in revision)
    ):
        raise Tau3DevelopmentScreeningError(
            "development source revision is not immutable"
        )
    tasks = source.get("tasks")
    if (
        not isinstance(tasks, list)
        or source.get("task_count") != len(tasks)
        or not tasks
    ):
        raise Tau3DevelopmentScreeningError(
            "development source task_count mismatch"
        )
    candidates: dict[str, list[dict[str, Any]]] = {
        domain: [] for domain in DOMAINS
    }
    for raw_task in tasks:
        if not isinstance(raw_task, dict):
            raise Tau3DevelopmentScreeningError(
                "development source task is not an object"
            )
        task = dict(raw_task)
        domain = task.get("domain")
        if domain not in candidates:
            raise Tau3DevelopmentScreeningError(
                f"development source contains unsupported domain {domain!r}"
            )
        for key in (
            "raw_id",
            "raw_id_sha256",
            "task_sha256",
            "prompt_sha256",
            "family_id",
        ):
            if not isinstance(task.get(key), str) or not task[key]:
                raise Tau3DevelopmentScreeningError(
                    f"development source task lacks {key}"
                )
        if (
            _sha256_text(f"{domain}:{task['raw_id']}")
            != task["raw_id_sha256"]
        ):
            raise Tau3DevelopmentScreeningError(
                "development source raw_id hash mismatch"
            )
        selection_key = _sha256_text(
            "\0".join(
                (
                    TAU3_DEVELOPMENT_SCREENING_SCHEMA_VERSION,
                    ALGORITHM,
                    str(domain),
                    task["task_sha256"],
                )
            )
        )
        candidates[str(domain)].append(
            {
                "domain": domain,
                "raw_id": task["raw_id"],
                "raw_id_sha256": task["raw_id_sha256"],
                "task_sha256": task["task_sha256"],
                "prompt_sha256": task["prompt_sha256"],
                "family_id": task["family_id"],
                "selection_key_sha256": selection_key,
            }
        )
    selected: list[dict[str, Any]] = []
    source_counts: dict[str, int] = {}
    for domain in DOMAINS:
        domain_tasks = candidates[domain]
        if not domain_tasks:
            raise Tau3DevelopmentScreeningError(
                f"development source lacks {domain}"
            )
        source_counts[domain] = len(domain_tasks)
        selected.append(
            min(
                domain_tasks,
                key=lambda task: (
                    task["selection_key_sha256"],
                    task["task_sha256"],
                    task["raw_id_sha256"],
                ),
            )
        )
    selected_task_set_sha256 = _canonical_sha256(selected)
    payload = {
        "schema_version": TAU3_DEVELOPMENT_SCREENING_SCHEMA_VERSION,
        "created_at": created_at,
        "algorithm": ALGORITHM,
        "source_revision": revision,
        "development_source": {
            "sha256": source_sha256,
            "size": source_size,
            "task_count": len(tasks),
            "domain_counts": source_counts,
        },
        "domains": list(DOMAINS),
        "seeds": list(SEEDS),
        "tasks_per_domain": 1,
        "task_count": len(selected),
        "expected_run_count": len(DOMAINS) * len(SEEDS),
        "selected_tasks": selected,
        "selected_task_set_sha256": selected_task_set_sha256,
        "candidate_eligible": False,
        "selection_locked": True,
        "qualification_requires_full_development": True,
        "sealed_payload_accessed": False,
        "sealed_task_ids_materialized": False,
    }
    schema = check_schema_contract(
        payload,
        name_or_id="tau3_development_screening",
    )
    if schema.get("passed") is not True:
        raise Tau3DevelopmentScreeningError(
            "screening plan schema failed: "
            + "; ".join(str(error) for error in schema.get("errors") or [])
        )
    return payload


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Tau3DevelopmentScreeningError(
            f"cannot read JSON object {path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise Tau3DevelopmentScreeningError(
            f"JSON artifact must be an object: {path}"
        )
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
