#!/usr/bin/env python3
"""Opt-in live smoke for production external-state verifier adapters.

The smoke is intentionally read-only. It builds one verifier config per
provider from environment variables, captures a state snapshot, validates the
snapshot, and writes a machine-readable summary. Network calls only happen when
``--allow-network`` is supplied.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flightrecorder.path_safety import (  # noqa: E402 - repo bootstrap precedes local import
    assert_safe_output_directory,
    json_marker_has_schema_version,
    replace_owned_output_directory,
)
from flightrecorder.redaction import redact_text  # noqa: E402 - repo bootstrap precedes local import
from flightrecorder.schema_registry import check_schema_contract  # noqa: E402 - repo bootstrap precedes local import
from flightrecorder.validation import validate_artifacts  # noqa: E402 - repo bootstrap precedes local import
from flightrecorder.verifiers import (  # noqa: E402 - repo bootstrap precedes local import
    VERIFIER_CONFIG_SCHEMA_VERSION,
    capture_verified_state,
)

LIVE_VERIFIER_SMOKE_SUMMARY_SCHEMA_VERSION = "hfr.live_verifier_smoke.summary.v1"
DEFAULT_OUT = ROOT / "runs" / "live_verifier_smoke"
DEFAULT_SECRET_PATTERNS = [
    r"(?i)(api[_-]?key|secret|token|password|authorization|bearer)",
    r"sk_(live|test)_[A-Za-z0-9_]+",
    r"xox[baprs]-[A-Za-z0-9-]+",
]


@dataclass(frozen=True)
class ProviderSpec:
    id: str
    source_type: str
    description: str
    required_env: tuple[str, ...]
    optional_env: tuple[str, ...]
    sensitive_env: tuple[str, ...]
    build_source: Callable[[Mapping[str, str]], dict[str, Any]]
    missing_env: Callable[[Mapping[str, str]], list[str]] | None = None


def main(argv: list[str] | None = None) -> int:
    specs = _provider_specs_by_id()
    parser = argparse.ArgumentParser(description="Run read-only live verifier smoke checks for configured SaaS providers.")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Directory for smoke artifacts")
    parser.add_argument("--provider", action="append", choices=sorted(specs), help="Provider id to run; may be repeated")
    parser.add_argument(
        "--configured-only",
        action="store_true",
        help="With no --provider filters, select only providers whose required environment is present",
    )
    parser.add_argument("--allow-network", action="store_true", help="Permit live HTTP/IMAP reads")
    parser.add_argument("--strict-live", action="store_true", help="Fail if any selected provider is skipped or fails")
    parser.add_argument("--require-live-provider", action="store_true", help="Fail unless at least one provider passes")
    parser.add_argument("--timeout-seconds", type=float, default=15.0, help="Default network timeout for generated sources")
    parser.add_argument("--secret-pattern", action="append", default=[], help="Additional regex to redact from outputs")
    parser.add_argument("--preserve-paths", action="store_true", help="Preserve local paths in generated state snapshots")
    parser.add_argument("--keep-existing", action="store_true", help="Do not delete an existing output directory first")
    parser.add_argument("--force", action="store_true", help="Replace a prior valid verifier smoke output")
    parser.add_argument("--list-providers", action="store_true", help="Print provider ids and required env vars, then exit")
    args = parser.parse_args(argv)

    if args.list_providers:
        _print_provider_catalog(specs.values())
        return 0

    env = dict(os.environ)
    selected = _select_specs(specs, args.provider, configured_only=args.configured_only, env=env)
    out_dir = Path(args.out)
    _prepare_smoke_output(out_dir, force=bool(args.force), keep_existing=bool(args.keep_existing))
    out_dir.mkdir(parents=True, exist_ok=True)

    runtime_secret_patterns = _runtime_secret_patterns(selected, env, args.secret_pattern)
    records = [
        _run_provider(
            spec,
            env=env,
            out_dir=out_dir,
            allow_network=bool(args.allow_network),
            timeout_seconds=float(args.timeout_seconds),
            preserve_paths=bool(args.preserve_paths),
            runtime_secret_patterns=runtime_secret_patterns,
        )
        for spec in selected
    ]
    summary = _summary(
        records,
        selected,
        out_dir,
        allow_network=bool(args.allow_network),
        configured_only=bool(args.configured_only),
        strict_live=bool(args.strict_live),
        require_live_provider=bool(args.require_live_provider),
    )
    summary_path = out_dir / "live_verifier_smoke_summary.json"
    _write_json(summary_path, summary)

    schema_check = check_schema_contract(summary)
    _write_json(out_dir / "live_verifier_smoke_summary.schema_check.json", schema_check)

    print(f"wrote {summary_path}")
    print(
        "providers: "
        f"passed={summary['passed_provider_count']} "
        f"failed={summary['failed_provider_count']} "
        f"skipped={summary['skipped_provider_count']}"
    )
    return 0 if summary["passed"] and schema_check["passed"] else 1


def _prepare_smoke_output(out_dir: Path, *, force: bool, keep_existing: bool) -> None:
    if force and keep_existing:
        raise SystemExit("--force and --keep-existing are mutually exclusive")
    def owned(path: Path) -> bool:
        return json_marker_has_schema_version(
            path,
            "live_verifier_smoke_summary.json",
            LIVE_VERIFIER_SMOKE_SUMMARY_SCHEMA_VERSION,
        )
    try:
        if keep_existing:
            assert_safe_output_directory(out_dir, repo_root=ROOT)
            if out_dir.exists() and any(out_dir.iterdir()) and not owned(out_dir):
                raise ValueError(f"refusing to reuse unrecognized verifier smoke output: {out_dir}")
            return
        replace_owned_output_directory(
            out_dir,
            repo_root=ROOT,
            force=force,
            label="verifier smoke output",
            is_owned=owned,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


def _run_provider(
    spec: ProviderSpec,
    *,
    env: Mapping[str, str],
    out_dir: Path,
    allow_network: bool,
    timeout_seconds: float,
    preserve_paths: bool,
    runtime_secret_patterns: list[str],
) -> dict[str, Any]:
    provider_dir = out_dir / spec.id
    provider_dir.mkdir(parents=True, exist_ok=True)
    record = _base_record(spec, env)

    if not allow_network:
        record.update({"status": "skipped", "reason": "network_disabled"})
        return record

    missing = _missing_required_env(spec, env)
    if missing:
        record.update({"status": "skipped", "reason": "missing_configuration", "missing_env": missing})
        return record

    config_path = provider_dir / "verifier_config.json"
    state_path = provider_dir / "state_snapshot.json"
    validation_path = provider_dir / "validation.json"
    try:
        source = spec.build_source(env)
        source.setdefault("timeout_seconds", timeout_seconds)
        config = {
            "schema_version": VERIFIER_CONFIG_SCHEMA_VERSION,
            "sources": [source],
        }
        _write_json(config_path, _sanitize_json(config, runtime_secret_patterns))
        capture_config = {
            "schema_version": VERIFIER_CONFIG_SCHEMA_VERSION,
            "secret_patterns": DEFAULT_SECRET_PATTERNS,
            "sources": [source],
        }
        snapshot = capture_verified_state(
            capture_config,
            preserve_paths=preserve_paths,
            secret_patterns=runtime_secret_patterns,
        )
        _write_json(state_path, snapshot)
        validation = validate_artifacts(state_snapshot_paths=[state_path], strict=True)
        _write_json(validation_path, validation)

        source_status = _source_status(snapshot, spec.id)
        passed = bool(validation["passed"]) and source_status == "ok"
        record.update(
            {
                "status": "passed" if passed else "failed",
                "reason": "ok" if passed else "validation_failed",
                "source_status": source_status,
                "source_count": snapshot.get("verifiers", {}).get("source_count", 0),
                "validation_passed": bool(validation["passed"]),
                "artifacts": {
                    "verifier_config": str(config_path),
                    "state_snapshot": str(state_path),
                    "validation": str(validation_path),
                },
            }
        )
    except Exception as exc:  # noqa: BLE001 - smoke summaries need safe failure records.
        record.update(
            {
                "status": "failed",
                "reason": "capture_failed",
                "error": redact_text(str(exc), runtime_secret_patterns),
                "artifacts": {"verifier_config": str(config_path)} if config_path.exists() else {},
            }
        )
    return record


def _summary(
    records: list[dict[str, Any]],
    selected: list[ProviderSpec],
    out_dir: Path,
    *,
    allow_network: bool,
    configured_only: bool,
    strict_live: bool,
    require_live_provider: bool,
) -> dict[str, Any]:
    passed_count = sum(1 for record in records if record["status"] == "passed")
    failed_count = sum(1 for record in records if record["status"] == "failed")
    skipped_count = sum(1 for record in records if record["status"] == "skipped")
    live_attempted_count = passed_count + failed_count
    passed = failed_count == 0
    if strict_live and skipped_count:
        passed = False
    if require_live_provider and passed_count == 0:
        passed = False
    return {
        "schema_version": LIVE_VERIFIER_SMOKE_SUMMARY_SCHEMA_VERSION,
        "passed": passed,
        "allow_network": allow_network,
        "configured_only": configured_only,
        "strict_live": strict_live,
        "require_live_provider": require_live_provider,
        "selected_provider_count": len(selected),
        "live_attempted_provider_count": live_attempted_count,
        "passed_provider_count": passed_count,
        "failed_provider_count": failed_count,
        "skipped_provider_count": skipped_count,
        "providers": records,
        "artifacts": {
            "summary": str(out_dir / "live_verifier_smoke_summary.json"),
            "schema_check": str(out_dir / "live_verifier_smoke_summary.schema_check.json"),
        },
        "environment": _environment(ROOT),
    }


def _base_record(spec: ProviderSpec, env: Mapping[str, str]) -> dict[str, Any]:
    return {
        "provider": spec.id,
        "source_type": spec.source_type,
        "description": spec.description,
        "required_env": _env_status(spec.required_env, env),
        "optional_env": _env_status(spec.optional_env, env),
        "status": "pending",
    }


def _select_specs(
    specs: dict[str, ProviderSpec],
    requested: list[str] | None,
    *,
    configured_only: bool,
    env: Mapping[str, str],
) -> list[ProviderSpec]:
    if requested:
        seen: set[str] = set()
        selected = []
        for provider_id in requested:
            if provider_id in seen:
                continue
            seen.add(provider_id)
            selected.append(specs[provider_id])
        return selected
    selected = [specs[key] for key in sorted(specs)]
    if configured_only:
        selected = [spec for spec in selected if not _missing_required_env(spec, env)]
    return selected


def _provider_specs_by_id() -> dict[str, ProviderSpec]:
    return {spec.id: spec for spec in _provider_specs()}


def _provider_specs() -> list[ProviderSpec]:
    return [
        ProviderSpec(
            id="discord",
            source_type="discord_messages",
            description="Read recent Discord channel messages.",
            required_env=("DISCORD_BOT_TOKEN", "HFR_DISCORD_CHANNEL_ID"),
            optional_env=("HFR_DISCORD_BASE_URL", "HFR_DISCORD_LIMIT"),
            sensitive_env=("DISCORD_BOT_TOKEN",),
            build_source=_discord_source,
        ),
        ProviderSpec(
            id="github",
            source_type="github_issue",
            description="Read one GitHub issue and its comments.",
            required_env=("HFR_GITHUB_OWNER", "HFR_GITHUB_REPO", "HFR_GITHUB_ISSUE_NUMBER"),
            optional_env=("GITHUB_TOKEN", "HFR_GITHUB_BASE_URL"),
            sensitive_env=("GITHUB_TOKEN",),
            build_source=_github_source,
        ),
        ProviderSpec(
            id="gitlab",
            source_type="gitlab_issues",
            description="Read GitLab issues for one project.",
            required_env=("GITLAB_TOKEN", "HFR_GITLAB_PROJECT_ID"),
            optional_env=("HFR_GITLAB_BASE_URL", "HFR_GITLAB_STATE", "HFR_GITLAB_LABELS", "HFR_GITLAB_SEARCH"),
            sensitive_env=("GITLAB_TOKEN",),
            build_source=_gitlab_source,
        ),
        ProviderSpec(
            id="gmail",
            source_type="gmail_threads",
            description="Read Gmail thread metadata/messages.",
            required_env=("GMAIL_ACCESS_TOKEN",),
            optional_env=("HFR_GMAIL_BASE_URL", "HFR_GMAIL_QUERY", "HFR_GMAIL_THREAD_ID", "HFR_GMAIL_MAX_THREADS"),
            sensitive_env=("GMAIL_ACCESS_TOKEN",),
            build_source=_gmail_source,
        ),
        ProviderSpec(
            id="google_calendar",
            source_type="google_calendar_events",
            description="Read Google Calendar events.",
            required_env=("GOOGLE_CALENDAR_ACCESS_TOKEN",),
            optional_env=("HFR_GOOGLE_CALENDAR_BASE_URL", "HFR_GOOGLE_CALENDAR_ID", "HFR_GOOGLE_CALENDAR_QUERY"),
            sensitive_env=("GOOGLE_CALENDAR_ACCESS_TOKEN",),
            build_source=_google_calendar_source,
        ),
        ProviderSpec(
            id="google_drive",
            source_type="google_drive_files",
            description="Read Google Drive file metadata.",
            required_env=("GOOGLE_DRIVE_ACCESS_TOKEN",),
            optional_env=("HFR_GOOGLE_DRIVE_BASE_URL", "HFR_GOOGLE_DRIVE_QUERY", "HFR_GOOGLE_DRIVE_PAGE_SIZE"),
            sensitive_env=("GOOGLE_DRIVE_ACCESS_TOKEN",),
            build_source=_google_drive_source,
        ),
        ProviderSpec(
            id="imap",
            source_type="imap",
            description="Read one mailbox through IMAP SELECT readonly.",
            required_env=("IMAP_HOST", "IMAP_USERNAME", "IMAP_PASSWORD"),
            optional_env=("IMAP_PORT", "IMAP_MAILBOX", "IMAP_SEARCH", "IMAP_MAX_MESSAGES"),
            sensitive_env=("IMAP_PASSWORD",),
            build_source=_imap_source,
        ),
        ProviderSpec(
            id="jira",
            source_type="jira_issues",
            description="Read Jira issues through search.",
            required_env=("JIRA_API_TOKEN", "HFR_JIRA_BASE_URL"),
            optional_env=("JIRA_EMAIL", "HFR_JIRA_JQL", "HFR_JIRA_MAX_RESULTS"),
            sensitive_env=("JIRA_API_TOKEN",),
            build_source=_jira_source,
        ),
        ProviderSpec(
            id="kubernetes",
            source_type="kubernetes_resources",
            description="Read Kubernetes API resources.",
            required_env=("HFR_K8S_RESOURCE_URL",),
            optional_env=("KUBERNETES_BEARER_TOKEN", "HFR_K8S_TOKEN_ENV"),
            sensitive_env=("KUBERNETES_BEARER_TOKEN",),
            build_source=_kubernetes_source,
        ),
        ProviderSpec(
            id="linear",
            source_type="linear_issues",
            description="Read Linear issues through GraphQL.",
            required_env=("LINEAR_API_KEY",),
            optional_env=("HFR_LINEAR_BASE_URL", "HFR_LINEAR_FIRST"),
            sensitive_env=("LINEAR_API_KEY",),
            build_source=_linear_source,
        ),
        ProviderSpec(
            id="microsoft_graph_events",
            source_type="microsoft_graph_events",
            description="Read Microsoft Graph calendar events.",
            required_env=("MICROSOFT_GRAPH_TOKEN",),
            optional_env=("HFR_MICROSOFT_GRAPH_BASE_URL", "HFR_GRAPH_USER_ID", "HFR_GRAPH_EVENTS_TOP"),
            sensitive_env=("MICROSOFT_GRAPH_TOKEN",),
            build_source=_microsoft_graph_events_source,
        ),
        ProviderSpec(
            id="microsoft_graph_messages",
            source_type="microsoft_graph_messages",
            description="Read Microsoft Graph mail messages.",
            required_env=("MICROSOFT_GRAPH_TOKEN",),
            optional_env=("HFR_MICROSOFT_GRAPH_BASE_URL", "HFR_GRAPH_USER_ID", "HFR_GRAPH_MAIL_FOLDER_ID"),
            sensitive_env=("MICROSOFT_GRAPH_TOKEN",),
            build_source=_microsoft_graph_messages_source,
        ),
        ProviderSpec(
            id="notion",
            source_type="notion_database",
            description="Read Notion database pages.",
            required_env=("NOTION_TOKEN", "HFR_NOTION_DATABASE_ID"),
            optional_env=("HFR_NOTION_BASE_URL", "HFR_NOTION_PAGE_SIZE"),
            sensitive_env=("NOTION_TOKEN",),
            build_source=_notion_source,
        ),
        ProviderSpec(
            id="pagerduty",
            source_type="pagerduty_incidents",
            description="Read PagerDuty incidents.",
            required_env=("PAGERDUTY_API_TOKEN",),
            optional_env=("HFR_PAGERDUTY_BASE_URL", "HFR_PAGERDUTY_LIMIT", "HFR_PAGERDUTY_STATUSES"),
            sensitive_env=("PAGERDUTY_API_TOKEN",),
            build_source=_pagerduty_source,
        ),
        ProviderSpec(
            id="s3",
            source_type="s3_objects",
            description="Read an S3-compatible object listing.",
            required_env=("HFR_S3_BUCKET",),
            optional_env=("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN", "HFR_S3_UNSIGNED"),
            sensitive_env=("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"),
            build_source=_s3_source,
            missing_env=_s3_missing_env,
        ),
        ProviderSpec(
            id="slack",
            source_type="slack_history",
            description="Read Slack channel history.",
            required_env=("SLACK_BOT_TOKEN", "HFR_SLACK_CHANNEL_ID"),
            optional_env=("HFR_SLACK_BASE_URL", "HFR_SLACK_LIMIT"),
            sensitive_env=("SLACK_BOT_TOKEN",),
            build_source=_slack_source,
        ),
        ProviderSpec(
            id="stripe",
            source_type="stripe_objects",
            description="Read Stripe object or collection state.",
            required_env=("STRIPE_SECRET_KEY",),
            optional_env=("HFR_STRIPE_BASE_URL", "HFR_STRIPE_RESOURCE", "HFR_STRIPE_OBJECT_ID", "HFR_STRIPE_LIMIT"),
            sensitive_env=("STRIPE_SECRET_KEY",),
            build_source=_stripe_source,
        ),
        ProviderSpec(
            id="zendesk",
            source_type="zendesk_tickets",
            description="Read Zendesk ticket/search state.",
            required_env=("ZENDESK_API_TOKEN", "HFR_ZENDESK_BASE_URL"),
            optional_env=("ZENDESK_EMAIL", "HFR_ZENDESK_TICKET_ID", "HFR_ZENDESK_QUERY"),
            sensitive_env=("ZENDESK_API_TOKEN",),
            build_source=_zendesk_source,
        ),
    ]


def _slack_source(env: Mapping[str, str]) -> dict[str, Any]:
    return _clean(
        {
            "id": "slack",
            "type": "slack_history",
            "base_url": env.get("HFR_SLACK_BASE_URL", "https://slack.com/api"),
            "channel_id": env["HFR_SLACK_CHANNEL_ID"],
            "token_env": "SLACK_BOT_TOKEN",
            "limit": _int_env(env, "HFR_SLACK_LIMIT", 5),
            "state_path": "slack",
        }
    )


def _google_calendar_source(env: Mapping[str, str]) -> dict[str, Any]:
    return _clean(
        {
            "id": "google_calendar",
            "type": "google_calendar_events",
            "base_url": env.get("HFR_GOOGLE_CALENDAR_BASE_URL", "https://www.googleapis.com/calendar/v3"),
            "calendar_id": env.get("HFR_GOOGLE_CALENDAR_ID", "primary"),
            "token_env": "GOOGLE_CALENDAR_ACCESS_TOKEN",
            "query": env.get("HFR_GOOGLE_CALENDAR_QUERY"),
            "max_results": _int_env(env, "HFR_GOOGLE_CALENDAR_MAX_RESULTS", 5),
            "state_path": "calendar",
        }
    )


def _google_drive_source(env: Mapping[str, str]) -> dict[str, Any]:
    return _clean(
        {
            "id": "google_drive",
            "type": "google_drive_files",
            "base_url": env.get("HFR_GOOGLE_DRIVE_BASE_URL", "https://www.googleapis.com/drive/v3"),
            "token_env": "GOOGLE_DRIVE_ACCESS_TOKEN",
            "query": env.get("HFR_GOOGLE_DRIVE_QUERY"),
            "page_size": _int_env(env, "HFR_GOOGLE_DRIVE_PAGE_SIZE", 5),
            "state_path": "drive",
        }
    )


def _kubernetes_source(env: Mapping[str, str]) -> dict[str, Any]:
    source = {
        "id": "kubernetes",
        "type": "kubernetes_resources",
        "url": env["HFR_K8S_RESOURCE_URL"],
        "state_path": "kubernetes",
    }
    token_env = env.get("HFR_K8S_TOKEN_ENV")
    if token_env:
        source["bearer_token_env"] = token_env
    elif env.get("KUBERNETES_BEARER_TOKEN"):
        source["bearer_token_env"] = "KUBERNETES_BEARER_TOKEN"
    return source


def _stripe_source(env: Mapping[str, str]) -> dict[str, Any]:
    object_id = env.get("HFR_STRIPE_OBJECT_ID")
    return _clean(
        {
            "id": "stripe",
            "type": "stripe_objects",
            "base_url": env.get("HFR_STRIPE_BASE_URL", "https://api.stripe.com/v1"),
            "resource": env.get("HFR_STRIPE_RESOURCE", "payment_intents"),
            "object_id": object_id,
            "limit": _int_env(env, "HFR_STRIPE_LIMIT", 1),
            "token_env": "STRIPE_SECRET_KEY",
            "state_path": "payments.payment" if object_id else "payments",
        }
    )


def _notion_source(env: Mapping[str, str]) -> dict[str, Any]:
    return _clean(
        {
            "id": "notion",
            "type": "notion_database",
            "base_url": env.get("HFR_NOTION_BASE_URL", "https://api.notion.com/v1"),
            "database_id": env["HFR_NOTION_DATABASE_ID"],
            "token_env": "NOTION_TOKEN",
            "page_size": _int_env(env, "HFR_NOTION_PAGE_SIZE", 5),
            "state_path": "notion",
        }
    )


def _linear_source(env: Mapping[str, str]) -> dict[str, Any]:
    return _clean(
        {
            "id": "linear",
            "type": "linear_issues",
            "base_url": env.get("HFR_LINEAR_BASE_URL", "https://api.linear.app/graphql"),
            "token_env": "LINEAR_API_KEY",
            "first": _int_env(env, "HFR_LINEAR_FIRST", 5),
            "state_path": "linear",
        }
    )


def _jira_source(env: Mapping[str, str]) -> dict[str, Any]:
    source = {
        "id": "jira",
        "type": "jira_issues",
        "base_url": env["HFR_JIRA_BASE_URL"],
        "jql": env.get("HFR_JIRA_JQL", "ORDER BY updated DESC"),
        "max_results": _int_env(env, "HFR_JIRA_MAX_RESULTS", 5),
        "state_path": "jira",
    }
    if env.get("JIRA_EMAIL"):
        source["email_env"] = "JIRA_EMAIL"
        source["api_token_env"] = "JIRA_API_TOKEN"
    else:
        source["bearer_token_env"] = "JIRA_API_TOKEN"
    return source


def _s3_source(env: Mapping[str, str]) -> dict[str, Any]:
    source = {
        "id": "s3",
        "type": "s3_objects",
        "bucket": env["HFR_S3_BUCKET"],
        "prefix": env.get("HFR_S3_PREFIX", ""),
        "region": env.get("HFR_AWS_REGION") or env.get("AWS_REGION", "us-east-1"),
        "max_keys": _int_env(env, "HFR_S3_MAX_KEYS", 5),
        "unsigned": _bool_env(env, "HFR_S3_UNSIGNED", False),
        "state_path": "s3",
    }
    if env.get("HFR_S3_URL"):
        source["url"] = env["HFR_S3_URL"]
    elif env.get("HFR_S3_ENDPOINT_URL"):
        source["endpoint_url"] = env["HFR_S3_ENDPOINT_URL"]
    return source


def _microsoft_graph_messages_source(env: Mapping[str, str]) -> dict[str, Any]:
    return _clean(
        {
            "id": "microsoft_graph_messages",
            "type": "microsoft_graph_messages",
            "base_url": env.get("HFR_MICROSOFT_GRAPH_BASE_URL", "https://graph.microsoft.com/v1.0"),
            "token_env": "MICROSOFT_GRAPH_TOKEN",
            "user_id": env.get("HFR_GRAPH_USER_ID", "me"),
            "folder_id": env.get("HFR_GRAPH_MAIL_FOLDER_ID"),
            "top": _int_env(env, "HFR_GRAPH_MESSAGES_TOP", 5),
            "state_path": "graph.mail",
        }
    )


def _microsoft_graph_events_source(env: Mapping[str, str]) -> dict[str, Any]:
    return _clean(
        {
            "id": "microsoft_graph_events",
            "type": "microsoft_graph_events",
            "base_url": env.get("HFR_MICROSOFT_GRAPH_BASE_URL", "https://graph.microsoft.com/v1.0"),
            "token_env": "MICROSOFT_GRAPH_TOKEN",
            "user_id": env.get("HFR_GRAPH_USER_ID", "me"),
            "top": _int_env(env, "HFR_GRAPH_EVENTS_TOP", 5),
            "state_path": "graph.calendar",
        }
    )


def _gitlab_source(env: Mapping[str, str]) -> dict[str, Any]:
    return _clean(
        {
            "id": "gitlab",
            "type": "gitlab_issues",
            "base_url": env.get("HFR_GITLAB_BASE_URL", "https://gitlab.com/api/v4"),
            "project_id": env["HFR_GITLAB_PROJECT_ID"],
            "token_env": "GITLAB_TOKEN",
            "state": env.get("HFR_GITLAB_STATE"),
            "labels": env.get("HFR_GITLAB_LABELS"),
            "search": env.get("HFR_GITLAB_SEARCH"),
            "per_page": _int_env(env, "HFR_GITLAB_PER_PAGE", 5),
            "state_path": "gitlab",
        }
    )


def _discord_source(env: Mapping[str, str]) -> dict[str, Any]:
    return _clean(
        {
            "id": "discord",
            "type": "discord_messages",
            "base_url": env.get("HFR_DISCORD_BASE_URL", "https://discord.com/api/v10"),
            "channel_id": env["HFR_DISCORD_CHANNEL_ID"],
            "token_env": "DISCORD_BOT_TOKEN",
            "limit": _int_env(env, "HFR_DISCORD_LIMIT", 5),
            "state_path": "discord",
        }
    )


def _zendesk_source(env: Mapping[str, str]) -> dict[str, Any]:
    source = {
        "id": "zendesk",
        "type": "zendesk_tickets",
        "base_url": env["HFR_ZENDESK_BASE_URL"],
        "ticket_id": env.get("HFR_ZENDESK_TICKET_ID"),
        "query": env.get("HFR_ZENDESK_QUERY", "type:ticket"),
        "state_path": "zendesk",
    }
    if env.get("ZENDESK_EMAIL"):
        source["email_env"] = "ZENDESK_EMAIL"
        source["api_token_env"] = "ZENDESK_API_TOKEN"
    else:
        source["bearer_token_env"] = "ZENDESK_API_TOKEN"
    return _clean(source)


def _pagerduty_source(env: Mapping[str, str]) -> dict[str, Any]:
    return _clean(
        {
            "id": "pagerduty",
            "type": "pagerduty_incidents",
            "base_url": env.get("HFR_PAGERDUTY_BASE_URL", "https://api.pagerduty.com"),
            "token_env": "PAGERDUTY_API_TOKEN",
            "limit": _int_env(env, "HFR_PAGERDUTY_LIMIT", 5),
            "statuses": _csv_env(env, "HFR_PAGERDUTY_STATUSES"),
            "state_path": "pagerduty",
        }
    )


def _github_source(env: Mapping[str, str]) -> dict[str, Any]:
    source = {
        "id": "github",
        "type": "github_issue",
        "owner": env["HFR_GITHUB_OWNER"],
        "repo": env["HFR_GITHUB_REPO"],
        "issue_number": _int_env(env, "HFR_GITHUB_ISSUE_NUMBER", 0),
        "base_url": env.get("HFR_GITHUB_BASE_URL", "https://api.github.com"),
        "include_comments": True,
        "state_path": "github.issue",
    }
    if env.get("GITHUB_TOKEN"):
        source["token_env"] = "GITHUB_TOKEN"
    return source


def _gmail_source(env: Mapping[str, str]) -> dict[str, Any]:
    return _clean(
        {
            "id": "gmail",
            "type": "gmail_threads",
            "base_url": env.get("HFR_GMAIL_BASE_URL", "https://gmail.googleapis.com/gmail/v1"),
            "token_env": "GMAIL_ACCESS_TOKEN",
            "query": env.get("HFR_GMAIL_QUERY"),
            "thread_id": env.get("HFR_GMAIL_THREAD_ID"),
            "max_threads": _int_env(env, "HFR_GMAIL_MAX_THREADS", 5),
            "format": env.get("HFR_GMAIL_FORMAT", "metadata"),
            "state_path": "gmail.threads",
        }
    )


def _imap_source(env: Mapping[str, str]) -> dict[str, Any]:
    return _clean(
        {
            "id": "imap",
            "type": "imap",
            "host": env["IMAP_HOST"],
            "port": _int_env(env, "IMAP_PORT", 993),
            "username_env": "IMAP_USERNAME",
            "password_env": "IMAP_PASSWORD",
            "mailbox": env.get("IMAP_MAILBOX", "INBOX"),
            "search": env.get("IMAP_SEARCH", "ALL"),
            "max_messages": _int_env(env, "IMAP_MAX_MESSAGES", 5),
            "include_body": False,
            "state_path": "mail.imap",
        }
    )


def _missing_required_env(spec: ProviderSpec, env: Mapping[str, str]) -> list[str]:
    if spec.missing_env is not None:
        return spec.missing_env(env)
    return [name for name in spec.required_env if not env.get(name)]


def _s3_missing_env(env: Mapping[str, str]) -> list[str]:
    missing = ["HFR_S3_BUCKET"] if not env.get("HFR_S3_BUCKET") else []
    if not _bool_env(env, "HFR_S3_UNSIGNED", False):
        if not env.get("AWS_ACCESS_KEY_ID"):
            missing.append("AWS_ACCESS_KEY_ID")
        if not env.get("AWS_SECRET_ACCESS_KEY"):
            missing.append("AWS_SECRET_ACCESS_KEY")
    return missing


def _source_status(snapshot: dict[str, Any], source_id: str) -> str | None:
    source = snapshot.get("verifiers", {}).get("sources", {}).get(source_id)
    return source.get("status") if isinstance(source, dict) else None


def _env_status(names: tuple[str, ...], env: Mapping[str, str]) -> list[dict[str, Any]]:
    return [{"name": name, "present": bool(env.get(name))} for name in names]


def _runtime_secret_patterns(selected: list[ProviderSpec], env: Mapping[str, str], extra: list[str]) -> list[str]:
    patterns = list(DEFAULT_SECRET_PATTERNS)
    patterns.extend(extra)
    sensitive_names = {name for spec in selected for name in spec.sensitive_env}
    for name in sorted(sensitive_names):
        value = env.get(name)
        if value and len(value) >= 4:
            patterns.append(re.escape(value))
    return patterns


def _clean(value: dict[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if item is not None}


def _int_env(env: Mapping[str, str], name: str, default: int) -> int:
    raw = env.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _bool_env(env: Mapping[str, str], name: str, default: bool) -> bool:
    raw = env.get(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _csv_env(env: Mapping[str, str], name: str) -> list[str] | None:
    raw = env.get(name)
    if not raw:
        return None
    return [item.strip() for item in raw.split(",") if item.strip()]


def _sanitize_json(value: Any, secret_patterns: list[str]) -> Any:
    if isinstance(value, dict):
        return {key: _sanitize_json(item, secret_patterns) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize_json(item, secret_patterns) for item in value]
    if isinstance(value, str):
        return redact_text(value, secret_patterns)
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _print_provider_catalog(specs: Any) -> None:
    for spec in sorted(specs, key=lambda item: item.id):
        required = ", ".join(spec.required_env) or "none"
        optional = ", ".join(spec.optional_env) or "none"
        print(f"{spec.id}\t{spec.source_type}\trequired=[{required}]\toptional=[{optional}]")


def _environment(root: Path) -> dict[str, Any]:
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "flight_recorder_root": str(root),
        "flight_recorder_git_commit": _git_output(root, ["rev-parse", "HEAD"]) or "unknown",
        "flight_recorder_git_dirty": bool(_git_output(root, ["status", "--short"])),
    }


def _git_output(root: Path, args: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=root,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
