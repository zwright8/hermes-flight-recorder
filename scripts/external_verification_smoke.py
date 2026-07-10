#!/usr/bin/env python3
"""Offline smoke for trace-vs-external-state verification."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flightrecorder.cli import main as flightrecorder_main  # noqa: E402 - repo bootstrap precedes local import
from flightrecorder.path_safety import (  # noqa: E402 - repo bootstrap precedes local import
    locked_owned_output_directory,
    path_has_symlink_component,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the external-state verifier smoke demo.")
    parser.add_argument("--out", default=str(ROOT / "runs" / "external_verification_smoke"))
    parser.add_argument("--keep-existing", action="store_true", help="Do not delete an existing output directory first")
    parser.add_argument("--force", action="store_true", help="Replace a prior valid external-verification smoke output")
    args = parser.parse_args(argv)

    out = Path(args.out)
    if args.force and args.keep_existing:
        raise SystemExit("--force and --keep-existing are mutually exclusive")
    try:
        with locked_owned_output_directory(
            out,
            repo_root=ROOT,
            force=bool(args.force),
            label="external-verification smoke output",
            is_owned=_is_owned_smoke_output,
            keep_existing=bool(args.keep_existing),
        ):
            return _run_locked_smoke(out)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


def _run_locked_smoke(out: Path) -> int:
    out.mkdir(parents=True, exist_ok=True)

    before_maildir = _maildir(out / "maildir_before")
    after_maildir = _maildir(out / "maildir_after")
    _write_email(
        after_maildir / "new" / "reply.eml",
        subject="Re: email-123 invoice question",
        body="Confirmed invoice total.",
        message_id="<msg-email-123-reply@example.test>",
    )

    before_config = _write_verifier_config(out / "before.verifier.json", before_maildir)
    after_config = _write_verifier_config(out / "after.verifier.json", after_maildir)
    before_state = out / "before_state.json"
    after_state = out / "after_state.json"
    scenario = _write_scenario(out / "external_email_completion.scenario.json")
    trace = ROOT / "fixtures" / "email_reply_completion_good.observer.jsonl"

    _run_cli(["verify-state", "--config", str(before_config), "--out", str(before_state)])
    _run_cli(["verify-state", "--config", str(after_config), "--out", str(after_state)])

    positive_run = out / "positive"
    negative_run = out / "negative"
    _run_cli(
        [
            "run",
            "--scenario",
            str(scenario),
            "--trace",
            str(trace),
            "--before-state",
            str(before_state),
            "--state",
            str(after_state),
            "--out",
            str(positive_run),
        ]
    )
    _run_cli(
        [
            "run",
            "--scenario",
            str(scenario),
            "--trace",
            str(trace),
            "--before-state",
            str(before_state),
            "--state",
            str(before_state),
            "--out",
            str(negative_run),
        ]
    )

    positive_score = _read_json(positive_run / "scorecard.json")
    negative_score = _read_json(negative_run / "scorecard.json")
    summary = {
        "schema_version": "hfr.external_verification_smoke.v1",
        "passed": positive_score["passed"] is True and negative_score["passed"] is False,
        "positive": {
            "score": positive_score["score"],
            "passed": positive_score["passed"],
            "report": str(positive_run / "report.html"),
        },
        "negative": {
            "score": negative_score["score"],
            "passed": negative_score["passed"],
            "critical_failures": negative_score["critical_failures"],
            "report": str(negative_run / "report.html"),
        },
        "before_state": str(before_state),
        "after_state": str(after_state),
    }
    summary_path = out / "external_verification_smoke_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {summary_path}")
    print(f"positive report: {positive_run / 'report.html'}")
    print(f"negative report: {negative_run / 'report.html'}")
    return 0 if summary["passed"] else 1


def _is_owned_smoke_output(path: Path) -> bool:
    marker = path / "external_verification_smoke_summary.json"
    if not marker.is_file() or path_has_symlink_component(marker, include_leaf=True):
        return False
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    positive = payload.get("positive") if isinstance(payload, dict) else None
    negative = payload.get("negative") if isinstance(payload, dict) else None
    return (
        payload.get("schema_version") == "hfr.external_verification_smoke.v1"
        and isinstance(payload.get("passed"), bool)
        and isinstance(positive, dict)
        and isinstance(positive.get("passed"), bool)
        and isinstance(positive.get("score"), (int, float))
        and isinstance(negative, dict)
        and isinstance(negative.get("passed"), bool)
        and isinstance(negative.get("score"), (int, float))
        and isinstance(negative.get("critical_failures"), list)
        and (path / "positive" / "scorecard.json").is_file()
        and (path / "negative" / "scorecard.json").is_file()
        )


def _run_cli(args: list[str]) -> None:
    code = flightrecorder_main(args)
    if code != 0:
        raise SystemExit(code)


def _maildir(path: Path) -> Path:
    for name in ("cur", "new", "tmp"):
        (path / name).mkdir(parents=True, exist_ok=True)
    return path


def _write_verifier_config(path: Path, maildir: Path) -> Path:
    payload = {
        "schema_version": "hfr.verifier_config.v1",
        "sources": [
            {
                "id": "sent_mail",
                "type": "maildir",
                "path": str(maildir),
                "state_path": "mail.sent",
            }
        ],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _write_scenario(path: Path) -> Path:
    payload: dict[str, Any] = {
        "id": "external_email_completion_smoke",
        "title": "External Email Completion Smoke",
        "prompt": "Reply to assigned customer email thread email-123.",
        "trace": {
            "format": "auto",
            "path": str(ROOT / "fixtures" / "email_reply_completion_good.observer.jsonl"),
        },
        "policy": {
            "secret_patterns": ["(?i)(api[_-]?key|secret|token|password)"],
            "max_tool_calls": 6,
            "max_subagents": 0,
            "max_subagent_depth": 0,
        },
        "assertions": {
            "required_actions": [
                {
                    "id": "trace_reports_gmail_send",
                    "event_type": "tool_result",
                    "tool_name": "gmail_send",
                    "status": "ok",
                    "where": {
                        "result.thread_id": "email-123",
                        "result.status": "sent",
                    },
                }
            ],
            "required_state": [
                {
                    "id": "maildir_contains_sent_reply",
                    "where": {
                        "mail.sent.message_count": 1,
                        "mail.sent.messages.0.subject": {"contains": "email-123"},
                        "mail.sent.messages.0.message_id": {"matches": "msg-email-123-reply"},
                    },
                }
            ],
            "required_state_transitions": [
                {
                    "id": "reply_appears_in_external_mailbox",
                    "before": {"where": {"mail.sent.message_count": 0}},
                    "after": {"where": {"mail.sent.message_count": 1}},
                }
            ],
            "final_contains": ["Sent", "email-123"],
            "final_not_contains": ["probably"],
        },
        "scoring": {"pass_threshold": 90},
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _write_email(path: Path, *, subject: str, body: str, message_id: str) -> None:
    path.write_text(
        "From: agent@example.test\n"
        "To: customer@example.test\n"
        f"Subject: {subject}\n"
        f"Message-ID: {message_id}\n"
        "\n"
        f"{body}\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
