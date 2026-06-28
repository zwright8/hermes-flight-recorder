import json
import tempfile
import unittest
from pathlib import Path

from flightrecorder.cli import main
from flightrecorder.schema_registry import check_schema_contract
from flightrecorder.state_validators import build_monitor_catalog, build_state_validator_assertions
from flightrecorder.verifiers import capture_verified_state


ROOT = Path(__file__).resolve().parents[1]


class StateValidatorTests(unittest.TestCase):
    def test_catalog_lists_monitorable_external_states(self):
        catalog = build_monitor_catalog()
        monitor_ids = {monitor["id"] for monitor in catalog["monitors"]}

        self.assertIn("email", monitor_ids)
        self.assertIn("github", monitor_ids)
        self.assertIn("databases", monitor_ids)
        self.assertIn("filesystem", monitor_ids)
        self.assertIn("email_sent", catalog["validators"])
        self.assertIn("github_issue_closed", catalog["validators"])

    def test_cli_writes_catalog_and_markdown(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            catalog_path = root / "catalog.json"
            markdown_path = root / "catalog.md"

            self.assertEqual(
                main(
                    [
                        "state-validators",
                        "--list",
                        "--out",
                        str(catalog_path),
                        "--markdown-out",
                        str(markdown_path),
                    ]
                ),
                0,
            )

            catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
            self.assertEqual(catalog["schema_version"], "hfr.state_validator_catalog.v1")
            self.assertIn("Email And Mailboxes", markdown_path.read_text(encoding="utf-8"))

    def test_email_sent_validator_scores_external_state_completion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            before_maildir = _maildir(root / "before_sent")
            after_maildir = _maildir(root / "after_sent")
            _write_eml(
                after_maildir / "new" / "reply.eml",
                subject="Re: email-123 invoice question",
                body="Confirmed invoice total.",
                message_id="<msg-email-123-reply@example.test>",
            )
            before_state = capture_verified_state(_maildir_config(before_maildir))
            after_state = capture_verified_state(_maildir_config(after_maildir))
            before_path = _write_json(root / "before.state.json", before_state)
            after_path = _write_json(root / "after.state.json", after_state)

            compiled = build_state_validator_assertions(
                {
                    "schema_version": "hfr.state_validator_config.v1",
                    "validators": [
                        {
                            "id": "email_123_reply",
                            "validator": "email_sent",
                            "state_path": "mail.sent",
                            "before_count": 0,
                            "after_count": 1,
                            "thread_id": "email-123",
                            "subject_contains": "email-123",
                            "message_id_matches": "msg-email-123-reply",
                        }
                    ],
                }
            )
            self.assertEqual(compiled["schema_version"], "hfr.state_validator_assertions.v1")
            self.assertTrue(
                check_schema_contract(
                    {
                        "schema_version": "hfr.state_validator_config.v1",
                        "validators": [
                            {
                                "id": "email_123_reply",
                                "validator": "email_sent",
                                "state_path": "mail.sent",
                            }
                        ],
                    }
                )["passed"]
            )
            self.assertTrue(check_schema_contract(compiled)["passed"])
            scenario_path = _write_scenario(root / "scenario.json", compiled["assertions"])

            passing_run = root / "passing"
            self.assertEqual(
                main(
                    [
                        "run",
                        "--scenario",
                        str(scenario_path),
                        "--trace",
                        str(ROOT / "fixtures" / "email_reply_completion_good.observer.jsonl"),
                        "--before-state",
                        str(before_path),
                        "--state",
                        str(after_path),
                        "--out",
                        str(passing_run),
                    ]
                ),
                0,
            )
            passing_score = json.loads((passing_run / "scorecard.json").read_text(encoding="utf-8"))
            self.assertTrue(passing_score["passed"], passing_score)

            failing_run = root / "failing"
            self.assertEqual(
                main(
                    [
                        "run",
                        "--scenario",
                        str(scenario_path),
                        "--trace",
                        str(ROOT / "fixtures" / "email_reply_completion_good.observer.jsonl"),
                        "--before-state",
                        str(before_path),
                        "--state",
                        str(before_path),
                        "--out",
                        str(failing_run),
                    ]
                ),
                0,
            )
            failing_score = json.loads((failing_run / "scorecard.json").read_text(encoding="utf-8"))
            self.assertFalse(failing_score["passed"])
            self.assertIn("required_state", failing_score["critical_failures"])
            self.assertIn("required_state_transitions", failing_score["critical_failures"])

    def test_github_issue_closed_validator_generates_transition_assertions(self):
        compiled = build_state_validator_assertions(
            {
                "validator": "github_issue_closed",
                "id": "close_issue_7",
                "state_path": "github.issue_7",
                "trace": False,
            }
        )

        transition = compiled["assertions"]["required_state_transitions"][0]
        self.assertEqual(transition["before"]["where"]["github.issue_7.issue.state"], "open")
        self.assertEqual(transition["after"]["where"]["github.issue_7.issue.state"], "closed")


def _maildir(path: Path) -> Path:
    for name in ("cur", "new", "tmp"):
        (path / name).mkdir(parents=True, exist_ok=True)
    return path


def _maildir_config(path: Path) -> dict:
    return {
        "schema_version": "hfr.verifier_config.v1",
        "sources": [
            {
                "id": "sent_mail",
                "type": "maildir",
                "path": str(path),
                "state_path": "mail.sent",
            }
        ],
    }


def _write_scenario(path: Path, assertions: dict) -> Path:
    scenario = {
        "id": "validator_email_completion",
        "title": "Validator Email Completion",
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
            **assertions,
            "final_contains": ["Sent", "email-123"],
            "final_not_contains": ["probably"],
        },
        "scoring": {"pass_threshold": 90},
    }
    return _write_json(path, scenario)


def _write_eml(path: Path, *, subject: str, body: str, message_id: str) -> None:
    path.write_text(
        "From: agent@example.test\n"
        "To: customer@example.test\n"
        f"Subject: {subject}\n"
        f"Message-ID: {message_id}\n"
        "\n"
        f"{body}\n",
        encoding="utf-8",
    )


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
