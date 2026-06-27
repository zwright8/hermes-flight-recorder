import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from flightrecorder.cli import main
from flightrecorder.state_diff import build_state_diff


ROOT = Path(__file__).resolve().parents[1]


def run_cli(args):
    with redirect_stdout(StringIO()):
        return main(args)


class StateDiffTests(unittest.TestCase):
    def test_build_state_diff_is_deterministic_and_bounded(self):
        diff = build_state_diff(
            {"b": 2, "nested": {"same": True, "removed": "x"}, "items": [{"id": 1}]},
            {"a": 1, "nested": {"same": True, "added": "y"}, "items": [{"id": 2}, {"id": 3}]},
            max_changes=3,
        )

        self.assertEqual(diff["schema_version"], "hfr.state_diff.v1")
        self.assertTrue(diff["changed"])
        self.assertEqual(diff["change_count"], 6)
        self.assertTrue(diff["truncated"])
        self.assertEqual(
            [(change["path"], change["kind"]) for change in diff["changes"]],
            [
                ("a", "added"),
                ("b", "removed"),
                ("items.0.id", "changed"),
            ],
        )

    def test_diff_state_cli_redacts_and_validates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            before = root / "before.json"
            after = root / "after.json"
            out = root / "state_diff.json"
            validation = root / "validation.json"
            before.write_text(json.dumps({"token": "api_key=before-secret", "status": "draft"}), encoding="utf-8")
            after.write_text(
                json.dumps({"token": "api_key=after-secret", "status": "sent", "notes": "secret=after-secret"}),
                encoding="utf-8",
            )

            self.assertEqual(
                run_cli(
                    [
                        "diff-state",
                        "--before",
                        str(before),
                        "--after",
                        str(after),
                        "--out",
                        str(out),
                    ]
                ),
                0,
            )
            diff = json.loads(out.read_text(encoding="utf-8"))
            rendered = json.dumps(diff)
            self.assertIn("[REDACTED]", rendered)
            self.assertNotIn("before-secret", rendered)
            self.assertNotIn("after-secret", rendered)
            self.assertEqual(run_cli(["validate", "--state-diff", str(out), "--out", str(validation), "--strict"]), 0)

    def test_run_emits_state_diff_for_before_after_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"

            self.assertEqual(run_cli(["run", "--scenario", str(ROOT / "scenarios" / "email_reply_completion_good.json"), "--out", str(out)]), 0)

            diff = json.loads((out / "state_diff.json").read_text(encoding="utf-8"))
            lineage = json.loads((out / "artifact_lineage.json").read_text(encoding="utf-8"))
            self.assertTrue(diff["changed"])
            self.assertEqual(diff["change_count"], 2)
            self.assertEqual(
                [(change["path"], change["kind"]) for change in diff["changes"]],
                [
                    ("gmail.threads.email-123.last_sent_message_id", "changed"),
                    ("gmail.threads.email-123.sent_replies.0", "added"),
                ],
            )
            self.assertIn("state_diff", {item["name"] for item in lineage["outputs"]})
            self.assertIn(
                {
                    "from": ["before_state_snapshot", "state_snapshot"],
                    "to": "state_diff",
                    "operation": "diff_state",
                },
                lineage["graph"],
            )
            self.assertEqual(run_cli(["validate", "--run", str(out), "--strict"]), 0)

    def test_report_command_renders_state_diff(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            report_path = Path(tmp) / "standalone_report.html"
            self.assertEqual(run_cli(["run", "--scenario", str(ROOT / "scenarios" / "email_reply_completion_good.json"), "--out", str(run_dir)]), 0)

            self.assertEqual(
                run_cli(
                    [
                        "report",
                        "--scenario",
                        str(ROOT / "scenarios" / "email_reply_completion_good.json"),
                        "--trace",
                        str(run_dir / "normalized_trace.json"),
                        "--score",
                        str(run_dir / "scorecard.json"),
                        "--state-diff",
                        str(run_dir / "state_diff.json"),
                        "--out",
                        str(report_path),
                    ]
                ),
                0,
            )

            report = report_path.read_text(encoding="utf-8")
            self.assertIn("State Changes", report)
            self.assertIn('data-label="Before"', report)
            self.assertIn("gmail.threads.email-123.last_sent_message_id", report)
            self.assertIn("msg-email-123-001", report)

    def test_validate_rejects_inconsistent_state_diff(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            validation = Path(tmp) / "validation.json"
            self.assertEqual(run_cli(["run", "--scenario", str(ROOT / "scenarios" / "email_reply_completion_good.json"), "--out", str(out)]), 0)
            diff_path = out / "state_diff.json"
            diff = json.loads(diff_path.read_text(encoding="utf-8"))
            diff["change_count"] = 99
            diff_path.write_text(json.dumps(diff, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            code = run_cli(["validate", "--run", str(out), "--out", str(validation)])

            self.assertEqual(code, 1)
            summary = json.loads(validation.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in summary["targets"] for error in target["errors"])
            self.assertIn("state_diff.truncated", errors)


if __name__ == "__main__":
    unittest.main()
