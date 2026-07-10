import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from flightrecorder.cli import main
from flightrecorder.report import _render_state_diff
from flightrecorder.state_diff import build_state_diff


ROOT = Path(__file__).resolve().parents[1]


def run_cli(args):
    with redirect_stdout(StringIO()):
        return main(args)


class StateDiffTests(unittest.TestCase):
    def test_build_state_diff_uses_json_scalar_equality(self):
        diff = build_state_diff({"value": False}, {"value": 0})

        self.assertTrue(diff["changed"])
        self.assertEqual(diff["change_count"], 1)
        self.assertEqual(diff["change_status"], "changed")
        self.assertEqual(diff["changes"][0]["path"], "value")

    def test_build_state_diff_is_deterministic_and_bounded(self):
        diff = build_state_diff(
            {"b": 2, "nested": {"same": True, "removed": "x"}, "items": [{"id": 1}]},
            {"a": 1, "nested": {"same": True, "added": "y"}, "items": [{"id": 2}, {"id": 3}]},
            max_changes=3,
        )

        self.assertEqual(diff["schema_version"], "hfr.state_diff.v1")
        self.assertTrue(diff["changed"])
        self.assertEqual(diff["change_count"], 4)
        self.assertTrue(diff["truncated"])
        self.assertFalse(diff["change_count_exact"])
        self.assertTrue(diff["comparison_truncated"])
        self.assertEqual(
            [(change["path"], change["kind"]) for change in diff["changes"]],
            [
                ("a", "added"),
                ("b", "removed"),
                ("items.0.id", "changed"),
            ],
        )

    def test_build_state_diff_stops_after_first_omitted_change(self):
        class ExplodingEquality:
            def __eq__(self, other):
                raise AssertionError("state diff traversed beyond its change limit")

        diff = build_state_diff(
            {"items": [0, 1, ExplodingEquality()]},
            {"items": [1, 2, ExplodingEquality()]},
            max_changes=1,
        )

        self.assertEqual(diff["change_count"], 2)
        self.assertTrue(diff["truncated"])
        self.assertFalse(diff["change_count_exact"])
        self.assertTrue(diff["comparison_truncated"])
        self.assertEqual([change["path"] for change in diff["changes"]], ["items.0"])

    def test_build_state_diff_bounds_large_values_in_change_records(self):
        diff = build_state_diff(
            {},
            {"payload": {"items": ["x" * 10_000 for _ in range(1_000)]}},
        )

        rendered_change = json.dumps(diff["changes"][0])
        self.assertLess(len(rendered_change), 20_000)
        self.assertIn("$hfr_summary", rendered_change)
        self.assertNotIn("x" * 2_000, rendered_change)

        branching_value: object = 0
        for _ in range(4):
            branching_value = [branching_value] * 16
        branching_diff = build_state_diff({}, {"payload": branching_value})

        self.assertLess(len(json.dumps(branching_diff["changes"][0])), 20_000)

    def test_build_state_diff_bounds_depth_and_node_comparison(self):
        class ExplodingLengthDict(dict):
            def __len__(self):
                raise AssertionError("state diff exceeded its node budget")

        before = {"items": [0, 0, 0, ExplodingLengthDict()]}
        after = {"items": [0, 0, 0, ExplodingLengthDict()]}

        node_limited = build_state_diff(before, after, max_nodes=4)

        self.assertFalse(node_limited["changed"])
        self.assertFalse(node_limited["change_count_exact"])
        self.assertTrue(node_limited["comparison_truncated"])
        self.assertEqual(node_limited["truncation_reason"], "max_nodes")

        depth_limited = build_state_diff(
            {"outer": {"inner": {"before": ["x" * 10_000]}}},
            {"outer": {"inner": {"after": ["y" * 10_000]}}},
            max_depth=2,
        )

        self.assertTrue(depth_limited["changed"])
        self.assertFalse(depth_limited["change_count_exact"])
        self.assertTrue(depth_limited["comparison_truncated"])
        self.assertEqual(depth_limited["truncation_reason"], "max_depth")
        self.assertLess(len(json.dumps(depth_limited)), 20_000)

    def test_incomplete_comparison_is_explicitly_unknown_and_schema_valid(self):
        before = {"outer": {"items": [{"value": "before"}] * 20}}
        after = {"outer": {"items": [{"value": "after"}] * 20}}

        diff = build_state_diff(before, after, max_depth=1)

        self.assertFalse(diff["changed"])
        self.assertEqual(diff["change_count"], 0)
        self.assertEqual(diff["change_status"], "unknown")
        self.assertFalse(diff["comparison_complete"])
        self.assertTrue(diff["comparison_truncated"])
        self.assertTrue(diff["truncated"])
        self.assertFalse(diff["change_count_exact"])

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state_diff.json"
            summary = Path(tmp) / "validation.json"
            path.write_text(json.dumps(diff), encoding="utf-8")
            self.assertEqual(
                run_cli(["validate", "--state-diff", str(path), "--strict", "--out", str(summary)]),
                0,
            )

    def test_incomplete_directory_capture_makes_comparison_unknown(self):
        captured = {
            "filesystem": {
                "directories": {
                    "workspace": {
                        "kind": "directory",
                        "scan_incomplete": True,
                        "entries": [{"name": "visible.txt", "kind": "file"}],
                    }
                }
            }
        }

        diff = build_state_diff(captured, json.loads(json.dumps(captured)))

        self.assertEqual(diff["change_status"], "unknown")
        self.assertFalse(diff["comparison_complete"])
        self.assertEqual(diff["truncation_reason"], "incomplete_snapshot")

    def test_legacy_directory_truncation_markers_make_comparison_unknown(self):
        for legacy_marker in ("entries_truncated", "entry_count_is_lower_bound"):
            with self.subTest(marker=legacy_marker):
                captured = {
                    "filesystem": {
                        "directories": {
                            "workspace": {
                                "kind": "directory",
                                legacy_marker: True,
                                "entries": [{"name": "visible.txt", "kind": "file"}],
                            }
                        }
                    }
                }

                diff = build_state_diff(captured, json.loads(json.dumps(captured)))

                self.assertFalse(diff["changed"])
                self.assertFalse(diff["comparison_complete"])
                self.assertEqual(diff["change_status"], "unknown")
                self.assertEqual(diff["truncation_reason"], "incomplete_snapshot")

    def test_report_renders_incomplete_comparison_as_unknown(self):
        state_diff = build_state_diff(
            {"outer": {"items": [{"value": "before"}] * 20}},
            {"outer": {"items": [{"value": "after"}] * 20}},
            max_depth=1,
        )

        rendered = _render_state_diff(state_diff, [])

        self.assertIn('<article class="rule unknown">', rendered)
        self.assertIn("<span>UNKNOWN</span>", rendered)
        self.assertNotIn("UNCHANGED", rendered)
        self.assertIn("comparison became incomplete", rendered)
        self.assertIn("result is unknown", rendered)

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
            self.assertNotIn("change_count_exact", diff)
            self.assertNotIn("comparison_truncated", diff)
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
