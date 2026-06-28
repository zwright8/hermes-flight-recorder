import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

from flightrecorder.cli import main


ROOT = Path(__file__).resolve().parents[1]


class CovenIntegrationTests(unittest.TestCase):
    def test_coven_example_generates_passing_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "coven_run"
            stdout = StringIO()
            stderr = StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                rc = main(
                    [
                        "run",
                        "--scenario",
                        str(ROOT / "examples" / "coven" / "detached_session_coven.json"),
                        "--out",
                        str(out_dir),
                    ]
                )

            self.assertEqual(rc, 0, stderr.getvalue())
            self.assertTrue((out_dir / "normalized_trace.json").exists())
            self.assertTrue((out_dir / "scorecard.json").exists())
            self.assertTrue((out_dir / "report.html").exists())
            self.assertIn("PASS", (out_dir / "report.html").read_text(encoding="utf-8"))

    def test_coven_jsonl_schema_check_accepts_fixture(self):
        stdout = StringIO()
        stderr = StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            rc = main(
                [
                    "schemas",
                    "--check-jsonl",
                    str(ROOT / "fixtures" / "coven_detached_good.coven.jsonl"),
                    "--name",
                    "coven_event",
                ]
            )

        self.assertEqual(rc, 0, stderr.getvalue())
        self.assertIn('"passed": true', stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
