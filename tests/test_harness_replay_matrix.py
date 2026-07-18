import json
import tempfile
import unittest
from pathlib import Path

from flightrecorder.cli import _run_scenario_artifacts
from flightrecorder.schema_registry import check_schema_file
from scripts.hermes_harness import (
    HARNESS_REPLAY_RESULT_SCHEMA_VERSION,
    publish_harness_artifacts,
    replay_trace,
    write_fake_secret_canaries,
)
from scripts.live_coven_smoke import _write_smoke_artifacts as write_coven_smoke_artifacts
from scripts.live_hermes_smoke import EXPECTED as HERMES_EXPECTED
from scripts.live_hermes_smoke import _write_smoke_artifacts as write_hermes_smoke_artifacts


ROOT = Path(__file__).resolve().parents[1]


class HarnessReplayMatrixTests(unittest.TestCase):
    def test_fixture_harness_outputs_replay_across_live_style_formats(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cases = [
                self._hermes_case(root / "hermes"),
                self._openclaw_case(root / "openclaw"),
                self._coven_case(root / "coven"),
            ]

            for case in cases:
                with self.subTest(runner=case["runner"]):
                    result = case["harness_result"]
                    self.assertTrue(result["scorecard"]["passed"])
                    self.assertEqual(result["runner"], case["runner"])
                    self.assertEqual(result["trace"]["format"], case["trace_format"])
                    self.assertTrue(check_schema_file(case["run_dir"] / "harness_manifest.json", "harness_run_manifest")["passed"])
                    self.assertTrue(check_schema_file(case["run_dir"] / "harness_result.json", "harness_run_result")["passed"])

                    replay = replay_trace(case["run_dir"] / "artifact_lineage.json", case["run_dir"] / "replay")

                    self.assertEqual(replay["schema_version"], HARNESS_REPLAY_RESULT_SCHEMA_VERSION)
                    self.assertEqual(replay["exit_code"], 0)
                    self.assertTrue(replay["passed"])
                    self.assertTrue(check_schema_file(case["run_dir"] / "replay" / "harness_replay_result.json", "harness_replay_result")["passed"])

    def test_failed_live_style_scorecard_still_publishes_valid_harness_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            observer = out / "live_observer.jsonl"
            self._write_hermes_observer(observer, final_answer="wrong answer")
            artifact_result = write_hermes_smoke_artifacts(observer, out)
            harness_result = self._publish(
                out,
                scenario_path=out / "live_scenario.json",
                artifact_result=artifact_result,
                trace_path=observer,
                trace_format="observer_jsonl",
                runner="hermes_live_fixture_failed",
                provider="custom",
                model="hfr-mock",
            )

            self.assertFalse(harness_result["scorecard"]["passed"])
            self.assertTrue(check_schema_file(out / "harness_manifest.json", "harness_run_manifest")["passed"])
            self.assertTrue(check_schema_file(out / "harness_result.json", "harness_run_result")["passed"])

    def _hermes_case(self, out: Path) -> dict:
        out.mkdir(parents=True)
        observer = out / "live_observer.jsonl"
        self._write_hermes_observer(observer, final_answer=HERMES_EXPECTED)
        artifact_result = write_hermes_smoke_artifacts(observer, out)
        harness_result = self._publish(
            out,
            scenario_path=out / "live_scenario.json",
            artifact_result=artifact_result,
            trace_path=observer,
            trace_format="observer_jsonl",
            runner="hermes_live_fixture",
            provider="custom",
            model="hfr-mock",
        )
        return {"runner": "hermes_live_fixture", "trace_format": "observer_jsonl", "run_dir": out, "harness_result": harness_result}

    def _openclaw_case(self, out: Path) -> dict:
        out.mkdir(parents=True)
        trace = out / "live_openclaw.openclaw.jsonl"
        trace.write_text((ROOT / "fixtures" / "openclaw_support_ticket_good.openclaw.jsonl").read_text(encoding="utf-8"), encoding="utf-8")
        artifact_result = _run_scenario_artifacts(
            ROOT / "examples" / "openclaw" / "support_ticket_completion_openclaw.json",
            out,
            trace_override=trace,
            trace_format="openclaw_jsonl",
            preserve_paths=True,
        )
        harness_result = self._publish(
            out,
            scenario_path=ROOT / "examples" / "openclaw" / "support_ticket_completion_openclaw.json",
            artifact_result=artifact_result,
            trace_path=trace,
            trace_format="openclaw_jsonl",
            runner="openclaw_live_fixture",
            provider="hfrmock",
            model="hfr-openclaw-fixture",
        )
        return {"runner": "openclaw_live_fixture", "trace_format": "openclaw_jsonl", "run_dir": out, "harness_result": harness_result}

    def _coven_case(self, out: Path) -> dict:
        out.mkdir(parents=True)
        trace = out / "live_coven.coven.jsonl"
        trace.write_text((ROOT / "fixtures" / "coven_detached_good.coven.jsonl").read_text(encoding="utf-8"), encoding="utf-8")
        fake_secret_files = write_fake_secret_canaries(out / "home")
        artifact_result = write_coven_smoke_artifacts(
            trace,
            out,
            sandbox=self._sandbox(out),
            fake_secret_files=fake_secret_files,
            process={"coven_exit_code": 0},
            metadata={"source": "test_harness_replay_matrix.py"},
        )
        return {
            "runner": "coven_live_smoke",
            "trace_format": "coven_jsonl",
            "run_dir": out,
            "harness_result": artifact_result["harness_result"],
        }

    def _publish(
        self,
        out: Path,
        *,
        scenario_path: Path,
        artifact_result: dict,
        trace_path: Path,
        trace_format: str,
        runner: str,
        provider: str,
        model: str,
    ) -> dict:
        fake_secret_files = write_fake_secret_canaries(out / "home")
        return publish_harness_artifacts(
            scenario_path=scenario_path,
            run_dir=out,
            artifact_result=artifact_result,
            trace_path=trace_path,
            trace_format=trace_format,
            runner=runner,
            provider=provider,
            model=model,
            sandbox=self._sandbox(out),
            fake_secret_files=fake_secret_files,
            process={"exit_code": 0},
            metadata={"source": "test_harness_replay_matrix.py"},
        )

    def _sandbox(self, out: Path) -> dict:
        return {
            "root": out / "sandbox-root",
            "home": out / "home",
            "workspace": out / "workspace",
            "events": out / "events",
            "ephemeral": True,
            "audit_artifacts_kept": True,
        }

    def _write_hermes_observer(self, path: Path, *, final_answer: str) -> None:
        rows = [
            {
                "hook": "pre_llm_call",
                "payload": {
                    "session_id": "live-session",
                    "model": "hfr-mock",
                    "user_message": "Reply exactly: flight recorder live smoke ok",
                },
            },
            {
                "hook": "post_api_request",
                "payload": {
                    "session_id": "live-session",
                    "model": "hfr-mock",
                    "finish_reason": "stop",
                },
            },
            {
                "hook": "post_llm_call",
                "payload": {
                    "session_id": "live-session",
                    "model": "hfr-mock",
                    "assistant_response": final_answer,
                },
            },
        ]
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
