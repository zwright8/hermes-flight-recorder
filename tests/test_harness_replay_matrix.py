import json
import shutil
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

from flightrecorder.cli import _run_scenario_artifacts, main
from flightrecorder.schema_registry import check_schema_file
from scripts.hermes_harness import (
    HARNESS_REPLAY_RESULT_SCHEMA_VERSION,
    HARNESS_RUN_RESULT_SCHEMA_VERSION,
    main as harness_main,
    publish_harness_artifacts,
    replay_trace,
    write_fake_secret_canaries,
)
from scripts.live_coven_smoke import _write_smoke_artifacts as write_coven_smoke_artifacts
from scripts.live_hermes_smoke import EXPECTED as HERMES_EXPECTED
from scripts.live_hermes_smoke import _write_smoke_artifacts as write_hermes_smoke_artifacts
from scripts.live_openclaw_smoke import MODEL_REF as OPENCLAW_MODEL_REF


ROOT = Path(__file__).resolve().parents[1]


def _run_cli(args: list[str]) -> tuple[int, str, str]:
    stdout = StringIO()
    stderr = StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        rc = main(args)
    return rc, stdout.getvalue(), stderr.getvalue()


def _run_harness_cli(args: list[str]) -> tuple[int, str, str]:
    stdout = StringIO()
    stderr = StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        rc = harness_main(args)
    return rc, stdout.getvalue(), stderr.getvalue()


class HarnessReplayMatrixTests(unittest.TestCase):
    def test_live_shaped_harness_artifacts_replay_across_supported_trace_formats(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cases = [
                self._hermes_case(root / "hermes"),
                self._openclaw_case(root / "openclaw"),
                self._coven_case(root / "coven"),
            ]

            for case in cases:
                with self.subTest(runner=case["runner"], trace_format=case["trace_format"]):
                    harness_result = self._publish_case(case)

                    self.assertEqual(harness_result["schema_version"], HARNESS_RUN_RESULT_SCHEMA_VERSION)
                    self.assertTrue(harness_result["scorecard"]["passed"])
                    self.assertTrue(harness_result["replay"]["self_contained"])
                    self.assertEqual(harness_result["trace"]["format"], case["trace_format"])
                    self._assert_harness_artifacts_validate(case["out_dir"])

                    replay_dir = case["out_dir"] / "replay"
                    replay = _replay_trace_quietly(case["out_dir"] / "artifact_lineage.json", replay_dir)

                    self.assertEqual(replay["schema_version"], HARNESS_REPLAY_RESULT_SCHEMA_VERSION)
                    self.assertEqual(replay["exit_code"], 0)
                    self.assertTrue(replay["passed"])
                    self.assertTrue(
                        check_schema_file(replay_dir / "harness_replay_result.json", "harness_replay_result")["passed"]
                    )
                    self._assert_cli_ok(
                        ["validate", "--harness-replay-result", str(replay_dir / "harness_replay_result.json"), "--strict"]
                    )

    def test_publish_trace_cli_records_codex_style_trajectory_harness_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "codex_style"

            rc, stdout, stderr = _run_harness_cli(
                [
                    "publish-trace",
                    "--scenario",
                    str(ROOT / "scenarios" / "prompt_injection_good.json"),
                    "--trace",
                    str(ROOT / "fixtures" / "prompt_injection_good.trajectory.jsonl"),
                    "--format",
                    "trajectory_jsonl",
                    "--runner",
                    "codex_style_trace",
                    "--provider",
                    "codex",
                    "--out",
                    str(out_dir),
                    "--relative-paths",
                ]
            )

            self.assertEqual(rc, 0, stderr or stdout)
            harness_result = _json_from_stdout(stdout)
            self.assertEqual(harness_result["runner"], "codex_style_trace")
            self.assertEqual(harness_result["provider"], "codex")
            self.assertEqual(harness_result["trace"]["format"], "trajectory_jsonl")
            self.assertEqual(harness_result["trace"]["path"], "inputs/prompt_injection_good.trajectory.jsonl")
            self.assertEqual(harness_result["scorecard"]["path"], "scorecard.json")
            self.assertEqual(harness_result["model"]["id"], "fixture-model")
            self.assertEqual(harness_result["process"]["mode"], "recorded_trace")
            self.assertTrue(harness_result["scorecard"]["passed"])
            self.assertTrue((out_dir / "inputs" / "scenario.json").exists())
            self.assertTrue((out_dir / "inputs" / "prompt_injection_good.trajectory.jsonl").exists())
            self.assertTrue((out_dir / "sandbox" / "home" / ".hermes" / ".env").exists())
            for name in ("harness_manifest.json", "harness_result.json"):
                text = (out_dir / name).read_text(encoding="utf-8")
                self.assertNotIn(str(ROOT), text)
                self.assertNotIn(str(out_dir.parent), text)
            lineage = json.loads((out_dir / "artifact_lineage.json").read_text(encoding="utf-8"))
            lineage_text = json.dumps(lineage, sort_keys=True)
            self.assertNotIn(str(ROOT), lineage_text)
            self.assertNotIn(str(out_dir.parent), lineage_text)
            self.assertTrue(lineage["replay"]["self_contained"])
            self.assertEqual(lineage["replay"]["argv"][lineage["replay"]["argv"].index("--scenario") + 1], "inputs/scenario.json")
            self.assertEqual(
                lineage["replay"]["argv"][lineage["replay"]["argv"].index("--trace") + 1],
                "inputs/prompt_injection_good.trajectory.jsonl",
            )
            self._assert_harness_artifacts_validate(out_dir)

            replay_dir = out_dir / "replay"
            replay = _replay_trace_quietly(out_dir / "artifact_lineage.json", replay_dir)

            self.assertEqual(replay["exit_code"], 0)
            self.assertTrue(replay["passed"])
            self.assertTrue(check_schema_file(replay_dir / "harness_replay_result.json", "harness_replay_result")["passed"])

    def test_failed_live_shaped_scorecard_still_publishes_valid_harness_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "hermes_failed"
            out_dir.mkdir()
            observer = out_dir / "live_observer.jsonl"
            _write_hermes_observer_trace(observer, final_answer="unexpected answer")
            artifact_result = write_hermes_smoke_artifacts(observer, out_dir)

            harness_result = self._publish_case(
                {
                    "artifact_result": artifact_result,
                    "base_url": "http://127.0.0.1:8123/v1",
                    "model": "hfr-mock",
                    "out_dir": out_dir,
                    "provider": "custom",
                    "runner": "hermes_live_smoke",
                    "scenario_path": out_dir / "live_scenario.json",
                    "trace_format": "observer_jsonl",
                    "trace_path": observer,
                }
            )

            self.assertFalse(artifact_result["scorecard"]["passed"])
            self.assertFalse(harness_result["scorecard"]["passed"])
            self.assertTrue((out_dir / "regression_scenario.json").exists())
            self.assertTrue(harness_result["replay"]["self_contained"])
            self._assert_harness_artifacts_validate(out_dir)

            replay_dir = out_dir / "replay"
            replay = _replay_trace_quietly(out_dir / "artifact_lineage.json", replay_dir)

            self.assertEqual(replay["exit_code"], 0)
            self.assertFalse(replay["passed"])
            self.assertTrue(check_schema_file(replay_dir / "harness_replay_result.json", "harness_replay_result")["passed"])

    def _hermes_case(self, out_dir: Path) -> dict:
        out_dir.mkdir()
        observer = out_dir / "live_observer.jsonl"
        _write_hermes_observer_trace(observer, final_answer=HERMES_EXPECTED)
        return {
            "artifact_result": write_hermes_smoke_artifacts(observer, out_dir),
            "base_url": "http://127.0.0.1:8123/v1",
            "model": "hfr-mock",
            "out_dir": out_dir,
            "provider": "custom",
            "runner": "hermes_live_smoke",
            "scenario_path": out_dir / "live_scenario.json",
            "trace_format": "observer_jsonl",
            "trace_path": observer,
        }

    def _openclaw_case(self, out_dir: Path) -> dict:
        out_dir.mkdir()
        trace = out_dir / "live_openclaw.openclaw.jsonl"
        shutil.copyfile(ROOT / "fixtures" / "openclaw_support_ticket_good.openclaw.jsonl", trace)
        scenario_path = out_dir / "live_openclaw_scenario.json"
        scenario = json.loads((ROOT / "examples" / "openclaw" / "support_ticket_completion_openclaw.json").read_text(encoding="utf-8"))
        scenario["trace"] = {"format": "openclaw_jsonl", "path": trace.name}
        scenario_path.write_text(json.dumps(scenario, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return {
            "artifact_result": _run_scenario_artifacts(
                scenario_path,
                out_dir,
                trace_format="openclaw_jsonl",
                preserve_paths=True,
            ),
            "base_url": "http://127.0.0.1:8124/v1",
            "model": OPENCLAW_MODEL_REF,
            "out_dir": out_dir,
            "provider": "hfrmock",
            "runner": "openclaw_live_smoke",
            "scenario_path": scenario_path,
            "trace_format": "openclaw_jsonl",
            "trace_path": trace,
        }

    def _coven_case(self, out_dir: Path) -> dict:
        out_dir.mkdir()
        trace = out_dir / "live_coven.coven.jsonl"
        shutil.copyfile(ROOT / "fixtures" / "coven_detached_good.coven.jsonl", trace)
        return {
            "artifact_result": write_coven_smoke_artifacts(trace, out_dir),
            "base_url": None,
            "model": "openai/gpt-5.5",
            "out_dir": out_dir,
            "provider": "coven",
            "runner": "coven_live_smoke",
            "scenario_path": out_dir / "live_coven_scenario.json",
            "trace_format": "coven_jsonl",
            "trace_path": trace,
        }

    def _publish_case(self, case: dict) -> dict:
        out_dir = Path(case["out_dir"])
        fake_secret_files = write_fake_secret_canaries(out_dir / "home")
        return publish_harness_artifacts(
            scenario_path=case["scenario_path"],
            run_dir=out_dir,
            artifact_result=case["artifact_result"],
            trace_path=case["trace_path"],
            trace_format=case["trace_format"],
            runner=case["runner"],
            provider=case["provider"],
            model=case["model"],
            base_url=case["base_url"],
            sandbox={
                "root": out_dir / "sandbox",
                "home": out_dir / "home",
                "workspace": out_dir / "workspace",
                "events": out_dir / "events",
                "ephemeral": True,
                "audit_artifacts_kept": True,
            },
            fake_secret_files=fake_secret_files,
            process={"exit_code": 0},
            metadata={"source": "tests/test_harness_replay_matrix.py", "fixture_backed": True},
            preserve_paths=False,
        )

    def _assert_harness_artifacts_validate(self, out_dir: Path) -> None:
        self.assertTrue(check_schema_file(out_dir / "harness_manifest.json", "harness_run_manifest")["passed"])
        self.assertTrue(check_schema_file(out_dir / "harness_result.json", "harness_run_result")["passed"])
        self._assert_cli_ok(
            [
                "validate",
                "--harness-manifest",
                str(out_dir / "harness_manifest.json"),
                "--harness-result",
                str(out_dir / "harness_result.json"),
                "--strict",
            ]
        )

    def _assert_cli_ok(self, args: list[str]) -> None:
        rc, stdout, stderr = _run_cli(args)
        self.assertEqual(rc, 0, stderr or stdout)


def _write_hermes_observer_trace(path: Path, *, final_answer: str) -> None:
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


def _replay_trace_quietly(lineage_path: Path, replay_dir: Path) -> dict:
    stdout = StringIO()
    with redirect_stdout(stdout):
        return replay_trace(lineage_path, replay_dir, preserve_paths=False)


def _json_from_stdout(stdout: str) -> dict:
    start = stdout.find("{")
    if start < 0:
        raise AssertionError(f"stdout did not contain JSON: {stdout!r}")
    return json.loads(stdout[start:])


if __name__ == "__main__":
    unittest.main()
