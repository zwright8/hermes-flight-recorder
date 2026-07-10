import json
import shlex
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock

from scripts import check_openai_serving, manage_openai_serving
from scripts.hermes_harness import (
    build_harness_manifest,
    main as harness_main,
    publish_trace_run,
    replay_trace,
    run_scenario,
)


ROOT = Path(__file__).resolve().parents[1]


class HarnessPublicPathTests(unittest.TestCase):
    def test_harness_and_replay_artifacts_use_relative_paths_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scenario = _write_scenario(root / "scenario.json")
            run_dir = root / "run"
            manifest = build_harness_manifest(
                scenario_path=scenario,
                out_dir=run_dir,
                mock_response="public surface hardening complete",
            )

            result = run_scenario(manifest)
            published_dir = root / "published"
            published = publish_trace_run(
                scenario_path=scenario,
                trace_path=run_dir / "mock_observer.jsonl",
                out_dir=published_dir,
            )
            replay = replay_trace(run_dir / "artifact_lineage.json", root / "replay")

            self.assertFalse(Path(result["trace"]["path"]).is_absolute())
            self.assertFalse(Path(result["replay"]["lineage"]).is_absolute())
            self.assertFalse(Path(published["trace"]["path"]).is_absolute())
            self.assertFalse(Path(published["replay"]["lineage"]).is_absolute())
            self.assertFalse(Path(replay["lineage"]).is_absolute())
            self.assertFalse(Path(replay["out_dir"]).is_absolute())
            for path in (
                run_dir / "harness_manifest.json",
                run_dir / "harness_result.json",
                run_dir / "artifact_lineage.json",
                published_dir / "harness_manifest.json",
                published_dir / "harness_result.json",
                published_dir / "artifact_lineage.json",
                root / "replay" / "harness_replay_result.json",
            ):
                self.assertNotIn(str(root), path.read_text(encoding="utf-8"), path.name)

    def test_preserve_paths_is_explicit_and_relative_paths_remains_compatible(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scenario = _write_scenario(root / "scenario.json")
            preserve_out = root / "preserve"
            relative_out = root / "relative"

            with redirect_stdout(StringIO()):
                preserve_code = harness_main(
                    [
                        "run-scenario",
                        "--scenario",
                        str(scenario),
                        "--out",
                        str(preserve_out),
                        "--mock-response",
                        "public surface hardening complete",
                        "--preserve-paths",
                    ]
                )
                relative_code = harness_main(
                    [
                        "run-scenario",
                        "--scenario",
                        str(scenario),
                        "--out",
                        str(relative_out),
                        "--mock-response",
                        "public surface hardening complete",
                        "--relative-paths",
                    ]
                )

            self.assertEqual(preserve_code, 0)
            self.assertEqual(relative_code, 0)
            preserved = _read_json(preserve_out / "harness_result.json")
            relative = _read_json(relative_out / "harness_result.json")
            self.assertTrue(Path(preserved["trace"]["path"]).is_absolute())
            self.assertFalse(Path(relative["trace"]["path"]).is_absolute())


class ServingPublicArtifactTests(unittest.TestCase):
    def test_command_secret_values_use_shared_redaction_classifier(self):
        command = [
            "server",
            "--auth",
            "auth-secret",
            "--authentication=authentication-secret",
            "--bearer",
            "bearer-secret",
            "--cookie",
            "cookie-secret",
            "--client-auth-token=composite-secret",
            "--api-key-env",
            "SAFE_ENV_NAME",
        ]

        self.assertEqual(
            manage_openai_serving._command_secret_values(command),
            [
                "auth-secret",
                "authentication-secret",
                "bearer-secret",
                "cookie-secret",
                "composite-secret",
            ],
        )

    def test_log_sanitization_preserves_evidence_larger_than_previous_tail_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            log_path = root / "server.stdout.log"
            private_path = root / "private-model"
            secret = "serving-log-secret"
            evidence_lines = [f"evidence-{index:05d}\n" for index in range(6_000)]
            evidence_lines.insert(1, f"token={secret}\n")
            evidence_lines.insert(2, f"model_path={private_path}\n")
            log_path.write_text("".join(evidence_lines), encoding="utf-8")

            self.assertGreater(log_path.stat().st_size, 64 * 1024)
            manage_openai_serving._sanitize_log_file(
                log_path,
                secret_values=[secret],
                path_replacements={str(private_path): "private-model"},
            )

            sanitized = log_path.read_text(encoding="utf-8")
            self.assertIn("evidence-00000", sanitized)
            self.assertIn("evidence-05999", sanitized)
            self.assertIn("token=[REDACTED]", sanitized)
            self.assertIn("model_path=private-model", sanitized)
            self.assertNotIn(secret, sanitized)
            self.assertNotIn(str(root), sanitized)

    def test_log_sanitization_marks_oversized_lines_and_preserves_following_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "server.stdout.log"
            log_path.write_text(
                "before\n" + ("x" * (manage_openai_serving._MAX_LOG_LINE_CHARS + 1)) + "\nafter\n",
                encoding="utf-8",
            )

            manage_openai_serving._sanitize_log_file(
                log_path,
                secret_values=[],
                path_replacements={},
            )

            self.assertEqual(
                log_path.read_text(encoding="utf-8"),
                "before\n[REDACTED: oversized log line]\nafter\n",
            )

    def test_log_sanitization_replaces_atomically(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            log_path = root / "server.stdout.log"
            original = "first\nsecond\n"
            log_path.write_text(original, encoding="utf-8")
            real_sanitizer = manage_openai_serving.sanitize_public_artifact

            def fail_on_second_line(value, **kwargs):
                if value == "second\n":
                    raise RuntimeError("injected sanitization failure")
                return real_sanitizer(value, **kwargs)

            with (
                mock.patch.object(
                    manage_openai_serving,
                    "sanitize_public_artifact",
                    side_effect=fail_on_second_line,
                ),
                self.assertRaisesRegex(RuntimeError, "injected sanitization failure"),
            ):
                manage_openai_serving._sanitize_log_file(
                    log_path,
                    secret_values=[],
                    path_replacements={},
                )

            self.assertEqual(log_path.read_text(encoding="utf-8"), original)
            self.assertEqual(list(root.glob(".server.stdout.log.*.sanitizing")), [])

    def test_failed_private_log_sanitization_leaves_no_public_secret_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            private_log = root / "private.log"
            public_log = root / "server.stdout.log"
            private_log.write_text("token=raw-secret\n", encoding="utf-8")
            public_log.write_text("token=stale-secret\n", encoding="utf-8")

            with (
                mock.patch.object(
                    manage_openai_serving,
                    "sanitize_public_artifact",
                    side_effect=RuntimeError("injected sanitization failure"),
                ),
                self.assertRaisesRegex(RuntimeError, "injected sanitization failure"),
            ):
                manage_openai_serving._sanitize_log_file(
                    private_log,
                    public_path=public_log,
                    secret_values=["raw-secret", "stale-secret"],
                    path_replacements={},
                )

            self.assertFalse(public_log.exists())
            self.assertEqual(private_log.read_text(encoding="utf-8"), "token=raw-secret\n")

    def test_managed_process_never_writes_raw_bytes_to_public_log_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / "serving"
            out.mkdir()
            public_logs = [out / "server.stdout.log", out / "server.stderr.log"]
            for path in public_logs:
                path.write_text("token=stale-secret\n", encoding="utf-8")
            observed_public_state: list[tuple[bool, bool]] = []

            def observe_private_staging(*_args, **_kwargs):
                observed_public_state.append(tuple(path.exists() for path in public_logs))
                return {"ready": False, "attempts": [], "summary": "injected_not_ready"}

            command = shlex.join(
                [
                    sys.executable,
                    "-c",
                    "import os; print(os.environ['PRIVATE_TOKEN'])",
                ]
            )
            with (
                mock.patch.object(
                    manage_openai_serving,
                    "_wait_until_ready",
                    side_effect=observe_private_staging,
                ),
                mock.patch.object(
                    manage_openai_serving,
                    "_sanitize_log_file",
                    side_effect=RuntimeError("injected sanitization failure"),
                ),
                redirect_stdout(StringIO()),
            ):
                code = manage_openai_serving.main(
                    [
                        "--command",
                        command,
                        "--cwd",
                        str(root),
                        "--base-url",
                        "http://127.0.0.1:9/v1",
                        "--env",
                        "PRIVATE_TOKEN=raw-secret",
                        "--out",
                        str(out),
                    ]
                )

            self.assertEqual(code, 1)
            self.assertEqual(observed_public_state, [(False, False)])
            self.assertTrue(all(not path.exists() for path in public_logs))
            self.assertNotIn("raw-secret", (out / "serving_lifecycle.json").read_text(encoding="utf-8"))

    def test_endpoint_artifacts_redact_adapter_path_url_credentials_and_secret_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            adapter = root / "private-adapter"
            adapter.mkdir()
            (adapter / "adapter_config.json").write_text("{}\n", encoding="utf-8")
            base_url = "http://private-user:url-password@127.0.0.1:8000/v1?api_key=url-secret"
            api_key = "endpoint-api-secret"
            requested_model = root / "private-model"
            served_model = "hfr-model+private-adapter"
            health = {
                "ok": True,
                "status_code": 200,
                "url": f"{base_url}/healthz",
                "elapsed_ms": 1,
                "json": {"model": served_model},
                "error": f"Bearer {api_key}",
                "attempts": [],
            }
            responses = [
                {"ok": True, "status_code": 200, "url": f"{base_url}/models", "elapsed_ms": 1, "json": {"data": [{"id": served_model}]}},
                {
                    "ok": True,
                    "status_code": 200,
                    "url": f"{base_url}/chat/completions",
                    "elapsed_ms": 1,
                    "json": {"model": served_model, "choices": [{"message": {"content": "ok"}}]},
                },
            ]
            capability = {"status": "supported", "response_ok": True, "error": None}
            streaming = {**capability, "event_count": 1, "done_seen": True, "text": "ok"}
            tools = {**capability, "tool_call_count": 1, "tool_calls": [{}]}
            structured = {**capability, "json_parse_passed": True, "parsed": {}, "text": "{}"}

            with (
                mock.patch.object(check_openai_serving, "_get_first_json", return_value=health),
                mock.patch.object(check_openai_serving, "_request_json", side_effect=responses),
                mock.patch.object(check_openai_serving, "_tool_call_check", return_value=tools),
                mock.patch.object(check_openai_serving, "_structured_output_check", return_value=structured),
                mock.patch.object(check_openai_serving, "_streaming_check", return_value=streaming),
            ):
                artifacts = check_openai_serving.check_endpoint(
                    base_url=base_url,
                    model=str(requested_model),
                    provider="custom",
                    arm="candidate",
                    engine="openai_compatible",
                    adapter=str(adapter),
                    api_key=api_key,
                    timeout=1.0,
                    out_dir=root / "artifacts",
                    require_streaming=True,
                    require_tool_call=True,
                    require_structured_output=True,
                )

            text = json.dumps(artifacts, sort_keys=True)
            for private_value in (str(root), "private-user", "url-password", "url-secret", api_key):
                self.assertNotIn(private_value, text)
            profile, _compatibility, report = artifacts
            self.assertEqual(profile["model_identity"]["requested_model"], "private-model")
            self.assertEqual(profile["model_identity"]["adapter"]["id"], "private-adapter")
            self.assertNotIn("@", profile["endpoint"]["base_url"])
            self.assertNotIn("?", report["base_url"])

    def test_lifecycle_and_referenced_logs_redact_paths_and_known_secrets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / "lifecycle"
            adapter = root / "adapter"
            adapter.mkdir()
            script = (
                "import os,sys; "
                "print(os.environ['PRIVATE_TOKEN']); "
                "print(os.environ['AMBIENT_SECRET']); "
                "print('token=log-secret', file=sys.stderr)"
            )
            command = shlex.join([sys.executable, "-c", script, "--access-token", "argv-secret"])
            base_url = "http://private-user:url-password@127.0.0.1:9/v1?token=url-secret"

            with mock.patch.dict("os.environ", {"AMBIENT_SECRET": "ambient-secret"}), redirect_stdout(StringIO()):
                code = manage_openai_serving.main(
                    [
                        "--command",
                        command,
                        "--cwd",
                        str(root),
                        "--adapter",
                        str(adapter),
                        "--base-url",
                        base_url,
                        "--api-key",
                        "endpoint-api-secret",
                        "--env",
                        "PRIVATE_TOKEN=env-secret",
                        "--out",
                        str(out),
                        "--startup-timeout",
                        "1",
                        "--poll-interval",
                        "0.05",
                    ]
                )

            self.assertEqual(code, 1)
            public_files = [
                out / "serving_lifecycle.json",
                out / "server.stdout.log",
                out / "server.stderr.log",
            ]
            combined = "\n".join(path.read_text(encoding="utf-8") for path in public_files)
            for private_value in (
                str(root),
                "private-user",
                "url-password",
                "url-secret",
                "endpoint-api-secret",
                "argv-secret",
                "env-secret",
                "ambient-secret",
                "log-secret",
            ):
                self.assertNotIn(private_value, combined)
            lifecycle = _read_json(out / "serving_lifecycle.json")
            self.assertEqual(lifecycle["adapter"], "adapter")
            self.assertFalse(Path(lifecycle["launch"]["cwd"]).is_absolute())
            self.assertNotIn("@", lifecycle["endpoint"]["base_url"])
            self.assertNotIn("?", lifecycle["endpoint"]["base_url"])


class SourceDistributionSurfaceTests(unittest.TestCase):
    def test_manifest_ships_live_smoke_imports_and_documented_serving_scripts(self):
        manifest_lines = set((ROOT / "MANIFEST.in").read_text(encoding="utf-8").splitlines())
        required = {
            "include scripts/hermes_harness.py",
            "include scripts/check_openai_serving.py",
            "include scripts/manage_openai_serving.py",
            "include scripts/mock_openai_server.py",
            "include scripts/build_serving_demo_report.py",
        }

        self.assertEqual(required - manifest_lines, set())


def _write_scenario(path: Path) -> Path:
    payload = {
        "id": "public_surface_hardening",
        "title": "Public Surface Hardening",
        "prompt": "Complete the public surface hardening task.",
        "policy": {"max_tool_calls": 1, "max_subagents": 0},
        "assertions": {
            "required_evidence": [
                {
                    "id": "completion",
                    "type": "final_matches",
                    "contains": "public surface hardening complete",
                }
            ],
            "final_contains": ["public surface hardening complete"],
        },
        "scoring": {"pass_threshold": 90},
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
