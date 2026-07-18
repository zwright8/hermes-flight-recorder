import importlib.util
import json
import shlex
import socket
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from flightrecorder.schema_registry import check_schema_file


ROOT = Path(__file__).resolve().parents[1]
SERVING_SCRIPT = ROOT / "scripts" / "check_openai_serving.py"
EVAL_SCRIPT = ROOT / "scripts" / "evaluate_hermes_heldout.py"
DEMO_SCRIPT = ROOT / "scripts" / "build_serving_demo_report.py"
LIFECYCLE_SCRIPT = ROOT / "scripts" / "run_managed_serving_eval.py"
MOCK_SERVER_SCRIPT = ROOT / "scripts" / "mock_openai_serving.py"
SUITE_SCRIPT = ROOT / "scripts" / "verify_serving_profiles.py"
RUNTIME_PREFLIGHT_SCRIPT = ROOT / "scripts" / "preflight_serving_runtime.py"


def _load_serving_script():
    spec = importlib.util.spec_from_file_location("check_openai_serving", SERVING_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_eval_script():
    spec = importlib.util.spec_from_file_location("evaluate_hermes_heldout", EVAL_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_demo_script():
    spec = importlib.util.spec_from_file_location("build_serving_demo_report", DEMO_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_lifecycle_script():
    spec = importlib.util.spec_from_file_location("run_managed_serving_eval", LIFECYCLE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_suite_script():
    spec = importlib.util.spec_from_file_location("verify_serving_profiles", SUITE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_runtime_preflight_script():
    spec = importlib.util.spec_from_file_location("preflight_serving_runtime", RUNTIME_PREFLIGHT_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ServingDemoTests(unittest.TestCase):
    def test_mock_serving_check_writes_ready_profile_and_compatibility_report(self):
        check_openai_serving = _load_serving_script()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            adapter = root / "adapter"
            adapter.mkdir()
            (adapter / "adapter_config.json").write_text('{"r": 8, "lora_alpha": 16}\n', encoding="utf-8")
            (adapter / "adapter_model.safetensors").write_bytes(b"hfr-test-adapter")

            with redirect_stdout(StringIO()):
                code = check_openai_serving.main(
                    [
                        "--out",
                        str(root / "serving"),
                        "--model",
                        "hfr-mock-model",
                        "--arm",
                        "flightrecorder",
                        "--adapter",
                        str(adapter),
                        "--mock-response",
                        "hfr serving smoke ok",
                        "--require-tool-call",
                        "--require-structured-output",
                    ]
                )

            self.assertEqual(code, 0)
            out = root / "serving"
            profile = _read_json(out / "serving_profile.json")
            compatibility = _read_json(out / "compatibility_report.json")
            report = _read_json(out / "serving_check.json")
            mock_requests = _read_json(out / "mock_requests.json")

            self.assertTrue(report["passed"], report)
            self.assertEqual(profile["schema_version"], "hfr.serving_profile.v1")
            self.assertEqual(profile["arm"], "flightrecorder")
            self.assertEqual(profile["model_identity"]["requested_model"], "hfr-mock-model")
            self.assertEqual(profile["model_identity"]["served_model_id"], "hfr-mock-model+adapter")
            self.assertTrue(profile["model_identity"]["adapter"]["local"])
            self.assertEqual(profile["capabilities"]["tool_calls"], "supported")
            self.assertEqual(profile["capabilities"]["structured_outputs"], "supported")
            self.assertTrue(profile["eval_preflight"]["ready"])
            self.assertEqual(compatibility["schema_version"], "hfr.serving_compatibility_report.v1")
            self.assertEqual(compatibility["checks"]["tool_calls"]["tool_call_count"], 1)
            self.assertEqual(compatibility["checks"]["structured_outputs"]["parsed"]["status"], "ok")
            self.assertIn("health", {item["id"] for item in report["checks"]})
            self.assertGreaterEqual(len(mock_requests["requests"]), 6)

    def test_evaluator_accepts_ready_serving_profile_with_adapter_identity(self):
        evaluate_hermes_heldout = _load_eval_script()
        with tempfile.TemporaryDirectory() as tmp:
            profile_path = Path(tmp) / "serving_profile.json"
            _write_profile(
                profile_path,
                base_url="http://127.0.0.1:8123/v1",
                requested_model="hfr-base",
                served_model_id="hfr-base+adapter",
                ready=True,
            )

            summary = evaluate_hermes_heldout._validate_serving_profile(
                profile_path,
                expected_model="hfr-base",
                expected_base_url="http://127.0.0.1:8123/v1",
            )

            self.assertTrue(summary["ready"])
            self.assertEqual(summary["served_model_id"], "hfr-base+adapter")
            self.assertEqual(summary["capabilities"]["chat_completions"], True)

    def test_evaluator_rejects_serving_profile_for_wrong_endpoint(self):
        evaluate_hermes_heldout = _load_eval_script()
        with tempfile.TemporaryDirectory() as tmp:
            profile_path = Path(tmp) / "serving_profile.json"
            _write_profile(
                profile_path,
                base_url="http://127.0.0.1:8123/v1",
                requested_model="hfr-base",
                served_model_id="hfr-base",
                ready=True,
            )

            with self.assertRaises(SystemExit) as caught:
                evaluate_hermes_heldout._validate_serving_profile(
                    profile_path,
                    expected_model="hfr-base",
                    expected_base_url="http://127.0.0.1:9999/v1",
                )

            self.assertIn("does not match eval base_url", str(caught.exception))

    def test_demo_report_links_claims_to_replay_artifacts(self):
        build_serving_demo_report = _load_demo_script()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline_eval = _write_eval_arm(root, "baseline", "hfr-base", passed=False, score=50)
            candidate_eval = _write_eval_arm(root, "flightrecorder", "hfr-base+adapter", passed=True, score=100)
            demo_json = root / "demo_run.json"
            report_md = root / "DEMO_REPORT.md"

            with redirect_stdout(StringIO()):
                code = build_serving_demo_report.main(
                    [
                        "--arm",
                        f"baseline={baseline_eval}",
                        "--arm",
                        f"flightrecorder={candidate_eval}",
                        "--endpoint-suite",
                        str(_write_endpoint_suite(root)),
                        "--out",
                        str(demo_json),
                        "--report",
                        str(report_md),
                    ]
                )

            self.assertEqual(code, 0)
            demo = _read_json(demo_json)
            report = report_md.read_text(encoding="utf-8")
            claim_ids = {claim["id"] for claim in demo["claims"]}
            self.assertEqual(demo["schema_version"], "hfr.serving_demo_run.v1")
            self.assertTrue(demo["same_scenario_ids"])
            self.assertTrue(demo["endpoint_suite"]["passed"])
            self.assertTrue(demo["endpoint_suite"]["demo_alignment"]["passed"])
            self.assertIn("flightrecorder_repairs_demo_scenario", claim_ids)
            self.assertIn("Serving Endpoint Readiness", report)
            self.assertIn("Demo aligned: True", report)
            self.assertIn("scorecard", report)
            self.assertIn("run_digest", report)
            self.assertIn("live_observer.jsonl", report)

    def test_managed_lifecycle_starts_preflights_passes_profile_and_stops_server(self):
        run_managed_serving_eval = _load_lifecycle_script()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            port = _free_port()
            eval_script = root / "fake_eval.py"
            eval_record = root / "eval_record.json"
            _write_fake_eval(eval_script, eval_record)

            server_command = shlex.join([sys.executable, str(MOCK_SERVER_SCRIPT), "--port", str(port), "--model", "hfr-managed-model"])
            eval_command = shlex.join([sys.executable, str(eval_script), "--profile", "{serving_profile}", "--base-url", "{base_url}"])
            with redirect_stdout(StringIO()):
                code = run_managed_serving_eval.main(
                    [
                        "--server-command",
                        server_command,
                        "--base-url",
                        f"http://127.0.0.1:{port}/v1",
                        "--model",
                        "hfr-managed-model",
                        "--out",
                        str(root / "managed"),
                        "--readiness-timeout",
                        "10",
                        "--probe-interval",
                        "0.2",
                        "--require-tool-call",
                        "--require-structured-output",
                        "--eval-command",
                        eval_command,
                    ]
                )

            self.assertEqual(code, 0)
            out = root / "managed"
            lifecycle_path = out / "serving_lifecycle_run.json"
            lifecycle = _read_json(lifecycle_path)
            profile = _read_json(out / "serving_profile.json")
            eval_data = _read_json(eval_record)
            schema_result = check_schema_file(lifecycle_path)
            self.assertTrue(schema_result["passed"], schema_result["errors"])
            self.assertEqual(schema_result["schema"]["name"], "serving_lifecycle_run")
            self.assertTrue(lifecycle["passed"], lifecycle)
            self.assertEqual(lifecycle["schema_version"], "hfr.serving_lifecycle_run.v1")
            self.assertTrue(lifecycle["preflight"]["passed"])
            self.assertEqual(lifecycle["eval"]["status"], "passed")
            self.assertTrue(lifecycle["cleanup"]["terminated"] or lifecycle["cleanup"]["exit_code_after_cleanup"] is not None)
            self.assertEqual(profile["model_identity"]["served_model_id"], "hfr-managed-model")
            self.assertEqual(Path(eval_data["profile"]).resolve(), (out / "serving_profile.json").resolve())
            self.assertEqual(eval_data["base_url"], f"http://127.0.0.1:{port}/v1")
            self.assertFalse(_port_is_listening(port), "managed server should not remain listening")

    def test_engine_profiles_define_future_vllm_and_sglang_readiness(self):
        check_openai_serving = _load_serving_script()

        for engine in ("vllm", "sglang"):
            profile = check_openai_serving.ENGINE_PROFILES[engine]
            self.assertTrue(profile["openai_compatible"])
            self.assertEqual(profile["status"], "profile_ready")
            self.assertIn("chat_completion", profile["required_checks"])
            self.assertIn("{model}", profile["launch_command_template"])
            self.assertIn("{adapter}", profile["launch_command_template"])

    def test_serving_profile_suite_verifies_required_arms_identity_and_lifecycle(self):
        verify_serving_profiles = _load_suite_script()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline_profile = root / "baseline_profile.json"
            candidate_profile = root / "candidate_profile.json"
            lifecycle_path = root / "candidate_lifecycle.json"
            report_path = root / "SERVING_ENDPOINTS.md"
            suite_path = root / "serving_endpoint_suite.json"
            _write_profile(baseline_profile, arm="baseline", base_url="http://127.0.0.1:8100/v1", requested_model="hfr-base", served_model_id="hfr-base", ready=True)
            _write_profile(candidate_profile, arm="flightrecorder", base_url="http://127.0.0.1:8101/v1", requested_model="hfr-base", served_model_id="hfr-base+adapter", ready=True)
            _write_lifecycle(lifecycle_path, profile_path=candidate_profile)

            with redirect_stdout(StringIO()):
                code = verify_serving_profiles.main(
                    [
                        "--profile",
                        f"baseline={baseline_profile}",
                        "--profile",
                        f"flightrecorder={candidate_profile}",
                        "--lifecycle",
                        f"flightrecorder={lifecycle_path}",
                        "--required-arm",
                        "baseline",
                        "--required-arm",
                        "flightrecorder",
                        "--expect-model",
                        "baseline=hfr-base",
                        "--expect-model",
                        "flightrecorder=hfr-base",
                        "--expect-adapter",
                        "flightrecorder=adapter",
                        "--require-tool-call",
                        "--require-structured-output",
                        "--strict-profile-arm",
                        "--out",
                        str(suite_path),
                        "--report",
                        str(report_path),
                    ]
                )

            self.assertEqual(code, 0)
            suite = _read_json(suite_path)
            schema_result = check_schema_file(suite_path)
            self.assertTrue(schema_result["passed"], schema_result["errors"])
            self.assertEqual(schema_result["schema"]["name"], "serving_endpoint_suite")
            self.assertTrue(suite["passed"], suite)
            self.assertEqual({arm["arm"] for arm in suite["arms"]}, {"baseline", "flightrecorder"})
            self.assertIn("Serving Endpoint Suite", report_path.read_text(encoding="utf-8"))

    def test_serving_profile_suite_fails_missing_required_arm(self):
        verify_serving_profiles = _load_suite_script()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profile_path = root / "baseline_profile.json"
            suite_path = root / "serving_endpoint_suite.json"
            _write_profile(profile_path, arm="baseline", base_url="http://127.0.0.1:8100/v1", requested_model="hfr-base", served_model_id="hfr-base", ready=True)

            with redirect_stdout(StringIO()):
                code = verify_serving_profiles.main(
                    [
                        "--profile",
                        f"baseline={profile_path}",
                        "--required-arm",
                        "baseline",
                        "--required-arm",
                        "flightrecorder",
                        "--out",
                        str(suite_path),
                    ]
                )

            self.assertEqual(code, 1)
            suite = _read_json(suite_path)
            self.assertFalse(suite["passed"])
            self.assertIn("missing_required_arm:flightrecorder", suite["failed_checks"])

    def test_serving_runtime_preflight_writes_ready_artifact_for_available_dependencies(self):
        preflight_serving_runtime = _load_runtime_preflight_script()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = root / "model-cache"
            adapter = root / "trace_adapter"
            cache.mkdir()
            _write_adapter_files(adapter)
            out = root / "runtime_preflight.json"
            report = root / "RUNTIME_PREFLIGHT.md"

            with redirect_stdout(StringIO()):
                code = preflight_serving_runtime.main(
                    [
                        "--model",
                        "local/test-model",
                        "--model-cache",
                        str(cache),
                        "--adapter",
                        f"trace_only={adapter}",
                        "--required-dependency",
                        "json",
                        "--required-dependency",
                        "pathlib",
                        "--out",
                        str(out),
                        "--report",
                        str(report),
                    ]
                )

            self.assertEqual(code, 0)
            artifact = _read_json(out)
            schema_result = check_schema_file(out)
            self.assertTrue(schema_result["passed"], schema_result["errors"])
            self.assertEqual(schema_result["schema"]["name"], "serving_runtime_preflight")
            self.assertTrue(artifact["passed"], artifact)
            self.assertEqual(artifact["readiness"], "ready")
            self.assertEqual(Path(artifact["runtime"]["path"]), Path(sys.executable).absolute())
            self.assertTrue(artifact["command_refs"][0]["server_command"].startswith(shlex.quote(str(Path(sys.executable).absolute()))))
            self.assertEqual(artifact["adapters"]["trace_only"]["files"]["adapter_config.json"]["bytes"], 9)
            self.assertIn("managed eval", report.read_text(encoding="utf-8"))

    def test_serving_runtime_preflight_records_missing_dependency_without_importing_ml_stack(self):
        preflight_serving_runtime = _load_runtime_preflight_script()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = root / "model-cache"
            cache.mkdir()
            out = root / "runtime_preflight.json"

            with redirect_stdout(StringIO()):
                code = preflight_serving_runtime.main(
                    [
                        "--model-cache",
                        str(cache),
                        "--required-dependency",
                        "definitely_missing_hfr_dependency",
                        "--out",
                        str(out),
                        "--allow-blocked",
                    ]
                )

            self.assertEqual(code, 0)
            artifact = _read_json(out)
            self.assertFalse(artifact["passed"])
            self.assertEqual(artifact["readiness"], "blocked")
            self.assertIn("missing_dependency:definitely_missing_hfr_dependency", artifact["blocked_checks"])


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_profile(path: Path, *, base_url: str, requested_model: str, served_model_id: str, ready: bool, arm: str = "candidate") -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": "hfr.serving_profile.v1",
                "profile_id": "test-profile",
                "arm": arm,
                "endpoint": {"base_url": base_url},
                "model_identity": {
                    "requested_model": requested_model,
                    "served_model_id": served_model_id,
                    "metadata_model": served_model_id,
                    "chat_response_model": served_model_id,
                    "observed_model_ids": [served_model_id],
                    "adapter": {"present": "+" in served_model_id, "id": "adapter", "path": "adapter"},
                },
                "capabilities": {
                    "health": True,
                    "models": True,
                    "model_metadata": True,
                    "chat_completions": True,
                    "tool_calls": "supported",
                    "structured_outputs": "supported",
                },
                "eval_preflight": {"ready": ready, "failed_checks": [] if ready else ["chat_completion"]},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_lifecycle(path: Path, *, profile_path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": "hfr.serving_lifecycle_run.v1",
                "generated_at": "2026-07-02T00:00:00Z",
                "started_at": "2026-07-02T00:00:00Z",
                "completed_at": "2026-07-02T00:00:01Z",
                "passed": True,
                "server": {"command": ["mock"], "command_text": "mock", "cwd": str(path.parent), "pid": 123, "exit_code_after_cleanup": 143},
                "preflight": {
                    "passed": True,
                    "readiness": "ready",
                    "attempt_count": 1,
                    "attempts": [{"at": "2026-07-02T00:00:00Z", "ready": True}],
                    "failed_checks": [],
                    "serving_profile": str(profile_path),
                },
                "eval": {"status": "passed", "exit_code": 0},
                "cleanup": {"attempted": True, "terminated": True, "killed": False, "exit_code_before_cleanup": None, "exit_code_after_cleanup": 143},
                "artifacts": {"serving_profile": str(profile_path), "lifecycle_run": str(path)},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_endpoint_suite(root: Path) -> Path:
    path = root / "serving_endpoint_suite.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "hfr.serving_endpoint_suite.v1",
                "generated_at": "2026-07-02T00:00:00Z",
                "passed": True,
                "failed_checks": [],
                "requirements": {"required_arms": ["baseline", "flightrecorder"]},
                "arms": [
                    {
                        "arm": "baseline",
                        "profile_path": str(root / "baseline_profile.json"),
                        "ready_for_eval": True,
                        "failed_checks": [],
                        "checks": [],
                        "endpoint": {"base_url": "http://127.0.0.1:8100/v1"},
                        "model_identity": {"requested_model": "hfr-base", "served_model_id": "hfr-base"},
                        "capabilities": {"tool_calls": "supported", "structured_outputs": "supported"},
                        "lifecycle": {"present": False, "path": ""},
                    },
                    {
                        "arm": "flightrecorder",
                        "profile_path": str(root / "candidate_profile.json"),
                        "ready_for_eval": True,
                        "failed_checks": [],
                        "checks": [],
                        "endpoint": {"base_url": "http://127.0.0.1:8101/v1"},
                        "model_identity": {"requested_model": "hfr-base", "served_model_id": "hfr-base+adapter"},
                        "capabilities": {"tool_calls": "supported", "structured_outputs": "supported"},
                        "lifecycle": {"present": True, "path": str(root / "candidate_lifecycle.json")},
                    },
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _write_adapter_files(path: Path) -> None:
    path.mkdir(parents=True)
    (path / "adapter_config.json").write_text("{\"r\": 8}\n", encoding="utf-8")
    (path / "adapter_model.safetensors").write_bytes(b"hfr-adapter")
    (path / "tokenizer_config.json").write_text("{}\n", encoding="utf-8")
    (path / "chat_template.jinja").write_text("{{ messages }}\n", encoding="utf-8")


def _write_eval_arm(root: Path, arm: str, model: str, *, passed: bool, score: int) -> Path:
    arm_dir = root / arm
    run_dir = arm_dir / "demo_scenario"
    run_dir.mkdir(parents=True)
    for name in ("live_observer.jsonl", "scorecard.json", "report.html", "artifact_lineage.json"):
        (run_dir / name).write_text("{}\n", encoding="utf-8")
    (run_dir / "run_digest.json").write_text(
        json.dumps(
            {
                "schema_version": "hfr.run_digest.v1",
                "outcome": {
                    "passed": passed,
                    "score": score,
                    "summary": "PASS" if passed else "FAIL",
                },
                "trace_signal": {"model": model, "event_count": 4},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    suite = {
        "total": 1,
        "passed": 1 if passed else 0,
        "failed": 0 if passed else 1,
        "error_count": 0,
        "metadata": {"arm": arm, "model": model, "base_url": "http://127.0.0.1:8000/v1"},
        "metrics": {
            "pass_rate": 1.0 if passed else 0.0,
            "average_score": score,
            "critical_failure_counts": [] if passed else [{"id": "final_answer", "count": 1}],
        },
        "runs": [
            {
                "scenario_id": "demo_scenario",
                "scenario_title": "Demo Scenario",
                "task_family": "demo",
                "passed": passed,
                "score": score,
                "critical_failures": [] if passed else ["final_answer"],
                "failed_rules": [] if passed else ["final_answer"],
                "trace_path": str(run_dir / "live_observer.jsonl"),
                "scorecard": str(run_dir / "scorecard.json"),
                "run_digest": str(run_dir / "run_digest.json"),
                "report": str(run_dir / "report.html"),
                "lineage": str(run_dir / "artifact_lineage.json"),
                "run_dir": str(run_dir),
            }
        ],
    }
    suite_path = arm_dir / "suite_summary.json"
    suite_path.write_text(json.dumps(suite, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    eval_summary = {
        "schema_version": "hfr.hermes_heldout_eval_summary.v1",
        "arm": arm,
        "model": model,
        "base_url": "http://127.0.0.1:8000/v1",
        "suite_summary": str(suite_path),
        "total": 1,
        "passed": suite["passed"],
        "failed": suite["failed"],
        "pass_rate": suite["metrics"]["pass_rate"],
        "average_score": suite["metrics"]["average_score"],
        "critical_failure_total": 0 if passed else 1,
        "serving_profile": {
            "path": str(arm_dir / "serving_profile.json"),
            "profile_id": f"{arm}-profile",
        },
    }
    eval_path = arm_dir / "evaluation_summary.json"
    eval_path.write_text(json.dumps(eval_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return eval_path


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _port_is_listening(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def _write_fake_eval(path: Path, record: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "import argparse, json",
                "from pathlib import Path",
                "parser = argparse.ArgumentParser()",
                "parser.add_argument('--profile', required=True)",
                "parser.add_argument('--base-url', required=True)",
                "args = parser.parse_args()",
                "profile = Path(args.profile)",
                "data = json.loads(profile.read_text(encoding='utf-8'))",
                f"Path({str(record)!r}).write_text(json.dumps({{'profile': str(profile), 'base_url': args.base_url, 'ready': data['eval_preflight']['ready']}}, sort_keys=True) + '\\n', encoding='utf-8')",
                "raise SystemExit(0 if data['eval_preflight']['ready'] else 1)",
                "",
            ]
        ),
        encoding="utf-8",
    )
