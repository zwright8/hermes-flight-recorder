import importlib.util
import hashlib
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock

from flightrecorder.schema_registry import check_schema_contract, check_schema_file
from flightrecorder.validation import validate_artifacts


ROOT = Path(__file__).resolve().parents[1]
SERVING_SCRIPT = ROOT / "scripts" / "check_openai_serving.py"
TRANSFORMERS_SERVING_SCRIPT = ROOT / "scripts" / "serve_transformers_openai.py"
DEMO_SCRIPT = ROOT / "scripts" / "build_serving_demo_report.py"


def _load_script(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ServingDemoTests(unittest.TestCase):
    def test_transformers_prompt_renderer_passes_tools_to_chat_template(self):
        serving = _load_script(TRANSFORMERS_SERVING_SCRIPT, "serve_transformers_openai_prompt_test")

        class Tokenizer:
            def __init__(self):
                self.kwargs = {}

            def apply_chat_template(self, messages, **kwargs):
                self.kwargs = kwargs
                return "rendered"

        tokenizer = Tokenizer()
        tools = [{"type": "function", "function": {"name": "read_file", "parameters": {"type": "object"}}}]

        prompt = serving._render_chat_prompt(tokenizer, [{"role": "user", "content": "read"}], tools)

        self.assertEqual(prompt, "rendered")
        self.assertEqual(tokenizer.kwargs["tools"], tools)

    def test_committed_demo_artifacts_honor_registered_schema_claims(self):
        catalog = _read_json(ROOT / "flightrecorder" / "schemas" / "manifest.json")
        registered_versions = {item["artifact_schema_version"] for item in catalog["schemas"]}
        failures: list[str] = []
        demo_root = ROOT / "experiments" / "qwen3_4b_flightrecorder"
        for path in sorted(demo_root.rglob("*.json")):
            payload = _read_json(path)
            if payload.get("schema_version") not in registered_versions:
                continue
            result = check_schema_contract(payload)
            if result["passed"] is not True:
                failures.append(f"{path.relative_to(ROOT)}: {'; '.join(result['errors'][:4])}")
        self.assertEqual(failures, [])

    def test_mock_serving_check_writes_ready_profile_and_compatibility_report(self):
        check_openai_serving = _load_script(SERVING_SCRIPT, "check_openai_serving")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            adapter = root / "adapter"
            adapter.mkdir()
            (adapter / "adapter_config.json").write_text('{"r": 8}\n', encoding="utf-8")
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
                        "--require-streaming",
                        "--require-tool-call",
                        "--require-structured-output",
                    ]
                )

            self.assertEqual(code, 0)
            out = root / "serving"
            profile = _read_json(out / "serving_profile.json")
            compatibility = _read_json(out / "compatibility_report.json")
            report = _read_json(out / "serving_check.json")
            schema_result = check_schema_file(out / "serving_profile.json")
            self.assertTrue(schema_result["passed"], schema_result["errors"])
            self.assertEqual(schema_result["schema"]["name"], "serving_profile")
            compatibility_result = check_schema_file(out / "compatibility_report.json")
            self.assertTrue(compatibility_result["passed"], compatibility_result["errors"])
            self.assertEqual(compatibility_result["schema"]["name"], "serving_compatibility_report")
            endpoint_result = check_schema_file(out / "serving_check.json")
            self.assertTrue(endpoint_result["passed"], endpoint_result["errors"])
            self.assertEqual(endpoint_result["schema"]["name"], "serving_endpoint_check")
            self.assertTrue(report["passed"], report)
            self.assertEqual(profile["artifacts"]["serving_profile"], "serving_profile.json")
            self.assertEqual(profile["model_identity"]["served_model_id"], "hfr-mock-model+adapter")
            self.assertTrue(profile["model_identity"]["adapter"]["local"])
            self.assertTrue(profile["model_identity"]["adapter"]["immutable"])
            self.assertRegex(profile["model_identity"]["adapter"]["sha256"], r"^[0-9a-f]{64}$")
            self.assertTrue(profile["model_identity"]["adapter"]["revision"].startswith("local-"))
            self.assertEqual(
                profile["model_identity"]["adapter"]["observation_source"],
                "local_artifact_sha256",
            )
            self.assertEqual(profile["capabilities"]["streaming"], "supported")
            self.assertEqual(profile["capabilities"]["tool_calls"], "supported")
            self.assertEqual(profile["capabilities"]["structured_outputs"], "supported")
            self.assertGreaterEqual(compatibility["checks"]["streaming"]["event_count"], 1)
            self.assertTrue(compatibility["checks"]["streaming"]["done_seen"])
            self.assertEqual(compatibility["checks"]["tool_calls"]["tool_call_count"], 1)
            validation = validate_artifacts(
                serving_profile_paths=[out / "serving_profile.json"],
                serving_compatibility_report_paths=[out / "compatibility_report.json"],
                serving_endpoint_check_paths=[out / "serving_check.json"],
                strict=True,
            )
            self.assertTrue(validation["passed"], validation)
            self.assertEqual(
                [target["type"] for target in validation["targets"]],
                ["serving_profile", "serving_compatibility_report", "serving_endpoint_check"],
            )
            for artifact_name in (
                "serving_profile.json",
                "compatibility_report.json",
                "serving_check.json",
                "mock_requests.json",
            ):
                self.assertEqual(
                    (out / artifact_name).stat().st_mode & 0o777,
                    0o600,
                    artifact_name,
                )

    def test_serving_check_refuses_occupied_output_before_endpoint_use(self):
        check_openai_serving = _load_script(
            SERVING_SCRIPT,
            "check_openai_serving_create_once",
        )
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "serving"
            out.mkdir()
            sentinel = out / "sentinel.json"
            sentinel.write_text('{"preserve": true}\n', encoding="utf-8")

            with (
                mock.patch.object(
                    check_openai_serving,
                    "_start_mock_server",
                    side_effect=AssertionError("endpoint must not be contacted"),
                ),
                self.assertRaisesRegex(
                    SystemExit,
                    "serving evidence output already exists",
                ),
            ):
                check_openai_serving.main(
                    [
                        "--out",
                        str(out),
                        "--model",
                        "hfr-mock-model",
                        "--mock-response",
                        "must not run",
                    ]
                )

            self.assertEqual(
                sentinel.read_text(encoding="utf-8"),
                '{"preserve": true}\n',
            )

    def test_serving_check_rejects_symlinked_output_parent_before_endpoint_use(self):
        check_openai_serving = _load_script(
            SERVING_SCRIPT,
            "check_openai_serving_symlink_parent",
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            real_parent = root / "real-parent"
            real_parent.mkdir()
            linked_parent = root / "linked-parent"
            linked_parent.symlink_to(real_parent, target_is_directory=True)

            with (
                mock.patch.object(
                    check_openai_serving,
                    "_start_mock_server",
                    side_effect=AssertionError("endpoint must not be contacted"),
                ),
                self.assertRaisesRegex(
                    SystemExit,
                    "serving evidence output parent",
                ),
            ):
                check_openai_serving.main(
                    [
                        "--out",
                        str(linked_parent / "serving"),
                        "--model",
                        "hfr-mock-model",
                        "--mock-response",
                        "must not run",
                    ]
                )

    def test_serving_check_rejects_public_output_parent_before_endpoint_use(self):
        check_openai_serving = _load_script(
            SERVING_SCRIPT,
            "check_openai_serving_public_parent",
        )
        with tempfile.TemporaryDirectory() as tmp:
            public_parent = Path(tmp) / "public-parent"
            public_parent.mkdir()
            public_parent.chmod(0o777)

            with (
                mock.patch.object(
                    check_openai_serving,
                    "_start_mock_server",
                    side_effect=AssertionError("endpoint must not be contacted"),
                ),
                self.assertRaisesRegex(
                    SystemExit,
                    "owned by the effective user and have mode 0700",
                ),
            ):
                check_openai_serving.main(
                    [
                        "--out",
                        str(public_parent / "serving"),
                        "--model",
                        "hfr-mock-model",
                        "--mock-response",
                        "must not run",
                    ]
                )

    def test_serving_check_does_not_publish_partial_output_and_can_retry(self):
        check_openai_serving = _load_script(
            SERVING_SCRIPT,
            "check_openai_serving_atomic_publication",
        )
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "serving"
            original_write = check_openai_serving._write_json
            write_count = 0

            def fail_after_second_write(*args):
                nonlocal write_count
                write_count += 1
                original_write(*args)
                if write_count == 2:
                    raise OSError("injected publication failure")

            with (
                mock.patch.object(
                    check_openai_serving,
                    "_write_json",
                    side_effect=fail_after_second_write,
                ),
                self.assertRaisesRegex(
                    OSError,
                    "injected publication failure",
                ),
                redirect_stdout(StringIO()),
            ):
                check_openai_serving.main(
                    [
                        "--out",
                        str(out),
                        "--model",
                        "hfr-mock-model",
                        "--mock-response",
                        "hfr serving smoke ok",
                    ]
                )

            self.assertFalse(out.exists())
            self.assertEqual(
                list(Path(tmp).glob(".serving.hfr-stage-*")),
                [],
            )
            with redirect_stdout(StringIO()):
                code = check_openai_serving.main(
                    [
                        "--out",
                        str(out),
                        "--model",
                        "hfr-mock-model",
                        "--mock-response",
                        "hfr serving smoke ok",
                    ]
                )
            self.assertEqual(code, 0)
            self.assertEqual(
                list(Path(tmp).glob(".serving.hfr-stage-*")),
                [],
            )

    def test_serving_check_enforces_private_modes_under_restrictive_umask(self):
        check_openai_serving = _load_script(
            SERVING_SCRIPT,
            "check_openai_serving_private_modes",
        )
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "serving"
            previous_umask = os.umask(0o777)
            try:
                with redirect_stdout(StringIO()):
                    code = check_openai_serving.main(
                        [
                            "--out",
                            str(out),
                            "--model",
                            "hfr-mock-model",
                            "--mock-response",
                            "hfr serving smoke ok",
                        ]
                    )
            finally:
                os.umask(previous_umask)

            self.assertEqual(code, 0)
            self.assertEqual(out.stat().st_mode & 0o777, 0o700)
            for artifact in out.iterdir():
                self.assertEqual(artifact.stat().st_mode & 0o777, 0o600)

    def test_serving_check_rejects_output_swap_before_atomic_publication(self):
        check_openai_serving = _load_script(
            SERVING_SCRIPT,
            "check_openai_serving_output_swap",
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / "serving"
            original_rename = (
                check_openai_serving.atomic_rename_entry_noreplace
            )

            def insert_target_then_rename(
                parent_descriptor,
                source_name,
                target_name,
            ):
                os.mkdir(
                    target_name,
                    mode=0o700,
                    dir_fd=parent_descriptor,
                )
                target_descriptor = os.open(
                    target_name,
                    os.O_RDONLY
                    | getattr(os, "O_DIRECTORY", 0)
                    | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=parent_descriptor,
                )
                try:
                    sentinel_descriptor = os.open(
                        "sentinel.json",
                        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                        0o600,
                        dir_fd=target_descriptor,
                    )
                    with os.fdopen(
                        sentinel_descriptor,
                        "w",
                        encoding="utf-8",
                    ) as handle:
                        handle.write('{"preserve": true}\n')
                finally:
                    os.close(target_descriptor)
                return original_rename(
                    parent_descriptor,
                    source_name,
                    target_name,
                )

            with (
                mock.patch.object(
                    check_openai_serving,
                    "atomic_rename_entry_noreplace",
                    side_effect=insert_target_then_rename,
                ),
                self.assertRaisesRegex(
                    ValueError,
                    "atomic publication target changed concurrently",
                ),
                redirect_stdout(StringIO()),
            ):
                check_openai_serving.main(
                    [
                        "--out",
                        str(out),
                        "--model",
                        "hfr-mock-model",
                        "--mock-response",
                        "hfr serving smoke ok",
                    ]
                )

            self.assertEqual(
                (out / "sentinel.json").read_text(encoding="utf-8"),
                '{"preserve": true}\n',
            )
            self.assertEqual(
                list(root.glob(".serving.hfr-stage-*")),
                [],
            )

    def test_serving_check_cannot_succeed_after_parent_rebinding(self):
        check_openai_serving = _load_script(
            SERVING_SCRIPT,
            "check_openai_serving_parent_rebinding",
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parent = root / "parent"
            parent.mkdir()
            parent.chmod(0o700)
            displaced_parent = root / "displaced-parent"
            victim = root / "victim"
            victim.mkdir()
            out = parent / "serving"
            original_check = check_openai_serving.check_endpoint

            def check_then_rebind_parent(*args, **kwargs):
                result = original_check(*args, **kwargs)
                parent.rename(displaced_parent)
                parent.symlink_to(victim, target_is_directory=True)
                return result

            with (
                mock.patch.object(
                    check_openai_serving,
                    "check_endpoint",
                    side_effect=check_then_rebind_parent,
                ),
                self.assertRaisesRegex(
                    SystemExit,
                    "serving evidence output parent changed",
                ),
                redirect_stdout(StringIO()),
            ):
                check_openai_serving.main(
                    [
                        "--out",
                        str(out),
                        "--model",
                        "hfr-mock-model",
                        "--mock-response",
                        "hfr serving smoke ok",
                    ]
                )

            self.assertTrue((displaced_parent / "serving").is_dir())
            self.assertFalse((victim / "serving").exists())

    def test_serving_check_rejects_staging_source_swap_before_publication(self):
        check_openai_serving = _load_script(
            SERVING_SCRIPT,
            "check_openai_serving_source_swap",
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / "serving"
            original_require_identity = (
                check_openai_serving._require_staging_identity
            )

            def swap_staging_source(
                parent_descriptor,
                staging_name,
                expected_identity,
            ):
                displaced_name = f"{staging_name}.displaced"
                os.rename(
                    staging_name,
                    displaced_name,
                    src_dir_fd=parent_descriptor,
                    dst_dir_fd=parent_descriptor,
                )
                os.mkdir(
                    staging_name,
                    mode=0o700,
                    dir_fd=parent_descriptor,
                )
                return original_require_identity(
                    parent_descriptor,
                    staging_name,
                    expected_identity,
                )

            with (
                mock.patch.object(
                    check_openai_serving,
                    "_require_staging_identity",
                    side_effect=swap_staging_source,
                ),
                self.assertRaises(BaseExceptionGroup) as captured,
                redirect_stdout(StringIO()),
            ):
                check_openai_serving.main(
                    [
                        "--out",
                        str(out),
                        "--model",
                        "hfr-mock-model",
                        "--mock-response",
                        "hfr serving smoke ok",
                    ]
                )

            messages = [
                str(error)
                for error in captured.exception.exceptions
            ]
            self.assertTrue(
                any("changed before publication" in item for item in messages),
                messages,
            )
            self.assertTrue(
                any("changed before cleanup" in item for item in messages),
                messages,
            )
            self.assertFalse(out.exists())
            self.assertEqual(
                len(list(root.glob(".serving.hfr-stage-*"))),
                2,
            )

    def test_serving_check_preserves_primary_and_cleanup_failures(self):
        check_openai_serving = _load_script(
            SERVING_SCRIPT,
            "check_openai_serving_cleanup_failure",
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / "serving"

            def fail_with_unexpected_entry(
                directory_descriptor,
                _name,
                _payload,
            ):
                os.mkdir(
                    "unexpected-entry",
                    mode=0o700,
                    dir_fd=directory_descriptor,
                )
                raise OSError("primary publication failure")

            with (
                mock.patch.object(
                    check_openai_serving,
                    "_write_json",
                    side_effect=fail_with_unexpected_entry,
                ),
                self.assertRaises(BaseExceptionGroup) as captured,
                redirect_stdout(StringIO()),
            ):
                check_openai_serving.main(
                    [
                        "--out",
                        str(out),
                        "--model",
                        "hfr-mock-model",
                        "--mock-response",
                        "hfr serving smoke ok",
                    ]
                )

            messages = [
                str(error)
                for error in captured.exception.exceptions
            ]
            self.assertIn("primary publication failure", messages)
            self.assertTrue(
                any("unexpected entry" in item for item in messages),
                messages,
            )
            self.assertFalse(out.exists())
            retained_stages = list(root.glob(".serving.hfr-stage-*"))
            self.assertEqual(len(retained_stages), 1)
            self.assertTrue(
                (retained_stages[0] / "unexpected-entry").is_dir()
            )

    def test_remote_adapter_identity_must_be_observed_from_endpoint_metadata(self):
        check_openai_serving = _load_script(SERVING_SCRIPT, "check_openai_serving_remote_identity")
        model = "Qwen/Qwen3-4B-Instruct-2507"
        adapter_id = "example/agent-adapter"
        revision = "adapter-revision-42"
        adapter_sha256 = "a" * 64
        served_model = f"{model}+{adapter_id}"
        health = {"ok": True, "status_code": 200, "url": "http://endpoint/health", "json": {"ok": True}}
        models = {
            "ok": True,
            "status_code": 200,
            "url": "http://endpoint/v1/models",
            "json": {"data": [{"id": served_model}]},
        }
        chat = {
            "ok": True,
            "status_code": 200,
            "url": "http://endpoint/v1/chat/completions",
            "json": {"model": served_model, "choices": [{"message": {"content": "ok"}}]},
        }
        capability = {"status": "supported", "response_ok": True, "error": None}
        streaming = {**capability, "event_count": 1, "done_seen": True, "text": "ok"}
        tools = {**capability, "tool_call_count": 1, "tool_calls": [{}]}
        structured = {**capability, "json_parse_passed": True, "parsed": {}, "text": "{}"}

        def run_check(root: Path, metadata_payload: dict) -> tuple[dict, dict]:
            metadata = {
                "ok": True,
                "status_code": 200,
                "url": "http://endpoint/v1/models/model",
                "json": metadata_payload,
            }
            with (
                mock.patch.object(check_openai_serving, "_get_first_json", side_effect=[health, metadata]),
                mock.patch.object(check_openai_serving, "_request_json", side_effect=[models, chat]),
                mock.patch.object(check_openai_serving, "_tool_call_check", return_value=tools),
                mock.patch.object(check_openai_serving, "_structured_output_check", return_value=structured),
                mock.patch.object(check_openai_serving, "_streaming_check", return_value=streaming),
            ):
                profile, _compatibility, report = check_openai_serving.check_endpoint(
                    base_url="http://endpoint/v1",
                    model=model,
                    provider="custom",
                    arm="flightrecorder",
                    engine="openai_compatible",
                    adapter=adapter_id,
                    adapter_id=adapter_id,
                    adapter_revision=revision,
                    adapter_sha256=adapter_sha256,
                    api_key="",
                    timeout=1.0,
                    out_dir=root,
                    require_streaming=True,
                    require_tool_call=True,
                    require_structured_output=True,
                )
            return profile, report

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            observed_profile, observed_report = run_check(
                root,
                {
                    "id": served_model,
                    "adapter": {
                        "id": adapter_id,
                        "revision": revision,
                        "sha256": adapter_sha256,
                    },
                },
            )
            self.assertTrue(observed_report["passed"], observed_report)
            self.assertEqual(
                observed_profile["model_identity"]["adapter"]["observation_source"],
                "endpoint_model_metadata",
            )

            declared_profile, declared_report = run_check(root, {"id": served_model})
            self.assertFalse(declared_report["passed"])
            self.assertEqual(
                declared_profile["model_identity"]["adapter"]["observation_source"],
                "unobserved",
            )
            self.assertIn("adapter_identity_observed", declared_report["failed_checks"])
            self.assertIn("adapter_identity_matches_expected", declared_report["failed_checks"])

    def test_mlx_adapter_requires_matching_endpoint_metadata(self):
        check_openai_serving = _load_script(
            SERVING_SCRIPT,
            "check_openai_serving_mlx_identity",
        )
        model = "mlx-community/Qwen3.5-9B-4bit"
        health = {
            "ok": True,
            "status_code": 200,
            "url": "http://endpoint/health",
            "json": {"status": "ok"},
        }
        models = {
            "ok": True,
            "status_code": 200,
            "url": "http://endpoint/v1/models",
            "json": {"data": [{"id": model}]},
        }
        chat = {
            "ok": True,
            "status_code": 200,
            "url": "http://endpoint/v1/chat/completions",
            "json": {
                "model": model,
                "choices": [{"message": {"content": "ok"}}],
            },
        }
        capability = {"status": "supported", "response_ok": True, "error": None}
        streaming = {
            **capability,
            "event_count": 1,
            "done_seen": True,
            "text": "ok",
        }
        tools = {
            **capability,
            "tool_call_count": 1,
            "tool_calls": [{}],
        }
        structured = {
            **capability,
            "json_parse_passed": True,
            "parsed": {},
            "text": "{}",
        }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            adapter = root / "adapter"
            adapter.mkdir()
            weights = adapter / "adapters.safetensors"
            weights.write_bytes(b"hfr-mlx-adapter")
            (adapter / "adapter_config.json").write_text(
                "{}\n",
                encoding="utf-8",
            )
            sha256 = hashlib.sha256(weights.read_bytes()).hexdigest()
            revision = f"local-{sha256[:16]}"

            def run_check(metadata_adapter: dict) -> tuple[dict, dict]:
                metadata = {
                    "ok": True,
                    "status_code": 200,
                    "url": "http://endpoint/v1/models/model",
                    "json": {
                        "id": model,
                        "adapter": metadata_adapter,
                    },
                }
                with (
                    mock.patch.object(
                        check_openai_serving,
                        "_get_first_json",
                        side_effect=[health, metadata],
                    ),
                    mock.patch.object(
                        check_openai_serving,
                        "_request_json",
                        side_effect=[models, chat],
                    ),
                    mock.patch.object(
                        check_openai_serving,
                        "_tool_call_check",
                        return_value=tools,
                    ),
                    mock.patch.object(
                        check_openai_serving,
                        "_structured_output_check",
                        return_value=structured,
                    ),
                    mock.patch.object(
                        check_openai_serving,
                        "_streaming_check",
                        return_value=streaming,
                    ),
                ):
                    profile, _compatibility, report = (
                        check_openai_serving.check_endpoint(
                            base_url="http://endpoint/v1",
                            model=model,
                            provider="local",
                            arm="adapter",
                            engine="mlx",
                            adapter=str(adapter),
                            adapter_id="candidate-a",
                            adapter_revision=revision,
                            adapter_sha256=sha256,
                            api_key="",
                            timeout=1.0,
                            out_dir=root,
                            require_streaming=False,
                            require_tool_call=False,
                            require_structured_output=False,
                        )
                    )
                return profile, report

            matched_profile, matched = run_check(
                {
                    "id": "candidate-a",
                    "revision": revision,
                    "sha256": sha256,
                }
            )
            self.assertTrue(matched["passed"], matched)
            self.assertNotIn(str(root), json.dumps(matched_profile))
            self.assertNotIn(str(root), json.dumps(matched))
            matched_check = next(
                item
                for item in matched["checks"]
                if item["id"] == "mlx_adapter_endpoint_identity"
            )
            self.assertTrue(matched_check["passed"])

            _mismatched_profile, mismatched = run_check(
                {
                    "id": "candidate-a",
                    "revision": revision,
                    "sha256": "f" * 64,
                }
            )
            self.assertFalse(mismatched["passed"])
            self.assertIn(
                "mlx_adapter_endpoint_identity",
                mismatched["failed_checks"],
            )

    def test_partial_sse_stream_without_done_marker_is_not_supported(self):
        check_openai_serving = _load_script(SERVING_SCRIPT, "check_openai_serving_partial_sse")
        partial_body = (
            b'data: {"choices":[{"delta":{"content":"partial"},"finish_reason":null}]}\n\n'
        )

        class PartialResponse(io.BytesIO):
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, _exc_type, _exc, _traceback):
                self.close()

        with mock.patch.object(
            check_openai_serving,
            "open_http_response",
            return_value=PartialResponse(partial_body),
        ):
            response = check_openai_serving._request_stream(
                "http://127.0.0.1:8000/v1/chat/completions",
                payload={"model": "hfr-mock-model", "stream": True},
                api_key="",
                timeout=1.0,
            )

        self.assertFalse(response["ok"])
        self.assertEqual(response["event_count"], 1)
        self.assertFalse(response["done_seen"])
        self.assertEqual(response["text"], "partial")
        self.assertIn("[DONE]", response["error"])

    def test_streaming_capability_defensively_requires_done_marker(self):
        check_openai_serving = _load_script(SERVING_SCRIPT, "check_openai_serving_done_requirement")
        partial_response = {
            "ok": True,
            "event_count": 1,
            "done_seen": False,
            "text": "partial",
            "error": None,
        }

        with mock.patch.object(
            check_openai_serving,
            "_request_stream",
            return_value=partial_response,
        ):
            capability = check_openai_serving._streaming_check(
                "http://127.0.0.1:8000/v1",
                "hfr-mock-model",
                api_key="",
                timeout=1.0,
            )

        self.assertEqual(capability["status"], "not_verified")
        self.assertFalse(capability["response_ok"])
        self.assertFalse(capability["done_seen"])

    def test_done_terminated_sse_without_text_is_not_supported(self):
        check_openai_serving = _load_script(SERVING_SCRIPT, "check_openai_serving_empty_sse")
        empty_body = (
            b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
            b"data: [DONE]\n\n"
        )

        class EmptyResponse(io.BytesIO):
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, _exc_type, _exc, _traceback):
                self.close()

        with mock.patch.object(
            check_openai_serving,
            "open_http_response",
            return_value=EmptyResponse(empty_body),
        ):
            response = check_openai_serving._request_stream(
                "http://127.0.0.1:8000/v1/chat/completions",
                payload={"model": "hfr-mock-model", "stream": True},
                api_key="",
                timeout=1.0,
            )

        self.assertFalse(response["ok"])
        self.assertEqual(response["event_count"], 1)
        self.assertTrue(response["done_seen"])
        self.assertEqual(response["text"], "")
        self.assertIn("text content", response["error"])

    def test_validate_replays_all_serving_capability_evidence(self):
        check_openai_serving = _load_script(
            SERVING_SCRIPT, "check_openai_serving_capability_replay"
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with redirect_stdout(StringIO()):
                code = check_openai_serving.main(
                    [
                        "--out",
                        str(root / "serving"),
                        "--model",
                        "hfr-mock-model",
                        "--mock-response",
                        "hfr serving smoke ok",
                    ]
                )
            self.assertEqual(code, 0)
            original = _read_json(root / "serving" / "compatibility_report.json")

            mutations = (
                ("streaming", "done_seen", False, "done_seen"),
                ("tool_calls", "tool_call_count", 0, "tool_call_count"),
                ("structured_outputs", "json_parse_passed", False, "json_parse_passed"),
            )
            for capability, field_name, forged_value, expected_error in mutations:
                with self.subTest(capability=capability):
                    forged = json.loads(json.dumps(original))
                    forged["checks"][capability][field_name] = forged_value
                    forged_path = root / f"forged_{capability}.json"
                    _write_json(forged_path, forged)
                    schema_result = check_schema_file(forged_path)
                    self.assertTrue(schema_result["passed"], schema_result)
                    validation = validate_artifacts(
                        serving_compatibility_report_paths=[forged_path],
                        strict=True,
                    )
                    self.assertFalse(validation["passed"], validation)
                    errors = "\n".join(validation["targets"][0]["errors"])
                    self.assertIn(expected_error, errors)

    def test_validate_rejects_serving_artifacts_with_unknown_control_plane_fields(self):
        check_openai_serving = _load_script(SERVING_SCRIPT, "check_openai_serving_unknown_fields")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with redirect_stdout(StringIO()):
                code = check_openai_serving.main(
                    [
                        "--out",
                        str(root / "serving"),
                        "--model",
                        "hfr-mock-model",
                        "--arm",
                        "flightrecorder",
                        "--mock-response",
                        "hfr serving smoke ok",
                        "--require-streaming",
                        "--require-tool-call",
                        "--require-structured-output",
                    ]
                )
            self.assertEqual(code, 0)
            out = root / "serving"

            profile = _read_json(out / "serving_profile.json")
            profile["provider_console_url"] = "https://example.invalid/job"
            profile["endpoint"]["signed_url"] = "https://example.invalid/signed"
            forged_profile = root / "forged_profile.json"
            _write_json(forged_profile, profile)

            compatibility = _read_json(out / "compatibility_report.json")
            compatibility["checks"]["streaming"]["provider_call"] = {"url": "https://example.invalid/grader"}
            forged_compatibility = root / "forged_compatibility.json"
            _write_json(forged_compatibility, compatibility)

            endpoint = _read_json(out / "serving_check.json")
            endpoint["provider_endpoint"] = "https://example.invalid/endpoint"
            endpoint["checks"][0]["provider_call"] = {"url": "https://example.invalid/check"}
            forged_endpoint = root / "forged_endpoint.json"
            _write_json(forged_endpoint, endpoint)

            for path in (forged_profile, forged_compatibility, forged_endpoint):
                schema_result = check_schema_file(path)
                self.assertFalse(schema_result["passed"], schema_result)

            validation = validate_artifacts(
                serving_profile_paths=[forged_profile],
                serving_compatibility_report_paths=[forged_compatibility],
                serving_endpoint_check_paths=[forged_endpoint],
                strict=True,
            )

            self.assertFalse(validation["passed"])
            errors = "\n".join(error for target in validation["targets"] for error in target["errors"])
            self.assertIn("serving_profile contains unknown field(s): ['provider_console_url']", errors)
            self.assertIn("serving_profile.endpoint contains unknown field(s): ['signed_url']", errors)
            self.assertIn(
                "serving_compatibility_report.checks.streaming contains unknown field(s): ['provider_call']",
                errors,
            )
            self.assertIn("serving_endpoint_check contains unknown field(s): ['provider_endpoint']", errors)
            self.assertIn("serving_endpoint_check.checks[0] contains unknown field(s): ['provider_call']", errors)

    def test_demo_report_links_claims_to_replay_artifacts(self):
        build_serving_demo_report = _load_script(DEMO_SCRIPT, "build_serving_demo_report")
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
                        "--out",
                        str(demo_json),
                        "--report",
                        str(report_md),
                    ]
                )

            self.assertEqual(code, 0)
            demo = _read_json(demo_json)
            schema_result = check_schema_file(demo_json)
            self.assertTrue(schema_result["passed"], schema_result["errors"])
            self.assertEqual(schema_result["schema"]["name"], "serving_demo_run")
            report = report_md.read_text(encoding="utf-8")
            claim_ids = {claim["id"] for claim in demo["claims"]}
            comparison = demo["comparisons"][0]
            self.assertTrue(demo["same_scenario_ids"])
            self.assertEqual(comparison["reference_arm"], "baseline")
            self.assertEqual(comparison["metric_deltas"]["pass_rate"], 1)
            self.assertEqual(comparison["metric_deltas"]["failed"], -1)
            self.assertEqual(comparison["scenario_outcomes"][0]["outcome"], "candidate_repaired")
            self.assertIn("flightrecorder_repairs_demo_scenario", claim_ids)
            self.assertFalse(Path(demo["arms"][0]["source"]).is_absolute())
            self.assertEqual(demo["arms"][0]["serving_profile"], "baseline/serving_profile.json")
            self.assertFalse(Path(demo["scenarios"][0]["arms"]["baseline"]["trace_path"]).is_absolute())
            self.assertIn("## Base Vs Candidate Comparisons", report)
            self.assertIn("candidate_repaired", report)
            self.assertIn("[serving_profile](baseline/serving_profile.json)", report)
            self.assertIn("scorecard", report)
            self.assertIn("run_digest", report)
            self.assertIn("live_observer.jsonl", report)
            validation = validate_artifacts(serving_demo_run_paths=[demo_json], strict=True)
            self.assertTrue(validation["passed"], validation)

    def test_validate_rejects_demo_run_with_unknown_control_plane_fields(self):
        build_serving_demo_report = _load_script(DEMO_SCRIPT, "build_serving_demo_report_unknown_fields")
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
                        "--out",
                        str(demo_json),
                        "--report",
                        str(report_md),
                    ]
                )
            self.assertEqual(code, 0)
            demo = _read_json(demo_json)
            demo["automation_thread_ref"] = "redacted-thread"
            _write_json(demo_json, demo)

            schema_result = check_schema_file(demo_json)
            self.assertFalse(schema_result["passed"], schema_result)
            validation = validate_artifacts(serving_demo_run_paths=[demo_json], strict=True)

            self.assertFalse(validation["passed"])
            errors = "\n".join(error for target in validation["targets"] for error in target["errors"])
            self.assertIn("serving_demo_run contains unknown field(s): ['automation_thread_ref']", errors)

    def test_validate_rejects_demo_run_with_inconsistent_arm_metrics(self):
        build_serving_demo_report = _load_script(DEMO_SCRIPT, "build_serving_demo_report_metrics_validation")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline_eval = _write_eval_arm(root, "baseline", "hfr-base", passed=False, score=50)
            candidate_eval = _write_eval_arm(root, "flightrecorder", "hfr-base+adapter", passed=False, score=20)
            demo_json = root / "demo_run.json"
            report_md = root / "DEMO_REPORT.md"
            with redirect_stdout(StringIO()):
                code = build_serving_demo_report.main(
                    [
                        "--arm",
                        f"baseline={baseline_eval}",
                        "--arm",
                        f"flightrecorder={candidate_eval}",
                        "--out",
                        str(demo_json),
                        "--report",
                        str(report_md),
                    ]
                )
            self.assertEqual(code, 0)
            demo = _read_json(demo_json)
            candidate = next(arm for arm in demo["arms"] if arm["name"] == "flightrecorder")
            candidate["metrics"].update(
                {
                    "total": 1,
                    "passed": 1,
                    "failed": 0,
                    "pass_rate": 1.0,
                    "average_score": 100,
                    "critical_failure_total": 0,
                }
            )
            demo_json.write_text(json.dumps(demo, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            validation = validate_artifacts(serving_demo_run_paths=[demo_json], strict=True)

            self.assertFalse(validation["passed"])
            errors = "\n".join(error for target in validation["targets"] for error in target["errors"])
            self.assertIn("serving_demo_run.arms[1].metrics.passed expected 0", errors)
            self.assertIn("serving_demo_run.arms[1].metrics.pass_rate expected 0.0", errors)

    def test_validate_rejects_inconsistent_serving_endpoint_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            check_path = root / "serving_check.json"
            check_path.write_text(
                json.dumps(
                    {
                        "schema_version": "hfr.serving_endpoint_check.v1",
                        "generated_at": "2026-01-01T00:00:00Z",
                        "passed": True,
                        "readiness": "ready",
                        "profile_id": "candidate-mock",
                        "arm": "candidate",
                        "model": "hfr-mock-model",
                        "served_model_id": "hfr-mock-model",
                        "base_url": "http://127.0.0.1:8000/v1",
                        "checks": [{"id": "chat_completion", "passed": False, "details": {}}],
                        "failed_checks": [],
                        "artifacts": {},
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )

            validation = validate_artifacts(serving_endpoint_check_paths=[check_path], strict=True)

            self.assertFalse(validation["passed"])
            errors = "\n".join(error for target in validation["targets"] for error in target["errors"])
            self.assertIn("serving_endpoint_check.failed_checks expected ['chat_completion']", errors)

    def test_validate_rejects_demo_run_with_unknown_claim_arm(self):
        build_serving_demo_report = _load_script(DEMO_SCRIPT, "build_serving_demo_report_for_validation")
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
                        "--out",
                        str(demo_json),
                        "--report",
                        str(report_md),
                    ]
                )
            self.assertEqual(code, 0)
            demo = _read_json(demo_json)
            demo["claims"][0]["evidence"][0]["arm"] = "ghost"
            demo_json.write_text(json.dumps(demo, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            validation = validate_artifacts(serving_demo_run_paths=[demo_json], strict=True)

            self.assertFalse(validation["passed"])
            errors = "\n".join(error for target in validation["targets"] for error in target["errors"])
            self.assertIn("serving_demo_run.claims[0].evidence[0].arm must match a demo arm", errors)


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: dict):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_eval_arm(root: Path, arm: str, model: str, *, passed: bool, score: int) -> Path:
    arm_dir = root / arm
    run_dir = arm_dir / "demo_scenario"
    run_dir.mkdir(parents=True)
    for name in ("live_observer.jsonl", "scorecard.json", "report.html", "artifact_lineage.json"):
        (run_dir / name).write_text("{}\n", encoding="utf-8")
    (arm_dir / "serving_profile.json").write_text('{"schema_version": "hfr.serving_profile.v1"}\n', encoding="utf-8")
    (run_dir / "run_digest.json").write_text(
        json.dumps({"schema_version": "hfr.serving_demo.run_digest.v1", "outcome": {"passed": passed, "score": score, "summary": "PASS" if passed else "FAIL"}, "trace_signal": {"model": model, "event_count": 4}}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    suite = {
        "total": 1,
        "passed": 1 if passed else 0,
        "failed": 0 if passed else 1,
        "metadata": {"arm": arm, "model": model, "base_url": "http://127.0.0.1:8000/v1", "serving_profile": "serving_profile.json"},
        "metrics": {"pass_rate": 1.0 if passed else 0.0, "average_score": score, "critical_failure_counts": [] if passed else [{"id": "final_answer", "count": 1}]},
        "runs": [
            {
                "scenario_id": "demo_scenario",
                "scenario_title": "Demo Scenario",
                "task_family": "demo",
                "passed": passed,
                "score": score,
                "critical_failures": [] if passed else ["final_answer"],
                "trace_path": str(run_dir / "live_observer.jsonl"),
                "scorecard": str(run_dir / "scorecard.json"),
                "run_digest": str(run_dir / "run_digest.json"),
                "report": str(run_dir / "report.html"),
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
        "serving_profile": "serving_profile.json",
        "suite_summary": str(suite_path),
        "total": 1,
        "passed": suite["passed"],
        "failed": suite["failed"],
        "pass_rate": suite["metrics"]["pass_rate"],
        "average_score": suite["metrics"]["average_score"],
        "critical_failure_total": 0 if passed else 1,
    }
    eval_path = arm_dir / "evaluation_summary.json"
    eval_path.write_text(json.dumps(eval_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return eval_path
