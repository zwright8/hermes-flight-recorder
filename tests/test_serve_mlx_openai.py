from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from types import SimpleNamespace


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "serve_mlx_openai.py"
SPEC = importlib.util.spec_from_file_location("serve_mlx_openai", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
serve_mlx_openai = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = serve_mlx_openai
SPEC.loader.exec_module(serve_mlx_openai)


class ServeMlxOpenAITests(unittest.TestCase):
    def test_parser_requires_local_model_and_alias_without_importing_mlx(self) -> None:
        before = set(sys.modules)
        args = serve_mlx_openai.parse_args(
            [
                "--model",
                "local-model",
                "--served-model-name",
                "candidate",
                "--allowed-origins",
                "http://localhost,https://example.test",
                "--prompt-cache-bytes",
                "2M",
            ]
        )
        self.assertEqual(args.model, "local-model")
        self.assertEqual(args.served_model_name, "candidate")
        self.assertEqual(
            args.allowed_origins,
            ["http://localhost", "https://example.test"],
        )
        self.assertEqual(args.prompt_cache_bytes, 2 * 1024 * 1024)
        self.assertFalse(any(name.startswith("mlx_lm") for name in set(sys.modules) - before))
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            serve_mlx_openai.parse_args(
                [
                    "--model",
                    "local-model",
                    "--served-model-name",
                    "candidate",
                    "--adapter-p",
                    "attacker-adapter",
                ]
            )

    def test_validates_local_paths_and_hashes_plural_adapters_weights(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model = root / "model"
            adapter = root / "adapter"
            model.mkdir()
            adapter.mkdir()
            weights = b"plural MLX adapter weights"
            (adapter / "adapters.safetensors").write_bytes(weights)
            expected_hash = hashlib.sha256(weights).hexdigest()
            (adapter / "adapter_config.json").write_text(
                json.dumps(
                    {
                        "adapter_id": "tau3-candidate",
                        "adapter_revision": "revision-7",
                        "adapter_sha256": expected_hash,
                    }
                ),
                encoding="utf-8",
            )

            identity = serve_mlx_openai.validate_configuration(
                _args(
                    model,
                    adapter_path=adapter,
                    adapter_id="tau3-candidate",
                    adapter_revision="revision-7",
                    adapter_sha256=expected_hash,
                )
            )

            self.assertEqual(identity.model_path, model.resolve())
            self.assertEqual(identity.adapter.path, adapter.resolve())
            self.assertEqual(identity.adapter.sha256, expected_hash)
            self.assertTrue(identity.adapter.metadata()["immutable"])

    def test_binding_survives_upstream_model_then_adapter_map_order(self) -> None:
        model = "/private/models/base"
        adapter = "/private/adapters/candidate"
        model_map, adapter_map, draft_map = serve_mlx_openai.bind_model_aliases(
            {"default_model": "wrong"},
            {"default_model": None},
            {"default_model": "draft"},
            model_path=model,
            served_model_name="candidate",
            adapter_path=adapter,
        )

        for request_name in ("default_model", "candidate"):
            resolved_model = model_map.get(request_name, request_name)
            resolved_adapter = adapter_map.get(resolved_model)
            self.assertEqual(resolved_model, model)
            self.assertEqual(resolved_adapter, adapter)
            self.assertIsNone(draft_map[request_name])

    def test_metadata_exposes_exact_alias_and_immutable_adapter_identity(self) -> None:
        adapter = serve_mlx_openai.AdapterIdentity(
            present=True,
            adapter_id="candidate-v1",
            revision="commit-abc123",
            sha256="a" * 64,
            path=Path("/private/adapters/candidate"),
        )
        identity = serve_mlx_openai.ServingIdentity(
            model_path=Path("/private/models/base"),
            served_model_name="flightrecorder-candidate",
            adapter=adapter,
        )

        metadata = serve_mlx_openai.build_model_metadata(identity)

        self.assertEqual(metadata["id"], "flightrecorder-candidate")
        self.assertEqual(metadata["model"], "flightrecorder-candidate")
        self.assertEqual(metadata["adapter_identity"]["id"], "candidate-v1")
        self.assertEqual(metadata["adapter_identity"]["revision"], "commit-abc123")
        self.assertEqual(metadata["adapter_identity"]["sha256"], "a" * 64)
        self.assertTrue(metadata["adapter_identity"]["local"])
        self.assertTrue(metadata["adapter_identity"]["immutable"])
        self.assertNotIn("model_path", metadata["metadata"])

    def test_handler_accepts_only_governed_alias_and_configured_adapter(self) -> None:
        class BaseHandler:
            def validate_model_parameters(self) -> None:
                self.base_validation_called = True

        adapter_path = Path("/private/adapters/candidate")
        identity = serve_mlx_openai.ServingIdentity(
            model_path=Path("/private/models/base"),
            served_model_name="mlx-community/Qwen3.5-9B-4bit",
            adapter=serve_mlx_openai.AdapterIdentity(
                present=True,
                adapter_id="candidate-a",
                revision="local-0123456789abcdef",
                sha256="a" * 64,
                path=adapter_path,
            ),
        )
        handler_type = serve_mlx_openai.make_handler(BaseHandler, identity)
        handler = handler_type()
        handler.requested_model = identity.served_model_name
        handler.requested_draft_model = serve_mlx_openai.DEFAULT_MODEL_KEY
        handler.adapter = None

        handler.validate_model_parameters()

        self.assertTrue(handler.base_validation_called)
        handler.requested_model = serve_mlx_openai.DEFAULT_MODEL_KEY
        with self.assertRaisesRegex(ValueError, "model must be"):
            handler.validate_model_parameters()
        handler.requested_model = identity.served_model_name
        handler.adapter = "/private/adapters/base-only"
        with self.assertRaisesRegex(ValueError, "does not match"):
            handler.validate_model_parameters()

    def test_hash_and_revision_mismatches_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model = root / "model"
            adapter = root / "adapter"
            model.mkdir()
            adapter.mkdir()
            (adapter / "adapters.safetensors").write_bytes(b"weights")
            (adapter / "adapter_config.json").write_text(
                json.dumps({"adapter_revision": "revision-good"}),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                serve_mlx_openai.ServingConfigurationError,
                "SHA-256 does not match",
            ):
                serve_mlx_openai.validate_configuration(
                    _args(model, adapter_path=adapter, adapter_sha256="0" * 64)
                )
            with self.assertRaisesRegex(
                serve_mlx_openai.ServingConfigurationError,
                "revision does not match",
            ):
                serve_mlx_openai.validate_configuration(
                    _args(
                        model,
                        adapter_path=adapter,
                        adapter_revision="revision-wrong",
                    )
                )
            with self.assertRaisesRegex(
                serve_mlx_openai.ServingConfigurationError,
                "must be immutable",
            ):
                serve_mlx_openai.validate_configuration(
                    _args(model, adapter_path=adapter, adapter_revision="main")
                )

    def test_missing_and_symlinked_paths_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model = root / "model"
            model.mkdir()
            with self.assertRaisesRegex(
                serve_mlx_openai.ServingConfigurationError,
                "does not exist",
            ):
                serve_mlx_openai.validate_configuration(_args(root / "missing"))

            model_link = root / "model-link"
            model_link.symlink_to(model, target_is_directory=True)
            with self.assertRaisesRegex(
                serve_mlx_openai.ServingConfigurationError,
                "symlink",
            ):
                serve_mlx_openai.validate_configuration(_args(model_link))

            adapter = root / "adapter"
            adapter.mkdir()
            weights = root / "weights"
            weights.write_bytes(b"weights")
            (adapter / "adapters.safetensors").symlink_to(weights)
            with self.assertRaisesRegex(
                serve_mlx_openai.ServingConfigurationError,
                "regular adapters.safetensors",
            ):
                serve_mlx_openai.validate_configuration(
                    _args(model, adapter_path=adapter)
                )

    def test_host_is_restricted_to_canonical_loopback(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model = Path(temp_dir) / "model"
            model.mkdir()
            exposed = _args(model)
            exposed.host = "0.0.0.0"
            with self.assertRaisesRegex(
                serve_mlx_openai.ServingConfigurationError,
                "loopback",
            ):
                serve_mlx_openai.validate_configuration(exposed)

            bracketed = _args(model)
            bracketed.host = "[::1]"
            serve_mlx_openai.validate_configuration(bracketed)
            self.assertEqual(
                serve_mlx_openai.canonical_loopback_host(bracketed.host),
                "::1",
            )


def _args(
    model: Path,
    *,
    adapter_path: Path | str = "",
    adapter_id: str = "",
    adapter_revision: str = "",
    adapter_sha256: str = "",
) -> SimpleNamespace:
    return SimpleNamespace(
        model=str(model),
        served_model_name="candidate",
        adapter_path=str(adapter_path),
        adapter_id=adapter_id,
        adapter_revision=adapter_revision,
        adapter_sha256=adapter_sha256,
        host="127.0.0.1",
        port=8080,
        num_draft_tokens=3,
        max_tokens=512,
        decode_concurrency=32,
        prompt_concurrency=8,
        prefill_step_size=2048,
        prompt_cache_size=10,
        top_p=1.0,
        min_p=0.0,
        temp=0.0,
    )


if __name__ == "__main__":
    unittest.main()
