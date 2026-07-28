#!/usr/bin/env python3
"""Serve one local MLX model through mlx_lm's OpenAI-compatible server.

The wrapper deliberately exposes only one governed model alias.  MLX and
``mlx_lm.server`` are imported only after all local paths and adapter identity
claims have been validated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


DEFAULT_MODEL_KEY = "default_model"
_MUTABLE_REVISIONS = {"head", "latest", "main", "master"}
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_MACOS_SYSTEM_ROOT_SYMLINKS = {
    Path("/etc"): Path("/private/etc"),
    Path("/tmp"): Path("/private/tmp"),
    Path("/var"): Path("/private/var"),
}


class ServingConfigurationError(ValueError):
    """Raised when a local serving configuration cannot be proven safe."""


@dataclass(frozen=True)
class AdapterIdentity:
    present: bool
    adapter_id: str
    revision: str
    sha256: str
    path: Path | None

    def metadata(self) -> dict[str, Any]:
        return {
            "present": self.present,
            "id": self.adapter_id,
            "revision": self.revision,
            "sha256": self.sha256,
            "local": self.present,
            "immutable": self.present,
        }


@dataclass(frozen=True)
class ServingIdentity:
    model_path: Path
    served_model_name: str
    adapter: AdapterIdentity


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        allow_abbrev=False,
    )
    parser.add_argument("--model", required=True, help="Existing local MLX model directory.")
    parser.add_argument("--served-model-name", required=True, help="OpenAI model alias to expose.")
    parser.add_argument("--adapter-path", default="", help="Existing local MLX adapter directory.")
    parser.add_argument("--adapter-id", default="", help="Stable adapter id.")
    parser.add_argument("--adapter-revision", default="", help="Immutable adapter revision.")
    parser.add_argument("--adapter-sha256", default="", help="Expected adapters.safetensors SHA-256.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--allowed-origins", type=_comma_separated, default=["*"])
    parser.add_argument("--num-draft-tokens", type=int, default=3)
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], default="INFO")
    parser.add_argument("--chat-template", default="")
    parser.add_argument("--use-default-chat-template", action="store_true")
    parser.add_argument("--temp", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--chat-template-args", type=_json_object, default={})
    parser.add_argument("--decode-concurrency", type=int, default=32)
    parser.add_argument("--prompt-concurrency", type=int, default=8)
    parser.add_argument("--prefill-step-size", type=int, default=2048)
    parser.add_argument("--prompt-cache-size", type=int, default=10)
    parser.add_argument("--prompt-cache-bytes", type=_parse_size)
    parser.add_argument("--pipeline", action="store_true")
    return parser.parse_args(argv)


def canonical_loopback_host(value: str) -> str:
    host = str(value).strip("[]").lower()
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise ServingConfigurationError(
            "--host must be a loopback address"
        )
    return host


def validate_configuration(args: argparse.Namespace) -> ServingIdentity:
    canonical_loopback_host(args.host)
    alias = str(args.served_model_name)
    if not alias or alias == DEFAULT_MODEL_KEY:
        raise ServingConfigurationError(
            f"--served-model-name must be non-empty and differ from {DEFAULT_MODEL_KEY!r}"
        )
    if len(alias) > 256 or any(character.isspace() or ord(character) < 32 for character in alias):
        raise ServingConfigurationError("--served-model-name contains unsafe characters")
    if "?" in alias or "#" in alias:
        raise ServingConfigurationError("--served-model-name must not contain URL query or fragment delimiters")

    model_path = _verified_local_directory(Path(args.model), "model")
    adapter_path_value = str(args.adapter_path or "")
    identity_fields = (
        str(args.adapter_id or ""),
        str(args.adapter_revision or ""),
        str(args.adapter_sha256 or ""),
    )
    if not adapter_path_value:
        if any(identity_fields):
            raise ServingConfigurationError(
                "adapter identity arguments require --adapter-path"
            )
        adapter = AdapterIdentity(False, "", "", "", None)
    else:
        adapter_path = _verified_local_directory(Path(adapter_path_value), "adapter")
        adapter = _adapter_identity(
            adapter_path,
            adapter_id=identity_fields[0],
            revision=identity_fields[1],
            expected_sha256=identity_fields[2],
        )

    _require_positive_int(args.port, "--port")
    if int(args.port) > 65535:
        raise ServingConfigurationError("--port must be at most 65535")
    for name in (
        "num_draft_tokens",
        "max_tokens",
        "decode_concurrency",
        "prompt_concurrency",
        "prefill_step_size",
        "prompt_cache_size",
    ):
        _require_nonnegative_int(getattr(args, name), f"--{name.replace('_', '-')}")
    if args.decode_concurrency == 0 or args.prompt_concurrency == 0 or args.prefill_step_size == 0:
        raise ServingConfigurationError("concurrency and prefill step values must be positive")
    if not 0 <= float(args.top_p) <= 1 or not 0 <= float(args.min_p) <= 1:
        raise ServingConfigurationError("--top-p and --min-p must be between 0 and 1")
    if float(args.temp) < 0:
        raise ServingConfigurationError("--temp must be non-negative")

    return ServingIdentity(model_path, alias, adapter)


def bind_model_aliases(
    model_map: Mapping[str, str | None],
    adapter_map: Mapping[str, str | None],
    draft_model_map: Mapping[str, str | None],
    *,
    model_path: str,
    served_model_name: str,
    adapter_path: str | None,
) -> tuple[dict[str, str | None], dict[str, str | None], dict[str, str | None]]:
    """Return provider maps that bind both request keys after MLX map rewriting.

    ``mlx_lm.server.ModelProvider.load`` looks up ``model_path`` first, then
    looks up the adapter using that already-rewritten path.  Binding the exact
    resolved model path in ``adapter_map`` is therefore essential; alias-only
    adapter entries are insufficient.  Inputs are not mutated.
    """

    del model_map, adapter_map, draft_model_map
    request_keys = (DEFAULT_MODEL_KEY, served_model_name, model_path)
    models: dict[str, str | None] = {
        key: model_path for key in request_keys
    }
    adapters: dict[str, str | None] = {
        key: adapter_path for key in request_keys
    }
    drafts: dict[str, str | None] = {
        key: None for key in request_keys
    }
    return models, adapters, drafts


def bind_provider(provider: Any, identity: ServingIdentity) -> Any:
    adapter_path = str(identity.adapter.path) if identity.adapter.path is not None else None
    models, adapters, drafts = bind_model_aliases(
        provider._model_map,
        provider._adapter_map,
        provider._draft_model_map,
        model_path=str(identity.model_path),
        served_model_name=identity.served_model_name,
        adapter_path=adapter_path,
    )
    provider._model_map.clear()
    provider._model_map.update(models)
    provider._adapter_map.clear()
    provider._adapter_map.update(adapters)
    provider._draft_model_map.clear()
    provider._draft_model_map.update(drafts)
    return provider


def build_model_metadata(identity: ServingIdentity) -> dict[str, Any]:
    adapter = identity.adapter.metadata()
    return {
        "id": identity.served_model_name,
        "model": identity.served_model_name,
        "object": "model",
        "owned_by": "local",
        "adapter_identity": adapter,
        "metadata": {
            "engine": "mlx_lm",
            "local": True,
            "adapter_identity": adapter,
        },
    }


def make_handler(base_handler: type[Any], identity: ServingIdentity) -> type[Any]:
    metadata = build_model_metadata(identity)
    alias = identity.served_model_name
    configured_adapter = str(identity.adapter.path) if identity.adapter.path is not None else None

    class GovernedMLXHandler(base_handler):
        def validate_model_parameters(self) -> None:
            super().validate_model_parameters()
            if self.requested_model != alias:
                raise ValueError(f"model must be {alias!r}")
            if self.requested_draft_model not in {DEFAULT_MODEL_KEY, alias}:
                raise ValueError("external draft models are disabled")
            if self.adapter is not None:
                requested_adapter = str(Path(self.adapter).expanduser().absolute())
                if configured_adapter is None or requested_adapter != configured_adapter:
                    raise ValueError("request adapter does not match the configured local adapter")

        def handle_models_request(self) -> None:
            path = urllib.parse.urlsplit(self.path).path.rstrip("/")
            collection_path = "/v1/models"
            requested_alias = ""
            if path.startswith(f"{collection_path}/"):
                requested_alias = urllib.parse.unquote(path[len(collection_path) + 1 :])
            if path == collection_path:
                self._send_governed_json({"object": "list", "data": [metadata]})
            elif requested_alias == alias:
                self._send_governed_json(metadata)
            else:
                self._send_governed_json(
                    {"error": {"message": f"Unknown model: {requested_alias}"}},
                    status=404,
                )

        def _send_governed_json(self, payload: dict[str, Any], *, status: int = 200) -> None:
            encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
            self._set_completion_headers(status)
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)
            self.wfile.flush()

    GovernedMLXHandler.__name__ = "GovernedMLXHandler"
    return GovernedMLXHandler


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        identity = validate_configuration(args)
    except (OSError, ServingConfigurationError) as exc:
        raise SystemExit(f"unsafe MLX serving configuration: {exc}") from exc

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    args.host = canonical_loopback_host(args.host)
    args.model = str(identity.model_path)
    args.adapter_path = str(identity.adapter.path) if identity.adapter.path is not None else None
    args.draft_model = None
    args.trust_remote_code = False

    try:
        import mlx.core as mx
        from mlx_lm import server as mlx_server
    except ImportError as exc:
        raise SystemExit(
            "mlx_lm is required to serve this model; install the repository's MLX serving environment"
        ) from exc

    if mx.metal.is_available():
        mx.set_wired_limit(
            mx.device_info()["max_recommended_working_set_size"]
        )
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    provider = bind_provider(mlx_server.ModelProvider(args), identity)
    handler = make_handler(mlx_server.APIHandler, identity)
    print(
        json.dumps(
            {
                "event": "hfr_mlx_openai_server_starting",
                "host": args.host,
                "port": args.port,
                "model": identity.served_model_name,
                "adapter_identity": identity.adapter.metadata(),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    mlx_server.run(
        args.host,
        args.port,
        provider,
        handler_class=handler,
    )
    return 0


def _adapter_identity(
    adapter_path: Path,
    *,
    adapter_id: str,
    revision: str,
    expected_sha256: str,
) -> AdapterIdentity:
    weights_path = adapter_path / "adapters.safetensors"
    if weights_path.is_symlink() or not weights_path.is_file():
        raise ServingConfigurationError(
            f"adapter must contain a regular adapters.safetensors file: {adapter_path}"
        )
    computed_sha256 = _sha256_file(weights_path)
    expected_sha256 = expected_sha256.lower()
    if expected_sha256 and not _SHA256_RE.fullmatch(expected_sha256):
        raise ServingConfigurationError("--adapter-sha256 must be 64 lowercase hexadecimal characters")
    if expected_sha256 and expected_sha256 != computed_sha256:
        raise ServingConfigurationError(
            "provided adapter SHA-256 does not match adapters.safetensors"
        )

    config = _read_adapter_config(adapter_path)
    observed_sha256 = _first_string(config, "adapter_sha256", "weights_sha256", "sha256").lower()
    if observed_sha256 and observed_sha256 != computed_sha256:
        raise ServingConfigurationError(
            "adapter_config.json SHA-256 does not match adapters.safetensors"
        )
    observed_revision = _first_string(config, "adapter_revision", "revision")
    if revision.lower() in _MUTABLE_REVISIONS:
        raise ServingConfigurationError(
            "provided adapter revision must be immutable, not a mutable alias"
        )
    if revision and observed_revision and revision != observed_revision:
        raise ServingConfigurationError(
            "provided adapter revision does not match adapter_config.json"
        )
    effective_revision = revision or observed_revision
    if not effective_revision or effective_revision.lower() in _MUTABLE_REVISIONS:
        effective_revision = f"local-{computed_sha256[:16]}"

    observed_id = _first_string(config, "adapter_id", "id")
    if adapter_id and observed_id and adapter_id != observed_id:
        raise ServingConfigurationError(
            "provided adapter id does not match adapter_config.json"
        )
    effective_id = adapter_id or observed_id or adapter_path.name
    if not effective_id or any(character.isspace() for character in effective_id):
        raise ServingConfigurationError("adapter id must be a non-empty identifier without whitespace")
    return AdapterIdentity(
        True,
        effective_id,
        effective_revision,
        computed_sha256,
        adapter_path,
    )


def _read_adapter_config(adapter_path: Path) -> dict[str, Any]:
    config_path = adapter_path / "adapter_config.json"
    if not config_path.exists():
        return {}
    if config_path.is_symlink() or not config_path.is_file():
        raise ServingConfigurationError("adapter_config.json must be a regular file")
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ServingConfigurationError(f"could not read adapter_config.json: {exc}") from exc
    if not isinstance(payload, dict):
        raise ServingConfigurationError("adapter_config.json must contain a JSON object")
    return payload


def _verified_local_directory(path: Path, label: str) -> Path:
    expanded = path.expanduser()
    if "\x00" in os.fspath(expanded):
        raise ServingConfigurationError(f"{label} path contains a null byte")
    if ".." in expanded.parts:
        raise ServingConfigurationError(f"{label} path must not contain parent traversal")
    absolute = expanded.absolute()
    if _has_symlink_component(absolute):
        raise ServingConfigurationError(f"{label} path must not contain symlink components: {path}")
    try:
        resolved = absolute.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ServingConfigurationError(f"{label} path does not exist: {path}") from exc
    if not resolved.is_dir():
        raise ServingConfigurationError(f"{label} path must be a directory: {path}")
    return resolved


def _has_symlink_component(path: Path) -> bool:
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        try:
            if current.is_symlink():
                allowed_target = _MACOS_SYSTEM_ROOT_SYMLINKS.get(current)
                if allowed_target is not None and current.resolve(strict=True) == allowed_target:
                    current = allowed_target
                    continue
                return True
        except OSError:
            return True
    return False


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _first_string(payload: Mapping[str, Any], *keys: str) -> str:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


def _comma_separated(value: str) -> list[str]:
    return value.split(",")


def _json_object(value: str) -> dict[str, Any]:
    try:
        payload = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if not isinstance(payload, dict):
        raise argparse.ArgumentTypeError("must be a JSON object")
    return payload


def _parse_size(value: str) -> int:
    match = re.fullmatch(r"([0-9]+)([KMGTP]?B?)?", value.strip().upper())
    if match is None:
        raise argparse.ArgumentTypeError("must be an integer byte count optionally followed by K, M, G, T, or P")
    number = int(match.group(1))
    suffix = (match.group(2) or "").removesuffix("B")
    power = {"": 0, "K": 1, "M": 2, "G": 3, "T": 4, "P": 5}[suffix]
    return number * (1024**power)


def _require_positive_int(value: Any, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ServingConfigurationError(f"{name} must be a positive integer")


def _require_nonnegative_int(value: Any, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ServingConfigurationError(f"{name} must be a non-negative integer")


if __name__ == "__main__":
    raise SystemExit(main())
