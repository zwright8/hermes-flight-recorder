from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import textwrap
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

from flightrecorder.schema_registry import check_schema_contract
from flightrecorder.tau3_mlx_training import (
    Tau3MlxTrainingConfig,
    Tau3MlxTrainingError,
    _config_within_bounds,
    _prefix_equivalence_sample_membership,
    _policy_complete_manifest_replays,
    _scan_mlx_data_dir,
    _write_telemetry,
    main as tau3_mlx_training_main,
    run_tau3_mlx_training,
)
from flightrecorder.tau3_exposure import build_tau3_exposure_ledger
from flightrecorder.mlx_prefix_cache_lora import (
    PrefixCacheTrainingError,
    _validation_indices,
    split_supervised_tokens,
)
from flightrecorder.tau3_model_identity import build_tau3_model_identity
from tests.test_tau3_training_artifacts import _base_bundle, _rewrite_manifest, _write_json, _write_jsonl
from tests.test_tau3_competitive_dataset import (
    _FakeTokenizer as _V3FakeTokenizer,
    _install_fake_transformers as _install_v3_fake_transformers,
    _grounded_validation_patch as _v3_grounded_validation_patch,
    _write_contamination_report as _write_v3_contamination_report,
    _write_grounded_bundle as _write_v3_grounded_bundle,
    _write_source_dataset as _write_v3_source_dataset,
    _write_tokenizer_config as _write_v3_tokenizer_config,
)
from tests.test_tau3_prefix_equivalence import equivalence_fixture
from flightrecorder.tau3_prefix_equivalence import (
    _target_accounting_from_rows,
)
from flightrecorder.tau3_competitive_dataset import build_tau3_competitive_dataset


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _mutate_json(path: Path, mutate) -> None:
    payload = _read_json(path)
    mutate(payload)
    _write_json(path, payload)


def _install_fake_python(root: Path, mode: str) -> None:
    python = root / ".venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    script = f"""#!/usr/bin/env python3
import os, sys, time
mode = {mode!r}
adapter = sys.argv[sys.argv.index('--adapter-path') + 1] if '--adapter-path' in sys.argv else None
print('MLX_DISABLE_COMPILE=' + os.environ.get('MLX_DISABLE_COMPILE', '<unset>'))
print('train loss 1.25')
print('validation loss 0.75', file=sys.stderr)
if mode == 'sleep':
    time.sleep(2)
elif mode == 'crash':
    print('simulated crash', file=sys.stderr)
    sys.exit(9)
elif mode == 'oom':
    print('out of memory while allocating adapter', file=sys.stderr)
    sys.exit(1)
elif mode == 'no_output':
    sys.exit(0)
elif mode == 'mlx_progress':
    print('Starting training..., iters: 100')
    print('Iter 1: Train loss 1.234, It/sec 0.1')
    print('Iter 1: Val loss 0.442, Val took 59.258s', file=sys.stderr)
    print('Training took 100s')
    os.makedirs(adapter, exist_ok=True)
    open(os.path.join(adapter, 'adapter_config.json'), 'w').write('{{"r": 16}}\\n')
    open(os.path.join(adapter, 'adapters.safetensors'), 'wb').write(b'fake-adapter')
else:
    os.makedirs(adapter, exist_ok=True)
    open(os.path.join(adapter, 'adapter_config.json'), 'w').write('{{"r": 16}}\\n')
    open(os.path.join(adapter, 'adapters.safetensors'), 'wb').write(b'fake-adapter')
    os.makedirs(os.path.join(adapter, 'checkpoint-0001'), exist_ok=True)
    open(os.path.join(adapter, 'checkpoint-0001', 'weights.npz'), 'wb').write(b'fake-checkpoint')
"""
    python.write_text(script, encoding="utf-8")
    python.chmod(0o755)


def _runner_bundle(root: Path, *, attestation: bool = True) -> Path:
    bundle = _base_bundle(root)
    input_export = bundle / "training" / "input_export"
    input_export.mkdir()
    rows = [
        {"messages": [{"role": "user", "content": "Fix itinerary"}, {"role": "assistant", "content": "I updated the itinerary."}]},
        {"messages": [{"role": "user", "content": "Check order"}, {"role": "assistant", "content": "The order is ready."}]},
    ]
    _write_jsonl(input_export / "train.jsonl", [rows[0]])
    _write_jsonl(input_export / "valid.jsonl", [rows[1]])
    files = {}
    for name in ("train", "valid"):
        path = input_export / f"{name}.jsonl"
        files[name] = {"path": path.name, "size": path.stat().st_size, "sha256": _sha256(path)}
    _write_json(
        input_export / "mlx_dataset_manifest.json",
        {
            "schema_version": "hfr.tau3_mlx_dataset.v1",
            "passed": True,
            "counts": {"train": 1, "valid": 1},
            "files": files,
            "sealed_rows": 0,
            "test_file_present": False,
            "training_started": False,
        },
    )
    dataset_manifest = _read_json(bundle / "training" / "dataset_manifest.json")
    dataset_manifest.update(
        {
            "local_only": True,
            "views": {
                "mlx_train": {"path": "input_export/train.jsonl", "row_count": 1, "schema_version": "mlx.chat_jsonl.v1"},
                "mlx_valid": {"path": "input_export/valid.jsonl", "row_count": 1, "schema_version": "mlx.chat_jsonl.v1"},
            },
            "mlx_dataset_manifest": {
                "path": "input_export/mlx_dataset_manifest.json",
                "sha256": _sha256(input_export / "mlx_dataset_manifest.json"),
                "sealed_rows": 0,
            },
        }
    )
    if attestation:
        dataset_manifest["training_target_quality"] = {
            "schema_version": "hfr.tau3_training_target_quality.v1",
            "passed": True,
            "evaluation_criteria_exposure": False,
            "exact_match_count": 0,
            "substantial_exposure_count": 0,
        }
    _write_json(bundle / "training" / "dataset_manifest.json", dataset_manifest)
    _write_json(
        bundle / "training" / "model_manifest.json",
        {
            "schema_version": "hfr.tau3.model_manifest.v1",
            "model_id": "local-dense-8b",
            "base_model": "local-dense-8b",
            "revision": "base-rev",
            "local_only": True,
        },
    )
    _rewrite_manifest(bundle)
    return bundle


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fake_model(root: Path) -> tuple[Path, Path]:
    model = root / "models" / "base"
    model.mkdir(parents=True)
    (model / "config.json").write_text('{"model_type":"fake"}\n', encoding="utf-8")
    (model / "tokenizer.json").write_text('{"version":"1.0"}\n', encoding="utf-8")
    (model / "model.safetensors").write_bytes(b"fake-weights")
    identity = build_tau3_model_identity(model, model_id="fake/base-9b", revision="1234567890abcdef")
    identity_path = root / "identity.json"
    _write_json(identity_path, identity)
    return model, identity_path


def _sync_v3_tokenizer_assets_to_model(
    model: Path,
    identity_path: Path,
    tokenizer_config_path: Path,
) -> None:
    tokenizer_config = _read_json(tokenizer_config_path)
    tokenizer_dir = Path(str(tokenizer_config["tokenizer_path"]))
    for filename in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        source = tokenizer_dir / filename
        if source.is_file():
            shutil.copy2(source, model / filename)
    identity = build_tau3_model_identity(
        model,
        model_id="fake/base-9b",
        revision="1234567890abcdef",
    )
    _write_json(identity_path, identity)


def _protocol_config(root: Path, identity_path: Path) -> Path:
    identity = _read_json(identity_path)
    protocol = {
        "schema_version": "hfr.tau3_protocol_config.v1",
        "protocol_manifest": {
            "schema_version": "hfr.tau3_protocol_manifest.v1",
            "title": "Tau-3 MLX mixture training fixture",
            "signature": "a" * 64,
        },
        "tau_revision": {"schema_version": "hfr.tau3_revision.v1", "revision": "tau-fixture-rev"},
        "split_manifest": {"schema_version": "hfr.tau3_split_manifest.v1", "splits": {}},
        "harness_contract": {"schema_version": "hfr.tau3_harness_contract.v1", "fixed": True, "no_test_time_search": True},
        "model_freeze": {
            "schema_version": "hfr.tau3_model_freeze.v1",
            "base_model": {
                "name": identity["model_id"],
                "revision": identity["revision"],
                "local_identity_sha256": _sha256(identity_path),
                "local_tree_sha256": identity["tree_sha256"],
                "quantization": "4-bit",
            },
        },
        "budget": {"schema_version": "hfr.tau3_budget.v1", "max_seconds": 604800, "local_only": True, "network": False},
        "sealed_manifest": {"schema_version": "hfr.tau3_sealed_manifest.v1", "quarantine_predates_generation": True},
        "mlx_qlora_plan": {
            "schema_version": "hfr.tau3_mlx_qlora_plan.v1",
            "passed": True,
            "backend": "mlx-lm",
            "method": "4-bit QLoRA LoRA adapter",
            "quantization": "4-bit",
            "local_only": True,
            "network": False,
            "output_contract": {"adapter_only": True, "base_revision_required": True, "quantization_identity_required": True},
            "command_argv": ["python", "-m", "mlx_lm", "lora", "--fine-tune-type", "lora"],
        },
        "recipe_space": {
            "schema_version": "hfr.tau3_recipe_space.v1",
            "bounded": True,
            "max_trials": 12,
            "sealed_used": False,
            "development_only": True,
            "bounds": {
                "rank": [8, 16, 32],
                "alpha": [16, 20, 32],
                "learning_rate": [0.000001, 0.0002],
                "sequence_length": [2048, 8192],
                "steps": [1, 2000],
            },
        },
        "candidate_selection_contract": {
            "schema_version": "hfr.tau3_candidate_selection.v1",
            "passed": True,
            "development_only": True,
            "sealed_used": False,
        },
        "contamination_attestation": {"passed": True},
        "redaction_attestation": {"passed": True},
        "licenses": [
            {"id": "base", "training_allowed": True},
            {"id": "data", "training_allowed": True},
            {"id": "tools", "training_allowed": True},
            {"id": "harness", "training_allowed": True},
        ],
        "environment_manifest": {"schema_version": "hfr.tau3_environment.v1", "network_allowed": False},
    }
    path = root / "protocol.json"
    _write_json(path, protocol)
    return path


def _mixture_variant(root: Path, *, protocol_path: Path | None = None, contaminated: bool = False) -> Path:
    source = root / "clean_source"
    source.mkdir()
    train_source = {
        "messages": [{"role": "user", "content": "Book travel"}, {"role": "assistant", "content": "I can help with that."}],
        "metadata": {"episode_id": "airline-train-family1-task1", "task_family": "family1", "source_fingerprint_status": "verified"},
        "tools": [],
    }
    valid_source = {
        "messages": [{"role": "user", "content": "Check order"}, {"role": "assistant", "content": "The order is ready."}],
        "metadata": {"episode_id": "retail-development-family2-task1", "task_family": "family2", "source_fingerprint_status": "verified"},
        "tools": [],
    }
    _write_jsonl(source / "train.jsonl", [train_source])
    _write_jsonl(source / "valid.jsonl", [valid_source])
    protocol_sha256 = _sha256(protocol_path) if protocol_path is not None else None
    source_manifest = {
        "schema_version": "hfr.tau3_conversation_import.v1",
        "passed": True,
        "files": {
            "train": {"path": "train.jsonl", "sha256": _sha256(source / "train.jsonl")},
            "valid": {"path": "valid.jsonl", "sha256": _sha256(source / "valid.jsonl")},
        },
        "generation_provenance": {"protocol_sha256": protocol_sha256},
    }
    _write_json(source / "manifest.json", source_manifest)
    variant = root / "mixtures" / "assistant_turn_targets"
    variant.mkdir(parents=True)
    train_row: dict[str, Any] = {"messages": train_source["messages"], "metadata": {"source_row_sha256": _sha256(source / "train.jsonl")}}
    if contaminated:
        train_row["metadata"]["invented_tau_tool"] = True
    valid_row = {"messages": valid_source["messages"], "metadata": {"source_row_sha256": _sha256(source / "valid.jsonl")}}
    _write_jsonl(variant / "train.jsonl", [train_row])
    _write_jsonl(variant / "valid.jsonl", [valid_row])
    _write_json(
        variant / "manifest.json",
        {
            "schema_version": "hfr.tau3_training_mixture.v1",
            "variant": "assistant_turn_targets",
            "format": "mlx-chat-jsonl",
            "passed": True,
            "source_binding": {
                "source_dir": str(source),
                "source_manifest": {"path": "manifest.json", "sha256": _sha256(source / "manifest.json")},
                "train": {"path": "train.jsonl", "sha256": _sha256(source / "train.jsonl")},
                "valid": {"path": "valid.jsonl", "sha256": _sha256(source / "valid.jsonl")},
            },
            "counts": {"train": 1, "valid": 1},
            "files": {
                "train": {"path": "train.jsonl", "size": (variant / "train.jsonl").stat().st_size, "sha256": _sha256(variant / "train.jsonl")},
                "valid": {"path": "valid.jsonl", "size": (variant / "valid.jsonl").stat().st_size, "sha256": _sha256(variant / "valid.jsonl")},
            },
            "tokenizer": {
                "checked": True,
                "method": "pinned_base_apply_chat_template",
                "row_count": 2,
                "max_rendered_tokens": 32,
                "max_seq_length": 8192,
                "harness_context_window": 8192,
                "over_max_seq_length_count": 0,
                "over_context_window_count": 0,
                "chat_template_sha256": "0" * 64,
            },
            "sealed_rows": 0,
            "test_rows": 0,
            "training_started": False,
        },
    )
    if protocol_path is not None:
        _mutate_json(
            variant / "manifest.json",
            lambda payload: payload["source_binding"].__setitem__("protocol_sha256", protocol_sha256),
        )
    return variant


def _exposure_row(index: int) -> dict[str, Any]:
    domains = ("airline", "retail", "telecom")
    behaviors = (
        "successful_completion",
        "clarification_refusal",
        "authentication",
        "confirmation_before_mutation",
        "later_task_completion_actions",
        "safe_stopping",
        "transfer_handoff",
        "empty_result_recovery",
        "error_result_recovery",
        "repeated_call_recovery",
        "hallucinated_tool_correction",
        "harmful_mutation_correction",
        "premature_completion_correction",
    )
    domain = domains[index % len(domains)]
    behavior = behaviors[index % len(behaviors)]
    return {
        "messages": [
            {"role": "user", "content": f"{domain} task {index}"},
            {"role": "assistant", "content": f"Handled {behavior}."},
        ],
        "metadata": {
            "schema_version": "hfr.tau3_competitive_dataset_row.v1",
            "domain": domain,
            "behavior": behavior,
            "target_tool_name": f"{domain}_tool_{index % 4}",
            "target_action_class": "tool_call",
            "preceding_result_class": "success" if index % 3 else "recoverable_error",
            "length_bucket": "short",
            "source_family_id": f"family_{index % 8}",
            "source_provenance": {"method": "synthetic_governed_fixture"},
            "token_counts": {
                "method": "pinned_local_apply_chat_template",
                "exact": True,
                "chat_template_aware": True,
                "prompt_tokens": 10 + (index % 128),
                "supervised_tokens": 4 + (index % 5),
            },
        },
        "tools": [],
    }


def _exposure_mixture_variant(
    root: Path,
    protocol_path: Path,
    *,
    row_count: int = 52,
    epochs: int = 2,
    batch_size: int = 2,
    gradient_accumulation_steps: int = 2,
    name: str = "exposure",
) -> tuple[Path, Path, Path]:
    variant = _mixture_variant(root, protocol_path=protocol_path)
    rows = [_exposure_row(index) for index in range(row_count)]
    _write_jsonl(variant / "train.jsonl", rows)
    _write_jsonl(variant / "valid.jsonl", rows[:3])
    manifest = _read_json(variant / "manifest.json")
    manifest["counts"] = {"train": len(rows), "valid": 3}
    manifest["files"] = {
        "train": {
            "path": "train.jsonl",
            "size": (variant / "train.jsonl").stat().st_size,
            "sha256": _sha256(variant / "train.jsonl"),
        },
        "valid": {
            "path": "valid.jsonl",
            "size": (variant / "valid.jsonl").stat().st_size,
            "sha256": _sha256(variant / "valid.jsonl"),
        },
    }
    _write_json(variant / "manifest.json", manifest)
    exposure_dir = root / name
    receipt = build_tau3_exposure_ledger(
        variant / "train.jsonl",
        exposure_dir,
        seed=101,
        epochs=epochs,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )
    return (
        variant,
        Path(receipt["receipt_path"]),
        exposure_dir / "training_exposure_ledger.jsonl",
    )


def _write_prefix_equivalence(
    root: Path,
    *,
    mixture: Path,
    protocol: Path,
    identity: Path,
    config: Tau3MlxTrainingConfig,
) -> Path:
    artifact = equivalence_fixture()
    bindings = artifact["bindings"]
    bindings["dataset_file_sha256"] = _sha256(mixture / "train.jsonl")
    bindings["protocol_file_sha256"] = _sha256(protocol)
    bindings["model_identity_file_sha256"] = _sha256(identity)
    bindings["recipe"] = {
        "rank": config.rank,
        "scale": config.scale,
        "learning_rate": config.learning_rate,
        "num_layers": config.num_layers,
        "max_seq_length": config.max_seq_length,
        "batch_size": config.batch_size,
        "grad_accumulation": config.grad_accumulation,
        "mask_prompt": config.mask_prompt,
        "allowed_seeds": [config.seed],
    }
    candidate_rows = [
        json.loads(line)
        for line in (mixture / "train.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    target_rows = artifact["target_accounting"]["full_gradient"]["rows"]
    sample_hashes = [
        hashlib.sha256(
            json.dumps(
                row,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")
        ).hexdigest()
        for row in candidate_rows[: len(target_rows)]
    ]
    for arm in ("full_gradient", "detached_prefix"):
        rows = artifact["target_accounting"][arm]["rows"]
        for row, row_sha256 in zip(rows, sample_hashes, strict=True):
            row["row_sha256"] = row_sha256
        artifact["target_accounting"][arm] = (
            _target_accounting_from_rows(rows)
        )
    artifact["sample"]["row_hashes"] = sample_hashes
    artifact["sample"]["row_count"] = len(sample_hashes)
    artifact["sample"]["row_hashes_sha256"] = hashlib.sha256(
        json.dumps(
            sample_hashes,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    path = root / "prefix-equivalence.json"
    _write_json(path, artifact)
    return path


def _refresh_protocol_signature(bundle: Path) -> None:
    import hashlib

    protocol_path = bundle / "protocol" / "protocol_manifest.json"
    protocol = _read_json(protocol_path)
    protocol.pop("signature", None)
    payload = {
        "protocol_manifest": protocol,
        "tau_revision": _read_json(bundle / "protocol" / "tau_revision.json"),
        "split_manifest": _read_json(bundle / "protocol" / "split_manifest.json"),
        "harness_contract": _read_json(bundle / "protocol" / "harness_contract.json"),
        "model_freeze": _read_json(bundle / "protocol" / "model_freeze.json"),
        "budget": _read_json(bundle / "protocol" / "budget.json"),
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()
    protocol["signature"] = digest
    _write_json(protocol_path, protocol)


class Tau3MlxTrainingRunnerTests(unittest.TestCase):
    def test_policy_complete_scan_allows_only_masked_failed_invented_tool(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp)
            tool = {
                "type": "function",
                "function": {
                    "name": "update_record",
                    "description": "Update a fixture record.",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
            for split in ("train", "valid"):
                row = {
                    "messages": [
                        {"role": "system", "content": "Exact policy."},
                        {"role": "user", "content": "Please update my record."},
                        {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "bad-call",
                                    "type": "function",
                                    "function": {
                                        "name": "invented_tau_tool",
                                        "arguments": {},
                                    },
                                }
                            ],
                        },
                        {
                            "role": "tool",
                            "name": "invented_tau_tool",
                            "tool_call_id": "bad-call",
                            "content": "{\"error\":\"unknown_tool\"}",
                        },
                        {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "safe-call",
                                    "type": "function",
                                    "function": {
                                        "name": "update_record",
                                        "arguments": {},
                                    },
                                }
                            ],
                        },
                    ],
                    "tools": [tool],
                    "metadata": {
                        "schema_version": "hfr.tau3_policy_complete_row.v1",
                        "split": split,
                        "mask_prompt_required": True,
                        "negative_prefix": True,
                    },
                }
                _write_jsonl(data / f"{split}.jsonl", [row])

            passed = _scan_mlx_data_dir(data, policy_complete=True)
            self.assertTrue(passed["passed"], passed)

            train = json.loads(
                (data / "train.jsonl").read_text(encoding="utf-8")
            )
            train["messages"][-1]["tool_calls"][0]["function"][
                "name"
            ] = "invented_tau_tool"
            _write_jsonl(data / "train.jsonl", [train])
            failed = _scan_mlx_data_dir(data, policy_complete=True)
            self.assertFalse(failed["passed"])
            self.assertTrue(
                any(
                    "supervised target" in finding["reason"]
                    for finding in failed["findings"]
                )
            )

    def test_policy_complete_manifest_seal_binds_protocol_and_fail_closed_gates(
        self,
    ) -> None:
        protocol_sha256 = "a" * 64
        manifest = {
            "schema_version": "hfr.tau3_policy_complete_dataset.v1",
            "parent_protocol": {"sha256": protocol_sha256},
            "coverage": {"passed": True},
            "contamination": {"passed": True},
            "balance": {"passed": True},
            "supervision": {
                "mask_prompt_required": True,
                "negative_actions_are_context_only": True,
            },
            "sealed": {"access_count": 0, "payload_accessed": False},
            "training_started": False,
        }
        manifest["manifest_sha256"] = hashlib.sha256(
            json.dumps(
                manifest,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")
        ).hexdigest()

        self.assertTrue(
            _policy_complete_manifest_replays(
                manifest,
                protocol_sha256=protocol_sha256,
            )
        )
        manifest["coverage"]["passed"] = False
        self.assertFalse(
            _policy_complete_manifest_replays(
                manifest,
                protocol_sha256=protocol_sha256,
            )
        )

    def test_compile_mode_is_explicit_and_overrides_parent_environment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _install_fake_python(root, "success")
            bundle = _runner_bundle(root)
            with mock.patch.dict(os.environ, {"MLX_DISABLE_COMPILE": "unexpected"}):
                for disabled, expected in ((False, "<unset>"), (True, "1")):
                    receipt = run_tau3_mlx_training(
                        bundle_dir=bundle,
                        output_dir=root / f"out-{disabled}",
                        workspace_root=root,
                        config=Tau3MlxTrainingConfig(
                            iters=2,
                            disable_compile=disabled,
                            timeout_seconds=5,
                        ),
                    )
                    telemetry = (root / f"out-{disabled}" / "telemetry.jsonl").read_text(encoding="utf-8")
                    self.assertIn(f"MLX_DISABLE_COMPILE={expected}", telemetry)
                    self.assertIs(receipt["config"]["disable_compile"], disabled)

    def test_fixed_shape_padding_uses_governed_mlx_driver_and_is_receipted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _install_fake_python(root, "success")
            model, identity = _fake_model(root)
            protocol = _protocol_config(root, identity)
            receipt = run_tau3_mlx_training(
                mixture_dir=_mixture_variant(root, protocol_path=protocol),
                protocol_path=protocol,
                model_path=model,
                model_identity_path=identity,
                output_dir=root / "out",
                workspace_root=root,
                config=Tau3MlxTrainingConfig(
                    iters=2,
                    fixed_shape_padding=True,
                    timeout_seconds=5,
                ),
            )

            command = receipt["command"]
            self.assertIn("flightrecorder.mlx_fixed_shape_lora", command)
            self.assertNotIn("mlx_lm", command)
            self.assertTrue(receipt["config"]["fixed_shape_padding"])
            self.assertTrue(
                receipt["training_binding"]["recipe"]["fixed_shape_padding"]
            )
            self.assertTrue(
                _read_json(root / "out" / "mlx_lora_config.json")[
                    "fixed_shape_padding"
                ]
            )

    def test_prefix_cache_training_uses_governed_driver_and_is_receipted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _install_fake_python(root, "success")
            model, identity = _fake_model(root)
            protocol = _protocol_config(root, identity)
            receipt = run_tau3_mlx_training(
                mixture_dir=_mixture_variant(root, protocol_path=protocol),
                protocol_path=protocol,
                model_path=model,
                model_identity_path=identity,
                output_dir=root / "out",
                workspace_root=root,
                config=Tau3MlxTrainingConfig(
                    iters=2,
                    grad_checkpoint=False,
                    disable_compile=True,
                    prefix_cache_training=True,
                    timeout_seconds=5,
                ),
            )

            self.assertIn("flightrecorder.mlx_prefix_cache_lora", receipt["command"])
            self.assertNotIn("mlx_lm", receipt["command"])
            self.assertTrue(receipt["config"]["prefix_cache_training"])
            self.assertTrue(
                receipt["training_binding"]["recipe"]["prefix_cache_training"]
            )
            self.assertTrue(
                _read_json(root / "out" / "mlx_lora_config.json")[
                    "prefix_cache_training"
                ]
            )

    def test_prefix_cache_training_requires_memory_safe_recipe_controls(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _install_fake_python(root, "success")
            bundle = _runner_bundle(root)
            invalid = {
                "compile": Tau3MlxTrainingConfig(
                    prefix_cache_training=True,
                    grad_checkpoint=False,
                ),
                "checkpoint": Tau3MlxTrainingConfig(
                    prefix_cache_training=True,
                    disable_compile=True,
                ),
                "accumulation": Tau3MlxTrainingConfig(
                    prefix_cache_training=True,
                    grad_checkpoint=False,
                    disable_compile=True,
                    grad_accumulation=2,
                ),
                "fixed-padding": Tau3MlxTrainingConfig(
                    prefix_cache_training=True,
                    grad_checkpoint=False,
                    disable_compile=True,
                    fixed_shape_padding=True,
                ),
            }
            for label, config in invalid.items():
                with self.subTest(label=label), self.assertRaisesRegex(
                    Tau3MlxTrainingError,
                    "prelaunch checks failed",
                ):
                    run_tau3_mlx_training(
                        bundle_dir=bundle,
                        output_dir=root / f"out-{label}",
                        workspace_root=root,
                        config=config,
                    )

    def test_exposure_ledger_training_uses_governed_full_gradient_driver_and_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _install_fake_python(root, "success")
            model, identity = _fake_model(root)
            protocol = _protocol_config(root, identity)
            mixture, exposure_receipt, exposure_ledger = _exposure_mixture_variant(
                root,
                protocol,
            )

            receipt = run_tau3_mlx_training(
                mixture_dir=mixture,
                protocol_path=protocol,
                model_path=model,
                model_identity_path=identity,
                output_dir=root / "out",
                workspace_root=root,
                config=Tau3MlxTrainingConfig(
                    iters=52,
                    batch_size=2,
                    grad_accumulation=2,
                    exposure_ledger_training=True,
                    timeout_seconds=5,
                ),
                exposure_dataset_path=mixture / "train.jsonl",
                exposure_receipt_path=exposure_receipt,
                exposure_ledger_path=exposure_ledger,
            )

            self.assertIn("flightrecorder.mlx_exposure_lora", receipt["command"])
            self.assertNotIn("mlx_lm", receipt["command"])
            self.assertIn("--exposure-dataset", receipt["command"])
            self.assertTrue(receipt["config"]["exposure_ledger_training"])
            self.assertTrue(
                receipt["training_binding"]["recipe"]["exposure_ledger_training"]
            )
            self.assertTrue(
                receipt["training_binding"]["recipe"]["full_gradient_objective"]
            )
            self.assertEqual(receipt["training_binding"]["recipe"]["optimizer_steps"], 26)
            self.assertEqual(receipt["training_binding"]["recipe"]["microbatch_iterations"], 52)
            exposure = receipt["training_binding"]["exposure"]
            self.assertEqual(exposure["receipt"]["sha256"], _sha256(exposure_receipt))
            self.assertEqual(exposure["ledger"]["sha256"], _sha256(exposure_ledger))
            self.assertEqual(exposure["ledger"]["optimizer_steps"], 26)
            self.assertEqual(exposure["ledger"]["microbatch_iterations"], 52)
            self.assertEqual(receipt["config"]["exposure_schedule"]["optimizer_steps"], 26)
            self.assertEqual(receipt["config"]["exposure_schedule"]["microbatch_iterations"], 52)
            self.assertTrue(exposure["objective"]["iterator_patch_only"])
            self.assertTrue(exposure["objective"]["full_gradient"])
            cfg = _read_json(root / "out" / "mlx_lora_config.json")
            self.assertTrue(cfg["exposure_ledger_training"])
            self.assertEqual(cfg["exposure"]["ledger"]["optimizer_steps"], 26)
            self.assertEqual(cfg["exposure"]["ledger"]["microbatch_iterations"], 52)

    def test_exposure_prefix_training_requires_passing_bound_equivalence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _install_fake_python(root, "success")
            model, identity = _fake_model(root)
            protocol = _protocol_config(root, identity)
            mixture, exposure_receipt, exposure_ledger = _exposure_mixture_variant(
                root,
                protocol,
                batch_size=1,
                gradient_accumulation_steps=4,
            )
            config = Tau3MlxTrainingConfig(
                iters=104,
                batch_size=1,
                grad_accumulation=4,
                grad_checkpoint=False,
                disable_compile=True,
                prefix_cache_training=True,
                exposure_ledger_training=True,
                timeout_seconds=5,
            )

            with self.assertRaisesRegex(
                Tau3MlxTrainingError,
                "prefix equivalence",
            ):
                run_tau3_mlx_training(
                    mixture_dir=mixture,
                    protocol_path=protocol,
                    model_path=model,
                    model_identity_path=identity,
                    output_dir=root / "out",
                    workspace_root=root,
                    config=config,
                    exposure_dataset_path=mixture / "train.jsonl",
                    exposure_receipt_path=exposure_receipt,
                    exposure_ledger_path=exposure_ledger,
                )

    def test_exposure_prefix_training_uses_combined_driver_and_binds_equivalence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _install_fake_python(root, "success")
            model, identity = _fake_model(root)
            protocol = _protocol_config(root, identity)
            mixture, exposure_receipt, exposure_ledger = _exposure_mixture_variant(
                root,
                protocol,
                batch_size=1,
                gradient_accumulation_steps=4,
            )
            config = Tau3MlxTrainingConfig(
                iters=104,
                batch_size=1,
                grad_accumulation=4,
                grad_checkpoint=False,
                disable_compile=True,
                prefix_cache_training=True,
                exposure_ledger_training=True,
                timeout_seconds=5,
            )
            equivalence = _write_prefix_equivalence(
                root,
                mixture=mixture,
                protocol=protocol,
                identity=identity,
                config=config,
            )

            receipt = run_tau3_mlx_training(
                mixture_dir=mixture,
                protocol_path=protocol,
                model_path=model,
                model_identity_path=identity,
                output_dir=root / "out",
                workspace_root=root,
                config=config,
                exposure_dataset_path=mixture / "train.jsonl",
                exposure_receipt_path=exposure_receipt,
                exposure_ledger_path=exposure_ledger,
                prefix_equivalence_path=equivalence,
            )

            self.assertIn(
                "flightrecorder.mlx_exposure_prefix_cache_lora",
                receipt["command"],
            )
            self.assertIn("--prefix-equivalence", receipt["command"])
            recipe = receipt["training_binding"]["recipe"]
            self.assertFalse(recipe["full_gradient_objective"])
            self.assertTrue(recipe["prefix_equivalence_passed"])
            exposure = receipt["training_binding"]["exposure"]
            self.assertFalse(exposure["objective"]["full_gradient"])
            self.assertTrue(exposure["objective"]["detached_prefix"])
            binding = receipt["training_binding"]["prefix_equivalence"]
            self.assertEqual(binding["sha256"], _sha256(equivalence))
            self.assertTrue(binding["validation_passed"])
            self.assertTrue(binding["sample_membership"]["passed"])

    def test_prefix_equivalence_membership_rejects_non_candidate_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "train.jsonl"
            _write_jsonl(dataset, [{"messages": [], "metadata": {}}])
            artifact = equivalence_fixture()
            path = root / "equivalence.json"
            _write_json(path, artifact)

            result = _prefix_equivalence_sample_membership(path, dataset)

            self.assertFalse(result["passed"])
            self.assertEqual(
                len(result["missing_row_hashes"]),
                artifact["sample"]["row_count"],
            )

    def test_exposure_prefix_training_rejects_tampered_binding(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _install_fake_python(root, "success")
            model, identity = _fake_model(root)
            protocol = _protocol_config(root, identity)
            mixture, exposure_receipt, exposure_ledger = _exposure_mixture_variant(
                root,
                protocol,
                batch_size=1,
                gradient_accumulation_steps=4,
            )
            config = Tau3MlxTrainingConfig(
                iters=104,
                batch_size=1,
                grad_accumulation=4,
                grad_checkpoint=False,
                disable_compile=True,
                prefix_cache_training=True,
                exposure_ledger_training=True,
                timeout_seconds=5,
            )
            equivalence = _write_prefix_equivalence(
                root,
                mixture=mixture,
                protocol=protocol,
                identity=identity,
                config=config,
            )
            _mutate_json(
                equivalence,
                lambda payload: payload["bindings"].__setitem__(
                    "dataset_file_sha256",
                    "f" * 64,
                ),
            )

            with self.assertRaisesRegex(
                Tau3MlxTrainingError,
                "prelaunch checks failed",
            ):
                run_tau3_mlx_training(
                    mixture_dir=mixture,
                    protocol_path=protocol,
                    model_path=model,
                    model_identity_path=identity,
                    output_dir=root / "out",
                    workspace_root=root,
                    config=config,
                    exposure_dataset_path=mixture / "train.jsonl",
                    exposure_receipt_path=exposure_receipt,
                    exposure_ledger_path=exposure_ledger,
                    prefix_equivalence_path=equivalence,
                )

    def test_exposure_protocol_steps_bound_uses_optimizer_steps_not_microbatches(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _install_fake_python(root, "success")
            model, identity = _fake_model(root)
            protocol = _protocol_config(root, identity)
            mixture, exposure_receipt, exposure_ledger = _exposure_mixture_variant(
                root,
                protocol,
                row_count=2002,
                epochs=2,
                batch_size=2,
                gradient_accumulation_steps=2,
            )

            receipt = run_tau3_mlx_training(
                mixture_dir=mixture,
                protocol_path=protocol,
                model_path=model,
                model_identity_path=identity,
                output_dir=root / "out",
                workspace_root=root,
                config=Tau3MlxTrainingConfig(
                    iters=2002,
                    batch_size=2,
                    grad_accumulation=2,
                    exposure_ledger_training=True,
                    timeout_seconds=5,
                ),
                exposure_dataset_path=mixture / "train.jsonl",
                exposure_receipt_path=exposure_receipt,
                exposure_ledger_path=exposure_ledger,
            )

            recipe = receipt["training_binding"]["recipe"]
            exposure = receipt["training_binding"]["exposure"]
            self.assertEqual(recipe["optimizer_steps"], 1001)
            self.assertEqual(recipe["microbatch_iterations"], 2002)
            self.assertEqual(exposure["ledger"]["optimizer_steps"], 1001)
            self.assertEqual(exposure["ledger"]["microbatch_iterations"], 2002)

    def test_exposure_protocol_steps_bound_rejects_optimizer_steps_over_bound(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _install_fake_python(root, "success")
            model, identity = _fake_model(root)
            protocol = _protocol_config(root, identity)
            mixture, exposure_receipt, exposure_ledger = _exposure_mixture_variant(
                root,
                protocol,
                row_count=4002,
                epochs=2,
                batch_size=2,
                gradient_accumulation_steps=2,
            )

            with self.assertRaisesRegex(Tau3MlxTrainingError, "prelaunch checks failed"):
                run_tau3_mlx_training(
                    mixture_dir=mixture,
                    protocol_path=protocol,
                    model_path=model,
                    model_identity_path=identity,
                    output_dir=root / "out",
                    workspace_root=root,
                    config=Tau3MlxTrainingConfig(
                        iters=4002,
                        batch_size=2,
                        grad_accumulation=2,
                        exposure_ledger_training=True,
                        timeout_seconds=5,
                    ),
                    exposure_dataset_path=mixture / "train.jsonl",
                    exposure_receipt_path=exposure_receipt,
                    exposure_ledger_path=exposure_ledger,
                )

    def test_non_exposure_protocol_steps_bound_still_uses_iters(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _install_fake_python(root, "success")
            model, identity = _fake_model(root)
            protocol = _protocol_config(root, identity)
            _mutate_json(
                protocol,
                lambda payload: payload["recipe_space"]["bounds"].__setitem__(
                    "steps",
                    [1, 2],
                ),
            )

            with self.assertRaisesRegex(Tau3MlxTrainingError, "prelaunch checks failed"):
                run_tau3_mlx_training(
                    mixture_dir=_mixture_variant(root, protocol_path=protocol),
                    protocol_path=protocol,
                    model_path=model,
                    model_identity_path=identity,
                    output_dir=root / "out",
                    workspace_root=root,
                    config=Tau3MlxTrainingConfig(iters=3, timeout_seconds=5),
                )

    def test_exposure_ledger_training_fails_closed_on_recipe_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _install_fake_python(root, "success")
            model, identity = _fake_model(root)
            protocol = _protocol_config(root, identity)
            mixture, exposure_receipt, exposure_ledger = _exposure_mixture_variant(
                root,
                protocol,
            )

            with self.assertRaisesRegex(Tau3MlxTrainingError, "prelaunch checks failed"):
                run_tau3_mlx_training(
                    mixture_dir=mixture,
                    protocol_path=protocol,
                    model_path=model,
                    model_identity_path=identity,
                    output_dir=root / "out",
                    workspace_root=root,
                    config=Tau3MlxTrainingConfig(
                        iters=25,
                        batch_size=2,
                        grad_accumulation=2,
                        exposure_ledger_training=True,
                        timeout_seconds=5,
                    ),
                    exposure_dataset_path=mixture / "train.jsonl",
                    exposure_receipt_path=exposure_receipt,
                    exposure_ledger_path=exposure_ledger,
                )

    def test_exposure_ledger_training_requires_exact_training_dataset_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _install_fake_python(root, "success")
            model, identity = _fake_model(root)
            protocol = _protocol_config(root, identity)
            mixture, _exposure_receipt, _exposure_ledger = _exposure_mixture_variant(
                root,
                protocol,
            )
            alternate = root / "alternate"
            alternate.mkdir()
            alternate_train = alternate / "train.jsonl"
            alternate_rows = [_exposure_row(index) for index in range(52)]
            for row in alternate_rows:
                row["fixture_variant"] = "alternate"
            _write_jsonl(alternate_train, alternate_rows)
            alternate_receipt = build_tau3_exposure_ledger(
                alternate_train,
                alternate / "exposure",
                seed=101,
                epochs=2,
                batch_size=2,
                gradient_accumulation_steps=2,
            )

            with self.assertRaisesRegex(Tau3MlxTrainingError, "prelaunch checks failed"):
                run_tau3_mlx_training(
                    mixture_dir=mixture,
                    protocol_path=protocol,
                    model_path=model,
                    model_identity_path=identity,
                    output_dir=root / "out",
                    workspace_root=root,
                    config=Tau3MlxTrainingConfig(
                        iters=52,
                        batch_size=2,
                        grad_accumulation=2,
                        exposure_ledger_training=True,
                        timeout_seconds=5,
                    ),
                    exposure_dataset_path=alternate_train,
                    exposure_receipt_path=alternate_receipt["receipt_path"],
                    exposure_ledger_path=alternate / "exposure" / "training_exposure_ledger.jsonl",
                )

    def test_exposure_ledger_training_requires_mask_prompt(self) -> None:
        config = Tau3MlxTrainingConfig(
            iters=52,
            batch_size=2,
            grad_accumulation=2,
            exposure_ledger_training=True,
            mask_prompt=False,
        )
        self.assertFalse(
            _config_within_bounds(config)
        )

    def test_competitive_v3_dataset_bundle_is_validated_before_launch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _install_fake_python(root, "success")
            model, identity = _fake_model(root)
            tokenizer_config_path = _write_v3_tokenizer_config(root)
            _sync_v3_tokenizer_assets_to_model(model, identity, tokenizer_config_path)
            protocol = _protocol_config(root, identity)
            tokenizer = _V3FakeTokenizer()
            with _install_v3_fake_transformers(tokenizer), _v3_grounded_validation_patch():
                v3_dataset = root / "v3"
                build_tau3_competitive_dataset(
                    source_dataset_dir=_write_v3_source_dataset(root),
                    out_dir=v3_dataset,
                    tokenizer_config_path=tokenizer_config_path,
                    grounded_generation_bundle=_write_v3_grounded_bundle(root),
                    contamination_report_path=_write_v3_contamination_report(root),
                )

            with _install_v3_fake_transformers(_V3FakeTokenizer()), _v3_grounded_validation_patch():
                receipt = run_tau3_mlx_training(
                    mixture_dir=v3_dataset,
                    protocol_path=protocol,
                    model_path=model,
                    model_identity_path=identity,
                    output_dir=root / "out",
                    workspace_root=root,
                    config=Tau3MlxTrainingConfig(iters=2, timeout_seconds=5),
                )

            checks = {check["id"]: check for check in receipt["checks"]}
            self.assertTrue(checks["competitive_v3_dataset_validation_passed"]["passed"])
            self.assertTrue(checks["competitive_v3_files_replay"]["passed"])
            self.assertTrue(checks["competitive_v3_tokenizer_exact_recorded"]["passed"])
            self.assertTrue(checks["competitive_v3_tokenizer_matches_model"]["passed"])
            self.assertTrue(checks["training_target_quality_direct_semantic_scan"]["passed"])
            self.assertEqual(
                receipt["training_binding"]["dataset"]["declared_protocol_sha256"],
                None,
            )
            self.assertTrue(receipt["weights_updated"])

    def test_competitive_v3_training_passes_external_grounded_validator(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _install_fake_python(root, "success")
            grounded_python = root / "local" / "tau3" / "venv" / "bin" / "python"
            grounded_python.parent.mkdir(parents=True)
            grounded_python.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
            grounded_python.chmod(0o755)
            model, identity = _fake_model(root)
            tokenizer_config_path = _write_v3_tokenizer_config(root)
            _sync_v3_tokenizer_assets_to_model(model, identity, tokenizer_config_path)
            protocol = _protocol_config(root, identity)
            with _install_v3_fake_transformers(_V3FakeTokenizer()), _v3_grounded_validation_patch():
                v3_dataset = root / "v3"
                build_tau3_competitive_dataset(
                    source_dataset_dir=_write_v3_source_dataset(root),
                    out_dir=v3_dataset,
                    tokenizer_config_path=tokenizer_config_path,
                    grounded_generation_bundle=_write_v3_grounded_bundle(root),
                    contamination_report_path=_write_v3_contamination_report(root),
                )

            with _install_v3_fake_transformers(_V3FakeTokenizer()), mock.patch(
                "flightrecorder.tau3_mlx_training.validate_tau3_competitive_dataset_bundle",
                return_value={"passed": True, "errors": [], "coverage": {"passed": True}},
            ) as validate:
                receipt = run_tau3_mlx_training(
                    mixture_dir=v3_dataset,
                    protocol_path=protocol,
                    model_path=model,
                    model_identity_path=identity,
                    output_dir=root / "out",
                    workspace_root=root,
                    config=Tau3MlxTrainingConfig(iters=2, timeout_seconds=5),
                    grounded_validator_python=grounded_python,
                )

            validate.assert_called_once()
            self.assertEqual(validate.call_args.kwargs["grounded_validator_python"], grounded_python)
            self.assertTrue(validate.call_args.kwargs["strict"])
            binding = receipt["training_binding"]["dataset"]["grounded_validation"]
            self.assertEqual(binding["grounded_replay"]["mode"], "external_subprocess")
            self.assertEqual(
                binding["grounded_replay"]["interpreter"]["path"],
                str(grounded_python.resolve()),
            )
            self.assertEqual(
                binding["grounded_replay"]["interpreter"]["sha256"],
                _sha256(grounded_python),
            )
            self.assertEqual(
                binding["grounded_replay"]["interpreter"]["sha256_policy"],
                "workspace_executable_content_hash",
            )
            self.assertEqual(
                _read_json(root / "out" / "prelaunch_receipt.json")["training_binding"]["dataset"]["grounded_validation"],
                binding,
            )
            self.assertTrue(receipt["weights_updated"])

    def test_competitive_v3_training_fails_closed_when_external_grounded_replay_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _install_fake_python(root, "success")
            grounded_python = root / "local" / "tau3" / "venv" / "bin" / "python"
            grounded_python.parent.mkdir(parents=True)
            grounded_python.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
            grounded_python.chmod(0o755)
            model, identity = _fake_model(root)
            tokenizer_config_path = _write_v3_tokenizer_config(root)
            _sync_v3_tokenizer_assets_to_model(model, identity, tokenizer_config_path)
            protocol = _protocol_config(root, identity)
            with _install_v3_fake_transformers(_V3FakeTokenizer()), _v3_grounded_validation_patch():
                v3_dataset = root / "v3"
                build_tau3_competitive_dataset(
                    source_dataset_dir=_write_v3_source_dataset(root),
                    out_dir=v3_dataset,
                    tokenizer_config_path=tokenizer_config_path,
                    grounded_generation_bundle=_write_v3_grounded_bundle(root),
                    contamination_report_path=_write_v3_contamination_report(root),
                )

            with _install_v3_fake_transformers(_V3FakeTokenizer()), mock.patch(
                "flightrecorder.tau3_mlx_training.validate_tau3_competitive_dataset_bundle",
                return_value={"passed": False, "errors": ["grounded_generation strict replay failed"], "coverage": {}},
            ) as validate:
                with self.assertRaisesRegex(Tau3MlxTrainingError, "prelaunch checks failed"):
                    run_tau3_mlx_training(
                        mixture_dir=v3_dataset,
                        protocol_path=protocol,
                        model_path=model,
                        model_identity_path=identity,
                        output_dir=root / "out",
                        workspace_root=root,
                        config=Tau3MlxTrainingConfig(iters=2, timeout_seconds=5),
                        grounded_validator_python=grounded_python,
                    )

            validate.assert_called_once()
            self.assertEqual(validate.call_args.kwargs["grounded_validator_python"], grounded_python)
            self.assertFalse((root / "out" / "prelaunch_receipt.json").exists())

    def test_competitive_v3_dataset_tokenizer_must_match_model_tokenizer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _install_fake_python(root, "success")
            model, identity = _fake_model(root)
            tokenizer_config_path = _write_v3_tokenizer_config(root)
            _sync_v3_tokenizer_assets_to_model(model, identity, tokenizer_config_path)
            (model / "tokenizer.json").write_text(
                '{"fixture": "tampered-model-tokenizer"}\n',
                encoding="utf-8",
            )
            identity_payload = build_tau3_model_identity(
                model,
                model_id="fake/base-9b",
                revision="1234567890abcdef",
            )
            _write_json(identity, identity_payload)
            protocol = _protocol_config(root, identity)
            with _install_v3_fake_transformers(_V3FakeTokenizer()), _v3_grounded_validation_patch():
                v3_dataset = root / "v3"
                build_tau3_competitive_dataset(
                    source_dataset_dir=_write_v3_source_dataset(root),
                    out_dir=v3_dataset,
                    tokenizer_config_path=tokenizer_config_path,
                    grounded_generation_bundle=_write_v3_grounded_bundle(root),
                    contamination_report_path=_write_v3_contamination_report(root),
                )

                with self.assertRaisesRegex(Tau3MlxTrainingError, "prelaunch checks failed"):
                    run_tau3_mlx_training(
                        mixture_dir=v3_dataset,
                        protocol_path=protocol,
                        model_path=model,
                        model_identity_path=identity,
                        output_dir=root / "out",
                        workspace_root=root,
                        config=Tau3MlxTrainingConfig(iters=2, timeout_seconds=5),
                    )

    def test_prefix_cache_split_preserves_first_target_prediction(self) -> None:
        prefix, suffix_inputs, targets = split_supervised_tokens(
            [10, 11, 12, 20, 21, 22],
            prompt_offset=3,
            max_seq_length=8,
        )
        self.assertEqual(prefix, [10, 11])
        self.assertEqual(suffix_inputs, [12, 20, 21])
        self.assertEqual(targets, [20, 21, 22])
        with self.assertRaisesRegex(PrefixCacheTrainingError, "truncation"):
            split_supervised_tokens(
                [10, 11, 12, 20, 21, 22],
                prompt_offset=3,
                max_seq_length=5,
            )

    def test_prefix_cache_validation_slice_is_domain_balanced(self) -> None:
        class RawDataset:
            _data = [
                {"metadata": {"domain": domain}}
                for domain in (
                    "airline",
                    "airline",
                    "retail",
                    "retail",
                    "telecom",
                    "telecom",
                )
            ]

        class CachedDataset:
            _data = RawDataset()

            def __len__(self) -> int:
                return len(self._data._data)

        dataset = CachedDataset()
        selected = _validation_indices(dataset, 6)
        domains = [dataset._data._data[index]["metadata"]["domain"] for index in selected]
        self.assertEqual(
            domains,
            ["airline", "retail", "telecom", "airline", "retail", "telecom"],
        )

    def test_telemetry_event_is_flushed_for_live_observation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "telemetry.jsonl"
            losses: dict[str, list[float]] = {"train": [], "validation": []}
            with path.open("x", encoding="utf-8") as handle:
                _write_telemetry(handle, "stdout", "Iter 1: Train loss 1.25", losses)
                self.assertGreater(path.stat().st_size, 0)
            self.assertEqual(losses["train"], [1.25])

    def test_success_writes_schema_checked_receipt_and_fingerprints_adapter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _install_fake_python(root, "success")
            bundle = _runner_bundle(root)
            receipt = run_tau3_mlx_training(
                bundle_dir=bundle,
                output_dir=root / "out",
                workspace_root=root,
                config=Tau3MlxTrainingConfig(iters=2, timeout_seconds=5),
                created_at="2026-07-22T00:00:00Z",
            )
            self.assertEqual(receipt["terminal_status"], "success")
            self.assertTrue(receipt["weights_updated"])
            self.assertGreaterEqual(receipt["adapter"]["file_count"], 3)
            self.assertEqual(receipt["losses"]["last_train"], 1.25)
            self.assertEqual(receipt["losses"]["last_validation"], 0.75)
            self.assertIsNone(receipt["training_binding"])
            schema = check_schema_contract(_read_json(root / "out" / "training_receipt.json"), name_or_id="tau3_mlx_training_run")
            self.assertTrue(schema["passed"], schema["errors"])
            prelaunch = _read_json(root / "out" / "prelaunch_receipt.json")
            self.assertIsNone(prelaunch["training_binding"])
            prelaunch_schema = check_schema_contract(prelaunch, name_or_id="tau3_mlx_training_run")
            self.assertTrue(prelaunch_schema["passed"], prelaunch_schema["errors"])

    def test_mlx_progress_parser_records_only_explicit_loss_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _install_fake_python(root, "mlx_progress")
            receipt = run_tau3_mlx_training(
                bundle_dir=_runner_bundle(root),
                output_dir=root / "out",
                workspace_root=root,
                config=Tau3MlxTrainingConfig(iters=2, timeout_seconds=5),
            )
            self.assertEqual(receipt["terminal_status"], "success")
            self.assertTrue(receipt["weights_updated"])
            self.assertEqual(receipt["losses"]["train"], [1.25, 1.234])
            self.assertEqual(receipt["losses"]["validation"], [0.75, 0.442])
            self.assertNotIn(100.0, receipt["losses"]["train"])
            self.assertNotIn(59.258, receipt["losses"]["validation"])

    def test_crash_is_classified_without_weights_updated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _install_fake_python(root, "crash")
            receipt = run_tau3_mlx_training(
                bundle_dir=_runner_bundle(root),
                output_dir=root / "out",
                workspace_root=root,
                config=Tau3MlxTrainingConfig(timeout_seconds=5),
            )
            self.assertEqual(receipt["terminal_status"], "crash")
            self.assertFalse(receipt["weights_updated"])

    def test_timeout_is_classified_and_terminates_child(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _install_fake_python(root, "sleep")
            receipt = run_tau3_mlx_training(
                bundle_dir=_runner_bundle(root),
                output_dir=root / "out",
                workspace_root=root,
                config=Tau3MlxTrainingConfig(timeout_seconds=1),
            )
            self.assertEqual(receipt["terminal_status"], "timeout")
            self.assertTrue(receipt["timed_out"])
            self.assertFalse(receipt["weights_updated"])

    def test_no_output_success_is_failure_for_weight_update_claim(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _install_fake_python(root, "no_output")
            receipt = run_tau3_mlx_training(
                bundle_dir=_runner_bundle(root),
                output_dir=root / "out",
                workspace_root=root,
                config=Tau3MlxTrainingConfig(timeout_seconds=5),
            )
            self.assertEqual(receipt["terminal_status"], "no_output")
            self.assertFalse(receipt["weights_updated"])

    def test_oom_is_classified_without_weights_updated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _install_fake_python(root, "oom")
            receipt = run_tau3_mlx_training(
                bundle_dir=_runner_bundle(root),
                output_dir=root / "out",
                workspace_root=root,
                config=Tau3MlxTrainingConfig(timeout_seconds=5),
            )
            self.assertEqual(receipt["terminal_status"], "oom")
            self.assertFalse(receipt["weights_updated"])

    def test_tampered_bundle_is_rejected_before_launch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _install_fake_python(root, "success")
            bundle = _runner_bundle(root)
            _mutate_json(bundle / "protocol/budget.json", lambda payload: payload.__setitem__("max_seconds", 604801))
            with self.assertRaisesRegex(Tau3MlxTrainingError, "strict production bundle validation failed"):
                run_tau3_mlx_training(bundle_dir=bundle, output_dir=root / "out", workspace_root=root)

    def test_forbidden_launch_flags_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _install_fake_python(root, "success")
            bundle = _runner_bundle(root)
            _mutate_json(
                bundle / "training/mlx_qlora_plan.json",
                lambda payload: payload["command_argv"].extend(["--push-to-hub", "--report-to", "wandb"]),
            )
            _rewrite_manifest(bundle)
            with self.assertRaisesRegex(Tau3MlxTrainingError, "strict production bundle validation failed|prelaunch checks failed"):
                run_tau3_mlx_training(bundle_dir=bundle, output_dir=root / "out", workspace_root=root)

    def test_symlinked_bundle_and_nonlocal_output_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as outside:
            root = Path(tmp)
            _install_fake_python(root, "success")
            bundle = _runner_bundle(root)
            link = root / "linked_bundle"
            link.symlink_to(bundle, target_is_directory=True)
            with self.assertRaisesRegex(Tau3MlxTrainingError, "symlink|workspace root"):
                run_tau3_mlx_training(bundle_dir=link, output_dir=root / "out1", workspace_root=root)
            with self.assertRaisesRegex(Tau3MlxTrainingError, "under workspace root"):
                run_tau3_mlx_training(bundle_dir=bundle, output_dir=Path(outside) / "out", workspace_root=root)

    def test_sealed_or_test_trainer_view_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _install_fake_python(root, "success")
            bundle = _runner_bundle(root)
            _mutate_json(
                bundle / "training/dataset_manifest.json",
                lambda payload: payload["views"].__setitem__("mlx_test", {"path": "input_export/test.jsonl", "row_count": 1}),
            )
            _rewrite_manifest(bundle)
            with self.assertRaisesRegex(Tau3MlxTrainingError, "prelaunch checks failed"):
                run_tau3_mlx_training(bundle_dir=bundle, output_dir=root / "out", workspace_root=root)

    def test_attestation_is_not_required_when_direct_semantic_scan_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _install_fake_python(root, "success")
            bundle = _runner_bundle(root, attestation=False)
            receipt = run_tau3_mlx_training(
                bundle_dir=bundle,
                output_dir=root / "out",
                workspace_root=root,
                config=Tau3MlxTrainingConfig(timeout_seconds=5),
            )
            self.assertTrue(receipt["weights_updated"])

    def test_mixture_variant_training_uses_mlx_0313_argv_and_readonly_lora_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _install_fake_python(root, "success")
            model, identity = _fake_model(root)
            protocol = _protocol_config(root, identity)
            receipt = run_tau3_mlx_training(
                mixture_dir=_mixture_variant(root, protocol_path=protocol),
                protocol_path=protocol,
                model_path=model,
                model_identity_path=identity,
                output_dir=root / "out",
                workspace_root=root,
                config=Tau3MlxTrainingConfig(iters=2, disable_compile=True, timeout_seconds=5),
            )
            command = receipt["command"]
            self.assertIn("--fine-tune-type", command)
            self.assertIn("lora", command)
            self.assertIn("--grad-accumulation-steps", command)
            self.assertIn("--steps-per-report", command)
            self.assertIn("--steps-per-eval", command)
            self.assertNotIn("--lora-rank", command)
            self.assertNotIn("--lora-scale", command)
            self.assertTrue(receipt["mlx_lora_config"]["read_only"])
            cfg = _read_json(root / "out" / "mlx_lora_config.json")
            self.assertEqual(cfg["adapter_path"], "adapter")
            self.assertIn(str((root / "out" / "adapter").resolve()), command)
            self.assertEqual(cfg["lora_parameters"], {"rank": 16, "scale": 20.0, "dropout": 0.0})
            self.assertEqual(receipt["training_binding"]["model"]["identity_sha256"], _sha256(identity))
            self.assertEqual(receipt["training_binding"]["protocol"]["sha256"], _sha256(protocol))
            self.assertEqual(receipt["training_binding"]["protocol"]["protocol_signature"], "a" * 64)
            self.assertEqual(
                receipt["training_binding"]["protocol"]["protocol_signature_provenance"],
                {"source": "protocol_manifest.signature", "algorithm": "sha256"},
            )
            self.assertRegex(receipt["training_binding"]["recipe"]["recipe_id"], r"^tau3-mlx-recipe-[0-9a-f]{16}$")
            self.assertTrue(receipt["training_binding"]["recipe"]["disable_compile"])
            self.assertTrue(receipt["weights_updated"])

    def test_training_binding_schema_requires_protocol_signature_lineage(self) -> None:
        binding = {
            "protocol": {
                "sha256": "b" * 64,
                "protocol_signature": "a" * 64,
                "protocol_signature_provenance": {
                    "source": "protocol_manifest.signature",
                    "algorithm": "sha256",
                },
            },
        }
        receipt = {
            "schema_version": "hfr.tau3_mlx_training_run.v1",
            "phase": "prelaunch",
            "created_at": "2026-07-23T00:00:00Z",
            "bundle": {},
            "output_dir": ".",
            "command": [],
            "config": {},
            "training_binding": binding,
            "checks": [],
            "weights_updated": False,
            "terminal_status": "prelaunch",
        }
        valid = check_schema_contract(receipt, name_or_id="tau3_mlx_training_run")
        self.assertTrue(valid["passed"], valid["errors"])

        for missing_field in ("sha256", "protocol_signature", "protocol_signature_provenance"):
            with self.subTest(missing_field=missing_field):
                invalid = json.loads(json.dumps(receipt))
                invalid["training_binding"]["protocol"].pop(missing_field)
                result = check_schema_contract(invalid, name_or_id="tau3_mlx_training_run")
                self.assertFalse(result["passed"], result["errors"])

        missing_protocol = json.loads(json.dumps(receipt))
        missing_protocol["training_binding"].pop("protocol")
        result = check_schema_contract(missing_protocol, name_or_id="tau3_mlx_training_run")
        self.assertFalse(result["passed"], result["errors"])

    def test_unsigned_protocol_uses_protocol_file_sha256_as_content_seal_signature(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _install_fake_python(root, "success")
            model, identity = _fake_model(root)
            protocol = _protocol_config(root, identity)
            _mutate_json(protocol, lambda payload: payload["protocol_manifest"].pop("signature", None))
            receipt = run_tau3_mlx_training(
                mixture_dir=_mixture_variant(root, protocol_path=protocol),
                protocol_path=protocol,
                model_path=model,
                model_identity_path=identity,
                output_dir=root / "out",
                workspace_root=root,
                config=Tau3MlxTrainingConfig(iters=2, timeout_seconds=5),
            )
            self.assertEqual(receipt["training_binding"]["protocol"]["protocol_signature"], _sha256(protocol))
            self.assertEqual(
                receipt["training_binding"]["protocol"]["protocol_signature_provenance"],
                {"source": "protocol_file_sha256_content_seal", "algorithm": "sha256"},
            )
            schema = check_schema_contract(receipt, name_or_id="tau3_mlx_training_run")
            self.assertTrue(schema["passed"], schema["errors"])

    def test_invalid_embedded_protocol_signature_fails_closed_without_content_seal_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _install_fake_python(root, "success")
            model, identity = _fake_model(root)
            protocol = _protocol_config(root, identity)
            _mutate_json(protocol, lambda payload: payload["protocol_manifest"].__setitem__("signature", "not-a-sha"))
            with self.assertRaisesRegex(Tau3MlxTrainingError, "prelaunch checks failed"):
                run_tau3_mlx_training(
                    mixture_dir=_mixture_variant(root, protocol_path=protocol),
                    protocol_path=protocol,
                    model_path=model,
                    model_identity_path=identity,
                    output_dir=root / "out",
                    workspace_root=root,
                    config=Tau3MlxTrainingConfig(iters=2, timeout_seconds=5),
                )

    def test_mixture_resume_cli_adds_resume_adapter_file_and_records_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cwd = Path.cwd()
            os.chdir(root)
            try:
                _install_fake_python(root, "success")
                model, identity = _fake_model(root)
                protocol = _protocol_config(root, identity)
                mixture = _mixture_variant(root, protocol_path=protocol)
                prior = run_tau3_mlx_training(
                    mixture_dir=mixture,
                    protocol_path=protocol,
                    model_path=model,
                    model_identity_path=identity,
                    output_dir=root / "prior",
                    workspace_root=root,
                    config=Tau3MlxTrainingConfig(iters=2, timeout_seconds=5),
                    created_at="2026-07-22T00:00:00Z",
                )
                self.assertTrue(prior["weights_updated"])
                prior_receipt = root / "prior" / "training_receipt.json"
                resume_file = root / "prior" / "adapter" / "checkpoint-0001" / "weights.npz"
                code = tau3_mlx_training_main(
                    [
                        "--mixture-dir",
                        str(mixture),
                        "--protocol",
                        str(protocol),
                        "--model-path",
                        str(model),
                        "--model-identity",
                        str(identity),
                        "--out",
                        str(root / "resumed"),
                        "--iters",
                        "3",
                        "--timeout-seconds",
                        "5",
                        "--resume-receipt",
                        str(prior_receipt),
                        "--resume-adapter-file",
                        str(resume_file),
                    ]
                )
            finally:
                os.chdir(cwd)
            self.assertEqual(code, 0)
            receipt = _read_json(root / "resumed" / "training_receipt.json")
            self.assertIn("--resume-adapter-file", receipt["command"])
            self.assertIn(str(resume_file.resolve()), receipt["command"])
            self.assertEqual(receipt["config"]["resume"]["receipt"]["sha256"], _sha256(prior_receipt))
            self.assertEqual(receipt["training_binding"]["resume"]["adapter_file"]["sha256"], _sha256(resume_file))
            self.assertEqual(receipt["training_binding"]["resume"]["adapter_file"]["relative_path"], "checkpoint-0001/weights.npz")
            prelaunch = _read_json(root / "resumed" / "prelaunch_receipt.json")
            self.assertEqual(prelaunch["config"]["resume"], receipt["config"]["resume"])
            self.assertEqual(prelaunch["training_binding"]["resume"], receipt["training_binding"]["resume"])

    def test_portable_receipt_paths_replay_after_output_relocation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _install_fake_python(root, "success")
            model, identity = _fake_model(root)
            protocol = _protocol_config(root, identity)
            mixture = _mixture_variant(root, protocol_path=protocol)
            prior = run_tau3_mlx_training(
                mixture_dir=mixture,
                protocol_path=protocol,
                model_path=model,
                model_identity_path=identity,
                output_dir=root / "prior",
                workspace_root=root,
                config=Tau3MlxTrainingConfig(iters=2, timeout_seconds=5),
                created_at="2026-07-22T00:00:00Z",
            )
            self.assertEqual(prior["output_dir"], ".")
            self.assertEqual(prior["prelaunch_receipt"]["path"], "prelaunch_receipt.json")
            self.assertEqual(prior["telemetry"]["path"], "telemetry.jsonl")
            self.assertEqual(prior["mlx_lora_config"]["path"], "mlx_lora_config.json")
            self.assertEqual(prior["adapter"]["path"], "adapter")

            relocated = root / "relocated" / "prior"
            shutil.copytree(root / "prior", relocated)
            resumed = run_tau3_mlx_training(
                mixture_dir=mixture,
                protocol_path=protocol,
                model_path=model,
                model_identity_path=identity,
                output_dir=root / "resumed",
                workspace_root=root,
                config=Tau3MlxTrainingConfig(iters=3, timeout_seconds=5),
                resume_receipt_path=relocated / "training_receipt.json",
                resume_adapter_file=relocated / "adapter" / "checkpoint-0001" / "weights.npz",
            )
            self.assertTrue(resumed["weights_updated"])

            tampered = root / "tampered" / "prior"
            shutil.copytree(root / "prior", tampered)
            (tampered / "adapter" / "checkpoint-0001" / "weights.npz").write_bytes(b"tampered")
            with self.assertRaisesRegex(Tau3MlxTrainingError, "prelaunch checks failed"):
                run_tau3_mlx_training(
                    mixture_dir=mixture,
                    protocol_path=protocol,
                    model_path=model,
                    model_identity_path=identity,
                    output_dir=root / "tampered-resume",
                    workspace_root=root,
                    config=Tau3MlxTrainingConfig(iters=3, timeout_seconds=5),
                    resume_receipt_path=tampered / "training_receipt.json",
                    resume_adapter_file=tampered / "adapter" / "checkpoint-0001" / "weights.npz",
                )

    def test_resume_tampered_adapter_file_fails_closed_before_launch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _install_fake_python(root, "success")
            model, identity = _fake_model(root)
            protocol = _protocol_config(root, identity)
            mixture = _mixture_variant(root, protocol_path=protocol)
            run_tau3_mlx_training(
                mixture_dir=mixture,
                protocol_path=protocol,
                model_path=model,
                model_identity_path=identity,
                output_dir=root / "prior",
                workspace_root=root,
                config=Tau3MlxTrainingConfig(iters=2, timeout_seconds=5),
            )
            resume_file = root / "prior" / "adapter" / "checkpoint-0001" / "weights.npz"
            resume_file.write_bytes(b"tampered-checkpoint")
            with self.assertRaisesRegex(Tau3MlxTrainingError, "prelaunch checks failed"):
                run_tau3_mlx_training(
                    mixture_dir=mixture,
                    protocol_path=protocol,
                    model_path=model,
                    model_identity_path=identity,
                    output_dir=root / "resumed",
                    workspace_root=root,
                    config=Tau3MlxTrainingConfig(iters=3, timeout_seconds=5),
                    resume_receipt_path=root / "prior" / "training_receipt.json",
                    resume_adapter_file=resume_file,
                )

    def test_resume_hyperparameter_mismatch_fails_closed_before_launch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _install_fake_python(root, "success")
            model, identity = _fake_model(root)
            protocol = _protocol_config(root, identity)
            mixture = _mixture_variant(root, protocol_path=protocol)
            run_tau3_mlx_training(
                mixture_dir=mixture,
                protocol_path=protocol,
                model_path=model,
                model_identity_path=identity,
                output_dir=root / "prior",
                workspace_root=root,
                config=Tau3MlxTrainingConfig(iters=2, timeout_seconds=5),
            )
            mismatches = {
                "rank": Tau3MlxTrainingConfig(iters=3, rank=32, timeout_seconds=5),
                "compile-mode": Tau3MlxTrainingConfig(iters=3, disable_compile=True, timeout_seconds=5),
                "padding-mode": Tau3MlxTrainingConfig(iters=3, fixed_shape_padding=True, timeout_seconds=5),
                "training-objective": Tau3MlxTrainingConfig(
                    iters=3,
                    grad_checkpoint=False,
                    disable_compile=True,
                    prefix_cache_training=True,
                    timeout_seconds=5,
                ),
            }
            for label, config in mismatches.items():
                with self.subTest(label=label), self.assertRaisesRegex(Tau3MlxTrainingError, "prelaunch checks failed"):
                    run_tau3_mlx_training(
                        mixture_dir=mixture,
                        protocol_path=protocol,
                        model_path=model,
                        model_identity_path=identity,
                        output_dir=root / f"resumed-{label}",
                        workspace_root=root,
                        config=config,
                        resume_receipt_path=root / "prior" / "training_receipt.json",
                        resume_adapter_file=root / "prior" / "adapter" / "checkpoint-0001" / "weights.npz",
                    )

    def test_normal_venv_python_leaf_symlink_is_preserved_for_activation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _install_fake_python(root, "success")
            python = root / ".venv" / "bin" / "python"
            target = root / "runtime-python"
            python.replace(target)
            python.symlink_to(target)
            model, identity = _fake_model(root)
            protocol = _protocol_config(root, identity)
            receipt = run_tau3_mlx_training(
                mixture_dir=_mixture_variant(root, protocol_path=protocol),
                protocol_path=protocol,
                model_path=model,
                model_identity_path=identity,
                output_dir=root / "out",
                workspace_root=root,
                config=Tau3MlxTrainingConfig(iters=2, timeout_seconds=5),
            )
            self.assertEqual(receipt["command"][0], str(root.resolve() / ".venv" / "bin" / "python"))
            self.assertTrue(receipt["weights_updated"])

    def test_mixture_semantic_scan_rejects_invented_tau_tool_even_with_manifest_pass(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _install_fake_python(root, "success")
            model, identity = _fake_model(root)
            protocol = _protocol_config(root, identity)
            with self.assertRaisesRegex(Tau3MlxTrainingError, "prelaunch checks failed"):
                run_tau3_mlx_training(
                    mixture_dir=_mixture_variant(root, protocol_path=protocol, contaminated=True),
                    protocol_path=protocol,
                    model_path=model,
                    model_identity_path=identity,
                    output_dir=root / "out",
                    workspace_root=root,
                    config=Tau3MlxTrainingConfig(timeout_seconds=5),
                )

    def test_mixture_protocol_drift_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _install_fake_python(root, "success")
            model, identity = _fake_model(root)
            protocol = _protocol_config(root, identity)
            mixture = _mixture_variant(root, protocol_path=protocol)
            _mutate_json(protocol, lambda payload: payload["harness_contract"].__setitem__("turn_limit", 99))
            with self.assertRaisesRegex(Tau3MlxTrainingError, "prelaunch checks failed"):
                run_tau3_mlx_training(
                    mixture_dir=mixture,
                    protocol_path=protocol,
                    model_path=model,
                    model_identity_path=identity,
                    output_dir=root / "out",
                    workspace_root=root,
                    config=Tau3MlxTrainingConfig(timeout_seconds=5),
                )

    def test_mixture_base_identity_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _install_fake_python(root, "success")
            model, identity = _fake_model(root)
            protocol = _protocol_config(root, identity)
            _mutate_json(protocol, lambda payload: payload["model_freeze"]["base_model"].__setitem__("revision", "different-revision"))
            with self.assertRaisesRegex(Tau3MlxTrainingError, "prelaunch checks failed"):
                run_tau3_mlx_training(
                    mixture_dir=_mixture_variant(root, protocol_path=protocol),
                    protocol_path=protocol,
                    model_path=model,
                    model_identity_path=identity,
                    output_dir=root / "out",
                    workspace_root=root,
                    config=Tau3MlxTrainingConfig(timeout_seconds=5),
                )

    def test_mixture_out_of_space_recipe_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _install_fake_python(root, "success")
            model, identity = _fake_model(root)
            protocol = _protocol_config(root, identity)
            with self.assertRaisesRegex(Tau3MlxTrainingError, "prelaunch checks failed"):
                run_tau3_mlx_training(
                    mixture_dir=_mixture_variant(root, protocol_path=protocol),
                    protocol_path=protocol,
                    model_path=model,
                    model_identity_path=identity,
                    output_dir=root / "out",
                    workspace_root=root,
                    config=Tau3MlxTrainingConfig(rank=64, timeout_seconds=5),
                )

    def test_mixture_missing_protocol_provenance_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _install_fake_python(root, "success")
            model, identity = _fake_model(root)
            protocol = _protocol_config(root, identity)
            with self.assertRaisesRegex(Tau3MlxTrainingError, "prelaunch checks failed"):
                run_tau3_mlx_training(
                    mixture_dir=_mixture_variant(root),
                    protocol_path=protocol,
                    model_path=model,
                    model_identity_path=identity,
                    output_dir=root / "out",
                    workspace_root=root,
                    config=Tau3MlxTrainingConfig(timeout_seconds=5),
                )

    def test_computed_evaluation_criteria_exposure_blocks_launch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _install_fake_python(root, "success")
            bundle = _runner_bundle(root, attestation=False)
            source = bundle / "training_source"
            source.mkdir()
            criterion = "Refund must be denied unless the passenger provides a refundable fare code."
            _write_jsonl(
                source / "train_tasks.jsonl",
                [{"id": "task-1", "evaluation_criteria": {"must": criterion}}],
            )
            _mutate_json(
                bundle / "protocol/split_manifest.json",
                lambda payload: payload.__setitem__("source_paths", {"train": "training_source/train_tasks.jsonl"}),
            )
            _refresh_protocol_signature(bundle)
            _write_jsonl(
                bundle / "training" / "input_export" / "train.jsonl",
                [{"messages": [{"role": "user", "content": "x"}, {"role": "assistant", "content": criterion}]}],
            )
            manifest = _read_json(bundle / "training" / "input_export" / "mlx_dataset_manifest.json")
            manifest["files"]["train"]["size"] = (bundle / "training" / "input_export" / "train.jsonl").stat().st_size
            manifest["files"]["train"]["sha256"] = _sha256(bundle / "training" / "input_export" / "train.jsonl")
            _write_json(bundle / "training" / "input_export" / "mlx_dataset_manifest.json", manifest)
            _mutate_json(
                bundle / "training/dataset_manifest.json",
                lambda payload: payload["mlx_dataset_manifest"].__setitem__("sha256", _sha256(bundle / "training" / "input_export" / "mlx_dataset_manifest.json")),
            )
            _rewrite_manifest(bundle)
            with self.assertRaisesRegex(Tau3MlxTrainingError, "prelaunch checks failed"):
                run_tau3_mlx_training(bundle_dir=bundle, output_dir=root / "out", workspace_root=root)


if __name__ == "__main__":
    unittest.main()
