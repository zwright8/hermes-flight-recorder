from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from flightrecorder import mlx_exposure_prefix_cache_lora as segmented_lora
from flightrecorder.mlx_exposure_lora import load_exposure_schedule
from flightrecorder.mlx_exposure_prefix_cache_lora import (
    ExposurePrefixCacheLoraError,
    _build_segment_config,
    _file_sha256,
    _parse_args,
    train_with_exposure_prefix_cache,
)
from flightrecorder.tau3_exposure import build_tau3_exposure_ledger


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _row(index: int) -> dict:
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
    return {
        "messages": [
            {"role": "user", "content": f"request {index}"},
            {"role": "assistant", "content": "done"},
        ],
        "metadata": {
            "schema_version": "hfr.tau3_competitive_dataset_row.v1",
            "domain": domains[index % len(domains)],
            "behavior": behaviors[index % len(behaviors)],
            "target_tool_name": f"tool_{index % 4}",
            "target_action_class": "tool_call",
            "preceding_result_class": "success",
            "length_bucket": "short",
            "source_family_id": f"family_{index % 8}",
            "source_provenance": {"method": "fixture"},
            "token_counts": {
                "method": "pinned_local_apply_chat_template",
                "exact": True,
                "chat_template_aware": True,
                "prompt_tokens": 8 + index,
                "supervised_tokens": 3 + index % 7,
            },
        },
    }


class ExposurePrefixCacheSegmentContractTests(unittest.TestCase):
    def test_parser_strips_all_child_segment_arguments_before_mlx_lm(self) -> None:
        argv = [
            "--exposure-dataset",
            "train.jsonl",
            "--exposure-receipt",
            "receipt.json",
            "--exposure-ledger",
            "ledger.jsonl",
            "--prefix-equivalence",
            "equivalence.json",
            "--batch-size",
            "1",
            "--grad-accumulation-steps",
            "4",
            "--iters",
            "40",
            "--hfr-child-segment-start",
            "20",
            "--hfr-child-segment-end=40",
            "--hfr-child-segment-adapter-input",
            "previous.safetensors",
            "--hfr-child-segment-adapter-sha256",
            "a" * 64,
            "--hfr-child-segment-optimizer-state-input",
            "previous-optimizer.safetensors",
            "--hfr-child-segment-optimizer-state-sha256",
            "b" * 64,
            "--hfr-child-segment-optimizer-state-output",
            "optimizer.safetensors",
            "--model",
            "local-model",
        ]

        args, passthrough = _parse_args(argv)

        self.assertTrue(args.hfr_child_segment_enabled)
        self.assertEqual(args.hfr_child_segment_start, 20)
        self.assertEqual(args.hfr_child_segment_end, 40)
        self.assertEqual(
            passthrough,
            [
                "--batch-size",
                "1",
                "--grad-accumulation-steps",
                "4",
                "--iters",
                "40",
                "--model",
                "local-model",
            ],
        )

    def test_resumed_segment_requires_both_hash_bound_inputs(self) -> None:
        args = argparse.Namespace(
            hfr_child_segment_enabled=True,
            hfr_child_segment_start=20,
            hfr_child_segment_end=40,
            hfr_child_segment_adapter_input=Path("adapter.safetensors"),
            hfr_child_segment_adapter_sha256="a" * 64,
            hfr_child_segment_optimizer_state_input=None,
            hfr_child_segment_optimizer_state_sha256=None,
            hfr_child_segment_optimizer_state_output=Path("optimizer.safetensors"),
        )

        with self.assertRaisesRegex(
            ExposurePrefixCacheLoraError,
            "requires both adapter and optimizer-state inputs",
        ):
            _build_segment_config(args, total_microbatches=40)

    def test_schedule_exposes_parallel_ledger_token_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "train.jsonl"
            rows = [_row(index) for index in range(52)]
            _write_jsonl(dataset, rows)
            receipt = build_tau3_exposure_ledger(
                dataset,
                root / "exposure",
                seed=101,
                epochs=2,
                batch_size=2,
                gradient_accumulation_steps=2,
            )

            schedule = load_exposure_schedule(
                dataset_jsonl=dataset,
                receipt_path=receipt["receipt_path"],
                ledger_path=root
                / "exposure"
                / "training_exposure_ledger.jsonl",
                batch_size=2,
                grad_accumulation_steps=2,
                iters=52,
            )

            self.assertEqual(
                [len(step) for step in schedule["microbatch_token_counts"]],
                [len(step) for step in schedule["steps"]],
            )
            first_indices = schedule["steps"][0][0]
            expected_prompt = sum(
                rows[index]["metadata"]["token_counts"]["prompt_tokens"]
                for index in first_indices
            )
            expected_supervised = sum(
                rows[index]["metadata"]["token_counts"]["supervised_tokens"]
                for index in first_indices
            )
            self.assertEqual(
                schedule["microbatch_token_counts"][0][0],
                {
                    "prompt_tokens": expected_prompt,
                    "supervised_tokens": expected_supervised,
                },
            )


try:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten, tree_map
except ImportError:  # pragma: no cover - exercised on non-Apple CI.
    mx = None
    nn = None
    optim = None
    tree_flatten = None
    tree_map = None


@unittest.skipIf(mx is None, "MLX is unavailable")
class ExposurePrefixCacheSegmentMlxTests(unittest.TestCase):
    @staticmethod
    def _schedule(segment: dict) -> dict:
        return {
            "receipt": {
                "sampler_config": {
                    "batch_size": 1,
                    "gradient_accumulation_steps": 2,
                }
            },
            "steps": [[[0], [1]], [[2], [3]]],
            "microbatch_token_counts": [
                [
                    {"prompt_tokens": 10, "supervised_tokens": 2},
                    {"prompt_tokens": 11, "supervised_tokens": 3},
                ],
                [
                    {"prompt_tokens": 12, "supervised_tokens": 4},
                    {"prompt_tokens": 13, "supervised_tokens": 5},
                ],
            ],
            "microbatch_iterations": 4,
            "optimizer_steps": 2,
            "segment": segment,
        }

    @staticmethod
    def _args(adapter_file: Path) -> types.SimpleNamespace:
        return types.SimpleNamespace(
            batch_size=1,
            grad_accumulation_steps=2,
            grad_checkpoint=False,
            iters=4,
            steps_per_eval=2,
            val_batches=1,
            max_seq_length=128,
            steps_per_report=2,
            steps_per_save=2,
            adapter_file=adapter_file,
        )

    @staticmethod
    def _dataset() -> list[tuple[list[int], int]]:
        return [([index + 1], 10 + index) for index in range(4)]

    @staticmethod
    def _new_model(initial_weights):
        model = nn.Linear(1, 1)
        model.load_weights(initial_weights)
        return model

    def _run(
        self,
        *,
        root: Path,
        name: str,
        initial_weights,
        segment: dict,
    ):
        output = root / name
        output.mkdir()
        model = self._new_model(initial_weights)
        optimizer = optim.Adam(learning_rate=0.01)
        segmented_lora._RUNTIME_SCHEDULE = self._schedule(segment)

        def fake_split(tokens, prompt_offset, max_seq_length):
            del prompt_offset, max_seq_length
            row_number = int(tokens[0]) - 1
            return None, tokens, row_number + 2

        def fake_target(model, suffix_inputs, targets, cache):
            del cache
            scale = mx.array(float(suffix_inputs[0]), dtype=mx.float32)
            gradients = tree_map(
                lambda parameter: mx.ones_like(parameter) * scale,
                model.trainable_parameters(),
            )
            return scale, mx.array(targets), gradients

        with (
            mock.patch.object(
                segmented_lora,
                "_assert_dataset_matches_receipt",
                return_value=None,
            ),
            mock.patch.object(
                segmented_lora,
                "split_supervised_tokens",
                side_effect=fake_split,
            ),
            mock.patch.object(
                segmented_lora,
                "_materialize_prompt_cache",
                return_value=None,
            ),
            mock.patch.object(
                segmented_lora,
                "_target_value_and_grad",
                side_effect=fake_target,
            ),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            train_with_exposure_prefix_cache(
                model,
                optimizer,
                self._dataset(),
                args=self._args(output / "adapters.safetensors"),
            )
        mx.eval(model.trainable_parameters(), optimizer.state)
        return model, optimizer, output

    def test_split_run_matches_uninterrupted_weights_and_adam_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            seed_model = nn.Linear(1, 1)
            initial_weights = [
                (name, mx.array(value))
                for name, value in tree_flatten(seed_model.parameters())
            ]
            mx.eval([value for _, value in initial_weights])

            uninterrupted_model, uninterrupted_optimizer, _ = self._run(
                root=root,
                name="uninterrupted",
                initial_weights=initial_weights,
                segment={"enabled": False, "start": 0, "end": 4},
            )
            first_model, first_optimizer, first_output = self._run(
                root=root,
                name="segment-1",
                initial_weights=initial_weights,
                segment={
                    "enabled": True,
                    "start": 0,
                    "end": 2,
                    "adapter_input": None,
                    "optimizer_state_input": None,
                    "optimizer_state_output": root
                    / "segment-1"
                    / "optimizer.safetensors",
                },
            )
            del first_model, first_optimizer
            adapter_input = first_output / "adapters.safetensors"
            optimizer_input = first_output / "optimizer.safetensors"
            resumed_model, resumed_optimizer, resumed_output = self._run(
                root=root,
                name="segment-2",
                initial_weights=initial_weights,
                segment={
                    "enabled": True,
                    "start": 2,
                    "end": 4,
                    "adapter_input": adapter_input,
                    "adapter_input_sha256": _file_sha256(adapter_input),
                    "optimizer_state_input": optimizer_input,
                    "optimizer_state_input_sha256": _file_sha256(optimizer_input),
                    "optimizer_state_output": root
                    / "segment-2"
                    / "optimizer.safetensors",
                },
            )

            for (name, expected), (actual_name, actual) in zip(
                tree_flatten(uninterrupted_model.trainable_parameters()),
                tree_flatten(resumed_model.trainable_parameters()),
                strict=True,
            ):
                self.assertEqual(name, actual_name)
                self.assertTrue(bool(mx.array_equal(expected, actual).item()), name)
            uninterrupted_state = dict(
                tree_flatten(uninterrupted_optimizer.state)
            )
            resumed_state = dict(tree_flatten(resumed_optimizer.state))
            self.assertEqual(set(uninterrupted_state), set(resumed_state))
            for name, expected in uninterrupted_state.items():
                actual = resumed_state[name]
                self.assertTrue(bool(mx.array_equal(expected, actual).item()), name)
            self.assertEqual(
                int(resumed_optimizer.step.item()),
                2,
            )
            self.assertTrue((resumed_output / "adapters.safetensors").is_file())
            self.assertTrue((resumed_output / "optimizer.safetensors").is_file())
            self.assertTrue(
                (first_output / "0000002_adapters.safetensors").is_file()
            )
            self.assertTrue(
                (resumed_output / "0000004_adapters.safetensors").is_file()
            )
            self.assertEqual(
                (resumed_output / "adapters.safetensors").stat().st_mode & 0o777,
                0o600,
            )

    def test_non_finite_loss_fails_before_optimizer_update(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "output"
            output.mkdir()
            model = nn.Linear(1, 1)
            optimizer = optim.Adam(learning_rate=0.01)
            segmented_lora._RUNTIME_SCHEDULE = self._schedule(
                {
                    "enabled": True,
                    "start": 0,
                    "end": 2,
                    "adapter_input": None,
                    "optimizer_state_input": None,
                    "optimizer_state_output": output
                    / "optimizer.safetensors",
                }
            )

            def fake_split(tokens, prompt_offset, max_seq_length):
                del prompt_offset, max_seq_length
                row_number = int(tokens[0]) - 1
                return None, tokens, row_number + 2

            def fake_target(model, suffix_inputs, targets, cache):
                del suffix_inputs, cache
                gradients = tree_map(
                    mx.zeros_like,
                    model.trainable_parameters(),
                )
                return mx.array(float("nan")), mx.array(targets), gradients

            with (
                mock.patch.object(
                    segmented_lora,
                    "_assert_dataset_matches_receipt",
                    return_value=None,
                ),
                mock.patch.object(
                    segmented_lora,
                    "split_supervised_tokens",
                    side_effect=fake_split,
                ),
                mock.patch.object(
                    segmented_lora,
                    "_materialize_prompt_cache",
                    return_value=None,
                ),
                mock.patch.object(
                    segmented_lora,
                    "_target_value_and_grad",
                    side_effect=fake_target,
                ),
                mock.patch.object(optimizer, "update", wraps=optimizer.update) as update,
                self.assertRaisesRegex(
                    ExposurePrefixCacheLoraError,
                    "non-finite training loss at global microbatch 0",
                ),
            ):
                train_with_exposure_prefix_cache(
                    model,
                    optimizer,
                    self._dataset(),
                    args=self._args(output / "adapters.safetensors"),
                )

            update.assert_not_called()
            self.assertFalse((output / "adapters.safetensors").exists())
            self.assertFalse((output / "optimizer.safetensors").exists())

    def test_non_finite_accumulated_gradients_fail_before_update(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "output"
            output.mkdir()
            model = nn.Linear(1, 1)
            optimizer = optim.Adam(learning_rate=0.01)
            segmented_lora._RUNTIME_SCHEDULE = self._schedule(
                {
                    "enabled": True,
                    "start": 0,
                    "end": 2,
                    "adapter_input": None,
                    "optimizer_state_input": None,
                    "optimizer_state_output": output
                    / "optimizer.safetensors",
                }
            )

            def fake_split(tokens, prompt_offset, max_seq_length):
                del prompt_offset, max_seq_length
                row_number = int(tokens[0]) - 1
                return None, tokens, row_number + 2

            def fake_target(model, suffix_inputs, targets, cache):
                del suffix_inputs, cache
                gradients = tree_map(
                    lambda parameter: mx.full_like(parameter, float("inf")),
                    model.trainable_parameters(),
                )
                return mx.array(1.0), mx.array(targets), gradients

            with (
                mock.patch.object(
                    segmented_lora,
                    "_assert_dataset_matches_receipt",
                    return_value=None,
                ),
                mock.patch.object(
                    segmented_lora,
                    "split_supervised_tokens",
                    side_effect=fake_split,
                ),
                mock.patch.object(
                    segmented_lora,
                    "_materialize_prompt_cache",
                    return_value=None,
                ),
                mock.patch.object(
                    segmented_lora,
                    "_target_value_and_grad",
                    side_effect=fake_target,
                ),
                mock.patch.object(optimizer, "update", wraps=optimizer.update) as update,
                self.assertRaisesRegex(
                    ExposurePrefixCacheLoraError,
                    "accumulated gradients at global microbatch 0 contains "
                    "non-finite tensor values",
                ),
            ):
                train_with_exposure_prefix_cache(
                    model,
                    optimizer,
                    self._dataset(),
                    args=self._args(output / "adapters.safetensors"),
                )

            update.assert_not_called()
            self.assertFalse((output / "adapters.safetensors").exists())
            self.assertFalse((output / "optimizer.safetensors").exists())

    def test_non_finite_post_update_model_fails_before_save(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "output"
            output.mkdir()
            model = nn.Linear(1, 1)
            optimizer = optim.Adam(learning_rate=0.01)
            segmented_lora._RUNTIME_SCHEDULE = self._schedule(
                {
                    "enabled": True,
                    "start": 0,
                    "end": 2,
                    "adapter_input": None,
                    "optimizer_state_input": None,
                    "optimizer_state_output": output
                    / "optimizer.safetensors",
                }
            )

            def fake_split(tokens, prompt_offset, max_seq_length):
                del prompt_offset, max_seq_length
                row_number = int(tokens[0]) - 1
                return None, tokens, row_number + 2

            def fake_target(model, suffix_inputs, targets, cache):
                del suffix_inputs, cache
                gradients = tree_map(
                    mx.ones_like,
                    model.trainable_parameters(),
                )
                return mx.array(1.0), mx.array(targets), gradients

            def corrupt_update(model, gradients):
                del gradients
                model.update(
                    tree_map(
                        lambda parameter: mx.full_like(
                            parameter,
                            float("nan"),
                        ),
                        model.trainable_parameters(),
                    )
                )

            with (
                mock.patch.object(
                    segmented_lora,
                    "_assert_dataset_matches_receipt",
                    return_value=None,
                ),
                mock.patch.object(
                    segmented_lora,
                    "split_supervised_tokens",
                    side_effect=fake_split,
                ),
                mock.patch.object(
                    segmented_lora,
                    "_materialize_prompt_cache",
                    return_value=None,
                ),
                mock.patch.object(
                    segmented_lora,
                    "_target_value_and_grad",
                    side_effect=fake_target,
                ),
                mock.patch.object(
                    optimizer,
                    "update",
                    side_effect=corrupt_update,
                ),
                self.assertRaisesRegex(
                    ExposurePrefixCacheLoraError,
                    "model parameters after optimizer step 1 contains "
                    "non-finite tensor values",
                ),
            ):
                train_with_exposure_prefix_cache(
                    model,
                    optimizer,
                    self._dataset(),
                    args=self._args(output / "adapters.safetensors"),
                )

            self.assertFalse((output / "adapters.safetensors").exists())
            self.assertFalse((output / "optimizer.safetensors").exists())

    def test_resumed_segment_rejects_non_finite_adapter_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            output_dir.mkdir()
            model = nn.Linear(1, 1)
            optimizer = optim.Adam(learning_rate=0.01)
            optimizer.init(model.trainable_parameters())
            adapter_input = input_dir / "adapters.safetensors"
            optimizer_input = input_dir / "optimizer.safetensors"
            weights = dict(tree_flatten(model.trainable_parameters()))
            first_name = next(iter(weights))
            weights[first_name] = mx.full_like(
                weights[first_name],
                float("nan"),
            )
            mx.save_safetensors(str(adapter_input), weights)
            mx.save_safetensors(
                str(optimizer_input),
                dict(tree_flatten(optimizer.state)),
            )
            segmented_lora._RUNTIME_SCHEDULE = self._schedule(
                {
                    "enabled": True,
                    "start": 2,
                    "end": 4,
                    "adapter_input": adapter_input,
                    "adapter_input_sha256": _file_sha256(adapter_input),
                    "optimizer_state_input": optimizer_input,
                    "optimizer_state_input_sha256": _file_sha256(optimizer_input),
                    "optimizer_state_output": output_dir
                    / "optimizer.safetensors",
                }
            )

            with (
                mock.patch.object(
                    segmented_lora,
                    "_assert_dataset_matches_receipt",
                    return_value=None,
                ),
                self.assertRaisesRegex(
                    ExposurePrefixCacheLoraError,
                    "segment adapter input contains non-finite tensor values",
                ),
            ):
                train_with_exposure_prefix_cache(
                    model,
                    optim.Adam(learning_rate=0.01),
                    self._dataset(),
                    args=self._args(output_dir / "adapters.safetensors"),
                )

            self.assertFalse((output_dir / "adapters.safetensors").exists())
            self.assertFalse((output_dir / "optimizer.safetensors").exists())

    def test_segment_outputs_are_create_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            seed_model = nn.Linear(1, 1)
            initial_weights = [
                (name, mx.array(value))
                for name, value in tree_flatten(seed_model.parameters())
            ]
            output = root / "occupied"
            output.mkdir()
            occupied_adapter = output / "adapters.safetensors"
            occupied_adapter.write_bytes(b"immutable")
            before = hashlib.sha256(occupied_adapter.read_bytes()).hexdigest()
            model = self._new_model(initial_weights)
            optimizer = optim.Adam(learning_rate=0.01)
            segmented_lora._RUNTIME_SCHEDULE = self._schedule(
                {
                    "enabled": True,
                    "start": 0,
                    "end": 2,
                    "adapter_input": None,
                    "optimizer_state_input": None,
                    "optimizer_state_output": output / "optimizer.safetensors",
                }
            )

            with (
                mock.patch.object(
                    segmented_lora,
                    "_assert_dataset_matches_receipt",
                    return_value=None,
                ),
                self.assertRaisesRegex(
                    ExposurePrefixCacheLoraError,
                    "segment adapter output already exists",
                ),
            ):
                train_with_exposure_prefix_cache(
                    model,
                    optimizer,
                    self._dataset(),
                    args=self._args(occupied_adapter),
                )

            self.assertEqual(
                hashlib.sha256(occupied_adapter.read_bytes()).hexdigest(),
                before,
            )
            self.assertFalse((output / "optimizer.safetensors").exists())

    def test_resumed_segment_rejects_wrong_optimizer_step(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            output_dir.mkdir()
            model = nn.Linear(1, 1)
            optimizer = optim.Adam(learning_rate=0.01)
            optimizer.init(model.trainable_parameters())
            adapter_input = input_dir / "adapters.safetensors"
            optimizer_input = input_dir / "optimizer.safetensors"
            mx.save_safetensors(
                str(adapter_input),
                dict(tree_flatten(model.trainable_parameters())),
            )
            mx.save_safetensors(
                str(optimizer_input),
                dict(tree_flatten(optimizer.state)),
            )
            segmented_lora._RUNTIME_SCHEDULE = self._schedule(
                {
                    "enabled": True,
                    "start": 2,
                    "end": 4,
                    "adapter_input": adapter_input,
                    "adapter_input_sha256": _file_sha256(adapter_input),
                    "optimizer_state_input": optimizer_input,
                    "optimizer_state_input_sha256": _file_sha256(optimizer_input),
                    "optimizer_state_output": output_dir
                    / "optimizer.safetensors",
                }
            )

            with (
                mock.patch.object(
                    segmented_lora,
                    "_assert_dataset_matches_receipt",
                    return_value=None,
                ),
                self.assertRaisesRegex(
                    ExposurePrefixCacheLoraError,
                    "optimizer step does not match",
                ),
            ):
                train_with_exposure_prefix_cache(
                    model,
                    optim.Adam(learning_rate=0.01),
                    self._dataset(),
                    args=self._args(output_dir / "adapters.safetensors"),
                )

            self.assertFalse((output_dir / "adapters.safetensors").exists())
            self.assertFalse((output_dir / "optimizer.safetensors").exists())


if __name__ == "__main__":
    unittest.main()
