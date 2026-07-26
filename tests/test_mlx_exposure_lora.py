from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from flightrecorder import mlx_exposure_lora
from flightrecorder.mlx_exposure_lora import (
    ExposureLoraError,
    exposure_iterate_batches,
    install_exposure_schedule,
    load_exposure_schedule,
    make_exposure_train,
)
from flightrecorder.tau3_exposure import build_tau3_exposure_ledger
from flightrecorder.tau3_prefix_equivalence_sample import SAMPLE_STRATA


class ChatDataset:
    def __init__(self, rows, tokenizer, mask_prompt=True):
        self._data = rows
        self.tokenizer = tokenizer
        self.mask_prompt = mask_prompt

    def __getitem__(self, index):
        return self._data[index]

    def process(self, row):
        return self.tokenizer.apply_chat_template(row["messages"]), 1


class CacheDataset:
    def __init__(self, dataset):
        self._data = dataset

    def __getitem__(self, index):
        return self._data[index]


class ConcatenatedDataset:
    def __init__(self, datasets):
        self._data = datasets


def _fake_mlx_dataset_modules() -> dict[str, types.ModuleType]:
    mlx_lm = types.ModuleType("mlx_lm")
    mlx_lm.__path__ = []
    tuner = types.ModuleType("mlx_lm.tuner")
    tuner.__path__ = []
    datasets = types.ModuleType("mlx_lm.tuner.datasets")
    datasets.CacheDataset = CacheDataset
    datasets.ChatDataset = ChatDataset
    datasets.ConcatenatedDataset = ConcatenatedDataset
    mlx_lm.tuner = tuner
    tuner.datasets = datasets
    return {
        "mlx_lm": mlx_lm,
        "mlx_lm.tuner": tuner,
        "mlx_lm.tuner.datasets": datasets,
    }


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


class FakeTokenizer:
    def apply_chat_template(
        self,
        messages,
        tools=None,
        add_generation_prompt=False,
        return_dict=False,
    ):
        del tools, add_generation_prompt, return_dict
        text = " ".join(str(message.get("content", "")) for message in messages)
        digits = "".join(character for character in text if character.isdigit())
        base = int(digits or 0) % 1000
        return [base, base + 1, base + 2]


def _chat_dataset(rows: list[dict]) -> ChatDataset:
    return ChatDataset(rows, FakeTokenizer(), mask_prompt=True)


def _cache_dataset(rows: list[dict]) -> CacheDataset:
    return CacheDataset(_chat_dataset(rows))


class MlxExposureLoraTests(unittest.TestCase):
    def setUp(self) -> None:
        self._mlx_modules = mock.patch.dict(
            sys.modules,
            _fake_mlx_dataset_modules(),
        )
        self._mlx_modules.start()

    def tearDown(self) -> None:
        self._mlx_modules.stop()

    def _schedule_fixture(self, root: Path) -> tuple[Path, dict]:
        dataset = root / "train.jsonl"
        _write_jsonl(dataset, [_row(index) for index in range(52)])
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
            ledger_path=root / "exposure" / "training_exposure_ledger.jsonl",
            batch_size=2,
            grad_accumulation_steps=2,
            iters=52,
        )
        return dataset, schedule

    def test_load_exposure_schedule_extracts_optimizer_microbatches(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "train.jsonl"
            _write_jsonl(dataset, [_row(index) for index in range(52)])
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
                ledger_path=root / "exposure" / "training_exposure_ledger.jsonl",
                batch_size=2,
                grad_accumulation_steps=2,
                iters=52,
            )

            self.assertEqual(len(schedule["steps"]), 26)
            self.assertEqual(schedule["microbatch_iterations"], 52)
            self.assertEqual(schedule["optimizer_steps"], 26)
            self.assertEqual(len(schedule["steps"][0]), 2)
            self.assertEqual(len(schedule["steps"][0][0]), 2)
            self.assertTrue(schedule["receipt"]["passed"])
            self.assertTrue(schedule["validation"]["passed"])

    def test_load_exposure_schedule_rejects_recipe_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "train.jsonl"
            _write_jsonl(dataset, [_row(index) for index in range(52)])
            receipt = build_tau3_exposure_ledger(
                dataset,
                root / "exposure",
                seed=101,
                epochs=2,
                batch_size=2,
                gradient_accumulation_steps=2,
            )

            with self.assertRaisesRegex(ExposureLoraError, "recipe does not match"):
                load_exposure_schedule(
                    dataset_jsonl=dataset,
                    receipt_path=receipt["receipt_path"],
                    ledger_path=root / "exposure" / "training_exposure_ledger.jsonl",
                    batch_size=4,
                    grad_accumulation_steps=1,
                    iters=52,
                )

    def test_bounded_smoke_schedule_allows_only_behavior_completeness_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "train.jsonl"
            rows = []
            for index, (domain, behavior) in enumerate(SAMPLE_STRATA):
                row = _row(index)
                row["metadata"]["domain"] = domain
                row["metadata"]["behavior"] = behavior
                rows.append(row)
            _write_jsonl(dataset, rows)
            receipt = build_tau3_exposure_ledger(
                dataset,
                root / "exposure",
                seed=101,
                epochs=2,
                batch_size=1,
                gradient_accumulation_steps=4,
            )

            with self.assertRaisesRegex(
                ExposureLoraError,
                "candidate eligible",
            ):
                load_exposure_schedule(
                    dataset_jsonl=dataset,
                    receipt_path=receipt["receipt_path"],
                    ledger_path=root
                    / "exposure"
                    / "training_exposure_ledger.jsonl",
                    batch_size=1,
                    grad_accumulation_steps=4,
                    iters=8,
                )

            schedule = load_exposure_schedule(
                dataset_jsonl=dataset,
                receipt_path=receipt["receipt_path"],
                ledger_path=root
                / "exposure"
                / "training_exposure_ledger.jsonl",
                batch_size=1,
                grad_accumulation_steps=4,
                iters=8,
                bounded_smoke=True,
            )

            self.assertFalse(schedule["receipt"]["passed"])
            self.assertEqual(schedule["microbatch_iterations"], 8)

    def test_exposure_iterator_yields_every_ledger_microbatch_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_path, schedule = self._schedule_fixture(root)
            raw_rows = [
                json.loads(line)
                for line in dataset_path.read_text(encoding="utf-8").splitlines()
            ]

            dataset = _cache_dataset(raw_rows)
            install_exposure_schedule(schedule)
            yielded: list[tuple[int, ...]] = []

            def fake_batch(dataset, indices, *, batch_size, max_seq_length):
                yielded.append(tuple(indices))
                return indices, []

            with mock.patch.object(mlx_exposure_lora, "_batch_from_indices", fake_batch):
                batches = list(
                    exposure_iterate_batches(
                        dataset,
                        batch_size=2,
                        max_seq_length=128,
                        loop=False,
                    )
                )

            expected = [
                tuple(microbatch)
                for step in schedule["steps"]
                for microbatch in step
            ]
            self.assertEqual(yielded, expected)
            self.assertEqual(len(batches), 52)

    def test_exposure_train_uses_standard_iterator_for_validation_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_path, schedule = self._schedule_fixture(root)
            raw_rows = [
                json.loads(line)
                for line in dataset_path.read_text(encoding="utf-8").splitlines()
            ]

            train_dataset = _cache_dataset(raw_rows)
            valid_dataset = _cache_dataset(raw_rows[:4])
            install_exposure_schedule(schedule)
            calls: list[str] = []

            def standard_iterate_batches(dataset, *args, **kwargs):
                calls.append("valid" if dataset is valid_dataset else "train")
                return iter([("standard", dataset)])

            def upstream_train(model, optimizer, train_dataset, val_dataset=None, iterate_batches=None):
                train_iter = iterate_batches(
                    train_dataset,
                    batch_size=2,
                    max_seq_length=128,
                    loop=True,
                )
                first_train = next(train_iter)
                valid_iter = iterate_batches(
                    val_dataset,
                    batch_size=2,
                    max_seq_length=128,
                    loop=False,
                )
                first_valid = next(valid_iter)
                return first_train, first_valid

            with mock.patch.object(
                mlx_exposure_lora,
                "_batch_from_indices",
                lambda dataset, indices, **kwargs: (tuple(indices), []),
            ):
                result = make_exposure_train(upstream_train, standard_iterate_batches)(
                    object(),
                    object(),
                    train_dataset,
                    valid_dataset,
                )

            self.assertIsInstance(result[0][0], tuple)
            self.assertEqual(result[1][0], "standard")
            self.assertEqual(calls, ["valid"])

    def test_wrong_train_dataset_cannot_bypass_exposure_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _dataset_path, schedule = self._schedule_fixture(root)

            install_exposure_schedule(schedule)

            def upstream_train(*args, **kwargs):
                raise AssertionError("upstream train should not be called")

            with self.assertRaisesRegex(ExposureLoraError, "dataset order/content"):
                make_exposure_train(upstream_train, lambda *args, **kwargs: iter(()))(
                    object(),
                    object(),
                    _cache_dataset([_row(999)]),
                    None,
                )

    def test_supported_mlx_cache_dataset_hashes_and_indexes_batches(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_path, schedule = self._schedule_fixture(root)
            raw_rows = [
                json.loads(line)
                for line in dataset_path.read_text(encoding="utf-8").splitlines()
            ]
            cache_dataset = _cache_dataset(raw_rows)
            chat_dataset = _chat_dataset(raw_rows)
            install_exposure_schedule(schedule)
            first_microbatch = schedule["steps"][0][0]
            with mock.patch.object(
                mlx_exposure_lora,
                "_batch_from_indices",
                lambda dataset, indices, **kwargs: (tuple(indices), []),
            ):
                batch = next(
                    exposure_iterate_batches(
                        cache_dataset,
                        batch_size=2,
                        max_seq_length=128,
                    )
                )
                direct_batch = next(
                    exposure_iterate_batches(
                        chat_dataset,
                        batch_size=2,
                        max_seq_length=128,
                    )
                )
            self.assertEqual(batch[0], tuple(first_microbatch))
            self.assertEqual(direct_batch[0], tuple(first_microbatch))

    def test_unknown_and_concatenated_dataset_wrappers_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_path, schedule = self._schedule_fixture(root)
            raw_rows = [
                json.loads(line)
                for line in dataset_path.read_text(encoding="utf-8").splitlines()
            ]
            install_exposure_schedule(schedule)

            class UnknownWrapper:
                _data = raw_rows

            with self.assertRaisesRegex(ExposureLoraError, "unsupported"):
                next(
                    exposure_iterate_batches(
                        UnknownWrapper(),
                        batch_size=2,
                        max_seq_length=128,
                    )
                )
            concatenated = ConcatenatedDataset([_chat_dataset(raw_rows)])
            with self.assertRaisesRegex(ExposureLoraError, "ConcatenatedDataset"):
                next(
                    exposure_iterate_batches(
                        concatenated,
                        batch_size=2,
                        max_seq_length=128,
                    )
                )

    def test_overlength_and_invalid_prompt_offset_fail_before_truncation(self) -> None:
        class Dataset:
            _data = []

            def __init__(self, item):
                self.item = item

            def __getitem__(self, index: int):
                return self.item

        with self.assertRaisesRegex(ExposureLoraError, "exceeds max_seq_length"):
            mlx_exposure_lora._batch_from_indices(
                Dataset(([1, 2, 3, 4], 1)),
                [0],
                batch_size=1,
                max_seq_length=3,
            )
        with self.assertRaisesRegex(ExposureLoraError, "prompt offset"):
            mlx_exposure_lora._batch_from_indices(
                Dataset(([1, 2, 3], 3)),
                [0],
                batch_size=1,
                max_seq_length=3,
            )

        _, lengths = mlx_exposure_lora._batch_from_indices(
            Dataset(([1, 2, 3, 4], 2)),
            [0],
            batch_size=1,
            max_seq_length=4,
        )
        self.assertEqual(lengths.tolist(), [[2, 3]])


if __name__ == "__main__":
    unittest.main()
