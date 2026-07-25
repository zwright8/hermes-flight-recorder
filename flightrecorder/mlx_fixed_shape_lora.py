"""Run MLX-LM LoRA training with one fixed token shape per batch.

MLX compiles a separate graph for each input shape. The upstream LoRA iterator
pads only to the longest sequence in a batch, which is efficient for ordinary
datasets but pathological for batch-size-one training over long, differently
sized agent transcripts. This entry point preserves every example's true loss
boundary while padding the tensor shape to ``max_seq_length`` so the compiled
training graph can be reused.

The module imports MLX-LM lazily because it is an optional training dependency,
not a dependency of the Flight Recorder core package.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from typing import Any


def fixed_shape_iterate_batches(
    dataset: Any,
    batch_size: int,
    max_seq_length: int,
    loop: bool = False,
    seed: int | None = None,
    comm_group: Any = None,
) -> Iterator[tuple[Any, Any]]:
    """Yield upstream-compatible batches padded to one governed fixed shape."""

    import mlx.core as mx
    import numpy as np
    from mlx_lm.tuner.datasets import CacheDataset

    if isinstance(dataset, CacheDataset):
        len_fn = lambda idx: dataset.itemlen(idx)
    else:
        len_fn = lambda idx: len(dataset[idx][0])
    ordered = sorted(range(len(dataset)), key=len_fn)
    if len(dataset) < batch_size:
        raise ValueError(
            f"Dataset must have at least batch_size={batch_size} examples "
            f"but only has {len(dataset)}."
        )

    if comm_group is not None:
        offset = comm_group.rank()
        step = comm_group.size()
    else:
        offset = 0
        step = 1
    if batch_size % step != 0:
        raise ValueError(
            "The batch size must be divisible by the number of workers"
        )

    batch_indices = [
        ordered[index + offset : index + offset + batch_size : step]
        for index in range(0, len(ordered) - batch_size + 1, batch_size)
    ]
    if seed is not None:
        np.random.seed(seed)

    while True:
        for shuffled_index in np.random.permutation(len(batch_indices)):
            batch = [dataset[index] for index in batch_indices[shuffled_index]]
            if len(batch[0]) == 2:
                batch, offsets = zip(*batch)
            else:
                offsets = [0] * len(batch)

            lengths = [len(tokens) for tokens in batch]
            batch_array = np.zeros(
                (batch_size // step, max_seq_length),
                dtype=np.int32,
            )
            for row_index, tokens in enumerate(batch):
                true_length = min(lengths[row_index], max_seq_length)
                batch_array[row_index, :true_length] = tokens[:true_length]
                lengths[row_index] = true_length

            # The true lengths and prompt offsets keep all fixed-shape padding
            # outside the loss mask even though the model input is rectangular.
            yield mx.array(batch_array), mx.array(list(zip(offsets, lengths)))

        if not loop:
            break


def main() -> None:
    """Patch only MLX-LM's training iterator, then run its normal LoRA CLI."""

    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    import mlx_lm.lora as lora
    from mlx_lm.tuner.trainer import train as upstream_train

    def fixed_shape_train(*args: Any, **kwargs: Any) -> Any:
        kwargs["iterate_batches"] = fixed_shape_iterate_batches
        return upstream_train(*args, **kwargs)

    lora.train = fixed_shape_train
    lora.main()


if __name__ == "__main__":
    main()
