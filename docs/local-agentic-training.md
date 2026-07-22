# Local Agentic LoRA Training

Hermes Flight Recorder can run a small, explicitly authorized LoRA trial from
registered local artifacts. The intended first use is a user-defined agentic
task such as exact-schema tool calling: capture the task as Flight Recorder
scenarios, export and review the trajectories, then select its `task_family`
for local action-SFT.

The core package remains dependency-free. The optional trainer lives in
[`scripts/train_agentic_lora.py`](../scripts/train_agentic_lora.py) and imports
PyTorch, Transformers, Datasets, TRL, PEFT, and Accelerate only after its plan
passes. No trainer dependency is imported by `--dry-run`.

## What was adapted from autoresearch

This path adapts the experimental controls—not the pretraining implementation—
from [karpathy/autoresearch](https://github.com/karpathy/autoresearch) and the
pinned Apple Silicon fork
[miolini/autoresearch-macos@537c6e6](https://github.com/miolini/autoresearch-macos/commit/537c6e6d0ecf7d28f9d70ce20bb05d8c7ed9cfce):

- a fixed trainer-active wall-clock budget;
- a frozen, content-addressed dataset and task selector;
- a bounded LoRA recipe surface;
- baseline/result artifacts with measured runtime and memory;
- explicit stop conditions instead of an unbounded overnight loop.

No upstream model, tokenizer, downloader, optimizer, or training-loop source is
copied. The referenced fork's pinned tree has no license file, despite its
README describing the project as MIT, so it is treated as a design reference.

## 1. Build governed task artifacts

Run the normal capture, review, governance, contamination, curation, action
credit, branch replay, and dataset export flow described in
[`agentic-training-huggingface.md`](agentic-training-huggingface.md). Each row
must carry the exact `task_family` you intend to select.

For tool-use imitation, use `fr_action_sft`. Its input must be the reviewed
`flightrecorder_action_sft` view. Flight Recorder admits only trajectories with
native assistant `tool_calls`, matching tool results and call IDs, recorded
exact tool schemas, complete governance, and accepted per-action credit. Plain
SFT rows are not accepted as a substitute by this executable mode.

## 2. Register an already-local model

Local mode never downloads a model or tokenizer. Put a Transformers-compatible
model directory on local storage and register that exact directory as the model
identity after reviewing its license:

```json
{
  "schema_version": "hfr.model_candidate.v1",
  "model_id": "Qwen/Qwen3-0.6B",
  "source": {
    "type": "huggingface_model",
    "revision": "immutable-reviewed-revision"
  },
  "license": {
    "status": "approved",
    "training_allowed": true
  },
  "compatibility": {
    "tokenizer": "available",
    "chat_template": "messages_and_tools",
    "serving": "transformers",
    "tool_calls": "native"
  }
}
```

The selected `--model` must match the registered canonical `model_id`.
`--local-model-path` supplies the existing, non-symlink directory used for
offline loading, so machine-specific cache paths do not replace the reviewed
model identity. Keep the optional ML stack in a separate trainer virtual
environment; it is not a Flight Recorder runtime dependency. The output must
be outside the base-model directory, and populated adapter targets are rejected
so an ordinary launch cannot overwrite prior weights or evidence.

## 3. Dry-run the exact task and recipe

```bash
.venv/bin/python scripts/train_agentic_lora.py \
  --mode fr_action_sft \
  --dry-run \
  --local-training \
  --model Qwen/Qwen3-0.6B \
  --local-model-path /absolute/path/to/local/model \
  --model-manifest runs/training/model_manifest.json \
  --dataset-manifest runs/training/dataset_training_manifest.json \
  --experiment-dir runs/training \
  --output-dir runs/local_tool_training \
  --task-family tool_calling \
  --device mps \
  --max-training-seconds 300 \
  --limit 32 \
  --disable-trackio
```

`--task-family` is an exact selector and may be repeated. The plan fails when
any requested family is absent, when the selected mode has no scoped rows, or
when any local/offline boundary is violated.

## 4. Preflight without loading a model

Run the same command with `--preflight-only` instead of `--dry-run`. It checks
that the optional trainer modules are discoverable and writes a classified
result without importing them, allocating a device, loading a model, or
starting training.

## 5. Explicitly execute the local trial

Real execution updates local adapter weights. It therefore requires both
`--local-training` and `--execute-local-training`:

```bash
path/to/trainer-venv/bin/python scripts/train_agentic_lora.py \
  --mode fr_action_sft \
  --local-training \
  --execute-local-training \
  --model Qwen/Qwen3-0.6B \
  --local-model-path /absolute/path/to/local/model \
  --model-manifest runs/training/model_manifest.json \
  --dataset-manifest runs/training/dataset_training_manifest.json \
  --experiment-dir runs/training \
  --output-dir runs/local_tool_training \
  --task-family tool_calling \
  --device mps \
  --max-training-seconds 300 \
  --limit 32 \
  --batch-size 1 \
  --gradient-accumulation-steps 8 \
  --lora-r 16 \
  --lora-alpha 32 \
  --disable-trackio
```

Local mode sets the Hugging Face and Transformers offline environment before
trainer imports, passes `local_files_only` to tokenizer/model loads, prohibits
Hub pushes and remote Trackio, and shares one wall-clock budget across SFT and
DPO phases. Device `auto` prefers CUDA, then Apple MPS, then CPU; `--device mps`
fails if Metal is unavailable. MPS uses float16, does not enable
`torch.compile`, and reports both process peak RSS and available Metal allocator
memory. The budget is checked at optimizer-step boundaries, so one in-flight
step may finish after the nominal deadline before the trainer stops.

The result fingerprints every final adapter file and records device, dtype,
trainer-active seconds, whether the time budget stopped the run, process
memory, accelerator memory, and selected task families. It is training
evidence—not promotion evidence. Evaluate the exact adapter against untouched
frozen, rolling, and adversarial Flight Recorder pools before any registry alias
or deployment decision.
