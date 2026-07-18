# Agentic Fine-Tune Training Layer

`scripts/train_agentic_lora.py` is the local TRL/PEFT trainer entry point for
Goal 5. It now writes a machine-readable plan before any heavy ML import and
defaults real launches to registry-backed safety checks.

## Supported Modes

- `trace_sft`: SFT over trace-only rows.
- `fr_sft`: SFT over Flight Recorder curated rows plus action rows.
- `fr_action_sft`: SFT over action/tool-decision rows only.
- `fr_dpo`: DPO over chosen/rejected rows.
- `fr_sft_dpo`: SFT, then DPO from the SFT adapter.
- `fr_reward_model`: dry-run extension point for reward-model training.
- `fr_step_rewards`: dry-run extension point for process/step rewards.

## Dry-Run Plans

Dry-run plans do not import `torch`, `datasets`, `transformers`, `peft`, or
`trl`.

```bash
python3 scripts/train_agentic_lora.py \
  --mode trace_sft \
  --dry-run \
  --experiment-dir experiments/qwen3_4b_flightrecorder \
  --output-dir experiments/qwen3_4b_flightrecorder/adapters \
  --disable-trackio

python3 scripts/train_agentic_lora.py \
  --mode fr_sft_dpo \
  --dry-run \
  --experiment-dir experiments/qwen3_4b_flightrecorder \
  --output-dir experiments/qwen3_4b_flightrecorder/adapters \
  --disable-trackio
```

Use `--limit N` for smoke plans. The plan records that a row limit is active
without mutating the full dataset files.

## Registered Smoke Fixture

Use the fixture writer when a worker needs a fully registered smoke path without
model downloads or GPU work:

```bash
python3 scripts/train_agentic_lora.py \
  --write-smoke-fixture /tmp/hfr-agentic-lora-smoke
```

The fixture writes a local model manifest, dataset manifest, and tiny trainer
views for trace SFT, curated SFT, action SFT, DPO, reward-model, and step-reward
planning. It also writes `smoke_fixture.json` with schema
`hfr.agentic_lora_smoke_fixture.v1`, so the fixture handoff itself can be
validated before downstream smoke plans consume it. Run a registry-required
smoke plan over it with a row limit:

```bash
python3 scripts/train_agentic_lora.py \
  --mode fr_sft_dpo \
  --dry-run \
  --require-registered-inputs \
  --experiment-dir /tmp/hfr-agentic-lora-smoke \
  --model-manifest /tmp/hfr-agentic-lora-smoke/registry/model_candidate.json \
  --dataset-manifest /tmp/hfr-agentic-lora-smoke/registry/dataset_version.json \
  --output-dir /tmp/hfr-agentic-lora-smoke/out \
  --limit 1 \
  --disable-trackio
```

The plan keeps both limited `prepared_counts` and `full_prepared_counts`, so the
smoke row limit is visible without hiding the complete fixture size.

## Registry-Backed Inputs

For registry-gated planning, pass both manifests and require them:

```bash
python3 scripts/train_agentic_lora.py \
  --mode fr_action_sft \
  --dry-run \
  --require-registered-inputs \
  --model-manifest experiments/registry/models/<candidate>.json \
  --dataset-manifest experiments/registry/datasets/<version>.json \
  --experiment-dir experiments/qwen3_4b_flightrecorder \
  --output-dir experiments/qwen3_4b_flightrecorder/adapters
```

The model manifest must provide a model id, known non-blocked license status,
`training_allowed` not false, and compatibility metadata. The dataset manifest
must provide a dataset identity/version, redacted or sanitized status, a passed
training gate, family-exclusive splits, clear quality flags, verified source
fingerprints, and trainer data-file paths or a wrapped Flight Recorder export
manifest.

Non-dry-run launches require registered model and dataset manifests by default.
The `--unsafe-allow-unregistered-launch` flag exists only as a legacy escape
hatch and should not be used for normal autonomous runs.

## Result Archives

Successful launches write `<mode>_result.json` with schema
`hfr.agentic_lora_training_result.v1`, the plan path, input manifests, adapter
directory, and training metrics. Runtime failures write the same result schema
with `status: failed` and a classified failure category such as
`missing_dependency`, `resource_exhausted`, `missing_training_rows`, or
`trainer_runtime_error`.

Non-dry-run launches that fail plan validation write a blocked result before any
heavy ML import:

```bash
python3 scripts/train_agentic_lora.py \
  --mode trace_sft \
  --experiment-dir experiments/qwen3_4b_flightrecorder \
  --output-dir experiments/qwen3_4b_flightrecorder/adapters \
  --result-registry experiments/registry/training_events.jsonl
```

Each non-dry-run result appends a JSONL event with schema
`hfr.agentic_lora_training_registry_event.v1` to `--result-registry`, or to
`<output-dir>/training_registry_events.jsonl` by default. These local registry
events record the mode, model, plan path, result path, input manifests, final
adapter directory when present, status, failed-check count, and classified
failure. They are handoff records, not promotion decisions.

## Launch Preflight

Use `--preflight-only` after a plan is expected to pass but before a smoke or
full launch. It writes the normal `<mode>_result.json` archive and registry
event with `status: preflight_passed` or `status: preflight_blocked`, then exits
before importing trainer modules, downloading model weights, allocating devices,
or starting training:

```bash
python3 scripts/train_agentic_lora.py \
  --mode fr_action_sft \
  --preflight-only \
  --require-registered-inputs \
  --model-manifest experiments/registry/models/<candidate>.json \
  --dataset-manifest experiments/registry/datasets/<version>.json \
  --experiment-dir experiments/qwen3_4b_flightrecorder \
  --output-dir experiments/qwen3_4b_flightrecorder/adapters \
  --disable-trackio
```

Preflight failures are classified like trainer failures. Missing local trainer
packages use `failure.category: missing_dependency` and are marked retryable.

## Backend Recipe Extensions

Use `--write-backend-recipes` to create side-effect-free handoff artifacts for
external trainer wrappers after a registry-backed plan passes. The command writes
the source `<mode>_plan.json`, a schema-checkable `backend_recipes.json`, and
inert JSON recipe files for Axolotl, LLaMA Factory, Unsloth, and reward/process
reward or future GRPO/RL extension points:

```bash
python3 scripts/train_agentic_lora.py \
  --mode fr_sft_dpo \
  --write-backend-recipes /tmp/hfr-agentic-lora-recipes \
  --require-registered-inputs \
  --model-manifest experiments/registry/models/<candidate>.json \
  --dataset-manifest experiments/registry/datasets/<version>.json \
  --experiment-dir experiments/qwen3_4b_flightrecorder \
  --output-dir experiments/qwen3_4b_flightrecorder/adapters \
  --limit 1 \
  --disable-trackio
```

The bundle uses `recommendation: ready_for_external_recipe_runner` only when the
source plan passes and both registered model and dataset manifests are present.
Flight Recorder does not execute the generated commands; external runners own
execution and must revalidate the source plan plus registry manifests before
launch.

## Model Registry Link Plans

Use `--write-model-registry-link-plan` with non-dry-run, preflight, or real
launch paths to write a side-effect-free plan for attaching the emitted
`<mode>_result.json` to a registered model entry:

```bash
python3 scripts/train_agentic_lora.py \
  --mode fr_action_sft \
  --preflight-only \
  --require-registered-inputs \
  --model-manifest experiments/registry/models/<candidate>.json \
  --dataset-manifest experiments/registry/datasets/<version>.json \
  --experiment-dir experiments/qwen3_4b_flightrecorder \
  --output-dir experiments/qwen3_4b_flightrecorder/adapters \
  --write-model-registry-link-plan experiments/registry/training_link_plan.json \
  --model-registry experiments/registry/model_registry.json \
  --model-registry-entry candidate \
  --disable-trackio
```

The link plan writes `hfr.agentic_lora_model_registry_link_plan.v1` and contains
exact `flightrecorder model-registry link` command arguments for the training
result and, after successful training, the final adapter directory. It never
mutates the registry and never moves `candidate`, `champion`, or `rollback`
aliases.

## Schema Checks

The training layer publishes bundled schemas for its direct artifacts:

- `agentic_lora_training_plan`
- `agentic_lora_training_result`
- `agentic_lora_training_registry_event`
- `agentic_lora_smoke_fixture`
- `agentic_lora_backend_recipes`
- `agentic_lora_model_registry_link_plan`

Use the existing schema checker on generated artifacts:

```bash
python3 -m flightrecorder schemas --check <fixture-dir>/smoke_fixture.json
python3 -m flightrecorder schemas --check <output-dir>/<mode>_plan.json
python3 -m flightrecorder schemas --check <output-dir>/<mode>_result.json
python3 -m flightrecorder schemas --check <recipe-dir>/backend_recipes.json
python3 -m flightrecorder schemas --check <link-plan.json>
python3 -m flightrecorder schemas --check-jsonl <registry-events.jsonl> --name agentic_lora_training_registry_event
```
