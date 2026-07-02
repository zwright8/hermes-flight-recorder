# Agentic Finetune Training Layer

Goal 5 owns reproducible, side-effect-free training paths from registered model
and dataset manifests into external trainer handoffs. Flight Recorder does not
download models, import trainer stacks, mutate weights, or train on unredacted
traces.

## Dry-Run Training Plans

Use `scripts/plan_agentic_training.py` to build a schema-checkable dry-run plan
before any trainer wrapper is allowed to run:

```bash
python3 scripts/plan_agentic_training.py \
  --mode sft_then_dpo \
  --model-manifest path/to/model_manifest.json \
  --dataset-manifest path/to/dataset_manifest.json \
  --trainer-backend axolotl \
  --output-dir runs/adapters/candidate \
  --limit 16 \
  --out runs/agentic_training_plan.json
```

The planner currently supports:

- `sft`
- `action_sft`
- `dpo`
- `sft_then_dpo`
- `reward_model`
- `process_rewards`
- `grpo`
- `rl`

`grpo` and `rl` are future extension points and remain blocked unless
`--allow-future-rl` is passed. That flag only allows planning; it still does not
launch a trainer.

## Required Registry Inputs

The model manifest must declare:

- a stable `model_id`, `id`, or `name`
- `license.status` in the approved/allowed/cleared/permissive/open family
- `license.allow_training: true` or `license.training_allowed: true`
- `compatibility.passed: true` or `training_compatibility.passed: true`

The dataset manifest must declare:

- a stable `dataset_id`, `id`, or `name`
- a known training license status and explicit training allowance
- `redaction.passed: true`
- `redaction.status` of `redacted`, `clean`, or `passed`
- `redaction.contains_unredacted_traces: false`
- mode-specific trainer views under `views`

Mode view requirements:

- `sft`: `sft`
- `action_sft`: `action_sft` or `sft`
- `dpo`: `dpo` or `preferences`
- `sft_then_dpo`: `sft` plus `dpo` or `preferences`
- `reward_model`: `reward_model` or `preferences`
- `process_rewards`: `process_rewards` or `step_rewards`
- `grpo` / `rl`: `episodes` or `rollouts`, plus explicit future-RL enablement

## Handoff Boundary

Plans use `hfr.agentic_training_plan.v1` and record:

- selected trainer views and stage sequence
- model and dataset manifest hashes
- readiness checks and blocked reasons
- trainer backend extension points
- the external runner command shape
- `training_started: false`
- `model_downloads_started: false`
- `flight_recorder_executed_training: false`

External runners must revalidate the plan, manifests, license status, redaction
status, and current file hashes immediately before launch.

## Durable Example

The repository includes a tiny synthetic fixture under
`examples/agentic_training/`:

- `model_manifest.json`
- `dataset_manifest.json`
- schema-conforming JSONL trainer views under `data/`
- `plans/sft_then_dpo_plan.json`

The committed plan can be regenerated with the command in
`examples/agentic_training/README.md`. It uses a fixed `--created-at` timestamp
so the sample is reproducible when the input manifests are unchanged.

## Trainer Preflight Bridge

Use `--agentic-training-plan` to carry a passed dry-run plan through the
existing trainer handoff pipeline:

```bash
flightrecorder trainer-preflight \
  --gate runs/evidence_bundle.json \
  --agentic-training-plan examples/agentic_training/plans/sft_then_dpo_plan.json \
  --trainer-command "python train.py --agentic-plan examples/agentic_training/plans/sft_then_dpo_plan.json" \
  --metadata launcher=agentic-dry-run \
  --out runs/trainer_preflight.json
```

The preflight records the plan as a hashed trainer artifact and validates it
against `hfr.agentic_training_plan.v1`. `trainer-archive`,
`trainer-archive-check`, and `trainer-consumer-plan` then copy the plan into
the portable archive and expose it as a runner input. None of these commands
execute the trainer command.

## Verification

Focused verification for this packet:

```bash
python3 -m unittest tests.test_agentic_training_plan
python3 -m unittest tests.test_schema_registry.SchemaRegistryTests.test_catalog_loads_public_artifact_contracts
python3 -m py_compile flightrecorder/agentic_training_plan.py scripts/plan_agentic_training.py tests/test_agentic_training_plan.py
python3 -m flightrecorder schemas --name agentic_training_plan --write-dir /tmp/hfr-agentic-training-schema --force
python3 -m unittest tests.test_trainer_preflight.TrainerPreflightTests.test_trainer_preflight_archives_agentic_training_plan
```

The dedicated worktree did not contain
`docs/agentic-finetune-24-7-goals.md`,
`docs/agentic-finetune-autonomous-operations.md`, or
`docs/agentic-finetune-autonomous-goals.md` at migration time, so this document
records the Goal 5 operating contract available in this branch.
