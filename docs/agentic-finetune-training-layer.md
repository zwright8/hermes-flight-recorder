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

The planner currently supports these default executable handoff modes:

- `sft`
- `action_sft`
- `dpo`
- `sft_then_dpo`

The planner also knows advanced extension modes, but they stay blocked by
default:

- `reward_model`
- `process_rewards`
- `grpo`
- `rl`

`reward_model` and `process_rewards` require `--allow-advanced-training`.
`grpo` and `rl` require `--allow-future-rl`. Those flags only allow planning;
they still do not launch a trainer, import trainer stacks, download models, or
update weights.
Every plan includes a `mode_contract` that makes this gate explicit. The
contract records the mode category, required trainer-view groups, selected data
evidence, reward-signal or reward-function requirements, and the side-effect
boundary that keeps Flight Recorder from starting trainers, cloud jobs, paid
grader calls, downloads, or weight updates.

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
- a `mode_contract` with data requirements and reward-contract details
- trainer backend extension points
- the external runner command shape
- `training_started: false`
- `model_downloads_started: false`
- `cloud_jobs_started: false`
- `paid_model_grader_calls_started: false`
- `weights_updated: false`
- `flight_recorder_executed_training: false`

External runners must revalidate the plan, manifests, license status, redaction
status, and current file hashes immediately before launch.
Run `flightrecorder validate --agentic-training-plan <plan.json>` before handing
the artifact to a trainer. The semantic validator closes the public plan shape
and rejects hidden provider job, credential, URL, live-training, model-download,
or weight-mutation fields in addition to the JSON Schema check.
For `grpo`, the reward contract is an interface record only:
`reward_fn(prompts, completions, **kwargs) -> list[float]`. For generic `rl`,
the recorded interface is `reward_fn(episodes, actions, **kwargs) -> list[float]`.
Flight Recorder does not provide, import, calibrate, or execute either
function; the external runner must supply and validate it before any live
training.

## Runtime Preflight for Tiny Smoke

Use `scripts/preflight_agentic_training_runtime.py` after a plan is ready and
before any bounded tiny-smoke trainer launch:

```bash
python3 scripts/preflight_agentic_training_runtime.py \
  --plan examples/agentic_training/plans/sft_then_dpo_plan.json \
  --require-module json \
  --skip-default-modules \
  --out runs/agentic_training_runtime_preflight.json
```

Runtime preflight emits `hfr.agentic_training_runtime_preflight.v1` and checks:

- the plan is schema-valid and still recommends
  `ready_for_external_trainer_plan`
- the plan `mode_contract` is present, matches the mode, has an open planning
  gate, satisfies its data requirements, keeps the reward contract external,
  and preserves fail-closed side-effect claims
- selected trainer-view JSONL files exist, pass their bundled row schemas, and
  match the manifest row counts
- required Python modules are discoverable with `importlib.util.find_spec`
- Flight Recorder did not import trainer stacks, download models, mutate
  weights, or launch training

Relative selected-view and input-manifest paths are resolved from the plan file,
its ancestor directories, and the dataset manifest location. The preflight does
not search the process CWD, so moved plans cannot pass by accidentally finding a
same-named local JSONL file.

By default the checker uses backend-specific dependency probes for common
external runners such as `axolotl`, `llama_factory`, `unsloth`, and process
reward wrappers. Use `--skip-default-modules` plus one or more
`--require-module` flags when CI needs a deterministic fixture or when an
external runner owns dependency resolution.

The command exits `0` for `ready_for_tiny_smoke_launch` and `1` for
`block_tiny_smoke_launch`; blocked artifacts are still schema-checkable so they
can be archived as failure evidence.

`flightrecorder agentic-training-flow` then binds a ready plan, runtime
preflight, and trainer consumer plan into `hfr.agentic_training_flow.v1`.
Default executable modes can reach `ready_for_delegated_trainer_execution`.
Advanced reward-model, process-reward, GRPO, and RL modes instead produce
`block_delegated_trainer_execution` receipts with a mirrored
`mode_contract_check` and `flow_mode_gate`. Those receipts are schema-checkable
evidence of why Flight Recorder did not delegate the trainer flow, including the
mode category, required opt-in flag, reward-contract obligations, and flow
promotion requirement.

## Result Receipts

After an external runner finishes, fails, or blocks before tiny smoke launch,
use `scripts/archive_agentic_training_result.py` to emit
`hfr.agentic_training_result.v1`:

```bash
python3 scripts/archive_agentic_training_result.py \
  --plan runs/agentic_training_plan.json \
  --runtime-preflight runs/agentic_training_runtime_preflight.json \
  --agentic-training-flow runs/agentic_training_flow.json \
  --status completed \
  --adapter runs/adapters/candidate/adapter.safetensors \
  --metrics runs/adapters/candidate/metrics.json \
  --out runs/agentic_training_result.json
```

Completed receipts require a ready runtime preflight, a ready delegated flow
receipt for the same plan/runtime pair, and at least one adapter or checkpoint
artifact. Failed, blocked, and aborted receipts require a
classified failure such as `dependency_missing`, `view_validation_failed`,
`trainer_crash`, `out_of_memory`, `timeout`, or `interrupted`. The receipt only
fingerprints supplied files and records lineage back to the plan, runtime
preflight, model manifest, and dataset manifest with SHA-256 plus byte-size
evidence; Flight Recorder still does not launch trainers, import trainer
stacks, download models, or mutate weights. Supplied artifact refs are recorded
relative to the receipt when possible, and validation reopens regular files to
reject hash or byte-size drift. The receipt also includes a byte-size-bound
`registry_update` proposal for `flightrecorder model-registry link`, leaving
`applied: false` until governance accepts the result. After the receipt exists,
validate it with `flightrecorder validate
--agentic-training-result runs/agentic_training_result.json --strict`, then
include it in the trainer-facing evidence bundle with `flightrecorder
evidence-bundle --agentic-training-result runs/agentic_training_result.json` so
the final preflight, wrapper dry-run, and result receipt are summarized under
`metrics.trainer_handoff`.

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
execute the trainer command. Agentic training plans bind model and dataset
manifest refs to SHA-256 plus byte-size evidence, and runtime preflight view
checks carry the same byte-size evidence for selected trainer views.

## Verification

Focused verification for this packet:

```bash
python3 -m unittest tests.test_agentic_training_plan
python3 -m flightrecorder validate --agentic-training-plan examples/agentic_training/plans/sft_then_dpo_plan.json --strict
python3 -m unittest tests.test_agentic_training_runtime
python3 -m unittest tests.test_schema_registry.SchemaRegistryTests.test_catalog_loads_public_artifact_contracts
python3 -m py_compile flightrecorder/agentic_training_plan.py flightrecorder/agentic_training_runtime.py scripts/plan_agentic_training.py scripts/preflight_agentic_training_runtime.py tests/test_agentic_training_plan.py tests/test_agentic_training_runtime.py
python3 -m flightrecorder schemas --name agentic_training_plan --write-dir /tmp/hfr-agentic-training-schema --force
python3 -m flightrecorder schemas --name agentic_training_runtime_preflight --write-dir /tmp/hfr-agentic-runtime-preflight-schema --force
python3 -m flightrecorder schemas --name agentic_training_result --write-dir /tmp/hfr-agentic-training-result-schema --force
python3 scripts/preflight_agentic_training_runtime.py --plan examples/agentic_training/plans/sft_then_dpo_plan.json --skip-default-modules --require-module json --out /tmp/hfr-agentic-runtime-preflight/ready.json
python3 -m unittest tests.test_agentic_training_result
python3 -m unittest tests.test_trainer_preflight.TrainerPreflightTests.test_trainer_preflight_archives_agentic_training_plan
```

The dedicated worktree was rebased onto `origin/main` on 2026-07-02 before the
runtime-preflight packet continued, so the shared autonomous Goal 5 docs are
available alongside this layer-specific contract.
