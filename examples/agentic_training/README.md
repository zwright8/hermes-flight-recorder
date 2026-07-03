# Agentic Training Plan Example

This directory contains a tiny registered-input fixture for the Goal 5 Training
Layer. It is safe to commit because it uses synthetic, redacted rows and a local
test model manifest.

Regenerate the sample plan with:

```bash
python3 scripts/plan_agentic_training.py \
  --mode sft_then_dpo \
  --model-manifest examples/agentic_training/model_manifest.json \
  --dataset-manifest examples/agentic_training/dataset_manifest.json \
  --trainer-backend axolotl \
  --output-dir runs/agentic_training/adapters \
  --limit 2 \
  --created-at 2026-07-02T00:00:00+00:00 \
  --out examples/agentic_training/plans/sft_then_dpo_plan.json
```

The plan is a handoff contract only. It does not import trainer packages,
download model weights, mutate aliases, or launch training.

Check the local tiny-smoke runtime boundary without importing trainer stacks:

```bash
python3 scripts/preflight_agentic_training_runtime.py \
  --plan examples/agentic_training/plans/sft_then_dpo_plan.json \
  --skip-default-modules \
  --require-module json \
  --created-at 2026-07-02T00:00:00+00:00 \
  --out examples/agentic_training/runtime_preflight/ready.json
```

The runtime preflight remains side-effect-free. It validates the plan and
selected trainer-view JSONL files, checks module discoverability with
`importlib.util.find_spec`, and records that Flight Recorder did not start
training or model downloads.

Record a synthetic external trainer result receipt with the committed
trainer-output fixture:

```bash
python3 scripts/archive_agentic_training_result.py \
  --plan examples/agentic_training/plans/sft_then_dpo_plan.json \
  --runtime-preflight examples/agentic_training/runtime_preflight/ready.json \
  --agentic-training-flow examples/agentic_training/agentic_training_flow.json \
  --status completed \
  --runner-id synthetic-example-runner \
  --run-id synthetic-completed-001 \
  --output-dir examples/agentic_training/trainer_outputs/adapter \
  --config examples/agentic_training/trainer_outputs/adapter/adapter_config.json \
  --adapter examples/agentic_training/trainer_outputs/adapter/adapter_model.safetensors \
  --metrics examples/agentic_training/trainer_outputs/metrics.json \
  --log examples/agentic_training/trainer_outputs/trainer.log \
  --created-at 2026-07-02T00:00:00+00:00 \
  --out examples/agentic_training/completed_result.json
```

The result receipt proposes size-bound model-registry links but does not mutate
registry entries, move aliases, download models, or train weights.

Bind the example receipts into a fail-closed loop contract:

```bash
flightrecorder agentic-loop plan \
  --iteration-id demo-loop-001 \
  --objective "Demonstrate a fail-closed closed-loop agentic training iteration contract." \
  --baseline local/mock-baseline \
  --candidate local/mock-candidate \
  --teacher local/mock-teacher \
  --provider mock \
  --region local \
  --gpu-class none \
  --budget max_cloud_cost_usd=0 \
  --budget max_gpu_hours=0 \
  --agentic-training-plan examples/agentic_training/plans/sft_then_dpo_plan.json \
  --agentic-training-runtime-preflight examples/agentic_training/runtime_preflight/ready.json \
  --agentic-training-result examples/agentic_training/completed_result.json \
  --cloud-training-provider-registry examples/agentic_training/cloud_training/provider_registry.json \
  --cloud-training-preflight examples/agentic_training/cloud_training/preflight.json \
  --cloud-training-artifact-manifest examples/agentic_training/cloud_training/artifact_manifest.json \
  --cloud-training-launch-plan examples/agentic_training/cloud_training/launch_plan.json \
  --cloud-training-launch-receipt examples/agentic_training/cloud_training/launch_receipt.json \
  --cloud-training-status-receipt examples/agentic_training/cloud_training/status_receipt.json \
  --created-at 2026-07-03T00:00:00+00:00 \
  --out examples/agentic_training/loop_plan.json
```

The committed plan is intentionally `planned_fail_closed` because this example
does not include rollout, review, local trainer-preflight, serving, held-out
eval, or promotion receipts. It does bind nested cloud-training receipts and
summarizes them in `cloud_training` without provider side effects.

Validate the committed receipt before including it in a trainer-facing evidence
bundle:

```bash
flightrecorder validate \
  --agentic-training-result examples/agentic_training/completed_result.json \
  --strict

flightrecorder schemas --check examples/agentic_training/loop_plan.json
flightrecorder validate \
  --agentic-loop-plan examples/agentic_training/loop_plan.json \
  --strict

flightrecorder agentic-loop ledger \
  --plan examples/agentic_training/loop_plan.json \
  --out examples/agentic_training/loop_ledger.json

flightrecorder validate \
  --agentic-loop-ledger examples/agentic_training/loop_ledger.json \
  --strict
```
