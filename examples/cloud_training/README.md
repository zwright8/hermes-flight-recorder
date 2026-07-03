# Cloud Training Contract Example

This directory contains deterministic, keyless cloud-training control-plane
receipts. They are safe to commit because they record provider capability and
dry-run launch contracts only. They do not call provider APIs, spend money,
download models, or update weights.
Each provider record includes an `adapter_contract` that names the implemented
receipt surfaces and attests that live launch is unsupported while live
preflight remains metadata-only.
The artifact manifest includes a `transfer_plan` for upload/download counts,
provider protocols, and explicit no-transfer/no-provider-call side-effect
flags.

Regenerate the fixtures with:

```bash
flightrecorder cloud-training providers \
  --provider modal \
  --provider huggingface_jobs \
  --created-at 2026-07-03T00:00:00+00:00 \
  --out examples/cloud_training/provider_registry.json

flightrecorder cloud-training artifacts \
  --provider modal \
  --upload examples/agentic_training/plans/sft_then_dpo_plan.json \
  --download adapters/candidate/adapter_model.safetensors \
  --created-at 2026-07-03T00:00:00+00:00 \
  --out examples/cloud_training/artifact_manifest.json

flightrecorder cloud-training preflight \
  --provider modal \
  --agentic-training-plan examples/agentic_training/plans/sft_then_dpo_plan.json \
  --region provider_default \
  --gpu-class a100 \
  --max-cost-usd 0 \
  --created-at 2026-07-03T00:00:00+00:00 \
  --out examples/cloud_training/preflight.json
```

The preflight is intentionally blocked because the example does not include
`trainer_preflight` or `trainer_launch_check` receipts. Continue the dry-run
chain with:

```bash
flightrecorder cloud-training plan \
  --preflight examples/cloud_training/preflight.json \
  --artifact-manifest examples/cloud_training/artifact_manifest.json \
  --created-at 2026-07-03T00:00:00+00:00 \
  --out examples/cloud_training/launch_plan.json

flightrecorder cloud-training launch \
  --launch-plan examples/cloud_training/launch_plan.json \
  --created-at 2026-07-03T00:00:00+00:00 \
  --out examples/cloud_training/launch_receipt.json

flightrecorder cloud-training launch \
  --launch-plan examples/cloud_training/launch_plan.json \
  --live \
  --created-at 2026-07-03T00:00:00+00:00 \
  --out examples/cloud_training/live_blocked_receipt.json

flightrecorder cloud-training status \
  --launch-receipt examples/cloud_training/launch_receipt.json \
  --cancel \
  --created-at 2026-07-03T00:00:00+00:00 \
  --out examples/cloud_training/status_receipt.json
```

Validate the contracts:

```bash
flightrecorder validate \
  --cloud-training-provider-registry examples/cloud_training/provider_registry.json \
  --cloud-training-artifact-manifest examples/cloud_training/artifact_manifest.json \
  --cloud-training-preflight examples/cloud_training/preflight.json \
  --cloud-training-launch-plan examples/cloud_training/launch_plan.json \
  --cloud-training-launch-receipt examples/cloud_training/launch_receipt.json \
  --cloud-training-launch-receipt examples/cloud_training/live_blocked_receipt.json \
  --cloud-training-status-receipt examples/cloud_training/status_receipt.json \
  --strict
```
