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
