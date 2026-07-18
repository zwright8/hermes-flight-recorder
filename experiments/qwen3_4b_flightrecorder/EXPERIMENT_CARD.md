# Qwen3-4B Flight Recorder Fine-Tune Experiment

- Model: `Qwen/Qwen3-4B-Instruct-2507`
- Generated: `2026-06-29T20:57:31.630099+00:00`
- Source runs: `runs`

## Current Evidence

- Episodes: 7 (2 passed, 5 failed)
- Pass rate: 0.2857
- Average score: 50.7143
- Raw trace-only SFT rows: 4
- Raw trace-only known failed row rate: 0.75
- Flight Recorder SFT rows: 1
- Flight Recorder action SFT rows: 3
- Flight Recorder combined DPO rows: 2
- Flight Recorder reward-model rows: 4
- Flight Recorder step-reward rows: 11

## Why This Bundle Matters

The trace-only arm imitates every completed final answer, including known failed runs.
The Flight Recorder arm separates accepted SFT examples, rejected behavior, DPO pairs, reward labels, step rewards, and held-out task families.

## Held-Out Splits

### Validation
- `prompt_injection`: scenarios prompt_injection_bad, prompt_injection_good

### Test
- `subagent_claim`: scenarios subagent_claim_bad

## Completion Status

- `data_bundle_ready`: True
- `baseline_model_eval_complete`: True
- `trace_only_finetune_complete`: True
- `flightrecorder_finetune_complete`: True
- `promotion_comparison_complete`: True

The data bundle is ready, but the model training/evaluation proof is not complete until the baseline, trace-only, and Flight Recorder fine-tuned models are all evaluated on the same held-out scenarios.

Use `scripts/serve_transformers_openai.py` to expose the base model or a PEFT adapter through a local OpenAI-compatible `/v1/chat/completions` endpoint.

Use `scripts/evaluate_hermes_heldout.py` to run those held-out scenarios through a live Hermes runtime and produce suite summaries for promotion comparison.
