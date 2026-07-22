# Qwen3-4B Browser LoRA: Paired Improvement Evidence

This public-safe evidence capsule records a real local LoRA training run and a
matched comparison against the unmodified base model. Flight Recorder governed
the recipe, training launch, development selection, one-shot adapter sealed
evaluation, validation, and release check.

## Result

Both arms use `Qwen/Qwen3-4B-Instruct-2507` at revision
`cdbee75f17c01a7cc42f958dc650907174af0554`, the same tokenizer and chat
template, deterministic inference, identical browser tasks, and the same
evaluator. The only functional difference is whether the frozen LoRA is
active.

| Split | Metric | Base | Browser LoRA | Absolute change |
| --- | --- | ---: | ---: | ---: |
| Development | Overall task success | 0/4 | 4/4 | +100 pp |
| Development | Functional tool calls | 0/4 | 4/4 | +100 pp |
| Development | Exact final answer | 0/4 | 4/4 | +100 pp |
| Sealed final | Overall task success | 0/9 | 9/9 | +100 pp |
| Sealed final | Functional tool calls | 0/9 | 9/9 | +100 pp |
| Sealed final | Exact final answer | 0/9 | 9/9 | +100 pp |
| Sealed final | Literal tool-call arguments | 0/9 | 0/9 | 0 pp |

The literal tool-call metric remains visible because this is not a claim of
byte-for-byte argument reproduction. The governed functional comparator accepts
only a trailing `headline` refinement on `browser.search`; tool names, order,
URLs, identifiers, commands, and non-null values remain strict.

The original one-shot sealed receipt evaluated only the already-selected
adapter. The base arm was run afterward as a post-hoc control against the same
frozen task IDs. It did not influence training, recipe selection, promotion, or
the original sealed decision. Its report is named
`sealed_base_evaluation_posthoc.json` so that provenance boundary cannot be
mistaken for a preregistered two-arm sealed run.

## Training

- Base: `Qwen/Qwen3-4B-Instruct-2507`, pinned revision above.
- Mode: native Flight Recorder action-SFT.
- Scope: `runtime_adapter_router_browser_train` / browser evaluation only.
- LoRA: rank 16, alpha 32, dropout 0.05.
- Optimizer recipe: 160 steps, learning rate `1e-4`, batch size 1, gradient
  accumulation 4, maximum length 1024.
- Curriculum: 71 reviewed source trajectories and four bounded action-turn
  repeats, producing 355 effective rows.
- Supervision: 355 fully visible rows, zero partially truncated rows, and zero
  zero-supervision rows.
- Runtime: 1783.586 seconds on Apple MPS; final aggregate train loss
  `0.0459748519`.
- Final weight file: 132,187,888 bytes, SHA-256
  `aeeb93af9a0b35a7c685942c320ae2465fd06edc73b9ee85d416918208a2e5e6`.
- Original training-directory fingerprint:
  `1db6865cd4013baeb5370264bcfa8406a0f8456eb5eba93e32469fac56ea87de`.
- Candidate identity:
  `a1495c41471e0b09230fcf1d9ef2556e88ca856dcab489aa110b302b669d3dc7`.

The public `adapter_config.json` intentionally replaces the local cache path
written by PEFT with the pinned Hugging Face model ID and revision. The trained
weights are byte-identical to the governed output.

## Artifact map

- `development_*_evaluation.json` and `sealed_*_evaluation*.json` are the exact
  schema-validated evaluator reports.
- `paired_task_scores.csv` proves that the arms cover identical task IDs and
  exposes every task-level outcome.
- `metrics_summary.csv` contains the aggregate base-to-adapter deltas.
- `training_result.json` preserves runtime, supervision, input-manifest, and
  adapter fingerprints.
- `training_curve.csv` contains all 160 loss-bearing trainer log entries.
- `campaign_validation.json` records the replayed campaign and release result.
- `dataset_manifest.json` and `model_manifest.json` pin governed inputs.
- `SHA256SUMS` fingerprints every machine-readable artifact in this capsule.

The loadable adapter is published at
[`zwright/qwen3-4b-hermes-flight-recorder-browser-lora`](https://huggingface.co/zwright/qwen3-4b-hermes-flight-recorder-browser-lora).
The exact public synthetic corpus is published at
[`zwright/hermes-flight-recorder-browser-tool-calling-trajectories`](https://huggingface.co/datasets/zwright/hermes-flight-recorder-browser-tool-calling-trajectories).

## Claim boundary

This is strong evidence of improvement on the frozen synthetic browser-tool
task family. It is not evidence of general reasoning improvement, literal
argument exactness, or safety behavior: the scoped four- and nine-task browser
subsets contain no write-denial, refusal, or failure-recovery cases. The sample
is small, the base control is post-hoc, and the adapter should remain routed
only to the evaluated browser scope. Broader deployment requires a new
preregistered mixed-domain evaluation with non-regression and safety coverage.

Raw observations, absolute-path launch records, checkpoints, optimizer state,
RNG state, and duplicate tokenizer files are intentionally excluded. They are
not necessary to audit the claim and are unsafe or wasteful publication
artifacts.
