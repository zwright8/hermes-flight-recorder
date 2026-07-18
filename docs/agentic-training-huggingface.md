# Agentic LoRA Training on Hugging Face

This is the executable path from Flight Recorder evidence to a tool-using LoRA
adapter. The core package stays dependency-free; TRL, PEFT, Trackio, and Hub
dependencies are isolated in the training job.

## What Flight Recorder Exports

`export-rl` now emits separate, versioned views:

- `sft.jsonl`: final-response supervised examples.
- `action_sft.jsonl`: complete native trajectories with assistant
  `tool_calls`, `tool` results, call IDs, ordering, and inferred JSON argument
  schemas in `tools`.
- `dpo.jsonl`: chosen and rejected native trajectories plus their shared tool
  definitions. Pairing is restricted to the same task family and exact prompt.
- `reward_model.jsonl` and `step_rewards.jsonl`: outcome and process signals for
  later reward-model work.
- `splits/train/*.jsonl`: the only views the experiment builder accepts for
  training when split artifacts are available.

Positive SFT/action-SFT rows require both a passing scorecard and configured,
complete task-evidence checks. Every generated artifact is redaction-scanned,
fingerprinted, and tied to a dataset version.

## 1. Build and Gate the Dataset

```bash
python3 -m flightrecorder run-suite \
  --scenarios scenarios \
  --out runs \
  --export-rl \
  --validate \
  --strict

python3 -m flightrecorder gate-export \
  --training-export runs/training_export \
  --policy examples/training_gate_policy.demo.json \
  --out runs/training_gate.json

python3 scripts/build_agentic_finetune_experiment.py \
  --runs-dir runs \
  --out experiments/qwen3_4b_flightrecorder
```

The builder fails closed if the split contract, leakage checks, redaction proof,
or training gate is absent or failed. It writes
`dataset_training_manifest.json`, whose trainer paths point only to filtered
training files and whose SHA-256 records are rechecked immediately before a
real launch.

## 2. Register the Base Model Decision

Create or select a model manifest that records the exact model ID, a reviewed
license status, training permission, and compatibility information. Do not mark
a license approved without reviewing the upstream model card and intended use.

```json
{
  "schema_version": "hfr.model_candidate.v1",
  "model_id": "Qwen/Qwen3-4B-Instruct-2507",
  "license_status": "approved",
  "training_allowed": true,
  "compatibility": {
    "tokenizer": "available",
    "chat_template": "messages_and_tools",
    "serving": "transformers_or_vllm"
  }
}
```

Validate the real training plan without importing the ML stack:

```bash
python3 scripts/train_agentic_lora.py \
  --mode fr_sft_dpo \
  --dry-run \
  --require-registered-inputs \
  --experiment-dir experiments/qwen3_4b_flightrecorder \
  --model-manifest path/to/model_manifest.json \
  --dataset-manifest experiments/qwen3_4b_flightrecorder/dataset_training_manifest.json \
  --output-dir experiments/qwen3_4b_flightrecorder/adapters \
  --hub-model-id YOUR_USER/hermes-agentic-adapter
```

Useful modes are `fr_action_sft` for pure tool-trajectory imitation,
`fr_dpo` for preference-only alignment, and `fr_sft_dpo` for the usual staged
path. The staged path trains accepted final responses and action trajectories,
then continues the adapter with DPO.

## 3. Prepare the Hugging Face Jobs Handoff

```bash
python3 scripts/prepare_huggingface_jobs_handoff.py \
  --experiment-dir experiments/qwen3_4b_flightrecorder \
  --model-manifest path/to/model_manifest.json \
  --dataset-repo YOUR_USER/hermes-agentic-training-data \
  --hub-model-id YOUR_USER/hermes-agentic-adapter \
  --mode fr_sft_dpo \
  --flavor a10g-large \
  --timeout 4h \
  --out experiments/qwen3_4b_flightrecorder/hf_jobs_handoff
```

This command performs no network write and starts no paid job. It produces:

- a private-dataset payload with the filtered rows, manifests, trainer, and job
  entry point;
- file hashes for review;
- `upload_to_hub.py`, which is the explicit upload boundary; and
- `job_request.template.json`, which requires an immutable dataset commit and
  references `HF_TOKEN` as a secret rather than embedding it.

After approving the upload, run the recorded `upload_command` from
`handoff.json`. The upload writes `job_request.json` bound to the returned Hub
commit. Submit that JSON through the Hugging Face Jobs integration. The job
uses Trackio, pushes every saved checkpoint, explicitly pushes the final PEFT
adapter, and records the final immutable Hub revision in its result.

Never place an HF token in a manifest, JSONL row, shell argument, trace, or
committed file. The job request must retain:

```json
{"secrets": {"HF_TOKEN": "$HF_TOKEN"}}
```

## 4. Evaluate the Adapter as an Agent

Serving must pass request tools into the tokenizer chat template; the included
Transformers server does so and fails clearly if the tokenizer cannot render
structured tools.

Run baseline, trace-only, and Flight Recorder arms against the same held-out
scenario files, then compare them:

```bash
python3 scripts/evaluate_hermes_heldout.py \
  --arm flightrecorder \
  --model YOUR_SERVED_MODEL \
  --base-url http://127.0.0.1:8000/v1 \
  --heldout experiments/qwen3_4b_flightrecorder/heldout_scenarios.json \
  --out experiments/qwen3_4b_flightrecorder/evaluations/flightrecorder

python3 scripts/compare_agentic_finetune_results.py \
  --baseline experiments/qwen3_4b_flightrecorder/evaluations/baseline/suite_summary.json \
  --trace-only experiments/qwen3_4b_flightrecorder/evaluations/trace_only/suite_summary.json \
  --flightrecorder experiments/qwen3_4b_flightrecorder/evaluations/flightrecorder/suite_summary.json
```

Comparative claims require a non-empty scenario set with identical scenario
IDs and content hashes across every arm. An empty run or same-named but changed
scenario is not governance-ready.

## Boundaries

- A successful training loss is not proof of agent quality. Promotion still
  depends on held-out task completion, tool correctness, safety regressions,
  serving compatibility, and rollback readiness.
- Inferred tool schemas describe observed argument shapes, not authoritative
  production tool semantics. Prefer recording the runtime's actual JSON tool
  schemas when the source adapter provides them.
- Hugging Face Jobs and Hub uploads are external, potentially paid operations;
  Flight Recorder prepares and fingerprints the handoff but does not submit it
  without explicit approval.
