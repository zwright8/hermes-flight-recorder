# Agentic LoRA Training on Hugging Face

This is the executable path from Flight Recorder evidence to a tool-using LoRA
adapter. The core package stays dependency-free; TRL, PEFT, Trackio, and Hub
dependencies are isolated in the training job.

## What Flight Recorder Exports

`export-rl` now emits separate, versioned views:

- `sft.jsonl`: final-response supervised examples.
- `action_sft.jsonl`: complete native trajectories with assistant
  `tool_calls`, `tool` results, call IDs, ordering, and recorded runtime tool
  schemas in `tools`. Inferred or incomplete schemas are quarantined from
  action imitation.
- `dpo.jsonl`: chosen and rejected native trajectories plus their shared tool
  definitions. Pairing is restricted to the same task family and exact prompt.
- `reward_model.jsonl` and `step_rewards.jsonl`: outcome and process signals for
  later reward-model work.
- `splits/train/*.jsonl`: the only views the experiment builder accepts for
  training when split artifacts are available.

Positive SFT/action-SFT rows require both a passing scorecard and configured,
complete task-evidence checks. Every generated artifact is redaction-scanned,
fingerprinted, and tied to a dataset version.

For a production self-improvement iteration, also require a passing
`hfr.data_governance_receipt.v1`, a passing
`hfr.dataset_contamination_report.v1`, native accepted action rows, and a
content-addressed curation recipe. The reproducible synthetic example is in
[`examples/self_improving_loop`](../examples/self_improving_loop/README.md).

## 1. Build and Gate the Dataset

For the public-safe end-to-end case study:

```bash
python3 scripts/prepare_self_improving_case_study.py \
  --out runs/self_improving_loop

python3 scripts/build_agentic_finetune_experiment.py \
  --runs-dir runs/self_improving_loop \
  --controls-dir runs/self_improving_loop \
  --out experiments/qwen3_4b_flightrecorder
```

For recorded production data, first build the normal export:

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
```

Then create the same six reviewed controls demonstrated by the case study:
`governance.json`, `contamination.json`, `curated.json`,
`action_credit.jsonl`, `branch_replay.json`, and `preferences.jsonl`. Pass their
directory through `--controls-dir`; the builder does not infer, synthesize, or
silently omit these production decisions. It fails closed if a control, split
contract, leakage check, redaction proof, or training gate is absent or failed.
It writes
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

SFT defaults to assistant-only loss. Some published chat templates, including
the current `Qwen/Qwen3-0.6B` template, do not expose the `{% generation %}`
mask that TRL requires for that mode. For a bounded compatibility run, pass
`--all-message-loss`; the launch plan records `assistant_only_loss: false` so
the broader loss scope remains explicit and auditable. Prefer a template with
assistant masks for production training when one is available.

## 3. Prepare the Hugging Face Jobs Handoff

```bash
python3 scripts/prepare_huggingface_jobs_handoff.py \
  --experiment-dir experiments/qwen3_4b_flightrecorder \
  --training-plan experiments/qwen3_4b_flightrecorder/training_plan.json \
  --model-manifest path/to/model_manifest.json \
  --reviewer YOUR_REVIEWER_ID \
  --dataset-repo YOUR_USER/hermes-agentic-training-data \
  --hub-model-id YOUR_USER/hermes-agentic-adapter \
  --mode fr_sft_dpo \
  --flavor a10g-large \
  --timeout 4h \
  --container-image YOUR_REGISTRY/hermes-trainer@sha256:YOUR_IMAGE_DIGEST \
  --out experiments/qwen3_4b_flightrecorder/hf_jobs_handoff
```

This command performs no network write and starts no paid job. It produces:

- a private-dataset payload with the filtered rows, manifests, trainer, and job
  entry point;
- a reviewed, fingerprinted plan that is the sole source of trainer arguments;
- exact dependency, container, base-model, tokenizer, chat-template, LoRA,
  loss-scope, seed, and checkpoint-cadence identities;
- file hashes for review;
- `upload_to_hub.py`, which is the explicit upload boundary; and
- `job_request.template.json`, which requires an immutable dataset commit and
  references `HF_TOKEN` as a secret rather than embedding it.

After approving the upload, run the recorded `upload_command` from
`handoff.json`. The upload writes `job_request.json` bound to the returned Hub
commit. Submit that JSON through the Hugging Face Jobs integration. The job
uses Trackio, pushes every saved checkpoint, explicitly pushes the final PEFT
adapter, and records the final immutable Hub revision in a durable completion
receipt before exit. A failed or interrupted job emits a terminal receipt with
resume identity rather than disappearing with its ephemeral filesystem.

After cloning the private model repository and checking out that exact
immutable revision, pass the clean Git checkout root to
`scripts/import_huggingface_job_completion.py`. The importer verifies the
checkout's Git commit and Hugging Face origin against the publication receipt;
an arbitrary directory name or unrelated 40-character revision is rejected
before it validates the remote
receipt and imports it into canonical `hfr.agentic_training_result.v1` and
`hfr.cloud_training_completion_receipt.v1` artifacts; job exit alone is never
treated as proof of completion.

The completion receipt must include observed CUDA, driver, Python, PyTorch,
Transformers, TRL, PEFT, Accelerate, and container-runtime identity. Declared
job requirements alone are not accepted as runtime proof. Canonical import
replays the reviewed training-plan binding even when the operator does not pass
an optional base directory.

Never place an HF token in a manifest, JSONL row, shell argument, trace, or
committed file. The job request must retain:

```json
{"secrets": {"HF_TOKEN": "$HF_TOKEN"}}
```

## 4. Evaluate the Adapter as an Agent

Serving must pass request tools into the tokenizer chat template; the included
Transformers server does so and fails clearly if the tokenizer cannot render
structured tools.

For a tuned arm, generate its serving profile with the exact immutable adapter
identity used in the arm manifest. Local adapters derive these values from the
weights automatically; remote adapters must supply them explicitly:

```bash
python3 scripts/check_openai_serving.py \
  --arm flightrecorder \
  --model YOUR_BASE_MODEL \
  --adapter YOUR_USER/hermes-agentic-adapter \
  --adapter-id YOUR_USER/hermes-agentic-adapter \
  --adapter-revision IMMUTABLE_HUB_COMMIT \
  --adapter-sha256 ADAPTER_WEIGHTS_SHA256 \
  --base-url http://127.0.0.1:8000/v1 \
  --out experiments/qwen3_4b_flightrecorder/serving/flightrecorder
```

Run baseline, trace-only, and Flight Recorder arms against the same held-out
scenario hashes for at least three predeclared seeds in each frozen, rolling,
and adversarial pool. Each invocation records an immutable arm identity,
repeat, decoding configuration, measured cost, risk tier, latency, and source
hashes. Then build canonical promotion evidence:

```bash
python3 scripts/evaluate_hermes_heldout.py \
  --arm baseline \
  --model YOUR_BASE_MODEL \
  --base-url http://127.0.0.1:8000/v1 \
  --heldout experiments/qwen3_4b_flightrecorder/heldout/frozen.json \
  --arm-identity experiments/qwen3_4b_flightrecorder/arm_identities/baseline.json \
  --repeat-index 0 \
  --seed 1729 \
  --pool-type frozen \
  --pool-id frozen-v1 \
  --risk-tier standard \
  --cost-usd-per-run 0.00 \
  --out experiments/qwen3_4b_flightrecorder/evaluations/runs/baseline-frozen-0

observation_args=()
for arm in baseline trace-only flightrecorder; do
  flag="--${arm}-observation"
  for pool in frozen rolling adversarial; do
    for repeat in 0 1 2; do
      observation_args+=("${flag}" "experiments/qwen3_4b_flightrecorder/evaluations/runs/${arm}-${pool}-${repeat}/agentic_eval_observation.json")
    done
  done
done

python3 scripts/compare_agentic_finetune_results.py \
  --baseline experiments/qwen3_4b_flightrecorder/evaluations/runs/baseline-frozen-0/suite_summary.json \
  --trace-only experiments/qwen3_4b_flightrecorder/evaluations/runs/trace-only-frozen-0/suite_summary.json \
  --flightrecorder experiments/qwen3_4b_flightrecorder/evaluations/runs/flightrecorder-frozen-0/suite_summary.json \
  "${observation_args[@]}" \
  --promotion-evidence-out experiments/qwen3_4b_flightrecorder/evaluations/promotion_evidence.json \
  --primary-metric score \
  --minimum-effect 0.05 \
  --minimum-repeats 3 \
  --required-pool frozen \
  --required-pool rolling \
  --required-pool adversarial
```

The first evaluator command shows one cell of the matrix. Run all nine cells
per arm (27 total) using the three predeclared seeds, the matching immutable
arm identity, the matching served model/adapter, and the held-out manifest for
each pool before building `observation_args`. Each output directory is
immutable; never reuse it for another arm, pool, repeat, or decoding
configuration.

Comparative claims require a non-empty scenario set with identical scenario
IDs and content hashes across every arm. An empty run or same-named but changed
scenario is not governance-ready. Promotion requires the lower confidence
bound to clear the minimum effect over both controls, zero new critical safety
failures, no tool-schema regression, bounded cost/latency, and no material
family or risk-tier regression.

## Boundaries

- A successful training loss is not proof of agent quality. Promotion still
  depends on held-out task completion, tool correctness, safety regressions,
  serving compatibility, and rollback readiness.
- Inferred tool schemas describe observed argument shapes, not authoritative
  production tool semantics, and are not eligible for action training.
- Hugging Face Jobs and Hub uploads are external, potentially paid operations;
  Flight Recorder prepares and fingerprints the handoff but does not submit it
  without explicit approval.
