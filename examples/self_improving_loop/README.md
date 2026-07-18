# Self-Improving Agent Loop Case Study

This public-safe synthetic case study exercises the operational loop without
production traces, credentials, provider calls, or paid training. It proves
the contracts and gates; it does not claim that a model generalizes.

The fixture contains two accepted native tool trajectories, one rejected
trajectory under the same complete task contract, one recovery trajectory for
step credit, a verified branch replay, a protected benchmark, a deterministic
curation recipe, and a failure cluster for whole-agent intervention routing.

## Reproduce the data path

Run from the repository root:

```bash
python3 scripts/prepare_self_improving_case_study.py \
  --out runs/self_improving_loop

python3 -m flightrecorder intervention-route \
  --cluster examples/self_improving_loop/failure_cluster.json \
  --out runs/self_improving_loop/intervention_route.json
```

The preparation command writes the governance, contamination, human curation,
per-action credit, verified branch replay, human rejection, deterministic split,
redaction, and training-gate controls as one internally consistent handoff. It
fails closed if any required control is empty or does not pass.

The rejected trajectory is excluded from positive curation, yet remains
available as a same-contract preference negative. The failed first lookup in
the recovery trajectory receives negative action credit even though the final
episode succeeds. The failure router selects a tool-schema intervention rather
than a model fine-tune because that is the least-cost adequate repair.

Governed deletion is a separate destructive operation and is intentionally not
run by this case study. `flightrecorder data-governance delete` consumes a
reviewed request containing `request_id`, `deletion_subject_ids`,
`dataset_entries`, `model_entries`, and `"erase_sources": true`. It writes
rebuilt descendants first, erases only the explicitly listed source files, and
quarantines models through transitive `parent_versions` lineage. Omitting the
erasure authorization leaves sources intact and produces a blocked receipt.

Validate every generated contract:

```bash
python3 -m flightrecorder schemas --check runs/self_improving_loop/governance.json
python3 -m flightrecorder schemas --check runs/self_improving_loop/contamination.json
python3 -m flightrecorder schemas --check-jsonl runs/self_improving_loop/action_credit.jsonl
python3 -m flightrecorder schemas --check-jsonl runs/self_improving_loop/preferences.jsonl
python3 -m flightrecorder schemas --check runs/self_improving_loop/branch_replay.json
python3 -m flightrecorder schemas --check runs/self_improving_loop/curated.json
python3 -m flightrecorder schemas --check runs/self_improving_loop/intervention_route.json
```

## Exercise the durable controller

```bash
python3 -m flightrecorder agentic-loop controller-plan \
  --controller-id synthetic-loop-v1 \
  --artifact-dir runs/self_improving_loop/controller \
  --candidate-model synthetic-candidate-v1 \
  --champion-model synthetic-champion-v1 \
  --canary-percentage 1 \
  --canary-percentage 10 \
  --canary-percentage 100 \
  --max-cost-usd 1 \
  --max-duration-seconds 300 \
  --max-attempts 40 \
  --deadline-at 2099-01-01T00:00:00+00:00 \
  --out runs/self_improving_loop/controller_plan.json

python3 -m flightrecorder agentic-loop execute \
  --plan runs/self_improving_loop/controller_plan.json \
  --state runs/self_improving_loop/controller_state.json \
  --owner-id case-study \
  --approve-all
```

Build the trainer-facing datasets. This excludes the accepted recovery episode
from action SFT because one of its tool actions received negative credit. It
also requires both the human rejection and verified branch replay in DPO:

```bash
python3 scripts/build_agentic_finetune_experiment.py \
  --runs-dir runs/self_improving_loop \
  --controls-dir runs/self_improving_loop \
  --model Qwen/Qwen3-0.6B \
  --out runs/self_improving_loop/experiment

python3 scripts/train_agentic_lora.py \
  --mode fr_sft_dpo \
  --dry-run \
  --experiment-dir runs/self_improving_loop/experiment \
  --model-manifest examples/case_studies/qwen3_0_6b_flightrecorder_lora/model_manifest.json \
  --dataset-manifest runs/self_improving_loop/experiment/dataset_training_manifest.json \
  --output-dir runs/self_improving_loop/adapters \
  --disable-trackio
```

The dry run imports no ML packages and launches no training. It replays all six
control gates and every registered file hash before declaring the plan ready.

The in-memory adapter is a disposable deployment proof. Real commands require
the explicit command adapter, immutable phase configuration, plan-bound
approvals, durable result and reconciliation commands, fencing-token echoing,
and `--allow-external` for phases marked as external side effects. Failures are
recorded through the failure-analysis and intervention-routing phases before a
repair iteration or approved rollback.

## Train, evaluate, and publish

Use the canonical trainer and private-by-default Hugging Face lifecycle in
[`docs/agentic-training-huggingface.md`](../../docs/agentic-training-huggingface.md).
Promotion evidence must use the baseline, trace-only LoRA, and Flight Recorder
LoRA against identical frozen, rolling, and adversarial scenario hashes with at
least three repeated seeds. The evaluator injects and attests the actual seed,
temperature, top-p, and token limit at the request boundary and bootstraps by
scenario/pool cluster rather than treating repeated seeds as independent data.
A private Hugging Face dataset/model repository is the distribution registry;
Hermes lineage receipts remain authoritative. GitHub stores the reproducible
case study and public-safe fixtures, while Hugging Face Hub is the right home
for adapter weights and immutable model revisions.
