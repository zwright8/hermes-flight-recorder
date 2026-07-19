# Case Study: A Self-Improving Hermes Agent

This case study tests whether Hermes Flight Recorder trajectories can teach a
small language model an organization-specific tool-routing convention and
improve its behavior on tasks that were never used for training.

It is designed to close the gap left by the earlier one-row LoRA mechanics
demo. A falling training loss is not considered success. The promotion gate
requires repeated held-out improvement with a positive 95% clustered bootstrap
confidence interval and no critical-safety regression.

The dataset manifest, held-out freeze receipt, contamination audit, training
result, raw evaluation arms, and statistical promotion report are registered
public Flight Recorder schema contracts. The case-study tests validate the
committed evidence against those contracts so malformed or incomplete proof
artifacts cannot silently enter the case study.

## Experiment design

- Base model: `Qwen/Qwen3-0.6B`
- Training data: 800 public-safe synthetic Flight Recorder trajectories
- Task families: inventory, calendar, email, filesystem, CRM, database,
  browser, support, payments, deployment, and critical safety
- Development evaluation: 120 separate rolling tasks, never used for gradients
- Frozen evaluation: 150 tasks with disjoint task IDs, record keys, exact
  prompts, and prompt templates
- Final evaluation pools: 90 frozen and 60 adversarial tasks
- Repeats: seeds 17, 29, and 43 for both arms
- Statistics: paired 10,000-sample bootstrap clustered by held-out task
- Safety gate: zero adapter critical-safety violations and no regression from
  the base model

The dispatch codes and tool names are intentionally opaque. Tool descriptions
do not reveal which route code maps to which tool. The convention can be
learned from recorded executions, while the held-out records and phrasings
cannot be memorized from the training rows.

## Reproduce the frozen dataset

```bash
python3 scripts/build_self_improving_agent_proof.py \
  --out examples/case_studies/self_improving_agent_proof/data
```

The command is deterministic. The committed manifest binds:

- training SHA-256: `31c5674497f33c44e28b84da7d25962f93e874a821cdc1e3481b13d5526a6917`
- development SHA-256: `bbef24a057c2135857cdabb4462a033ce01fe4a05b1b60b450b587d2b6d8c57c`
- held-out SHA-256: `5500c67defa9c7ef7db2626f887712900795057bf5a64f24d2e1db91cb57b55b`
- dataset identity: `74bad2f15f29df0e9048b0ef081cd5ffbab523979e5a760167532d78d23642bf`

`frozen_heldout_manifest.json` states that held-out prompts, outputs, and task
IDs must never enter training or hyperparameter selection. Candidate repair and
selection may use `development_tasks.jsonl`, which is also excluded from
gradients. The trainer checks both non-training file hashes but only loads
`train_trajectories.jsonl` into the training dataset.

## Train the LoRA adapter

The script is a self-contained `uv` program and reports the real run to
Trackio unless tracking is explicitly disabled for a local smoke test.

```bash
uv run scripts/train_self_improving_agent_proof.py \
  --data-dir examples/case_studies/self_improving_agent_proof/data \
  --output-dir runs/self_improving_agent_proof/adapter-v2 \
  --model Qwen/Qwen3-0.6B \
  --max-steps 100 \
  --batch-size 1 \
  --gradient-accumulation-steps 8 \
  --gradient-checkpointing \
  --max-length 640 \
  --lora-r 16 \
  --lora-alpha 32
```

## Run repeated baseline-versus-adapter evaluation

```bash
python3 scripts/evaluate_self_improving_agent_proof.py run \
  --heldout examples/case_studies/self_improving_agent_proof/data/heldout_tasks.jsonl \
  --model Qwen/Qwen3-0.6B \
  --arm baseline \
  --out runs/self_improving_agent_proof/baseline.json

python3 scripts/evaluate_self_improving_agent_proof.py run \
  --heldout examples/case_studies/self_improving_agent_proof/data/heldout_tasks.jsonl \
  --model Qwen/Qwen3-0.6B \
  --adapter runs/self_improving_agent_proof/adapter-v2 \
  --arm adapter \
  --out runs/self_improving_agent_proof/adapter.json

python3 scripts/evaluate_self_improving_agent_proof.py compare \
  --baseline runs/self_improving_agent_proof/baseline.json \
  --adapter-results runs/self_improving_agent_proof/adapter.json \
  --out examples/case_studies/self_improving_agent_proof/evaluation.json \
  --report-md examples/case_studies/self_improving_agent_proof/EVALUATION.md
```

The comparison exits non-zero if the confidence interval, safety, repeat,
frozen-hash, or per-family non-regression gates fail.

## Verified result

Candidate v2 passed the untouched final gate:

| Metric | Qwen3-0.6B baseline | Flight Recorder LoRA |
| --- | ---: | ---: |
| Overall exact task pass rate | 17.11% | 94.44% |
| Action-only exact tool-call rate | 10.00% | 93.06% |
| Critical-safety pass rate | 45.56% | 100.00% |
| Critical-safety violations | 31 | 0 |

The paired overall improvement was **+77.33 percentage points**, with a 95%
task-clustered bootstrap confidence interval of **[+70.67, +83.56]** across
150 held-out tasks and three repeated seeds per arm. All eleven task families
were non-regressing. See [EVALUATION.md](EVALUATION.md), the replayable
[`evaluation.json`](evaluation.json), and the raw per-arm observations under
[`evidence/`](evidence/).

## Publication targets

- Dataset: `zwright/hermes-flight-recorder-self-improving-agent-trajectories`
- Adapter: `zwright/qwen3-0.6b-hermes-flight-recorder-agent`
- Demo Space: `zwright/hermes-flight-recorder-agent-demo`

GitHub is the reproducibility and decision-record home. Hugging Face Hub is the
right distribution home for immutable dataset and model revisions and for the
deployed demonstration.
