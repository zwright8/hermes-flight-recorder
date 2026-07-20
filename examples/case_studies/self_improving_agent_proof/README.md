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

The JSONL records use the dedicated `hfr.self_improving_agent_episode.v1`
contract. They do not impersonate production `export-rl` action-SFT rows, and
their explicit split roles keep training, development, and final evaluation
semantics auditable.

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

- training SHA-256: `9cc907b0ce04d5f4b21f905357ed7f277fc640c70a8f31c90f76a187a14b2df5`
- development SHA-256: `d91951e52c8f09127e2a66f845e16a1bfb485a1c0294c11962f168eafe23882d`
- held-out SHA-256: `efc53b6035f763ec15a116d649d12e64b9fc2b9acaeae8fcd37bdb9ec4da5771`
- dataset identity: `e2af62c0a6668b3c7152579ac6c654052dd07ced6b0d0bda6da8dc387284419b`

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
  --output-dir runs/self_improving_agent_proof/adapter-v3 \
  --model Qwen/Qwen3-0.6B \
  --model-revision c1899de289a04d12100db370d81485cdf75e47ca \
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
  --model-revision c1899de289a04d12100db370d81485cdf75e47ca \
  --arm baseline \
  --out runs/self_improving_agent_proof/baseline.json

python3 scripts/evaluate_self_improving_agent_proof.py run \
  --heldout examples/case_studies/self_improving_agent_proof/data/heldout_tasks.jsonl \
  --model Qwen/Qwen3-0.6B \
  --model-revision c1899de289a04d12100db370d81485cdf75e47ca \
  --adapter runs/self_improving_agent_proof/adapter-v3 \
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

Candidate v3 passed the untouched final gate:

| Metric | Qwen3-0.6B baseline | Flight Recorder LoRA |
| --- | ---: | ---: |
| Overall exact task pass rate | 17.11% | 96.22% |
| Action-only exact tool-call rate | 10.00% | 95.28% |
| Critical-safety pass rate | 45.56% | 100.00% |
| Critical-safety violations | 31 | 0 |

The paired overall improvement was **+79.11 percentage points**, with a 95%
task-clustered bootstrap confidence interval of **[+72.67, +85.33]** across
150 held-out tasks and three repeated seeds per arm. All eleven task families
were non-regressing. See [EVALUATION.md](EVALUATION.md), the replayable
[`evaluation.json`](evaluation.json), and the raw per-arm observations under
[`evidence/`](evidence/).

## Published artifacts

- [Dataset](https://huggingface.co/datasets/zwright/hermes-flight-recorder-self-improving-agent-trajectories/tree/82cbbb6ec1d6dbf47803b9a32201171e2926dc00): revision `82cbbb6ec1d6dbf47803b9a32201171e2926dc00`
- [Adapter](https://huggingface.co/zwright/qwen3-0.6b-hermes-flight-recorder-agent/tree/5c4b3eb6e8540be59ecfea563b2f2f12b9bd1877): revision `5c4b3eb6e8540be59ecfea563b2f2f12b9bd1877`
- [Demo Space](https://zwright-hermes-flight-recorder-agent-demo.hf.space): runtime revision `88c00606f9ef87a4c03bd4658853dde76b80ce3c`

GitHub is the reproducibility and decision-record home. Hugging Face Hub is the
right distribution home for immutable dataset and model revisions and for the
deployed demonstration.

See [PUBLICATION.md](PUBLICATION.md) for byte-level integrity checks, the exact
live API probes, runtime evidence, and the documented deployment limitation.

The reviewed upload set intentionally excludes optimizer checkpoints,
`training_args.bin`, and the trainer-generated generic README:

```bash
hf upload zwright/hermes-flight-recorder-self-improving-agent-trajectories \
  examples/case_studies/self_improving_agent_proof/data . --repo-type dataset
hf upload zwright/hermes-flight-recorder-self-improving-agent-trajectories \
  examples/case_studies/self_improving_agent_proof/DATASET_CARD.md README.md --repo-type dataset

hf upload zwright/qwen3-0.6b-hermes-flight-recorder-agent \
  runs/self_improving_agent_proof/adapter-v3 . \
  --exclude 'checkpoint-*' --exclude 'checkpoint-*/**' \
  --exclude README.md --exclude training_args.bin --exclude training_result.json
hf upload zwright/qwen3-0.6b-hermes-flight-recorder-agent \
  examples/case_studies/self_improving_agent_proof/MODEL_CARD.md README.md
hf upload zwright/qwen3-0.6b-hermes-flight-recorder-agent \
  examples/case_studies/self_improving_agent_proof/evidence/training_result.json training_result.json
hf upload zwright/qwen3-0.6b-hermes-flight-recorder-agent \
  examples/case_studies/self_improving_agent_proof/evaluation.json evaluation.json

python -c 'from huggingface_hub import HfApi; HfApi().upload_folder(\
repo_id="zwright/hermes-flight-recorder-agent-demo", repo_type="space", \
folder_path="examples/case_studies/self_improving_agent_proof/space", \
ignore_patterns=["__pycache__/**", "*.pyc"])'
```

The Space command commits directly to the existing ZeroGPU repository. This
avoids the CLI's repository-creation preflight, which can report a PRO billing
error even when the destination Space already exists and is writable.
