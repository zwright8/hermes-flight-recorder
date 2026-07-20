# Case Study: Flight Recorder LoRA Improves Held-Out Agent Tasks

This case study demonstrates that Hermes Flight Recorder artifacts can support
a complete, evidence-backed agent-improvement loop: collect reviewed agent
trajectories, fine-tune an open-weight model, evaluate the adapted model on an
untouched held-out task set, and publish the decision evidence.

The result is intentionally bounded. It proves task-specific improvement for a
small Qwen3-0.6B LoRA adapter on an opaque tool-routing and approval-safety
benchmark. It is not a claim of broad agent intelligence or frontier-scale
generalization.

## Summary

| Item | Value |
| --- | --- |
| Base model | `Qwen/Qwen3-0.6B` |
| Adaptation | LoRA action-SFT |
| Training data | 800 public-safe synthetic Flight Recorder trajectories |
| Held-out set | 150 frozen tasks, disjoint task IDs, record keys, prompts, and templates |
| Evaluation repeats | 3 seeds per arm: 17, 29, 43 |
| Promotion statistic | Paired 10,000-sample bootstrap clustered by held-out task |
| Promotion gate | Positive 95% CI, no task-family regression, zero adapter critical-safety violations |

## Defined Agent Task

The benchmark asks a tool-using agent to learn an organization-specific
dispatch convention from recorded executions. Each task provides an opaque HFR
route code, a fresh record key, and a set of plausible tools whose descriptions
do not reveal the correct mapping.

The model must:

- select the exact tool associated with the dispatch code,
- copy the correct record key into the tool call,
- include approval tokens only when a write-capable route has a valid
  `APPROVED-` token, and
- refuse unsafe write requests with exactly `POLICY_REFUSAL`.

The held-out records use unseen task IDs, record keys, phrasings, and prompt
templates. A frozen manifest states that held-out prompts, outputs, and task
IDs must never enter training or hyperparameter selection.

## Result

Candidate v3 passed the frozen promotion gate.

| Metric | Base model | Flight Recorder LoRA |
| --- | ---: | ---: |
| Overall exact task pass rate | 17.11% | 96.22% |
| Action-only exact tool-call rate | 10.00% | 95.28% |
| Critical-safety pass rate | 45.56% | 100.00% |
| Critical-safety violations | 31 | 0 |

The paired overall improvement was **+79.11 percentage points**, with a 95%
task-clustered bootstrap confidence interval of **[+72.67, +85.33]** across
150 held-out tasks and three repeated seeds per arm. All eleven task families
were non-regressing.

## Evidence Trail

The committed artifacts are schema-checked public Flight Recorder contracts:

- [Dataset manifest](../../examples/case_studies/self_improving_agent_proof/data/dataset_manifest.json)
- [Frozen held-out manifest](../../examples/case_studies/self_improving_agent_proof/data/frozen_heldout_manifest.json)
- [Contamination audit](../../examples/case_studies/self_improving_agent_proof/data/contamination_audit.json)
- [Training result receipt](../../examples/case_studies/self_improving_agent_proof/evidence/training_result.json)
- [Baseline raw observations](../../examples/case_studies/self_improving_agent_proof/evidence/baseline_results.json)
- [Adapter raw observations](../../examples/case_studies/self_improving_agent_proof/evidence/adapter_results.json)
- [Statistical evaluation report](../../examples/case_studies/self_improving_agent_proof/evaluation.json)
- [Markdown evaluation summary](../../examples/case_studies/self_improving_agent_proof/EVALUATION.md)
- [Publication receipt](../../examples/case_studies/self_improving_agent_proof/PUBLICATION.md)

The Hugging Face publication receipt records immutable public revisions for
the dataset, adapter, and ZeroGPU demo source, plus byte-level checks that the
remote dataset manifest, training receipt, and evaluation report matched the
committed local files.

## Reproduce The Evidence

Regenerate the deterministic dataset:

```bash
python3 scripts/build_self_improving_agent_proof.py \
  --out examples/case_studies/self_improving_agent_proof/data
```

Train the LoRA adapter:

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

Run baseline and adapter evaluation, then compare:

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

The comparison command exits non-zero if the confidence interval, safety,
repeat-count, frozen-hash, base-model identity, decoding-policy, or per-family
non-regression gates fail.

## Boundary

This case study demonstrates that Flight Recorder can bind training data,
frozen evaluation data, trainer receipts, repeated model outputs, statistical
promotion criteria, and publication receipts into an auditable improvement
claim. It does not make Flight Recorder itself a model trainer or model host:
weight updates still happen in the external TRL/PEFT training stack, and
serving still happens in a dedicated runtime.
