---
base_model: Qwen/Qwen3-4B-Instruct-2507
library_name: peft
license: apache-2.0
pipeline_tag: text-generation
inference: false
tags:
- peft
- lora
- agents
- tool-calling
- browser
- flight-recorder
datasets:
- zwright/hermes-flight-recorder-browser-tool-calling-trajectories
model-index:
- name: Qwen3-4B Hermes Flight Recorder Browser LoRA
  results:
  - task:
      type: text-generation
      name: Governed browser tool calling
    dataset:
      type: zwright/hermes-flight-recorder-browser-tool-calling-trajectories
      name: HFR browser sealed-final scope
      split: test
    metrics:
    - type: accuracy
      name: Functional task success
      value: 1.0
    - type: accuracy
      name: Exact final answer
      value: 1.0
    - type: accuracy
      name: Literal tool-call arguments
      value: 0.0
---

# Qwen3-4B Hermes Flight Recorder Browser LoRA

This is a rank-16 LoRA for native browser-tool calling with
`Qwen/Qwen3-4B-Instruct-2507`. It was trained locally from public synthetic
Flight Recorder trajectories and selected under fail-closed development and
sealed-final evidence gates.

## Measured behavior

On identical frozen browser tasks, the unmodified base scored 0/4 development
and 0/9 sealed-final overall success. The adapter scored 4/4 and 9/9,
respectively. Functional tool-call and exact final-answer rates moved by the
same amount. Literal tool-call argument exactness remained 0/4 and 0/9 because
the adapter emits the evaluator's narrowly permitted trailing `headline`
search refinement.

The adapter evaluation was the governed one-shot sealed action. The matched
base control was run post-hoc after weights and recipe were frozen and is
clearly labeled as such in the evidence. Full reports, paired per-task scores,
training curve, manifests, checksums, and limitations are in the associated
GitHub evidence capsule.

## Load

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

revision = "cdbee75f17c01a7cc42f958dc650907174af0554"
base_id = "Qwen/Qwen3-4B-Instruct-2507"
adapter_id = "zwright/qwen3-4b-hermes-flight-recorder-browser-lora"

tokenizer = AutoTokenizer.from_pretrained(base_id, revision=revision)
base = AutoModelForCausalLM.from_pretrained(base_id, revision=revision)
model = PeftModel.from_pretrained(base, adapter_id)
```

Use the base model's native chat template and tool definitions. A deterministic
external router should activate this adapter only for the evaluated browser
scope; model text must not choose its own adapter, tools, or write authority.

## Training configuration

- LoRA rank 16, alpha 32, dropout 0.05.
- 160 optimizer steps at `1e-4` with batch size 1 and gradient accumulation 4.
- Maximum sequence length 1024.
- 71 reviewed source rows; 355 effective rows after bounded action-turn
  weighting.
- Zero truncated or zero-supervision rows.
- Apple MPS, float16, 1783.586 seconds.
- Final aggregate training loss: `0.0459748519`.
- Weight SHA-256:
  `aeeb93af9a0b35a7c685942c320ae2465fd06edc73b9ee85d416918208a2e5e6`.

## Limitations

- Browser-only synthetic scope; not a general-purpose agent adapter.
- Four development and nine sealed tasks are a small evaluation.
- No safety, refusal, write-denial, or failure-recovery rows occur in the
  scoped evaluation subset.
- Functional correctness is not literal argument equality.
- The base comparison is a post-hoc audit, not part of the original one-shot
  sealed receipt.
- No quantized runtime or production serving deployment was evaluated.

Do not merge or stack this adapter with another adapter without treating the
composition as a new candidate with its own immutable hash and evaluation.

## Provenance

- Base revision: `cdbee75f17c01a7cc42f958dc650907174af0554`.
- Chat-template SHA-256:
  `64f85b198065d0fba2a81f37e10ed68161ce2c19a754c7100e67e0ca2ee9c326`.
- Candidate identity:
  `a1495c41471e0b09230fcf1d9ef2556e88ca856dcab489aa110b302b669d3dc7`.
- GitHub evidence: [Hermes Flight Recorder PR #34](https://github.com/zwright8/hermes-flight-recorder/pull/34).
- Dataset: [public synthetic browser-tool trajectories](https://huggingface.co/datasets/zwright/hermes-flight-recorder-browser-tool-calling-trajectories).

The base model and this adapter are distributed under Apache-2.0. Users remain
responsible for reviewing upstream terms and validating behavior for their own
deployment.
