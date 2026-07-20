---
base_model: Qwen/Qwen3-0.6B
library_name: peft
pipeline_tag: text-generation
license: apache-2.0
tags:
  - lora
  - trl
  - tool-use
  - agentic
  - hermes-flight-recorder
---

# Hermes Flight Recorder Qwen3-0.6B LoRA Demo

This is the model card prepared for the adapter produced by the
[Qwen3-0.6B Flight Recorder case study](README.md). The planned Hub ID is
`zwright/hermes-flight-recorder-qwen3-0.6b-demo`; it was not published by the
GitHub case-study commit.

## Model details

- Base model: `Qwen/Qwen3-0.6B`
- Adaptation: PEFT LoRA
- Rank / alpha / dropout: 8 / 16 / 0
- Training objective: action-trajectory SFT over all message tokens
- Training steps: 20
- Maximum sequence length: 512
- Dataset: one synthetic, redacted Flight Recorder trajectory
- Dataset version: `hfrds-a801878214397fed`

The example preserves two assistant tool calls, two matching tool results, and
the final assistant response. Flight Recorder admitted it only after the
scorecard, configured task-completion evidence, redaction checks, split checks,
and source-fingerprint verification passed.

## Intended use

Use this adapter to reproduce or inspect the Flight Recorder → TRL/PEFT
integration. It is not intended for production email access or autonomous
actions.

After the adapter is published, it can be loaded with PEFT:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_id = "Qwen/Qwen3-0.6B"
adapter_id = "zwright/hermes-flight-recorder-qwen3-0.6b-demo"

tokenizer = AutoTokenizer.from_pretrained(base_id)
base = AutoModelForCausalLM.from_pretrained(base_id)
model = PeftModel.from_pretrained(base, adapter_id)
```

## Results

On the single training sequence, loss decreased from 3.0694 for the base model
to 0.1691 with the adapter, a 94.49% reduction. Training token accuracy reached
1.0 by step 18. The adapter SHA-256 is
`5642c36e91c69098666e19c653ea43639169adfaf1cdf226f16501b5165cc891`.

## Limitations

This is a one-row memorization and integration demonstration. It does not prove
held-out generalization, improved tool selection, structurally valid tool-call
generation, safe email behavior, or production readiness. The tuned sampled
completion was not structurally better than the base completion. Evaluate a
larger adapter on family-exclusive held-out scenarios and enforce tool parser,
task-completion, safety, and promotion gates before broader use.

The run used all-message loss because the current base-model chat template does
not provide the generation masks required by TRL's assistant-only loss.

## Data and provenance

The exact redacted row, training curve, portable evaluation receipt, reviewed
model manifest, and reproduction commands are versioned beside this card in
the GitHub case study. No raw production trace, secret, checkpoint, or base
model weight is included.
