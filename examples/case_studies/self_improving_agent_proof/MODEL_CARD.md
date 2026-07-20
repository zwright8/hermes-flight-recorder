---
base_model: Qwen/Qwen3-0.6B
datasets:
- zwright/hermes-flight-recorder-self-improving-agent-trajectories
library_name: peft
license: apache-2.0
pipeline_tag: text-generation
tags:
- agents
- tool-calling
- lora
- peft
- flight-recorder
---

# Qwen3-0.6B Hermes Flight Recorder Agent

This rank-16 LoRA adapter teaches `Qwen/Qwen3-0.6B` an opaque,
organization-specific Hermes tool-routing convention from 800 governed Flight
Recorder trajectories. The final candidate was selected on a separate rolling
development set and evaluated once on a frozen 150-task benchmark.

## Results

Across three seeds per task (450 paired observations per arm):

- baseline exact pass rate: 17.11%
- adapter exact pass rate: 96.22%
- paired improvement: +79.11 percentage points
- task-clustered 95% bootstrap CI: [+72.67, +85.33]
- baseline critical-safety violations: 31
- adapter critical-safety violations: 0
- per-family regressions: 0 of 11 families

The action-only effect was +85.28 points with 95% CI [+78.89, +91.11].
Repeated seeds are averaged within each task before resampling, so correlated
repeats are not treated as independent samples.

## Artifact identity

- base model revision: `c1899de289a04d12100db370d81485cdf75e47ca`
- dataset identity: `e2af62c0a6668b3c7152579ac6c654052dd07ced6b0d0bda6da8dc387284419b`
- training SHA-256: `9cc907b0ce04d5f4b21f905357ed7f277fc640c70a8f31c90f76a187a14b2df5`
- frozen held-out SHA-256: `efc53b6035f763ec15a116d649d12e64b9fc2b9acaeae8fcd37bdb9ec4da5771`
- adapter weights SHA-256: `099714d7c5db5988ec9819a7d71d572bfe6c72eb4931963dfe7ed6e9274bcfb4`

## Load the adapter

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_id = "Qwen/Qwen3-0.6B"
adapter_id = "zwright/qwen3-0.6b-hermes-flight-recorder-agent"

tokenizer = AutoTokenizer.from_pretrained(adapter_id)
model = AutoModelForCausalLM.from_pretrained(base_id)
model = PeftModel.from_pretrained(model, adapter_id)
```

Use the Qwen chat template with native `tools`, `enable_thinking=False`, and the
same policy system prompt documented in the dataset. The adapter emits native
`<tool_call>` JSON and `POLICY_REFUSAL` for unapproved write requests.

## Training

- SFT rows: 800
- optimizer steps: 100 (one complete corpus pass)
- assistant-only loss: enabled
- LoRA rank/alpha/dropout: 16 / 32 / 0.05
- effective batch size: 8
- maximum sequence length: 640
- learning rate: 2e-4 with linear decay
- final train loss: 0.01916
- tracking: Trackio enabled

Exact library versions and the full adapter artifact manifest are in
`evidence/training_result.json` in the GitHub case study.

## Scope and limitations

This is a rigorous bounded proof of learning from recorded agent behavior, not
a claim of general-purpose agent intelligence. The corpus is synthetic and the
convention is intentionally controlled. Production adoption still requires
reviewed real-world trajectories, domain-specific privacy governance, broader
red-team suites, canary rollout, monitoring, and rollback controls.
