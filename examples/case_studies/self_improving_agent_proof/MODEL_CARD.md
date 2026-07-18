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
- adapter exact pass rate: 94.44%
- paired improvement: +77.33 percentage points
- task-clustered 95% bootstrap CI: [+70.67, +83.56]
- baseline critical-safety violations: 31
- adapter critical-safety violations: 0
- per-family regressions: 0 of 11 families

The action-only effect was +83.06 points with 95% CI [+76.39, +89.17].
Repeated seeds are averaged within each task before resampling, so correlated
repeats are not treated as independent samples.

## Artifact identity

- base model revision: `c1899de289a04d12100db370d81485cdf75e47ca`
- dataset identity: `74bad2f15f29df0e9048b0ef081cd5ffbab523979e5a760167532d78d23642bf`
- training SHA-256: `31c5674497f33c44e28b84da7d25962f93e874a821cdc1e3481b13d5526a6917`
- frozen held-out SHA-256: `5500c67defa9c7ef7db2626f887712900795057bf5a64f24d2e1db91cb57b55b`
- adapter weights SHA-256: `90097459dd4ac441d231507c8a0c55ca7edd03ea0120a1c5f250275af3256b16`

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

