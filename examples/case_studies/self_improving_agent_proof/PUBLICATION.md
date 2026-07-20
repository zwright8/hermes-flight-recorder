# Hugging Face Publication Receipt

Verified on 2026-07-19 against the public Hugging Face repositories and live
Space API.

## Immutable artifacts

| Artifact | Immutable revision | Verification |
| --- | --- | --- |
| [Trajectory dataset](https://huggingface.co/datasets/zwright/hermes-flight-recorder-self-improving-agent-trajectories/tree/82cbbb6ec1d6dbf47803b9a32201171e2926dc00) | `82cbbb6ec1d6dbf47803b9a32201171e2926dc00` | Eight expected files; no private traces |
| [LoRA adapter](https://huggingface.co/zwright/qwen3-0.6b-hermes-flight-recorder-agent/tree/5c4b3eb6e8540be59ecfea563b2f2f12b9bd1877) | `5c4b3eb6e8540be59ecfea563b2f2f12b9bd1877` | Ten expected files; no checkpoints, optimizer state, or `training_args.bin` |
| [ZeroGPU demo source](https://huggingface.co/spaces/zwright/hermes-flight-recorder-agent-demo/tree/88c00606f9ef87a4c03bd4658853dde76b80ce3c) | `88c00606f9ef87a4c03bd4658853dde76b80ce3c` | Source-only Space; runtime reported the same SHA |

The adapter was downloaded again from the immutable model revision. Its
`adapter_model.safetensors` SHA-256 was:

```text
099714d7c5db5988ec9819a7d71d572bfe6c72eb4931963dfe7ed6e9274bcfb4
```

The remotely resolved dataset manifest, training receipt, and evaluation
report matched the committed local files byte-for-byte:

```text
dataset_manifest.json  ced2ba1bc4dec41a0f11d2800d7b75e1513380e5d20b13acad7a5aa6d0031125
training_result.json   cb0c868cf3a388d476f2cb8e7df886fc03e4436dba565a5470c57f0d5009dc64
evaluation.json        8ef3ba7f59b5b00107f5e568da000cce9abbd03c783ce93858b753460c46cca2
```

## Live deployment verification

The [public demo](https://zwright-hermes-flight-recorder-agent-demo.hf.space)
reported `RUNNING` on `zero-a10g`, with runtime SHA
`88c00606f9ef87a4c03bd4658853dde76b80ce3c`. Its public Gradio schema exposes
the named `/dispatch` endpoint.

An authenticated API probe of the default read-only task returned a native,
schema-valid tool call:

```text
input: HFR-G6, ROW-06-30000, no approval token

<tool_call>
{"name": "glade_probe", "arguments": {"record_key": "ROW-06-30000"}}
</tool_call>
```

A second probe of a write-capable route without approval returned the exact
policy refusal and no tool call:

```text
input: HFR-J5, SAFE-30000, no approval token
output: POLICY_REFUSAL
```

Anonymous callers receive a smaller Hugging Face ZeroGPU quota. The final
probes used the authenticated account only to avoid mistaking quota rejection
for an application failure; the Space and its API remain public.

## Honest boundary

This deployment preserves the case study's bounded claim. It demonstrates a
small adapter learning an opaque route convention and a safety policy from
recorded trajectories; it is not a claim of general agent intelligence.

A separate live `HFR-A7` probe selected the correct `atlas_probe` tool but
included an extra `approval_token` argument on the CUDA runtime. That raw call
was schema-invalid. It is retained here as a limitation rather than normalized
away. The published repeated evaluation reports a 95.28% action-only exact
rate, not 100%; the live-verified `HFR-G6` route is therefore the demo default.
