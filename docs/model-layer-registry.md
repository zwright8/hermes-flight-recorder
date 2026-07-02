# Model Layer Registry

The model layer records base-model candidates, license review posture,
metadata-only compatibility probes, registry aliases, and dry-run training
plans. These commands do not download model weights, import heavy ML packages,
or launch GPU work.

## Artifacts

- `experiments/registry/model_candidates/*.json`: model candidate metadata,
  source, license review, terms posture, and compatibility notes.
- `experiments/registry/compatibility/*.json`: metadata-only compatibility
  reports for tokenizer, chat template, serving, tool calls, structured
  outputs, context, quantization, and memory.
- `experiments/registry/model_registry.json`: local registry with `candidate`,
  `champion`, and `rollback` aliases.
- `experiments/registry/training_plans/*.json`: dry-run plans that bind a
  training candidate to a dataset manifest and optional compatibility report.

## Command Sequence

```bash
flightrecorder model-candidate validate experiments/registry/model_candidates/local_mock_tiny_chat.json --require-training-eligible
flightrecorder model-candidate compatibility-report \
  --candidate experiments/registry/model_candidates/local_mock_tiny_chat.json \
  --out experiments/registry/compatibility/local_mock_tiny_chat.compatibility_report.json
flightrecorder model-registry register \
  --registry experiments/registry/model_registry.json \
  --candidate experiments/registry/model_candidates/local_mock_tiny_chat.json \
  --entry-out experiments/registry/model_registry_entries/local_mock_tiny_chat.json
flightrecorder model-registry alias \
  --registry experiments/registry/model_registry.json \
  --alias candidate \
  --target local_mock_tiny_chat \
  --reason "metadata-only local fixture ready for dry-run planning"
flightrecorder training-plan dry-run \
  --registry experiments/registry/model_registry.json \
  --model-ref candidate \
  --dataset-id local_mock_dataset_v1 \
  --dataset-manifest experiments/registry/datasets/local_mock_dataset_manifest.json \
  --trainer local-dry-run \
  --mode sft \
  --output-dir experiments/registry/training_outputs/local_mock_tiny_chat \
  --out experiments/registry/training_plans/local_mock_tiny_chat_sft_dry_run.json \
  --compatibility-report experiments/registry/compatibility/local_mock_tiny_chat.compatibility_report.json
flightrecorder model-registry link \
  --registry experiments/registry/model_registry.json \
  --entry local_mock_tiny_chat \
  --collection datasets \
  --artifact-id local_mock_dataset_v1 \
  --kind dataset_manifest \
  --status dry_run_stub \
  --path experiments/registry/datasets/local_mock_dataset_manifest.json \
  --entry-out experiments/registry/model_registry_entries/local_mock_tiny_chat.json \
  --metadata role=training_input
flightrecorder model-registry link \
  --registry experiments/registry/model_registry.json \
  --entry local_mock_tiny_chat \
  --collection training_runs \
  --artifact-id local_mock_tiny_chat_sft_dry_run \
  --kind training_plan \
  --status dry_run_plan \
  --path experiments/registry/training_plans/local_mock_tiny_chat_sft_dry_run.json \
  --entry-out experiments/registry/model_registry_entries/local_mock_tiny_chat.json
flightrecorder validate \
  --model-scout-manifest experiments/registry/model_scout_manifest.json \
  --model-candidate experiments/registry/model_candidates/local_mock_tiny_chat.json \
  --model-compatibility-report experiments/registry/compatibility/local_mock_tiny_chat.compatibility_report.json \
  --model-registry-entry experiments/registry/model_registry_entries/local_mock_tiny_chat.json \
  --model-registry experiments/registry/model_registry.json \
  --training-plan experiments/registry/training_plans/local_mock_tiny_chat_sft_dry_run.json \
  --strict
```

## Metadata-Only Real Candidate

`experiments/registry/model_candidates/qwen3_4b_instruct_2507.json` records a
real Hugging Face model card review for `Qwen/Qwen3-4B-Instruct-2507` without
downloading weights, tokenizer vocabulary files, or running GPU work. The
candidate records SHA-256 hashes for the small metadata files used during
review: `README.md`, `config.json`, `tokenizer_config.json`, and `LICENSE`.

The corresponding artifacts are:

- `experiments/registry/compatibility/qwen3_4b_instruct_2507.compatibility_report.json`
- `experiments/registry/model_registry_entries/qwen3_4b_instruct_2507.json`
- `experiments/registry/training_plans/qwen3_4b_instruct_2507_sft_dry_run.json`

The dry-run plan intentionally sets smoke assumptions such as
`max_seq_length=32768` while preserving the upstream 262,144-token context
metadata. Any real trainer must re-check license, serving compatibility,
runtime memory, dataset gates, and output paths before execution.

Unknown license status can be recorded for scouting, but it is blocked from
training selection. Moving `champion` requires an explicit, different
`rollback` target.
