# Model Layer Registry

Goal 4 model-layer artifacts live under `experiments/registry/` and are safe to
produce without model-weight downloads, heavy ML imports, or GPU execution.

## Artifacts

- `model_candidate`: base-model scouting metadata, source, license review,
  accepted terms, and compatibility notes.
- `model_scout_manifest`: scouting intake list with candidate artifact refs,
  selection policy, license posture, registry entry refs, and compatibility
  proof links.
- `model_registry_entry`: a validated candidate snapshot plus lifecycle links
  for datasets, training runs, adapters, evals, and promotion decisions.
- `model_compatibility_report`: no-download report over context length,
  tokenizer, chat template, serving, tool calls, structured outputs,
  quantization, and memory metadata.
- `model_registry`: local registry container with explicit `candidate`,
  `champion`, and `rollback` aliases plus alias history.
- `training_plan`: dry-run plan referencing model, dataset, trainer,
  hyperparameters, output paths, compute assumptions, and optional
  compatibility-report proof.

## License Gate

Training selection uses an allowlist. A candidate is training-eligible only when
all of these are true:

- `license.status == "approved"`
- `license.review_status == "approved"`
- `license.accepted_terms == true`
- `license.training_allowed == true`

`unknown`, `restricted`, `rejected`, `noncommercial`, and future non-allowlisted
statuses can be recorded for scouting but are blocked from training selection
and dry-run training-plan generation.

## Commands

Validate a candidate:

```bash
python3 -m flightrecorder model-scout validate \
  --manifest experiments/registry/model_scout_manifest.json \
  --strict

python3 -m flightrecorder model-candidate validate \
  --candidate experiments/registry/model_candidates/local_mock_tiny_chat.json \
  --require-training-eligible
```

Register a candidate and set the candidate alias:

```bash
python3 -m flightrecorder model-registry register \
  --registry experiments/registry/model_registry.json \
  --candidate experiments/registry/model_candidates/local_mock_tiny_chat.json

python3 -m flightrecorder model-registry alias \
  --registry experiments/registry/model_registry.json \
  --alias candidate \
  --target local_mock_tiny_chat
```

Attach lifecycle artifacts without moving aliases:

```bash
python3 -m flightrecorder model-registry link \
  --registry experiments/registry/model_registry.json \
  --entry candidate \
  --type dataset \
  --artifact-id hfrds-example \
  --path runs/training_export/manifest.json

python3 -m flightrecorder model-registry link \
  --registry experiments/registry/model_registry.json \
  --entry local_mock_tiny_chat \
  --type eval \
  --artifact-id eval-flightrecorder-heldout \
  --path experiments/qwen3_4b_flightrecorder/evaluations/flightrecorder/evaluation_summary.json \
  --metadata arm=flightrecorder
```

Write a no-download compatibility report:

```bash
python3 -m flightrecorder model-candidate compatibility-report \
  --candidate experiments/registry/model_candidates/local_mock_tiny_chat.json \
  --out experiments/registry/compatibility_reports/local_mock_tiny_chat.json
```

Attach lifecycle links to a registry entry:

```bash
python3 -m flightrecorder model-registry link \
  --registry experiments/registry/model_registry.json \
  --entry candidate \
  --type dataset \
  --artifact-id mock-dataset-v1 \
  --path experiments/registry/datasets/mock_dataset_manifest.json \
  --metadata redaction_status=redacted

python3 -m flightrecorder model-registry link \
  --registry experiments/registry/model_registry.json \
  --entry candidate \
  --type training-run \
  --artifact-id local_mock_tiny_chat_sft_dry_run \
  --path experiments/registry/training_plans/local_mock_tiny_chat_sft_dry_run.json \
  --metadata mode=sft \
  --metadata dry_run=true
```

Move `champion` only with an explicit rollback target:

```bash
python3 -m flightrecorder model-registry alias \
  --registry experiments/registry/model_registry.json \
  --alias champion \
  --target new_candidate_entry \
  --rollback-target previous_champion_entry
```

Write a dry-run training plan:

```bash
python3 -m flightrecorder training-plan dry-run \
  --registry experiments/registry/model_registry.json \
  --model candidate \
  --dataset-id mock-dataset-v1 \
  --dataset-manifest experiments/registry/datasets/mock_dataset_manifest.json \
  --compatibility-report experiments/registry/compatibility_reports/local_mock_tiny_chat.json \
  --trainer local-test-trainer \
  --mode sft \
  --output-dir experiments/registry/adapters/local_mock_tiny_chat_sft \
  --out experiments/registry/training_plans/local_mock_tiny_chat_sft_dry_run.json
```

When `--compatibility-report` is provided, the plan fingerprints that exact
report and blocks if the report does not match the selected candidate/model,
did not pass, or records any weight/tokenizer download, heavy ML import, or GPU
execution.

Validate generated artifacts:

```bash
python3 -m flightrecorder validate \
  --model-scout-manifest experiments/registry/model_scout_manifest.json \
  --model-candidate experiments/registry/model_candidates/local_mock_tiny_chat.json \
  --model-compatibility-report experiments/registry/compatibility_reports/local_mock_tiny_chat.json \
  --model-registry experiments/registry/model_registry.json \
  --training-plan experiments/registry/training_plans/local_mock_tiny_chat_sft_dry_run.json \
  --strict
```
