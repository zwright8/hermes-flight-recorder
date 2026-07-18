# Case Study: Fine-tuning Qwen3-0.6B from a Flight Recorder Trajectory

This bounded demonstration shows that a trajectory recorded and gated by
Hermes Flight Recorder can be consumed by the repository's TRL/PEFT training
path and produce a real LoRA update. It is an integration and memorization
proof, not evidence that one example improves general agent quality.

## What was trained

- Base model: [`Qwen/Qwen3-0.6B`](https://huggingface.co/Qwen/Qwen3-0.6B)
- Trainer mode: `fr_action_sft`
- Method: LoRA, rank 8, alpha 16, dropout 0
- Hardware: Apple Silicon through PyTorch MPS
- Training data: one synthetic, redacted email-tool trajectory
- Source dataset: `hfrds-a801878214397fed`, built from commit
  `76e144e38526d88fb35cd755a3cac400cc81bdf8`

The source suite contained seven recorded episodes across five task families.
Flight Recorder produced two passing and five failing episodes, enforced
family-exclusive train/validation/test splits, and passed all 50 dataset-gate
checks. The positive action-SFT row in
[`data/action_sft.jsonl`](data/action_sft.jsonl) came from the train-only
`email_reply_completion` family. Its scorecard, configured task-completion
evidence, and source fingerprints all passed verification.

The row preserves the native agent interaction rather than flattening it into
prompt/answer text:

```text
user → assistant tool call → tool result →
assistant tool call → tool result → assistant
```

## Reproduce the data path

The exact generated run directories are intentionally ignored. Starting from
the repository root, rebuild and gate the data with:

```bash
python3 -m flightrecorder run-suite \
  --scenarios scenarios \
  --out runs/qwen3_0_6b_case_study \
  --export-rl \
  --validate \
  --strict

python3 -m flightrecorder gate-export \
  --training-export runs/qwen3_0_6b_case_study/training_export \
  --policy examples/training_gate_policy.demo.json \
  --out runs/qwen3_0_6b_case_study/training_gate.json

python3 scripts/build_agentic_finetune_experiment.py \
  --runs-dir runs/qwen3_0_6b_case_study \
  --out runs/qwen3_0_6b_case_study/experiment \
  --model Qwen/Qwen3-0.6B
```

Review the generated dataset manifest and model license decision before a real
launch. A dry run records the full launch plan without loading the ML stack:

```bash
python3 scripts/train_agentic_lora.py \
  --mode fr_action_sft \
  --dry-run \
  --require-registered-inputs \
  --experiment-dir runs/qwen3_0_6b_case_study/experiment \
  --model-manifest examples/case_studies/qwen3_0_6b_flightrecorder_lora/model_manifest.json \
  --dataset-manifest runs/qwen3_0_6b_case_study/experiment/dataset_training_manifest.json \
  --output-dir runs/qwen3_0_6b_case_study/adapter \
  --limit 1 \
  --max-steps 20 \
  --max-length 512 \
  --batch-size 1 \
  --gradient-accumulation-steps 1 \
  --sft-learning-rate 5e-4 \
  --lora-r 8 \
  --lora-alpha 16 \
  --all-message-loss \
  --disable-trackio
```

Remove `--dry-run` only after reviewing the plan and choosing an appropriate
local or Hugging Face Jobs runtime. This run used PyTorch 2.12.1,
Transformers 4.57.6, PEFT 0.19.1, and a TRL version below 1. The current Qwen3
chat template does not expose the `{% generation %}` masks required by TRL's
assistant-only loss, so the demo explicitly records `assistant_only_loss:
false` through `--all-message-loss`.

## Result

The 20-step run completed in 22.7 seconds. The complete curve is preserved in
[`training_curve.csv`](training_curve.csv), and the portable evaluation receipt
is in [`evaluation.json`](evaluation.json).

| Metric | Base | LoRA-tuned | Change |
| --- | ---: | ---: | ---: |
| Sequence loss on the training trajectory | 3.0694 | 0.1691 | -94.49% |
| Training token accuracy | 0.6167 at step 1 | 1.0000 at step 20 | +0.3833 |
| Training loss | 3.1740 at step 1 | 0.0227 at step 20 | -99.28% |

The adapter contained 20,236,472 bytes and had SHA-256
`5642c36e91c69098666e19c653ea43639169adfaf1cdf226f16501b5165cc891`.
That digest is recorded for provenance; the adapter itself is not stored in
Git.

## What failed, and why

Two failures made the reproducibility boundary clearer:

1. An unconstrained dependency resolution selected incompatible next-major ML
   packages and was unstable on MPS. The successful run stayed inside the
   repository's Transformers `<5`, TRL `<1`, and PEFT `<1` compatibility band.
2. TRL rejected `assistant_only_loss=True` because Qwen3's current chat
   template does not return an assistant mask. The trainer now exposes
   `--all-message-loss` and records the selected behavior in its launch plan.

A Hugging Face Jobs launch was also attempted but rejected before compute with
HTTP 402 because the account had insufficient prepaid credit. No managed job,
charge, model repository, or external artifact was created.

## Claim boundary

This case study proves the end-to-end interface:

```text
recorded evidence → validation and gating → native tool trajectory →
TRL/PEFT LoRA update → portable metrics and adapter digest
```

It does **not** establish held-out task improvement. The single-row model can
memorize the training trajectory, and its sampled tuned completion was not
structurally better than the base completion. A production claim requires more
diverse accepted trajectories, held-out family evaluation, tool-call parsing
checks, and promotion gates.

## Artifact placement

GitHub is the system of record for the reproducible case study, redacted data,
manifests, metrics, and commands. Hugging Face Hub is the better home for the
LoRA adapter because it provides model versioning, model cards, large-file
storage, and direct loading through PEFT. See [`MODEL_CARD.md`](MODEL_CARD.md)
for the publication-ready metadata and limitations. Checkpoints and model
weights must remain outside this repository.
