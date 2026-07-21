# Runtime Adapter Router Case Study

This case study supplies the realistic training and held-out data for Flight
Recorder's generalist-plus-specialists architecture.

The generated corpus contains:

- 360 native action-SFT trajectories;
- 288 train, 36 development, and 36 sealed-final rows;
- generalist, browser, database, code/terminal, and shared-safety scopes;
- exact immutable tool definitions, call IDs, arguments, results, and order;
- multi-call, cross-domain, ambiguity, refusal, failure-recovery, approved
  write, and denied-write examples;
- replayed governance, contamination, curation, action-credit, review, and
  dataset-registration controls.

`data/flightrecorder_action_sft.jsonl` and each scoped trainer file contain only
train rows. `data/all_splits_action_sft.jsonl` is an audit view and must not be
passed to a trainer. `data/sealed_final_action_sft.jsonl` is reserved for final
candidate evaluation.

Rebuild deterministically:

```bash
python3.11 scripts/build_runtime_adapter_training_corpus.py \
  --output-dir examples/case_studies/runtime_adapter_router \
  --count 360 \
  --seed 17
```

Validate the governed router and dispatch boundary:

```bash
python3.11 scripts/validate_runtime_adapter_router.py
```

The optional local trainer and evaluator are:

```bash
python3.11 scripts/train_agentic_lora.py --help
python3.11 scripts/evaluate_runtime_adapter_candidates.py --help
```

Local adapter weights and raw evaluation observations belong under ignored
`runs/` directories. Each candidate must have its own training result, adapter
directory fingerprint, raw observations, promotion decision, and route
descriptor. The completed integration run trained all four
candidates but promoted none: strict final-answer and safety checks failed even
where tool-call shape improved. No aggregate result authorizes routing.

The synthetic, schema-validated sealed report from that actual local MPS run is
committed at `evaluation/actual_local_evaluation.json`. It intentionally omits
adapter weights and preserves the fail-closed result: all five evaluated arms
(base, generalist, browser, database, and code/terminal) failed the strict gate,
so zero candidates are eligible for promotion. Its fingerprint binds the
held-out identity, candidate identities, metrics, and per-task scores.
