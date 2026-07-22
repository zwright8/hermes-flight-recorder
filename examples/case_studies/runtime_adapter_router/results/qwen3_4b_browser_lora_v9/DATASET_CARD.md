---
pretty_name: Hermes Flight Recorder Browser Tool-Calling Trajectories
license: apache-2.0
task_categories:
- text-generation
tags:
- agents
- tool-calling
- browser
- synthetic
- flight-recorder
configs:
- config_name: governed_browser
  data_files:
  - split: train
    path: data/browser/flightrecorder_action_sft.jsonl
  - split: validation
    path: data/development_action_sft.jsonl
  - split: test
    path: data/sealed_final_action_sft.jsonl
---

# Hermes Flight Recorder Browser Tool-Calling Trajectories

This dataset repository publishes the exact public-synthetic artifacts used by
the Qwen3-4B browser LoRA case study.

- `data/browser/flightrecorder_action_sft.jsonl`: governed browser train view.
- `data/development_action_sft.jsonl`: frozen multi-scope development file; the
  evaluator selects the browser task scope.
- `data/sealed_final_action_sft.jsonl`: original frozen multi-scope final file;
  the evaluator selects the browser task scope.
- `registry/browser_dataset_version.json`: immutable dataset registry entry.
- `controls/`: governance, curation, contamination, replay, credit, and review
  evidence referenced by the registry entry.
- `corpus_manifest.json`: deterministic corpus identity and split metadata.

All rows declare `public-synthetic` sensitivity and Apache-2.0 synthetic
fixture licensing. They contain no production traces, credentials, personal
data, live endpoints, or real tool side effects.

The published development and sealed-final files are now burned for future
model selection. They exist to make the historical claim auditable and must not
be reused as hidden evaluation data for later recipes.

Build logic lives in
[`scripts/build_runtime_adapter_training_corpus.py`](https://github.com/zwright8/hermes-flight-recorder/blob/codex/runtime-adapter-router/scripts/build_runtime_adapter_training_corpus.py).
The paired model evidence is in
[PR #34](https://github.com/zwright8/hermes-flight-recorder/pull/34).
