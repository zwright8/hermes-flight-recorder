---
license: apache-2.0
language:
- en
task_categories:
- text-generation
pretty_name: Hermes Flight Recorder Self-Improving Agent Trajectories
size_categories:
- n<1K
tags:
- agents
- tool-calling
- synthetic
- flight-recorder
---

# Hermes Flight Recorder Self-Improving Agent Trajectories

This public-safe synthetic dataset contains 800 governed agent trajectories
for supervised tool-use training, 120 development tasks, and a separately
frozen set of 150 final evaluation tasks. It demonstrates how recorded
successful executions and reviewed safety
refusals can become training data without publishing user traces.

## Files

- `train_trajectories.jsonl`: 800 training-only conversational tool-use rows
- `development_tasks.jsonl`: 120 non-gradient candidate-selection tasks
- `heldout_tasks.jsonl`: 150 immutable final evaluation-only tasks
- `frozen_heldout_manifest.json`: artifact hashes and exclusion policy
- `contamination_audit.json`: task ID, record key, prompt hash, and template
  disjointness evidence
- `dataset_manifest.json`: counts, task-family distribution, and dataset identity

Each training row includes native `messages`, exact `tools` JSON schemas,
governance metadata, review binding, task family, and the expected action. The
held-out rows use different task IDs, record IDs, prompt text, and prompt
templates. Route codes are shared because the benchmark measures whether the
model learned the organization convention from experience.

Every JSONL row conforms to the registered
`hfr.self_improving_agent_episode.v1` contract. The `split`, `pool`, and
`training_role` fields distinguish gradient-bearing trajectories from
development and final held-out evaluation tasks; these rows are intentionally
not labeled as production `export-rl` action-SFT artifacts.

## Intended use

Use `train_trajectories.jsonl` for SFT and `heldout_tasks.jsonl` only for final
evaluation. Do not tune hyperparameters, prompts, decoding, or selection rules
against the held-out outputs. The dataset is a bounded research proof, not a
claim that synthetic trajectories replace diverse reviewed production data.

## Privacy and safety

Every identifier and trajectory is deterministic synthetic data. No mailbox,
filesystem, browser, customer, credential, or production trace data is
included. Write-capable examples require an `APPROVED-` token; adversarial
held-out examples test that missing or injected approval is refused.
