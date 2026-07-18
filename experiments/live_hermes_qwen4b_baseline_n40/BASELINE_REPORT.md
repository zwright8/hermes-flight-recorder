# Live Hermes Qwen3-4B Baseline N40

This baseline evaluates the unfine-tuned `Qwen/Qwen3-4B-Instruct-2507` model as a Hermes agent on a larger held-out Flight Recorder task set before training on the new live Flight Recorder dataset.

## Scope

- Baseline model: `Qwen/Qwen3-4B-Instruct-2507`
- Runtime: Hermes CLI with Flight Recorder observer plugin
- Serving path: local `scripts/serve_transformers_openai.py`
- Generation cap: `--max-new-tokens 128`
- Hermes turn cap: `--max-turns 4`
- Per-task timeout: `120` seconds
- Held-out selection: 40 scenarios, 4 per task family
- Task families: `artifact_verification`, `code_edit_test`, `cron_async_completion`, `delegation_claim`, `file_read_write`, `git_repo_inspection`, `local_research`, `prompt_injection_resistance`, `terminal_command`, `unsupported_claim`

## Leakage Check

- Training catalog: `experiments/live_hermes_qwen4b_data/scenario_catalog.json`
- Held-out catalog: `experiments/live_hermes_qwen4b_baseline_n40/scenario_catalog.json`
- Held-out selected scenario count: 40
- Selected scenario id overlap with training: 0
- Full held-out catalog scenario id overlap with training: 0

Held-out task ids use the `101`-series index range; training task ids use the `001`-series range.

## Baseline Metrics

- Runs: 40
- Passed: 0
- Failed: 40
- Pass rate: 0.0
- Average score: 47.88
- Max score: 70
- Min score: 15

Critical failures:

- `required_evidence`: 40
- `final_answer`: 38
- `required_actions`: 3
- `forbidden_actions`: 1

Per-family coverage:

- `artifact_verification`: 4
- `code_edit_test`: 4
- `cron_async_completion`: 4
- `delegation_claim`: 4
- `file_read_write`: 4
- `git_repo_inspection`: 4
- `local_research`: 4
- `prompt_injection_resistance`: 4
- `terminal_command`: 4
- `unsupported_claim`: 4

## Artifacts

- Collection summary: `collection_summary.json`
- Suite summary: `suite_summary.json`
- Strict validation report: `validation_report.json`
- Scenario check: `scenario_check.json`
- Per-run traces and scorecards: `runs/*/`

## Comparison Use

Use `suite_summary.json` as the baseline summary when comparing a future LoRA adapter trained on `experiments/live_hermes_qwen4b_data`.

The comparison should use the same selected held-out scenario ids, same Hermes/Flight Recorder runner, same local serving shim, and the same generation/turn caps unless the baseline is rerun.

