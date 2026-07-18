# Live Hermes Qwen3-4B Baseline N100 MPS32

This baseline evaluates the unfine-tuned `Qwen/Qwen3-4B-Instruct-2507` model as a Hermes agent on the full 100-task held-out Flight Recorder suite before training on the new live Flight Recorder dataset.

## Scope

- Baseline model: `Qwen/Qwen3-4B-Instruct-2507`
- Runtime: Hermes CLI with Flight Recorder observer plugin
- Serving path: local `scripts/serve_transformers_openai.py`
- Device/dtype: `mps:0`, `torch.float16`
- Generation cap: `--max-new-tokens 32`
- Hermes turn cap: `--max-turns 4`
- Per-task timeout: `120` seconds
- Held-out selection: 100 scenarios, 10 per task family
- Task families: `artifact_verification`, `code_edit_test`, `cron_async_completion`, `delegation_claim`, `file_read_write`, `git_repo_inspection`, `local_research`, `prompt_injection_resistance`, `terminal_command`, `unsupported_claim`

## Stability Note

The earlier N40 baseline used `--max-new-tokens 128`, but the local MPS server was repeatedly killed during full N100 attempts under that cap. CPU serving completed model load but timed out on a single live Hermes task. The full N100 baseline therefore uses a stable 32-token generation cap.

For a valid prove/disprove comparison, evaluate the baseline, trace-only adapter, and Flight Recorder adapter with the same run config in `RUN_CONFIG.json`.

## Leakage Check

- Training catalog: `experiments/live_hermes_qwen4b_data/scenario_catalog.json`
- Held-out catalog: `experiments/live_hermes_qwen4b_baseline_n100_mps32/scenario_catalog.json`
- Held-out selected scenario count: 100
- Selected scenario id overlap with training: 0
- Full held-out catalog scenario id overlap with training: 0
- Selected prompt overlap with training: 0
- Full held-out catalog prompt overlap with training: 0

Held-out task ids use the `101`-series index range; training task ids use the `001`-series range.

## Baseline Metrics

- Runs: 100
- Execution errors: 0
- Passed: 0
- Failed: 100
- Pass rate: 0.0
- Average score: 47.0
- Max score: 50
- Min score: 20

Critical failures:

- `final_answer`: 100
- `required_evidence`: 100
- `required_actions`: 10

Per-family coverage:

- `artifact_verification`: 10
- `code_edit_test`: 10
- `cron_async_completion`: 10
- `delegation_claim`: 10
- `file_read_write`: 10
- `git_repo_inspection`: 10
- `local_research`: 10
- `prompt_injection_resistance`: 10
- `terminal_command`: 10
- `unsupported_claim`: 10

## Validation

- Strict artifact validation: passed
- Validation target count: 100
- Validation errors: 0
- Validation warnings: 0
- Scenario check: passed
- Scenario check errors: 0
- Scenario check warnings: 100

The scenario-check warnings are expected static advisories because the live runner injects trace paths at execution time. Strict validation of the produced run artifacts is clean.

## Artifacts

- Run config: `RUN_CONFIG.json`
- Collection summary: `collection_summary.json`
- Suite summary: `suite_summary.json`
- Strict validation report: `validation_report.json`
- Scenario check: `scenario_check.json`
- Leakage check: `leakage_check.json`
- Per-run traces and scorecards: `runs/*/`

## Comparison Use

Use `suite_summary.json` as the baseline summary when comparing future LoRA adapters trained on `experiments/live_hermes_qwen4b_data`.

Example:

```bash
python3.11 scripts/compare_agentic_finetune_results.py \
  --baseline experiments/live_hermes_qwen4b_baseline_n100_mps32/suite_summary.json \
  --trace-only <trace-only-suite-summary.json> \
  --flightrecorder <flightrecorder-suite-summary.json> \
  --out <comparison.json> \
  --report <comparison.md>
```

The trace-only and Flight Recorder adapter arms should use the same 100 held-out scenario ids, same Hermes/Flight Recorder runner, same local serving shim, and the same generation/turn caps unless all arms are rerun.
