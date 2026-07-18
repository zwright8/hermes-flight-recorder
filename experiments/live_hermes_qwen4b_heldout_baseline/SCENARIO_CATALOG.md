# Live Hermes Flight Recorder Scenario Catalog

This catalog bootstraps executable Hermes tasks for Qwen3-4B LoRA data collection.

- Tasks: 100
- Expected pass/fail: {'fail': 20, 'pass': 80}
- Projected action-SFT rows from expected-pass tasks: 312

## Task Families

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

## Data Quality Contract

- Passing tasks require tool-result evidence.
- Failed tasks are retained for DPO/reward negatives, not SFT.
- Workspaces are local, synthetic, and safe to execute.
- Prompt-injection tasks include a dummy secret and forbidden-action policy.
