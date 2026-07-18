# Flight Recorder Dataset Card

This card summarizes a Flight Recorder training export. It is generated from the same canonical artifacts as the JSONL views.

## Summary

- Source runs: `experiments/live_hermes_qwen4b_data/runs`
- Reward scale: `score`
- Episodes: 100 (80 passed, 20 failed)
- Pass rate: 0.8
- Average score: 89.4
- Average reward: 0.894

## Experiment Metadata

| Key | Value |
| --- | --- |
| `base_model_target` | Qwen/Qwen3-4B-Instruct-2507 |
| `collection` | live_hermes_flightrecorder |

## Source Fingerprints

- Fully verified episodes: 100 / 100
- Scenario fingerprints: 100
- Source-trace fingerprints: 100
- Fully verified trainer-view rows: 340 / 340
- Unverified trainer-view rows: 0

## Trace Signal

- Average events per episode: 13.28
- Event types: 5
- Final-answer rate: 1.0
- Tool/API episode rate: 1.0
- Trace risk count: 0

## Dataset Splits

- Task families: 10
- Family exclusive: True
- Train episodes: 80
- Validation episodes: 10
- Test episodes: 10

## Artifact Counts

| Artifact | Count |
| --- | ---: |
| `dpo` | 160 |
| `episodes` | 100 |
| `failure_modes` | 42 |
| `preferences` | 160 |
| `reward_model` | 100 |
| `rewards` | 100 |
| `sft` | 80 |
| `step_rewards` | 62 |

## Task Families

| Family | Episodes | Passed | Failed | Avg Score | Step Rewards | Failures | SFT | DPO | Reward Model |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| artifact_verification | 10 | 8 | 2 | 90.0 | 6 | 4 | 8 | 16 | 10 |
| code_edit_test | 10 | 8 | 2 | 90.0 | 6 | 4 | 8 | 16 | 10 |
| cron_async_completion | 10 | 8 | 2 | 90.0 | 6 | 4 | 8 | 16 | 10 |
| delegation_claim | 10 | 8 | 2 | 90.0 | 6 | 4 | 8 | 16 | 10 |
| file_read_write | 10 | 8 | 2 | 90.0 | 6 | 4 | 8 | 16 | 10 |
| git_repo_inspection | 10 | 8 | 2 | 90.0 | 6 | 4 | 8 | 16 | 10 |
| local_research | 10 | 8 | 2 | 90.0 | 6 | 4 | 8 | 16 | 10 |
| prompt_injection_resistance | 10 | 8 | 2 | 84.0 | 8 | 6 | 8 | 16 | 10 |
| terminal_command | 10 | 8 | 2 | 90.0 | 6 | 4 | 8 | 16 | 10 |
| unsupported_claim | 10 | 8 | 2 | 90.0 | 6 | 4 | 8 | 16 | 10 |

## Failure Pressure

| Rule | Count |
| --- | ---: |
| `final_answer` | 20 |
| `required_actions` | 2 |
| `required_evidence` | 20 |

## Quality Flags

- No dataset-level quality flags were emitted.

## Boundaries

- These artifacts are deterministic eval evidence and trainer-ready views, not a trainer.
- Reward labels are only as strong as the scenario policies and observable trace evidence.
- Review HTML reports and scorecards before using exported rows for model updates.
