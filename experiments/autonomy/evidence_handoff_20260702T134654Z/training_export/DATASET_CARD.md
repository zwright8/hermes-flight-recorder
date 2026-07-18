# Flight Recorder Dataset Card

This card summarizes a Flight Recorder training export. It is generated from the same canonical artifacts as the JSONL views.

## Summary

- Dataset version: `hfrds-8c428c4dcf70041d`
- Source runs: `experiments/autonomy/evidence_handoff_20260702T134654Z`
- Registry: `experiments/autonomy/evidence_handoff_20260702T134654Z/training_export/dataset_registry.json`
- Reward scale: `score`
- Episodes: 7 (2 passed, 5 failed)
- Pass rate: 0.2857
- Average score: 50.7143
- Average reward: 0.5071

## Source Fingerprints

- Fully verified episodes: 7 / 7
- Scenario fingerprints: 7
- Source-trace fingerprints: 7
- Fully verified trainer-view rows: 11 / 11
- Unverified trainer-view rows: 0

## Redaction

- Passed: True
- Unredacted secret findings: 0
- Scanner: `flightrecorder.generic_secret_assignment.v1`

## Label Provenance

- Positive episodes: 2
- Eligible positive episodes: 2
- Final-answer-only success exclusions: 0
- SFT labels: 2
- DPO pairs: 2
- Reward-model labels: 7

## Trace Signal

- Average events per episode: 6.0
- Event types: 6
- Final-answer rate: 1.0
- Tool/API episode rate: 0.8571
- Trace risk count: 2

## Dataset Splits

- Task families: 5
- Family exclusive: True
- Train episodes: 4
- Validation episodes: 2
- Test episodes: 1

## Artifact Counts

| Artifact | Count |
| --- | ---: |
| `dpo` | 2 |
| `episodes` | 7 |
| `failure_modes` | 14 |
| `preferences` | 2 |
| `reward_model` | 7 |
| `rewards` | 7 |
| `sft` | 2 |
| `step_rewards` | 23 |

## Task Families

| Family | Episodes | Passed | Failed | Avg Score | Step Rewards | Failures | SFT | DPO | Reward Model |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| budget_runaway | 1 | 0 | 1 | 75.0 | 3 | 1 | 0 | 0 | 1 |
| cron_async_delegation | 1 | 0 | 1 | 10.0 | 3 | 3 | 0 | 0 | 1 |
| email_reply_completion | 2 | 1 | 1 | 50.0 | 5 | 5 | 1 | 1 | 2 |
| prompt_injection | 2 | 1 | 1 | 50.0 | 11 | 4 | 1 | 1 | 2 |
| subagent_claim | 1 | 0 | 1 | 70.0 | 1 | 1 | 0 | 0 | 1 |

## Failure Pressure

| Rule | Count |
| --- | ---: |
| `budget` | 1 |
| `final_answer` | 1 |
| `forbidden_actions` | 1 |
| `required_action_sequences` | 2 |
| `required_actions` | 1 |
| `required_event_counts` | 2 |
| `required_evidence` | 3 |
| `required_state` | 1 |
| `required_state_transitions` | 1 |
| `secret_exposure` | 1 |

## Quality Flags

- No dataset-level quality flags were emitted.

## Boundaries

- These artifacts are deterministic eval evidence and trainer-ready views, not a trainer.
- Reward labels are only as strong as the scenario policies and observable trace evidence.
- Review HTML reports and scorecards before using exported rows for model updates.
