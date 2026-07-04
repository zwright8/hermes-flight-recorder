# Flight Recorder Improvement Pair Card

This card summarizes baseline-vs-candidate preference artifacts generated from paired Flight Recorder runs.

## Summary

- Baseline runs: `examples/agentic_training/promotion_governance/compare_runs/baseline`
- Candidate runs: `examples/agentic_training/promotion_governance/compare_runs/candidate`
- Pair count: 1
- Candidate wins: 1
- Baseline wins: 0
- Contract drift: 0
- Unverified contracts: 0
- Skipped pairs: 0

## Experiment Metadata

| Key | Value |
| --- | --- |
| `baseline` | local/mock-baseline |
| `candidate` | local/mock-candidate |
| `contract` | shared-email-reply-scenario |

## Pairs

| Scenario | Candidate Outcome | Contract | Delta | Chosen | Rejected | Reason |
| --- | --- | --- | ---: | --- | --- | --- |
| `email_reply_completion` | improved | matched | 100 | candidate | baseline | candidate score 100 beat baseline score 0; candidate_delta=100. Fixed rules: required_action_sequences, required_actions, required_event_counts, required_state, required_state_transitions. |

## Boundaries

- These rows preserve deterministic comparison evidence; they are not proof of causal model improvement.
- Use candidate wins as improvement examples and baseline wins as regression-avoidance examples.
- Review the source reports before using preference rows for model updates.
