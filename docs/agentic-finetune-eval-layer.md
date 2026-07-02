# Agentic Finetune Eval Layer

Goal 7 eval outputs must be directly consumable by Governance without turning
raw comparison movement into promotion evidence by accident.

## Held-Out Scenario Invariant

Cross-arm claims are valid only when every evaluated arm provides the identical
held-out scenario list. A candidate win, task-completion improvement, or rule fix
from a comparison export remains raw movement until the scenario-set check passes.

`flightrecorder eval-summary` enforces that boundary:

```bash
flightrecorder eval-summary \
  --suite-summary baseline=runs_baseline/suite_summary.json \
  --suite-summary candidate=runs_candidate/suite_summary.json \
  --compare-export candidate=runs/compare_rl_export \
  --compare-gate candidate=runs/compare_gate.json \
  --out runs/eval_summary.json
```

The summary reports:

- per-arm suite status and scenario IDs,
- whether held-out scenario lists are identical,
- raw compare-export movement,
- governance claims with candidate improvements suppressed when claims are not
  allowed,
- compare-gate failures,
- external adapter readiness blockers when adapter plans are included.

Validate summaries before handoff:

```bash
flightrecorder validate --eval-summary runs/eval_summary.json --strict
```

## Blocking Conditions

Governance claims are suppressed when any of these hold:

- missing suite summaries for a cross-arm comparison,
- single-arm or mismatched held-out scenario sets,
- missing or new scenarios in the compare manifest,
- contract fingerprint drift,
- unverified contract fingerprints,
- zero comparison pairs.

The artifact may still show raw movement, but `governance_claims` stays empty and
`passed` is false until the blockers are resolved.
