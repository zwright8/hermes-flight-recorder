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
- per-arm operational metrics for cost, latency, token usage, and
  task-completion status when suite summaries provide them,
- compare-gate failures,
- external adapter readiness blockers when adapter plans are included.

## Held-Out Manifests

Use suite summaries to build the canonical scenario manifest that downstream
evals must share:

```bash
flightrecorder heldout-manifest \
  --suite-summary baseline=runs_baseline/suite_summary.json \
  --suite-summary candidate=runs_candidate/suite_summary.json \
  --out runs/heldout_scenarios.json
```

A single suite source can seed external adapter planning. Cross-arm claims are
allowed only when two or more suite summaries prove the exact same scenario IDs.
Mismatched manifests validate as honest blocked artifacts, return nonzero from
the CLI, and block external adapter readiness when used as `--scenario-manifest`.

## External Adapter Plans

External harness adapters are represented by a readiness artifact before any
external eval claim can be made:

```bash
flightrecorder external-eval-plan \
  --scenario-manifest runs/heldout_scenarios.json \
  --model-endpoint http://127.0.0.1:8000/v1 \
  --out runs/external_eval_plan.json
```

The supported adapter IDs are `bfcl`, `inspect_ai`, `lm_eval_harness`, and
`swe_bench`. Plans fail closed by default: even when optional dependencies are
installed, an adapter is not ready unless `--allow-installed` is present and all
required inputs for that adapter are supplied. The plan records dependency
status, held-out scenario manifest SHA-256, required inputs, and blocking
reasons.

Include the plan in the governance summary:

```bash
flightrecorder eval-summary \
  --suite-summary baseline=runs_baseline/suite_summary.json \
  --suite-summary candidate=runs_candidate/suite_summary.json \
  --external-adapter-plan external=runs/external_eval_plan.json \
  --out runs/eval_summary.json
```

Validate summaries before handoff:

```bash
flightrecorder validate --eval-summary runs/eval_summary.json --strict
flightrecorder validate --heldout-manifest runs/heldout_scenarios.json --strict
flightrecorder validate --external-eval-plan runs/external_eval_plan.json --strict
```

## Blocking Conditions

Governance claims are suppressed when any of these hold:

- missing suite summaries for a cross-arm comparison,
- single-arm or mismatched held-out scenario sets,
- missing or new scenarios in the compare manifest,
- contract fingerprint drift,
- unverified contract fingerprints,
- zero comparison pairs.
- held-out manifests with mismatched or empty scenario sets.
- external adapter plans that are not ready.

The artifact may still show raw movement, but `governance_claims` stays empty and
`passed` is false until the blockers are resolved.

## Operational Metrics

Each eval-summary arm includes an `operational_metrics` object with `cost`,
`latency`, `tokens`, and `task_completion` sections. These fields are
best-effort summaries of values already present in suite summary run rows or
suite-level metrics. Missing values remain explicit through `source: "missing"`
and `missing_run_count`; they do not become evidence of low cost, low latency,
or task completion.
