# Rollout Generation Plan Example

This fixture is a deterministic, keyless rollout-generation plan. It schedules
baseline, candidate, and teacher harness batches over committed scenarios, but
does not call model providers, run harnesses, invoke graders, or write dataset
rows.
The fixture includes an `external_state_verifier_gate` that proves the declared
verifier config exists and that no verifier side effects or credential values
were recorded.

Regenerate it with:

```bash
flightrecorder agentic-rollout-plan \
  --iteration-id rollout-demo-001 \
  --scenario scenarios/prompt_injection_good.json \
  --scenario scenarios/email_reply_completion_good.json \
  --policy baseline=local/mock-baseline \
  --policy candidate=local/mock-candidate \
  --policy teacher=local/mock-teacher \
  --max-rollouts 6 \
  --verifier examples/external_verification/sqlite_task_state.verifier.json \
  --created-at 2026-07-03T00:00:00+00:00 \
  --out examples/rollout_generation/rollout_plan.json
```

Validate it with:

```bash
flightrecorder schemas --check examples/rollout_generation/rollout_plan.json
flightrecorder validate \
  --agentic-rollout-plan examples/rollout_generation/rollout_plan.json \
  --strict
```
