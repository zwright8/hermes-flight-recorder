# State Validator Examples

State validators compile common external-action checks into normal scenario
assertions. They do not replace scoring; they generate `required_actions`,
`required_state`, and `required_state_transitions` blocks that existing
scorecards already understand.

List monitorable external tools and states:

```bash
flightrecorder state-validators --list --markdown-out monitor-catalog.md
```

Compile a validator config:

```bash
flightrecorder state-validators \
  --config examples/state_validators/email_sent.validator.json \
  --out email_sent.assertions.json
```

Copy the generated `assertions` object into a scenario, then run that scenario
with `--before-state` and `--state` snapshots captured by `verify-state` or
`capture-state`.

