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

For unordered API results, use validators such as `collection_item_exists`,
`slack_message_sent`, `calendar_event_created`, or `k8s_resource_ready`. They
compile to `where_any` assertions, which require all field checks to match the
same item in a collection.
