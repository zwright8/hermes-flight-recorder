# External Verification Examples

Use `verify-state` when task completion must be proven outside the agent/model.
Capture state before the run, run the agent, capture state after the run, then
score the trace with both snapshots.

```bash
flightrecorder verify-state \
  --config examples/external_verification/imap_sent_email.verifier.json \
  --out before_state.json

# Run the agent here.

flightrecorder verify-state \
  --config examples/external_verification/imap_sent_email.verifier.json \
  --out after_state.json

flightrecorder run \
  --scenario scenarios/email_reply_completion_good.json \
  --trace path/to/agent.trace.jsonl \
  --before-state before_state.json \
  --state after_state.json \
  --out runs/email_reply_verified
```

Available source types include `eml`, `maildir`, `imap`, `gmail_threads`,
`github_issue`, `http_json`, and `sqlite`. All bundled adapters are read-only.

For JSON APIs, set `state_path` to the state name your scenario will read. If
the useful data is nested inside the API response, set `state_value_path` to copy
only that sub-tree into the snapshot. For example, Slack-style channel history
can map `json.messages` to `slack.messages`, which then works directly with the
`slack_message_sent` validator.
