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
`github_issue`, `slack_history`, `google_calendar_events`,
`google_drive_files`, `microsoft_graph_messages`, `microsoft_graph_events`,
`kubernetes_resources`, `stripe_objects`, `notion_database`, `linear_issues`,
`jira_issues`, `gitlab_issues`, `discord_messages`, `zendesk_tickets`,
`pagerduty_incidents`, `s3_objects`, `http_json`, and `sqlite`. All bundled
adapters are read-only.

For JSON APIs, set `state_path` to the state name your scenario will read. If
the useful data is nested inside the API response, set `state_value_path` to copy
only that sub-tree into the snapshot. For example, Slack-style channel history
can map `json.messages` to `slack.messages`, which then works directly with the
`slack_message_sent` validator.

Provider-shaped adapters generally do that mapping for you. For example,
`slack_history` can write to `slack`, making messages available at
`slack.messages`, while `stripe_objects` can write a single payment intent to
`payments.payment.status`. List-shaped sources such as `linear_issues`,
`jira_issues`, and `s3_objects` can use `state_value_path` to copy one issue or
the object list into the exact path your validator expects.
