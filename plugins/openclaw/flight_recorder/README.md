# Flight Recorder OpenClaw Plugin

Read-only OpenClaw hook collector for Hermes Flight Recorder.

```bash
openclaw plugins install plugins/openclaw/flight_recorder --link
openclaw plugins enable flight-recorder

openclaw config patch --stdin <<'JSON'
{
  "plugins": {
    "entries": {
      "flight-recorder": {
        "enabled": true,
        "hooks": { "allowConversationAccess": true },
        "config": { "outputDir": ".hfr-openclaw" }
      }
    }
  }
}
JSON

openclaw gateway run
openclaw agent --message "hello" --json

flightrecorder run \
  --scenario examples/openclaw/support_ticket_completion_openclaw.json \
  --trace .hfr-openclaw/<session>.openclaw.jsonl \
  --format openclaw_jsonl \
  --out runs/openclaw
```

The plugin observes OpenClaw hook payloads and appends bounded JSONL rows. It
does not block, rewrite, approve, or deny agent actions.

Conversation hook access is required to capture prompts and final answers. Raw
`.openclaw.jsonl` files should be treated as sensitive traces.
