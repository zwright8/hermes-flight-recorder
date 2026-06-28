# OpenClaw Example

This example shows how OpenClaw hook JSONL can be evaluated by Flight Recorder.
The passing fixture proves a support ticket was created from observable tool
events, not just from the agent's final answer.

```bash
flightrecorder run \
  --scenario examples/openclaw/support_ticket_completion_openclaw.json \
  --out runs/openclaw_support_ticket_completion
```

For a live OpenClaw smoke, install the read-only plugin in
`plugins/openclaw/flight_recorder` or run:

```bash
python3.11 scripts/live_openclaw_smoke.py --out live_openclaw_smoke_artifacts/latest
```
