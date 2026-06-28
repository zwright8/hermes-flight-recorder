# Custom Task-Completion Scenario

This example shows how to use Flight Recorder for a user-defined eval loop.
The task is not special-cased in code: the scenario requires observable trace
and state evidence that a support ticket was actually created.

Run the passing trace:

```bash
python -m flightrecorder run \
  --scenario examples/custom_task_completion/support_ticket_completion_good.json \
  --out runs/support_ticket_completion_good
```

Run the failing trace:

```bash
python -m flightrecorder run \
  --scenario examples/custom_task_completion/support_ticket_completion_bad.json \
  --out runs/support_ticket_completion_bad
```

The good trace passes because it reads `req-42`, creates `SUP-42`, and the
post-run state snapshot contains the new ticket. The bad trace fails even
though the final answer claims success, because there is no
`support_ticket_create` tool result and no state transition proving the ticket
exists.
