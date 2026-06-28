# Coven Example

This example scores a Coven `--stream-json --detach` trace. Detached Coven
sessions prove the runtime recorded a project-scoped session and prompt, but
they do not prove that an agent completed model work.

```bash
flightrecorder run \
  --scenario examples/coven/detached_session_coven.json \
  --out runs/coven_detached_session
```

For a real Coven daemon/session smoke, run:

```bash
python scripts/live_coven_smoke.py \
  --out live_coven_smoke_artifacts/latest
```
