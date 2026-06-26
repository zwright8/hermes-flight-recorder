# Deployment Guide

This guide describes a local or CI deployment for Hermes Flight Recorder.

## Install

From the project root:

```bash
python -m pip install . --no-deps
flightrecorder --help
```

For editable development:

```bash
python -m pip install -e . --no-deps
```

## Offline Verification

```bash
python -m unittest discover
./demo.sh
```

The demo generates `runs/index.html` with one passing and three failing
scenarios. No API keys or network are required.

## Live Hermes Collection

The collector is an optional read-only Hermes observer plugin adapter. It
captures observer-hook payloads as JSONL so the standalone CLI can score them.

Example plugin bootstrap:

```python
from flightrecorder.hermes_plugin import register as register_flight_recorder

def register(ctx):
    return register_flight_recorder(ctx)
```

Environment variables:

```bash
export HERMES_FLIGHT_RECORDER_OUTPUT_DIR=/secure/hermes-flight-recorder/events
export HERMES_FLIGHT_RECORDER_MAX_FIELD_CHARS=20000
```

The collector writes one JSONL file per session id. It never blocks tools,
never rewrites requests, and fails open if writing is impossible.

## Operational Checklist

- Store raw Hermes exports in a restricted directory.
- Run `flightrecorder run` with scenario policies that match your credential
  formats.
- Use `flightrecorder run --fail-on-score` for CI gates.
- Publish `report.html` and `scorecard.json`; avoid publishing raw traces.
- Run `flightrecorder audit --runs runs --fail-on-leak --forbid-text <secret>`
  before publishing generated artifacts.
- Keep failing `regression_scenario.json` files with the test suite.
- Run `python -m unittest discover` and `./demo.sh` in CI before release.

## Rollback

The standalone CLI does not mutate Hermes runtime state. To roll back a
deployment, uninstall the package or disable the optional collector plugin.
