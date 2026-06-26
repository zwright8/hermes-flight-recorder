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

The demo runs `flightrecorder run-suite` and generates `runs/index.html` with
two passing reports, three failing adversarial reports, `runs/suite_summary.json`,
single-run and suite compare reports, and `runs/training_export/` training
artifacts. It also writes `runs/validation.json` to prove the generated
contracts are internally consistent. No API keys or network are required.

## Live Hermes Collection

The collector is an optional read-only Hermes observer plugin adapter. It
captures observer-hook payloads as JSONL so the standalone CLI can score them.

Example plugin bootstrap:

```python
from flightrecorder.hermes_plugin import register as register_flight_recorder

def register(ctx):
    return register_flight_recorder(ctx)
```

Generate the same wrapper from the CLI:

```bash
flightrecorder observer-template --out flight_recorder_plugin.py
```

Environment variables:

```bash
export HERMES_FLIGHT_RECORDER_OUTPUT_DIR=/secure/hermes-flight-recorder/events
export HERMES_FLIGHT_RECORDER_MAX_FIELD_CHARS=20000
```

The collector writes one JSONL file per session id. It never blocks tools,
never rewrites requests, and fails open if writing is impossible.

## Live Runtime Smoke

Use the live smoke when a Hermes Agent source checkout is available:

```bash
python scripts/live_hermes_smoke.py \
  --hermes-root ../upstream-hermes-agent \
  --out live_smoke_artifacts/latest
```

The smoke runs a real `uv run hermes chat` session against a local mock model
endpoint and an isolated temporary `HERMES_HOME`. It proves the optional
observer plugin can be loaded by Hermes, receives observer hooks, and produces
`live_observer.jsonl`, `normalized_trace.json`, `scorecard.json`, and
`report.html` without external API keys or network.

## Operational Checklist

- Store raw Hermes exports in a restricted directory.
- Run `flightrecorder check-scenarios --scenarios <dir> --require-traces
  --strict` before publishing or running a custom scenario suite.
- Run `flightrecorder run-suite --scenarios <dir> --out runs --validate
  --strict` as the default CI evidence-bundle command.
- Add `--junit`, `--markdown`, and `--export-rl` when CI should publish native
  test reports, job summaries, and downstream training/failure-mode artifacts.
- Add `--fail-on-failed` when any failed scenario should fail the CI job.
- Use `flightrecorder run --fail-on-score` for targeted single-scenario gates.
- Use `flightrecorder compare --fail-on-regression` to gate candidate runs
  against a baseline scorecard.
- Use `flightrecorder compare-suite --fail-on-regression` to gate an entire
  candidate run directory against a baseline suite.
- Commit a suite gate policy JSON file and use `flightrecorder gate-suite
  --suite-summary runs/suite_summary.json --policy <policy.json>` for absolute
  CI acceptance gates. CLI threshold flags can tighten scalar policy values or
  add forbidden failure classes per job.
- Use `flightrecorder export-rl --runs runs --out runs/training_export` when
  downstream SFT, preference, reward-model, curriculum, or RL jobs need
  deterministic episode/reward/preference/failure-mode artifacts.
- Use `flightrecorder validate --runs runs --training-export runs/training_export
  --suite-summary runs/suite_summary.json --strict` before publishing artifacts
  or using them downstream.
- Publish `report.html` and `scorecard.json`; avoid publishing raw traces.
- Run `flightrecorder audit --runs runs --fail-on-leak --forbid-text <secret>`
  before publishing generated artifacts.
- Keep failing `regression_scenario.json` files with the test suite.
- Run `python -m unittest discover` and `./demo.sh` in CI before release.
- Run `python scripts/live_hermes_smoke.py --hermes-root <checkout>` before
  deploying the optional observer plugin into a real Hermes environment.

## Rollback

The standalone CLI does not mutate Hermes runtime state. To roll back a
deployment, uninstall the package or disable the optional collector plugin.
