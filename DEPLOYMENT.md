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
two passing reports, three failing adversarial reports, per-run
`artifact_lineage.json` provenance manifests, `runs/suite_summary.json`,
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
`report.html`, plus `artifact_lineage.json`, without external API keys or
network.

## Operational Checklist

- Store raw Hermes exports in a restricted directory.
- Run `flightrecorder check-scenarios --scenarios <dir> --require-traces
  --strict` before publishing or running a custom scenario suite.
- Use `flightrecorder draft-scenario --trace <trace> --out <scenario.json>` to
  bootstrap a scenario from a known-good run, then review and tighten the
  generated assertions before adding it to a release suite.
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
  add forbidden failure classes per job. Use task-family gates in the policy
  when CI must protect specific behavior classes, such as email completion or
  prompt-injection resistance, from being hidden by aggregate metrics.
- Use `flightrecorder export-rl --runs runs --out runs/training_export` when
  downstream SFT, preference, reward-model, curriculum, or RL jobs need
  deterministic episode/reward/step-reward/preference/trainer-view/failure-mode
  artifacts plus dataset-level quality metrics and a dataset card.
- Use `flightrecorder export-review --runs runs --out runs/review_queue` when
  maintainers need a human-curation queue before deterministic score labels are
  trusted as training signal.
- Use `flightrecorder validate --runs runs --training-export runs/training_export
  --review-export runs/review_queue --suite-summary runs/suite_summary.json
  --strict` before publishing artifacts or using them downstream.
- Use `flightrecorder gate-export --training-export runs/training_export
  --policy <policy.json>` when CI must block training jobs unless an export has
  enough examples, preferences, attribution, task-family coverage, and no
  forbidden quality flags.
- Publish `report.html`, `scorecard.json`, and `artifact_lineage.json`; avoid
  publishing raw traces.
- Run `flightrecorder audit --runs runs --fail-on-leak --forbid-text <secret>`
  before publishing generated artifacts.
- Keep failing `regression_scenario.json` files with the test suite.
- Run `python -m unittest discover` and `./demo.sh` in CI before release.
- Run `python scripts/live_hermes_smoke.py --hermes-root <checkout>` before
  deploying the optional observer plugin into a real Hermes environment.

## Rollback

The standalone CLI does not mutate Hermes runtime state. To roll back a
deployment, uninstall the package or disable the optional collector plugin.
