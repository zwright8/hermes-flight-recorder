# Hermes Autonomy Flight Recorder

Hermes Autonomy Flight Recorder is a standalone adversarial eval harness for
Hermes Agent runs. It converts existing run artifacts into a normalized trace,
scores the trace against explicit autonomy boundaries, and renders a static
HTML report that a maintainer can inspect quickly.

This is accountability and regression infrastructure. It is not a sandbox, not
prompt-injection prevention, and not a security boundary. Real containment
still belongs at the OS, process, and network layers.

## Why It Exists

Hermes already has skills, plugins, MCP, memory, subagents, cron/goals,
observer hooks, middleware, and trajectory export. The missing layer is a way
to prove what an autonomous run did, whether it stayed within stated
boundaries, and whether a failure can become a rerunnable regression.

## Quick Demo

The demo is offline and deterministic. It needs no API key and no network.

```bash
git clone https://github.com/zwright8/hermes-flight-recorder.git
cd hermes-flight-recorder
python -m unittest discover
./demo.sh
open runs/index.html
```

Expected demo output:

- One passing prompt-injection report.
- Three failing reports:
  - prompt-injection obedience and secret exposure,
  - unsupported subagent side-effect claim,
  - runaway delegation and budget violation.

## Install

```bash
python -m pip install . --no-deps
flightrecorder --help
```

Editable development:

```bash
python -m pip install -e . --no-deps
```

## CLI

```bash
flightrecorder run \
  --scenario scenarios/prompt_injection_good.json \
  --out runs/prompt_injection_good

flightrecorder normalize \
  --trace fixtures/prompt_injection_good.trajectory.jsonl \
  --format auto \
  --out runs/normalized_trace.json

flightrecorder score \
  --scenario scenarios/prompt_injection_good.json \
  --trace runs/normalized_trace.json \
  --out runs/scorecard.json

flightrecorder report \
  --scenario scenarios/prompt_injection_good.json \
  --trace runs/normalized_trace.json \
  --score runs/scorecard.json \
  --out runs/report.html

flightrecorder audit \
  --runs runs \
  --forbid-text hfr_fixture_secret_value_123 \
  --fail-on-leak
```

## Inputs

Canonical scenarios are JSON. YAML is supported only when `PyYAML` is already
installed.

Supported trace inputs:

- Hermes trajectory JSONL from `agent.save_trajectories` or batch runner output.
- Observer hook JSONL with events such as `pre_tool_call`, `post_tool_call`,
  `post_llm_call`, `subagent_start`, and `subagent_stop`.
- Minimal ATOF JSONL and ATIF JSON for compatibility demos.

## Outputs

Each run directory contains:

- `normalized_trace.json`: canonical `hfr.trace.v1` trace, redacted by default.
- `scorecard.json`: deterministic pass/fail rule results.
- `report.html`: self-contained static flight-recorder report.
- `regression_scenario.json`: emitted only for failing runs.

Raw evidence is intentionally not written by default. Use
`--write-sensitive-trace` only in a restricted directory when you need
`raw_trace.sensitive.json` for debugging.

For CI gates, add `--fail-on-score` to `flightrecorder run` so failing
scenarios return a nonzero exit code after writing their artifacts.
Use `flightrecorder audit --fail-on-leak` to scan generated run artifacts for
literal strings that must not ship.

## Scoring Rules

- `Forbidden Actions`: forbidden tools, commands, URLs, or final-answer leaks.
- `Secret Exposure`: configured secret-like patterns in outputs or answers.
- `Budget And Delegation`: tool-call, API-call, subagent, and depth limits.
- `Required Evidence`: checks that claims have matching event evidence.
- `Final Answer`: simple contains and not-contains assertions.

Scores start at 100. Critical rule failures force the run to fail even if the
numeric score remains above the threshold.

## Architecture

```text
scenario.json + trace artifact
          |
          v
  trace adapter -> raw trace in memory
          |
          v
  deterministic scorers -> scorecard.json
          |
          v
  redactor -> normalized_trace.json
          |
          v
  static report renderer -> report.html
          |
          v
  failed run -> regression_scenario.json
```

## Hackathon Pitch

Hermes can already act. This project helps Hermes prove. The Flight Recorder
turns autonomous runs into inspectable, scoreable, rerunnable evidence so users
and maintainers can catch prompt-injection obedience, unsupported subagent
claims, missing completion evidence, and budget runaway before those failures
become invisible.

For the maintainer-facing contribution framing, demo evidence, and upstream PR
draft, see [HERMES_CONTRIBUTION.md](HERMES_CONTRIBUTION.md).

## Limitations

- The scorer is deterministic and intentionally conservative.
- The MVP does not execute Hermes or mutate Hermes runtime behavior.
- The optional plugin path should remain read-only and fail-open.
- Regex policies are only as good as the scenarios that define them.

## Live Observer Collection

`flightrecorder.hermes_plugin` exposes a read-only Hermes observer adapter that
can be wrapped by a Hermes plugin. It writes observer JSONL files configured by:

```bash
export HERMES_FLIGHT_RECORDER_OUTPUT_DIR=/secure/hermes-flight-recorder/events
export HERMES_FLIGHT_RECORDER_MAX_FIELD_CHARS=20000
```

The collector never blocks tools, never rewrites model or tool requests, and
fails open if writing is unavailable.

## Release Check

```bash
./release_check.sh
```

This runs unit tests, bytecode compilation, the offline demo, report redaction
checks through `flightrecorder audit`, CI failure-mode checks, and a package
install smoke.
