# Coding Agent Guide

This file is the operating contract for coding agents working in this
repository. Keep changes small, preserve unrelated work, and prove behavior
with the narrowest relevant offline checks before reporting completion.

## Project Map

- `flightrecorder/`: dependency-free Python package and CLI implementation.
- `flightrecorder/schemas/`: versioned JSON contracts shipped with the package.
- `tests/`: `unittest`-compatible regression and integration tests.
- `scenarios/` and `fixtures/`: deterministic evaluation inputs.
- `examples/`: policies, verifier configurations, and integration examples.
- `scripts/`: optional live-runtime, verifier, serving, and training utilities.
- `plugins/`: optional Hermes/OpenClaw collection integrations.
- `demo.sh`: offline evidence demo; it replaces the root `runs/` directory.
- `release_check.sh`: comprehensive release verification; it also replaces
  generated `runs/`, `replay_runs/`, build, and schema-check artifacts.

Check for a more specific `AGENTS.md` before editing a nested subtree. More
specific instructions override this file for files in their scope.

## Setup

Python 3.11 or newer is required. The core package has no third-party runtime
dependencies and no required environment variables.

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -e . --no-deps
.venv/bin/python -m flightrecorder --help
```

Optional YAML support can be installed with
`.venv/bin/python -m pip install -e '.[yaml]'`. See `.env.example` for optional
collector, live-runtime, and verifier configuration. Never put real secrets in
tracked files, command-line arguments, fixtures, logs, or issue reports.

## Development Workflow

1. Inspect the working tree with `git status --short` and do not overwrite,
   delete, or reformat unrelated user changes.
2. Reuse existing helpers and schema conventions. Avoid new dependencies unless
   the task explicitly requires one.
3. When changing a JSON artifact, update its schema, schema registry/manifest,
   validation path, tests, and documentation together.
4. Add a regression test for behavior changes. Tests use temporary directories
   for generated artifacts whenever practical.
5. Prefer `python -m flightrecorder` in checks so the local checkout is tested.

## Verification

Use the repository virtual environment when present:

```bash
.venv/bin/python -m unittest discover
.venv/bin/python -m compileall -q flightrecorder scripts tests
```

Run a focused test while iterating, for example:

```bash
.venv/bin/python -m unittest tests.test_validation
```

Before a release-affecting handoff, run `./release_check.sh` in a clean,
disposable worktree. The script is the CI entry point, but it deletes and
regenerates ignored output directories; do not run it where `runs/`,
`replay_runs/`, `build/`, or `dist/` contain data that must be preserved.

## Risky-Action Boundaries

Agents may inspect files, edit files inside this repository, and run offline
tests that write only to temporary or ignored project directories.

Explicit user approval is required before any action that:

- enables network or external-provider access, including `--allow-network`,
  live verifier/provider smoke tests, or non-local model endpoints;
- uses credentials or reads live user, mailbox, cloud, cluster, or SaaS data;
- sends, uploads, publishes, deploys, pushes to a hub, or changes external
  state, including `--push-to-hub`;
- starts paid or long-running GPU/model training or managed serving;
- records unredacted evidence with `--write-sensitive-trace`;
- passes `--force`, overwrites a non-empty output directory, or deletes data
  outside a disposable temporary directory; or
- installs/enables plugins in a real Hermes, OpenClaw, Coven, or host-runtime
  configuration.

Never publish production traces or credentials, commit `.env`, weaken
redaction/security checks to make a test pass, run destructive commands outside
the repository, or claim an external side effect without independently
verifiable evidence. Treat trace, state-snapshot, model, and training artifacts
as sensitive until reviewed. Follow `SECURITY.md` for disclosure and artifact
handling.

## Pull Requests

Keep pull requests focused. Summarize behavior and contract changes, list the
exact verification commands run, identify generated or sensitive artifacts,
and call out any checks not run. Do not commit normal contents of `runs/`,
`replay_runs/`, local virtual environments, caches, credentials, or raw live
traces.
