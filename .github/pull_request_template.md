## Summary

<!-- Explain why this change is needed and the observable behavior it changes. -->

## Contract and compatibility impact

<!-- List affected CLI behavior, JSON schemas/artifacts, scenarios, plugins, or integrations. Write "None" when not applicable. -->

## Verification

<!-- List the exact commands run and their results. -->

- [ ] Focused tests cover behavior changes.
- [ ] `.venv/bin/python -m unittest discover` passes, or omitted checks are explained below.
- [ ] Schema, registry/manifest, validation, tests, and docs were updated together when an artifact contract changed.

## Safety and data handling

- [ ] No credentials, `.env` contents, raw production traces, or other sensitive artifacts are included.
- [ ] Network access, external side effects, destructive flags, live data, model training, and publishing were not used, or are documented below with explicit authorization.
- [ ] Generated `runs/`, `replay_runs/`, build outputs, caches, and virtual environments are not committed.

## Risks and omitted checks

<!-- Describe remaining risks, compatibility concerns, generated artifacts, and checks not run. -->
