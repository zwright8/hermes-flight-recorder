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
two passing reports, four failing adversarial reports, per-run
`artifact_lineage.json` provenance manifests, `runs/suite_summary.json`,
single-run and suite compare reports, `runs/scenario_quality.json`,
`runs/evidence_coverage.json`, `runs/evidence_bundle.json`, and
`runs/training_export/` training artifacts. It also writes
`runs/validation.json` to prove the generated contracts are internally
consistent. No API keys or network are required.

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
`live_scenario.json`, `live_observer.jsonl`, `normalized_trace.json`,
`scorecard.json`, `task_completion.json`, `report.html`, and
`artifact_lineage.json`, without external API keys or network.

## Operational Checklist

- Store raw Hermes exports in a restricted directory.
- Run `flightrecorder check-scenarios --scenarios <dir> --require-traces
  --strict` before publishing or running a custom scenario suite.
- Run `flightrecorder scenario-quality --scenarios <dir> --require-traces
  --min-average-score 80 --min-observable-rate 0.8 --max-weak-scenarios 0`
  when CI should reject shallow scenario contracts before they become
  benchmark or training signal.
- Use `flightrecorder draft-scenario --trace <trace> --out <scenario.json>` to
  bootstrap a scenario from a known-good run, then review and tighten the
  generated assertions before adding it to a release suite.
- Run `flightrecorder run-suite --scenarios <dir> --out runs --validate
  --strict --evidence-handoff` as the default CI evidence-bundle command. It
  creates the suite summary, per-run artifacts, scenario-quality summary,
  evidence-coverage summary, trace-observability summary, repair queue,
  validation summary, and top-level evidence bundle in one pass.
- Inspect `artifact_lineage.json.replay.self_contained` before relying on a
  rerun command. Use `flightrecorder replay` with
  `--lineage <run>/artifact_lineage.json --out <fresh-run>` to regenerate
  evidence from a self-contained lineage contract after input-hash verification.
  Use `flightrecorder replay-bundle` before publishing or moving evidence
  packages so scenario, trace, and state inputs travel with the lineage contract.
  Validate those packages with `flightrecorder validate` and
  `--replay-bundle <bundle-dir> --strict`. Shared artifacts redact paths by
  default; use `--preserve-paths` only in private CI when exact source paths are
  required.
- Add `--junit`, `--markdown`, and `--export-rl` when CI should publish native
  test reports, job summaries, and downstream training/failure-mode artifacts.
- Add repeated `--metadata key=value` flags to label the evaluated agent, model,
  prompt revision, skill revision, tool policy, or deployment candidate. The
  values are written to `suite_summary.json` and training-export artifacts.
- Add `--fail-on-failed` when any failed scenario should fail the CI job.
- Use `flightrecorder run --fail-on-score` for targeted single-scenario gates.
- Use `flightrecorder compare --fail-on-regression` to gate candidate runs
  against a baseline scorecard.
- Use `flightrecorder compare-suite --fail-on-regression` to gate an entire
  candidate run directory against a baseline suite. If both directories include
  `suite_summary.json` metadata, the comparison JSON and HTML show the
  baseline/candidate config identity side by side. Use aggregate failed-rule and
  critical-failure deltas to prioritize which failure classes need repair.
  Add `--fail-on-contract-drift --fail-on-unverified-contracts` when CI must
  prove paired scenarios used the same scenario contract. Keep the default
  `--contract-scope scenario` for live baseline/candidate behavior comparisons;
  use `--contract-scope scenario-and-trace` only for strict fixture replay where
  the source trace must also match.
- Use `flightrecorder trend-suite --suite-summary <old> --suite-summary <new>
  --out <trend.json>` when CI stores multiple suite summaries and maintainers
  need to review progress across an improvement run, not just one comparison.
- For task-oriented agents, inspect each run's `task_completion.json` or the
  training export's `dataset_metrics.task_completion` block before promoting a
  candidate. A high final-answer score is weaker evidence than a `complete`
  verdict backed by required tool-result, action-sequence, and event-count
  checks. When a scenario includes `state.path`, also review `state_snapshot.json`
  and the `source_state_snapshot` lineage hash; Flight Recorder verifies the
  supplied snapshot deterministically, but the connector or collector that
  produced that snapshot remains part of the trust boundary.
  Validate the resulting trend with `flightrecorder validate --suite-trend
  <trend.json> --strict` before treating it as release evidence.
- Use `flightrecorder evidence-coverage --runs runs --out
  runs/evidence_coverage.json --min-failed-rule-evidence-rate 1.0
  --max-failed-rules-without-evidence 0` when CI must prove failed scorecard
  judgments have structured evidence refs before those failures feed review,
  regression, or training loops.
- Use `flightrecorder evidence-bundle --runs runs --suite-summary
  runs/suite_summary.json --scenario-quality runs/scenario_quality.json
  --evidence-coverage runs/evidence_coverage.json --validation
  runs/validation.json --training-export runs/training_export --out
  runs/evidence_bundle.json` when CI should publish one readiness manifest over
  the generated evidence package. Route automation from
  `decision.recommendation`, route repair tickets or curricula from
  `decision.next_actions`, route rule-level work from `repair_queue.json`, and
  remember it summarizes the included gates and does not replace policy review.
- Use `flightrecorder validate --state-snapshot <snapshot.json> --strict` for
  `capture-state` outputs before they become required-state evidence or
  downstream training signal. The validator checks the captured schema and
  recomputes file hashes when source paths are still available.
- Commit a suite gate policy JSON file and use `flightrecorder gate-suite
  --suite-summary runs/suite_summary.json --policy <policy.json>` for absolute
  CI acceptance gates. CLI threshold flags can tighten scalar policy values or
  add forbidden failure classes per job. Use task-family gates in the policy
  when CI must protect specific behavior classes, such as email completion or
  prompt-injection resistance, from being hidden by aggregate metrics.
- Use `flightrecorder export-rl --runs runs --out runs/training_export` when
  downstream SFT, preference, reward-model, curriculum, or RL jobs need
  deterministic episode/reward/step-reward/preference/trainer-view/failure-mode
  artifacts plus dataset-level quality metrics, source-fingerprint coverage, and
  a dataset card.
- Use `flightrecorder export-compare-rl --baseline <old-runs> --candidate
  <new-runs> --out runs/compare_rl_export` when downstream jobs need
  baseline/candidate preference rows that preserve improvement or regression
  direction.
- Use `flightrecorder gate-compare-export --compare-export
  runs/compare_rl_export --policy <policy.json>` when CI must block comparison
  preference handoffs unless they contain enough candidate wins, required
  scenario coverage, expected task-completion improvements, expected rule fixes,
  no forbidden score or task-completion regression signal, and optionally no
  drifted or unverified comparison contracts via `--max-contract-drifts 0
  --max-unverified-contracts 0`. Use compare-policy `task_family_gates` when
  CI must protect specific workflow families from being hidden by aggregate
  candidate-win counts.
- Use `flightrecorder export-review --runs runs --out runs/review_queue` when
  maintainers need a human-curation queue before deterministic score labels are
  trusted as training signal.
- Use `flightrecorder apply-review --review-export runs/review_queue --labels
  <completed-labels.jsonl> --out runs/reviewed_export` to turn completed human
  labels into reviewed SFT, reward-model, preference, and DPO views.
- Use `flightrecorder review-calibration --reviewed-export
  runs/reviewed_export --out runs/review_calibration.json
  --min-agreement-rate 0.9 --max-false-positives 0` when CI should measure
  scorecard/human agreement before labels feed training or release decisions.
- Use `flightrecorder validate --runs runs --training-export runs/training_export
  --review-export runs/review_queue --reviewed-export runs/reviewed_export
  --compare-export runs/compare_rl_export --evidence-coverage
  runs/evidence_coverage.json --evidence-bundle runs/evidence_bundle.json
  --replay-bundle replay_bundles/prompt_injection_good --review-calibration
  runs/review_calibration.json --scenario-quality runs/scenario_quality.json
  --suite-summary runs/suite_summary.json --suite-trend runs/suite_trend.json
  --strict` before publishing artifacts or
  using them downstream.
- Use `flightrecorder gate-export --training-export runs/training_export
  --policy <policy.json>` when CI must block training jobs unless an export has
  enough examples, preferences, attribution, task-family coverage, complete
  task evidence via `--min-task-completion-complete` and
  `--min-task-completion-check-pass-rate`, complete source-fingerprint coverage
  via `--min-source-fingerprint-rate 1.0 --max-unverified-source-fingerprints
  0`, and no forbidden quality flags.
- Use `flightrecorder gate-reviewed --reviewed-export runs/reviewed_export
  --policy <policy.json>` when trainer jobs should wait for enough completed
  human labels, accepted and negative examples, reviewed trainer views,
  task-family coverage, and zero unresolved review labels.
- Publish `report.html`, `scorecard.json`, `artifact_lineage.json`, and
  `evidence_bundle.json`; avoid publishing raw traces.
- Run `flightrecorder audit --runs runs --fail-on-leak --forbid-text <secret>`
  before publishing generated artifacts.
- Keep failing `regression_scenario.json` files with the test suite.
- Run `python -m unittest discover` and `./demo.sh` in CI before release.
- Run `python scripts/live_hermes_smoke.py --hermes-root <checkout>` before
  deploying the optional observer plugin into a real Hermes environment.

## Rollback

The standalone CLI does not mutate Hermes runtime state. To roll back a
deployment, uninstall the package or disable the optional collector plugin.
