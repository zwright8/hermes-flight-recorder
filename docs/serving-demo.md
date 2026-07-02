# Serving And Demo Layer

Goal 6 starts with a dependency-free preflight for OpenAI-compatible endpoints
and a replayable demo report over held-out evaluation artifacts.

## Endpoint Preflight

For local or CI verification without a model server, use the managed mock:

```bash
python3 scripts/check_openai_serving.py \
  --mock-response "hfr serving smoke ok" \
  --require-tool-call \
  --require-structured-output \
  --model hfr-mock-model \
  --out experiments/qwen3_4b_flightrecorder/serving/mock_openai_check
```

The command writes:

- `serving_profile.json`: endpoint, model identity, adapter identity,
  capability summary, and Eval readiness.
- `compatibility_report.json`: OpenAI core, tool-call, and structured-output
  smoke results.
- `serving_check.json`: pass/fail summary and failed checks for automation.

For a real endpoint, pass `--base-url http://127.0.0.1:<port>/v1` instead of
`--mock-response`.

## Replayable Demo Report

After at least two held-out eval summaries or suite summaries exist, build a
human-readable replay report:

```bash
python3 scripts/build_serving_demo_report.py \
  --arm baseline=<baseline-evaluation-summary> \
  --arm flightrecorder=<candidate-evaluation-summary> \
  --out experiments/qwen3_4b_flightrecorder/serving/demo_run.json \
  --report experiments/qwen3_4b_flightrecorder/serving/DEMO_REPORT.md
```

The report links each behavior claim to evaluation summaries, suite summaries,
traces, scorecards, run digests, and HTML reports. Local artifact links are
written relative to the Markdown report directory so committed demo bundles can
be replayed from a fresh checkout.

## Verification

Validate generated artifacts with the bundled schema registry:

```bash
python3 -m flightrecorder schemas --check experiments/qwen3_4b_flightrecorder/serving/mock_openai_check/serving_profile.json
python3 -m flightrecorder schemas --check experiments/qwen3_4b_flightrecorder/serving/demo_run.json
```
