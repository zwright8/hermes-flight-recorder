# Serving And Demo Layer

Goal 6 starts with a dependency-free preflight for OpenAI-compatible endpoints
and a replayable demo report over held-out evaluation artifacts.

## Endpoint Preflight

For local or CI verification without a model server, use the managed mock:

```bash
python3 scripts/check_openai_serving.py \
  --mock-response "hfr serving smoke ok" \
  --require-streaming \
  --require-tool-call \
  --require-structured-output \
  --model hfr-mock-model \
  --out experiments/qwen3_4b_flightrecorder/serving/mock_openai_check
```

The command writes:

- `serving_profile.json`: endpoint, model identity, adapter identity,
  capability summary, and Eval readiness.
- `compatibility_report.json`: OpenAI core, streaming, tool-call, and
  structured-output smoke results.
- `serving_check.json`: pass/fail summary and failed checks for automation.

For a real endpoint, pass `--base-url http://127.0.0.1:<port>/v1` instead of
`--mock-response`.

## Managed Lifecycle

Use the lifecycle wrapper when the serving process should be owned by the
verification run. It starts the server, polls readiness, runs the endpoint
preflight, captures logs, writes lifecycle metadata, and tears the process down.

```bash
python3 scripts/manage_openai_serving.py \
  --profile mock \
  --host 127.0.0.1 \
  --port 18080 \
  --base-url http://127.0.0.1:18080/v1 \
  --model hfr-managed-mock \
  --served-model-name hfr-managed-mock \
  --adapter-load-strategy auto \
  --require-streaming \
  --require-tool-call \
  --require-structured-output \
  --out experiments/qwen3_4b_flightrecorder/serving/managed_mock_lifecycle
```

The command writes `serving_lifecycle.json`, `server.stdout.log`,
`server.stderr.log`, and a nested `preflight/` directory containing the same
serving profile, compatibility report, and endpoint check emitted by
`check_openai_serving.py`. Semantic validation recomputes lifecycle readiness
from the readiness probe, smoke check, teardown, and required preflight artifact
links so forged ready lifecycle records cannot stand on schema shape alone.
Persisted serving artifacts use stable adapter IDs instead of absolute adapter
paths, redact non-default working directories, remove URL userinfo/query
credentials, and scrub known API-key, command-argument, environment-value, and
log-text secrets. Execution still receives the original inputs; sanitization is
applied to the public receipts and referenced log files.

For vLLM, use the built-in launch profile or pass an explicit command:

```bash
python3 scripts/manage_openai_serving.py \
  --profile vllm \
  --host 127.0.0.1 \
  --port 8000 \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --served-model-name qwen3-flightrecorder \
  --require-streaming \
  --require-tool-call \
  --require-structured-output \
  --out experiments/qwen3_4b_flightrecorder/serving/vllm_lifecycle
```

For SGLang, use:

```bash
python3 scripts/manage_openai_serving.py \
  --profile sglang \
  --host 127.0.0.1 \
  --port 30000 \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --served-model-name qwen3-flightrecorder \
  --require-streaming \
  --require-tool-call \
  --require-structured-output \
  --out experiments/qwen3_4b_flightrecorder/serving/sglang_lifecycle
```

If upstream launch flags need local tuning, pass `--command "<server command>"`
and keep `--base-url` pointed at the resulting OpenAI-compatible endpoint.
When `--adapter` is supplied, the lifecycle artifact records
`adapter_strategy` with the requested and resolved load strategy. For built-in
vLLM/SGLang profiles this is metadata only; pass engine-specific adapter flags
with `--extra-engine-arg` or use `--command`.

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
be replayed from a fresh checkout. The JSON and Markdown outputs also include
base-vs-candidate comparison rows with metric deltas and per-scenario outcomes
such as `candidate_repaired` or `candidate_regressed`. Semantic validation
recomputes each arm's aggregate pass/fail metrics from those scenario rows, so
forged or stale demo summaries cannot claim better serving outcomes than the
linked evidence supports.

## Eval Preflight Handoff

Require the serving preflight before summarizing held-out eval results:

```bash
flightrecorder eval-summary \
  --suite-summary candidate=<candidate-suite-summary> \
  --serving-check candidate=<serving-check-json> \
  --require-serving-preflight \
  --out <eval-summary-json>
```

The `LABEL` before `=` must match a suite-summary label. Missing, unmatched, or
blocked serving checks make the eval summary a valid blocked artifact instead of
letting eval or governance loops assume the endpoint was ready.
Semantic validation recomputes the attached serving-preflight count from the
arm entries and requires blocked arm preflights to carry
`serving_preflight_blocked`, so compact summaries cannot hide stale or failed
serving readiness.

## Verification

Validate generated artifacts with the bundled schema registry:

```bash
python3 -m flightrecorder schemas --check experiments/qwen3_4b_flightrecorder/serving/mock_openai_check/serving_profile.json
python3 -m flightrecorder schemas --check experiments/qwen3_4b_flightrecorder/serving/mock_openai_check/compatibility_report.json
python3 -m flightrecorder schemas --check experiments/qwen3_4b_flightrecorder/serving/mock_openai_check/serving_check.json
python3 -m flightrecorder schemas --check experiments/qwen3_4b_flightrecorder/serving/managed_mock_lifecycle/serving_lifecycle.json
python3 -m flightrecorder schemas --check experiments/qwen3_4b_flightrecorder/serving/demo_run.json
```

Run semantic validation before handing serving artifacts to Eval or Governance:

```bash
python3 -m flightrecorder validate \
  --serving-profile experiments/qwen3_4b_flightrecorder/serving/mock_openai_check/serving_profile.json \
  --serving-compatibility-report experiments/qwen3_4b_flightrecorder/serving/mock_openai_check/compatibility_report.json \
  --serving-endpoint-check experiments/qwen3_4b_flightrecorder/serving/mock_openai_check/serving_check.json \
  --serving-lifecycle experiments/qwen3_4b_flightrecorder/serving/managed_mock_lifecycle/serving_lifecycle.json \
  --serving-demo-run experiments/qwen3_4b_flightrecorder/serving/demo_run.json \
  --strict
```

Include the lifecycle receipt in evidence bundles when serving readiness is part
of the handoff:

```bash
python3 -m flightrecorder evidence-bundle \
  --serving-lifecycle experiments/qwen3_4b_flightrecorder/serving/managed_mock_lifecycle/serving_lifecycle.json \
  --out experiments/qwen3_4b_flightrecorder/evidence_bundle.json
```
