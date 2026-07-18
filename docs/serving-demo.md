# Serving And Demo Layer

The serving/demo layer gives Eval a preflight artifact before a held-out suite
uses an OpenAI-compatible endpoint. It also gives humans a replayable report
that ties base-vs-candidate claims back to traces, scorecards, run digests, and
HTML reports.

## Endpoint Preflight

Use a repo-local uv virtualenv for real local serving so server processes do
not depend on the global Python installation:

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python torch transformers peft accelerate
```

Use the local Transformers shim for the MVP serving path:

```bash
.venv/bin/python scripts/serve_transformers_openai.py \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --adapter <adapter-dir> \
  --host 127.0.0.1 \
  --port 8000
```

In another shell, write the serving artifacts:

```bash
.venv/bin/python scripts/check_openai_serving.py \
  --engine transformers \
  --arm flightrecorder \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --adapter <adapter-dir> \
  --base-url http://127.0.0.1:8000/v1 \
  --out experiments/qwen3_4b_flightrecorder/serving/flightrecorder
```

The command writes:

- `serving_profile.json`: endpoint, engine profile, model identity, adapter
  identity, capability summary, and eval preflight readiness.
- `compatibility_report.json`: OpenAI core checks plus tool-call and structured
  output smoke results.
- `serving_check.json`: pass/fail summary and failed checks for automation.

For local CI or offline smoke verification, use a managed mock endpoint. The
script starts and shuts it down itself:

```bash
.venv/bin/python scripts/check_openai_serving.py \
  --mock-response "hfr serving smoke ok" \
  --require-tool-call \
  --require-structured-output \
  --model hfr-mock-model \
  --out experiments/qwen3_4b_flightrecorder/serving/mock_openai_check
```

## Managed Lifecycle

Use `scripts/run_managed_serving_eval.py` when an eval loop should own the
server lifecycle. The runner starts the server command, polls the endpoint with
`check_openai_serving.py` logic, writes profile/check artifacts, optionally
runs an Eval command with the generated profile, and always terminates the
server process.

Lightweight lifecycle smoke:

```bash
uv run python scripts/run_managed_serving_eval.py \
  --server-command "uv run python scripts/mock_openai_serving.py --port 18000 --model hfr-mock-model" \
  --base-url http://127.0.0.1:18000/v1 \
  --model hfr-mock-model \
  --out experiments/qwen3_4b_flightrecorder/serving/managed_mock_lifecycle \
  --require-tool-call \
  --require-structured-output
```

Managed local Transformers eval shape:

```bash
.venv/bin/python scripts/run_managed_serving_eval.py \
  --server-command ".venv/bin/python scripts/serve_transformers_openai.py --model Qwen/Qwen3-4B-Instruct-2507 --adapter <adapter-dir> --host 127.0.0.1 --port 8000" \
  --base-url http://127.0.0.1:8000/v1 \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --adapter <adapter-dir> \
  --arm flightrecorder \
  --out experiments/qwen3_4b_flightrecorder/serving/flightrecorder \
  --eval-command ".venv/bin/python scripts/evaluate_hermes_heldout.py --arm flightrecorder --model Qwen/Qwen3-4B-Instruct-2507 --base-url {base_url} --serving-profile {serving_profile} --out experiments/qwen3_4b_flightrecorder/evaluations/flightrecorder --force"
```

The lifecycle runner writes:

- `serving_lifecycle_run.json`: server command, PID, logs, preflight attempts,
  eval command result, and cleanup result.
- `server_stdout.txt` / `server_stderr.txt`: serving process logs.
- `eval_stdout.txt` / `eval_stderr.txt`: Eval command logs when an eval command
  is supplied.

Validate the lifecycle handoff before promoting it into Eval or demo evidence:

```bash
flightrecorder schemas --check experiments/qwen3_4b_flightrecorder/serving/flightrecorder/serving_lifecycle_run.json
```

## Real Runtime Preflight

Before launching a heavyweight local model server, write a dependency and
artifact preflight. This does not import `torch`, allocate devices, download
models, or start a server:

```bash
.venv/bin/python scripts/preflight_serving_runtime.py \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --runtime-python .venv/bin/python \
  --adapter trace_only=experiments/qwen3_4b_flightrecorder/adapters/qwen3_4b_local_trace_sft/trace_sft_adapter \
  --adapter flightrecorder=experiments/qwen3_4b_flightrecorder/adapters/qwen3_4b_local_fr_sft_dpo/fr_sft_dpo_adapter \
  --out experiments/qwen3_4b_flightrecorder/serving/real_runtime_preflight/serving_runtime_preflight.json \
  --report experiments/qwen3_4b_flightrecorder/serving/real_runtime_preflight/SERVING_RUNTIME_PREFLIGHT.md \
  --allow-blocked
```

The command writes `hfr.serving_runtime_preflight.v1` with model-cache status,
runtime dependency availability, local adapter file hashes, server-script
metadata, and ready-to-run managed lifecycle command references. Use it as the
gate before starting real baseline, trace-only, or Flight Recorder serving
processes.

## Endpoint Suite Guard

Before comparing arms, verify that every required profile is present, ready,
identity-matched, and compatible with tool calls or structured outputs when the
demo requires them:

```bash
.venv/bin/python scripts/verify_serving_profiles.py \
  --profile baseline=experiments/qwen3_4b_flightrecorder/serving/baseline/serving_profile.json \
  --profile trace_only=experiments/qwen3_4b_flightrecorder/serving/trace_only/serving_profile.json \
  --profile flightrecorder=experiments/qwen3_4b_flightrecorder/serving/flightrecorder/serving_profile.json \
  --lifecycle flightrecorder=experiments/qwen3_4b_flightrecorder/serving/flightrecorder/serving_lifecycle_run.json \
  --required-arm baseline \
  --required-arm trace_only \
  --required-arm flightrecorder \
  --expect-model baseline=Qwen/Qwen3-4B-Instruct-2507 \
  --expect-model trace_only=Qwen/Qwen3-4B-Instruct-2507 \
  --expect-model flightrecorder=Qwen/Qwen3-4B-Instruct-2507 \
  --require-structured-output \
  --strict-profile-arm \
  --out experiments/qwen3_4b_flightrecorder/serving/serving_endpoint_suite.json \
  --report experiments/qwen3_4b_flightrecorder/serving/SERVING_ENDPOINTS.md
```

The suite guard writes `hfr.serving_endpoint_suite.v1`, which Eval and humans
can use as the endpoint readiness matrix for baseline-vs-candidate demos.

## Eval Guard

Pass the profile into held-out evaluation so Eval refuses a wrong or unchecked
endpoint:

```bash
.venv/bin/python scripts/evaluate_hermes_heldout.py \
  --arm flightrecorder \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --base-url http://127.0.0.1:8000/v1 \
  --serving-profile experiments/qwen3_4b_flightrecorder/serving/flightrecorder/serving_profile.json \
  --out experiments/qwen3_4b_flightrecorder/evaluations/flightrecorder \
  --force
```

The evaluator checks that the profile is `eval_preflight.ready`, that the base
URL matches, and that the model identity matches either the requested base model
or an explicit `base+adapter` served model id.

## vLLM And SGLang Profiles

`scripts/check_openai_serving.py` already emits engine profile metadata for
`--engine vllm` and `--engine sglang`. Those profiles are readiness contracts:
they define expected OpenAI-compatible checks, adapter strategy, and launch
command templates. They do not install or launch those engines.

## Replayable Demo Report

After baseline, trace-only, and candidate eval summaries exist, build a human
inspection report:

```bash
.venv/bin/python scripts/build_serving_demo_report.py \
  --arm baseline=experiments/qwen3_4b_flightrecorder/evaluations/baseline/evaluation_summary.json \
  --arm trace_only=experiments/qwen3_4b_flightrecorder/evaluations/trace_only/evaluation_summary.json \
  --arm flightrecorder=experiments/qwen3_4b_flightrecorder/evaluations/flightrecorder/evaluation_summary.json \
  --endpoint-suite experiments/qwen3_4b_flightrecorder/serving/serving_endpoint_suite.json \
  --out experiments/qwen3_4b_flightrecorder/serving/demo_run.json \
  --report experiments/qwen3_4b_flightrecorder/serving/DEMO_REPORT.md
```

The report includes arm metrics, evidence-backed claims, and a scenario replay
index linking to trace, scorecard, run digest, and HTML report artifacts. When
`--endpoint-suite` is supplied, it also includes the endpoint readiness matrix
used to trust the compared arms.
