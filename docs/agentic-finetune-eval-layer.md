# Agentic Fine-Tune Eval Layer

The Eval layer produces held-out, regression, red-team, harness-compatibility,
and external-adapter evidence for Governance. Eval artifacts are evidence, not
promotion decisions.

## Required Invariant

Cross-arm claims are valid only when every compared arm uses the same held-out
scenario id list. If baseline, trace-only, champion, or candidate scenario ids
differ, comparison artifacts must report `comparison_status: "not_comparable"`
and block pass-rate, score, regression, or improvement claims.

## Held-Out Summaries

`scripts/evaluate_hermes_heldout.py` writes:

- `evaluation_plan.json`
- `suite_summary.json`
- `evaluation_summary.json`

The eval summary includes scenario ids, scenario fingerprint, model metadata,
pass rate, average score, critical failures, failed rules, task-completion
metrics, cost and latency availability, artifact hashes, optional serving
profile identity, and a `governance_handoff` block.

## Cross-Arm Comparison

Use:

```bash
.venv/bin/python scripts/compare_agentic_finetune_results.py \
  --baseline experiments/qwen3_4b_flightrecorder/evaluations/baseline/suite_summary.json \
  --trace-only experiments/qwen3_4b_flightrecorder/evaluations/trace_only/suite_summary.json \
  --flightrecorder experiments/qwen3_4b_flightrecorder/evaluations/flightrecorder/suite_summary.json \
  --out experiments/qwen3_4b_flightrecorder/promotion_comparison.json \
  --report experiments/qwen3_4b_flightrecorder/PROMOTION_REPORT.md
```

Despite the historical filename, this is an eval comparison handoff. Governance
must still apply evidence, data, model, serving, safety, license, rollback, and
card gates before any promotion.

The comparison command also writes standalone repair artifacts beside `--out`
unless overridden:

- `eval_repair_work_items.json`
- `eval_curriculum_suggestions.json`

These files contain the same failed-check pressure in smaller downstream
handoff shapes. Scenario-mismatch items instruct the loop to rerun every arm on
an identical held-out scenario list rather than treating any metric delta as a
valid improvement.

## External Adapter Plans

External adapters are optional and fail closed by default. Use the adapter plan
script before wiring BFCL, Inspect AI, lm-evaluation-harness, or SWE-bench into
promotion evidence:

```bash
.venv/bin/python scripts/plan_external_eval_adapters.py \
  --scenario-manifest experiments/qwen3_4b_flightrecorder/heldout_scenarios.json \
  --model <served-model-or-model-args> \
  --base-url <openai-compatible-base-url> \
  --out experiments/qwen3_4b_flightrecorder/evaluations/external_adapters/external_eval_adapters.json
```

The artifact schema is `hfr.external_eval_adapters.v1`. It records each
adapter's domain, optional dependency import names, required inputs, suite
tags, readiness, blocking reasons, next actions, and a runner contract that
requires identical scenario manifests before any comparative claim.

Run with `--allow-installed` only when optional dependencies and required input
artifacts intentionally exist. A blocked adapter must stay out of promotion
evidence.

## Verification

Use:

```bash
.venv/bin/python -m unittest tests.test_agentic_eval_layer
.venv/bin/python -m flightrecorder validate \
  --suite-summary experiments/qwen3_4b_flightrecorder/evaluations/baseline/suite_summary.json \
  --suite-summary experiments/qwen3_4b_flightrecorder/evaluations/trace_only/suite_summary.json \
  --suite-summary experiments/qwen3_4b_flightrecorder/evaluations/flightrecorder/suite_summary.json \
  --strict
```
