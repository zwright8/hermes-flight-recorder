# Agentic Finetune Eval Layer

Goal 7 eval outputs must be directly consumable by Governance without turning
raw comparison movement into promotion evidence by accident.

## Held-Out Scenario Invariant

Cross-arm claims are valid only when distinct evaluated arms provide the same
held-out scenario IDs and the same replayed scenario-content SHA-256 values. A
candidate win, task-completion improvement, or rule fix from a comparison export
remains raw movement until both the source-identity and content-identity checks pass.

`flightrecorder eval-summary` enforces that boundary:

```bash
flightrecorder eval-summary \
  --suite-summary baseline=runs/baseline/suite_summary.json \
  --suite-summary candidate=runs/candidate/suite_summary.json \
  --serving-check candidate=runs/serving_check.json \
  --require-serving-preflight \
  --compare-export candidate=runs/compare_rl_export \
  --compare-gate candidate=runs/compare_gate.json \
  --out runs/eval_summary.json \
  --markdown-out runs/eval_summary.md
```

The summary reports:

- per-arm suite status and scenario IDs,
- whether held-out scenario IDs and replayed content fingerprints are identical,
- raw compare-export movement,
- governance claims with candidate improvements suppressed when claims are not
  allowed,
- per-arm operational metrics for cost, latency, token usage, and
  task-completion status when suite summaries provide them,
- optional per-arm serving preflight readiness from `serving_check.json`,
- compare-gate failures,
- repair/curriculum work items for candidate regressions, failed rules,
  critical failures, gate failures, and adapter blockers,
- external adapter plan, execution-result, and coverage blockers when external
  adapters are included.

Use the Markdown report for human review. The JSON artifact remains the
contract; the report mirrors raw movement and approved claims in separate
columns so reviewers do not treat suppressed candidate wins as promotion
evidence.

## Held-Out Manifests

Use suite summaries to build the canonical scenario manifest that downstream
evals must share:

```bash
flightrecorder heldout-manifest \
  --suite-summary baseline=runs/baseline/suite_summary.json \
  --suite-summary candidate=runs/candidate/suite_summary.json \
  --out runs/heldout_scenarios.json
```

A single suite source can seed external adapter planning. Cross-arm claims are
allowed only when two or more distinctly labeled, distinctly sourced suite
summaries prove the exact same scenario IDs and scenario-file SHA-256 values.
The suite-summary files must also have distinct content fingerprints, so a copy
or hard link of one arm cannot impersonate an independent arm. Missing hashes,
stale hashes, duplicate arm sources, and mismatched content all produce honest
blocked artifacts. External adapter planning replays the held-out manifest
semantically instead of trusting its stored `ready` field.

Migration note: historical `hfr.heldout_scenario_manifest.v1` files whose source
rows omit `scenario_fingerprints` must be regenerated before they can be used as
readiness evidence. Blocked mismatch rows now include explicit fingerprint
differences so reviewers can distinguish ID-set drift from content drift.

## Red-Team Suite Manifests

Public-safe eval suite manifests live under `eval_suites/` and use
`hfr.eval_suite_manifest.v1`. Each manifest names an explicit `scenario_ids`
list to select from the normal scenario root:

```bash
flightrecorder run-suite \
  --scenarios scenarios \
  --suite-manifest eval_suites/red_team_prompt_injection.json \
  --out runs/red_team_prompt_injection
```

Use the same manifest and scenario IDs for every evaluated arm before making
cross-arm held-out claims. Suite manifests are selectors, not evidence that
different arms covered identical held-out work.

Validate manifests before handing them to automation:

```bash
flightrecorder validate --eval-suite-manifest eval_suites/red_team_prompt_injection.json --strict
flightrecorder schemas --check eval_suites/red_team_prompt_injection.json
```

## External Adapter Plans And Imported Results

An external adapter plan is a pre-execution handoff, not benchmark evidence.
Plan readiness never enables external eval claims. The dry-run receipt records
whether that handoff was prepared without side effects, but it also cannot
prove that a runner started or completed a benchmark.

```bash
flightrecorder external-eval-plan \
  --adapter local_mock \
  --scenario-manifest runs/heldout_scenarios.json \
  --model-endpoint local/candidate \
  --model local/candidate \
  --allow-installed \
  --out runs/external_eval_plan.json

flightrecorder external-eval-receipt \
  --plan runs/external_eval_plan.json \
  --out runs/external_eval_receipt.json
```

The supported adapter IDs are `bfcl`, `inspect_ai`, `lm_eval_harness`,
`local_mock`, and `swe_bench`. Plans fail closed by default: even when optional
dependencies are installed, an adapter is not ready unless `--allow-installed`
is present and all required inputs for that adapter are supplied. The
`local_mock` adapter is built in for deterministic offline dry-run receipts; it
does not start live benchmarks, call providers, download models, record
credentials, incur cloud spend, or update weights. The plan records dependency
status, held-out scenario manifest SHA-256 and byte size, required inputs, and
blocking reasons. Scenario manifests are replayable only when the plan can point
to them with a safe path relative to the plan output; absolute or traversal refs
are redacted and treated as missing, even with `--preserve-paths`. Validation
resolves the manifest reference from the plan file location before trusting
those fingerprints.
Each adapter row also includes an `adapter_contract` that attests to
plan-and-receipt-only dry-run transport, disabled live benchmark support, no
provider API calls, no model downloads, no credential values, and the need for
separate external runner evidence before any benchmark claim.
The committed examples in `examples/external_eval/` cover all five adapter
contracts and remain schema- and validation-checkable while blocked.
That directory also carries the held-out scenario, its input fixtures, and the
passing run evidence referenced by `heldout_suite_summary.json`. Every source
and run reference is relative to `examples/external_eval/`, so the example
replays and validates from a clean clone without relying on ignored top-level
`runs/` output.

Receipts are separate from plans. They archive dry-run readiness, or a blocked
live request, without starting BFCL, Inspect AI, lm-eval, SWE-bench, provider
APIs, model downloads, cloud spend, or weight updates. A receipt can be valid
with either `passed: true` or `passed: false`; neither value is proof of
benchmark completion. External benchmark claims remain disabled until Flight
Recorder imports a matching result produced by the external runner.

Run the adapter outside Flight Recorder. Then import its bounded raw result and
public runner metadata as `hfr.external_eval_result.v1`:

```bash
flightrecorder external-eval-result \
  --plan runs/external_eval_plan.json \
  --heldout-manifest runs/heldout_scenarios.json \
  --raw-result runs/external_runner/candidate_suite_summary.json \
  --runner-metadata runs/external_runner/runner_metadata.json \
  --adapter local_mock \
  --execution-id eval-001 \
  --model-id local/candidate \
  --normalizer-id hfr.local_mock.run_suite \
  --normalizer-version 1 \
  --raw-format hfr.run_suite.v1 \
  --status completed \
  --out runs/external_eval_result.json
```

The metadata file binds `adapter_id`, `execution_id`, and `model_id` to a
public `runner_observation`, including runner identity, exit status, timestamps,
cost reporting, and observed side effects. The importer does not load adapter
packages or run benchmark code. It rejects unsafe or oversized inputs,
fingerprints every source, applies an allowlisted normalizer, and stores only
normalized case identifiers, outcomes, metrics, and digests. Prompts, outputs,
tool arguments, patches, logs, provider URLs, and credentials are not embedded.
Executable plans require both `--model-endpoint` and a separate public
`--model` identity, so URL endpoints remain hash-bound without being copied into
the result identity.

Completed execution requires duplicate-free per-case evidence with exact
held-out coverage. Aggregate-only evidence and partial, duplicate, unexpected,
or unmapped cases remain incomplete. Result integrity and benchmark outcome
are independent: an integrity-valid, completed result may truthfully report a
`failed` benchmark outcome, but that outcome blocks external-eval claims and
governance readiness. Only a passing outcome with explicit safe runner
observations can unlock review.

Include both the plan and its result in the governance summary:

```bash
flightrecorder eval-summary \
  --suite-summary baseline=runs/baseline/suite_summary.json \
  --suite-summary candidate=runs/candidate/suite_summary.json \
  --external-adapter-plan local_mock=runs/external_eval_plan.json \
  --external-adapter-result local_mock=runs/external_eval_result.json \
  --out runs/eval_summary.json
```

For every adapter selected by a plan, `eval-summary` requires exactly one
result bound by plan SHA-256 and adapter ID. Labels are display metadata, not
the security boundary. Missing, duplicate, unexpected, or unmatched results
block the summary, as does any mismatch between the summary result set and the
result set supplied to the loop plan.

Validate summaries before handoff:

```bash
flightrecorder validate --eval-summary runs/eval_summary.json --strict
flightrecorder schemas --check runs/eval_summary.json
flightrecorder validate --heldout-manifest runs/heldout_scenarios.json --strict
flightrecorder schemas --check runs/heldout_scenarios.json
flightrecorder validate --external-eval-plan runs/external_eval_plan.json --strict
flightrecorder schemas --check runs/external_eval_plan.json
flightrecorder validate --external-eval-receipt runs/external_eval_receipt.json --strict
flightrecorder schemas --check runs/external_eval_receipt.json
flightrecorder validate --external-eval-result runs/external_eval_result.json --strict
flightrecorder schemas --check runs/external_eval_result.json
```

### Beta V1 Fail-Closed Correction

These public contracts are still beta. The v1 validators now deliberately
enforce the safer boundary: `hfr.external_eval_adapters.v1` plans always keep
claims disabled, `hfr.eval_summary.v1` requires matching imported results when
external adapter plans are included, and
`hfr.agentic_training_loop_plan.v1` separates plan readiness, execution
completion, and governance readiness. `hfr.promotion_decision.v1` now requires
the direct eval summary, its exact positive external-result set, candidate-model
binding, and the evidence bundle's matching summary fingerprint. Regenerate
older beta v1 eval summaries, evidence bundles, promotion policies, loop plans,
promotion decisions, promotion alias receipts, and promotion release records
that treated a plan, dry-run receipt, stored pass summary, or unbound source as
completion evidence; they now fail closed instead of being grandfathered into
governance readiness, alias authorization, or release readiness.

## Closed-Loop State Boundary

Bind the imported result and the eval summary into the full loop plan. This
focused command shows the execution-bearing inputs; a governance-ready loop
must also supply the other required rollout, review, curation, cloud, serving,
improvement, and promotion artifacts described by `agentic-loop plan --help`.

```bash
flightrecorder agentic-loop plan \
  --iteration-id loop-001 \
  --objective "Review externally executed held-out evaluation" \
  --agentic-training-plan runs/agentic_training_plan.json \
  --agentic-training-result runs/agentic_training_result.json \
  --heldout-manifest runs/heldout_scenarios.json \
  --external-eval-plan runs/external_eval_plan.json \
  --external-eval-receipt runs/external_eval_receipt.json \
  --external-eval-result runs/external_eval_result.json \
  --eval-summary runs/eval_summary.json \
  --out runs/agentic_training_loop_plan.json
```

The loop exposes three independent states:

- `plan_readiness`: `ready_to_execute` or `blocked`. This covers the
  pre-execution contracts and handoffs; it is not proof of execution.
- `execution_completion`: `completed`, `incomplete`, or `failed`. Completion
  requires a completed, plan-bound training result and exactly one completed,
  plan-bound external result for every selected adapter. A completed run may
  have a failed benchmark outcome; a runner execution whose status is `failed`
  makes execution completion `failed`.
- `governance_readiness`: `ready_for_review` or `blocked`. Review readiness
  requires a ready plan, completed execution, the exact result set in the eval
  summary, passing benchmark outcomes, and every remaining governance check to
  pass.

The legacy `readiness` field is derived from `governance_readiness`. Ledgers and
governance receipts carry all three states forward and require the completed,
review-ready combination before approval.

Include the eval summary in the evidence bundle that Governance or
improvement planning consumes:

```bash
flightrecorder evidence-bundle \
  --eval-summary runs/eval_summary.json \
  --out runs/evidence_bundle.json
```

The bundle stores compact eval-summary metrics and validation recomputes them
from the referenced `eval_summary.json` when that file is available, so stale
or forged bundle counts cannot hide held-out eval blockers.

Pass that same summary directly to `promotion-decision` with `--eval-summary`,
and repeat `--external-eval-result` once for every result named by the summary.
Promotion requires a positive, unique result set: the supplied result hashes
and projections must exactly match the summary, every result `model_id` must
match `--candidate-id`, and the evidence bundle must fingerprint the same
summary by hash and size. Each result must also be completed, passed,
coverage-complete, and ready for review.

`validate --promotion-decision` resolves those recorded sources relative to the
decision, rehashes them, reruns semantic validation for the evidence bundle,
eval summary, and every external result, then rebuilds the external-eval
lineage and required checks. Missing, replaced, or mutated sources therefore
block validation even when the persisted decision still says it passed.

## Blocking Conditions

Governance claims are suppressed when any of these hold:

- missing suite summaries for a cross-arm comparison;
- single-arm or mismatched held-out scenario sets;
- missing or new scenarios in the compare manifest;
- contract fingerprint drift;
- unverified contract fingerprints;
- zero comparison pairs;
- held-out manifests with mismatched or empty scenario sets;
- required serving preflight checks that are missing or blocked;
- external adapter plans that are not ready;
- missing, duplicate, unexpected, or plan-mismatched external eval results;
- incomplete or failed external runner execution;
- aggregate-only results or non-exact held-out case coverage;
- eval summaries whose external result set differs from the loop or promotion
  result set;
- promotion results for a different candidate, or an evidence bundle that
  fingerprints a different eval summary.

The artifact may still show raw movement, but `governance_claims` stays empty and
`passed` is false until the blockers are resolved.

## Operational Metrics

Each eval-summary arm includes an `operational_metrics` object with `cost`,
`latency`, `tokens`, and `task_completion` sections. These fields are
best-effort summaries of values already present in suite summary run rows or
suite-level metrics. Missing values remain explicit through `source: "missing"`
and `missing_run_count`; they do not become evidence of low cost, low latency,
or task completion.

## Repair And Curriculum Handoff

`eval-summary` always emits a `repair_curriculum` section. These work items are
deterministic routing hints derived from suite summaries, comparison manifests,
compare gates, and external adapter plans.

The section includes:

- `repair` items for baseline wins, task-completion regressions, suite critical
  failures, and new critical failures,
- `curriculum` items for repeated failed-rule and regressed-rule patterns,
- `eval_gate` items for failed comparison-gate checks,
- `eval_harness` items for held-out mismatch, contract, and external adapter
  readiness blockers.

These items are not promotion evidence and do not unsuppress raw candidate
movement. They exist so repair, scenario, curriculum, or adapter-readiness work
can be queued without manually interpreting raw eval artifacts.

To include eval-derived work in the longitudinal improvement loop, pass the
summary into the existing improvement-plan command:

```bash
flightrecorder improvement-plan \
  --evidence-bundle runs/evidence_bundle.json \
  --eval-summary runs/eval_summary.json \
  --out runs/improvement_plan.json
```

Eval `repair` and `curriculum` items keep their improvement-plan categories.
Eval gate and harness blockers are normalized as `bundle_action` items so the
ledger can track recurring operational blockers without adding a separate
category family.
When an improvement plan lists an `eval_summary` source artifact, validation
checks that the plan still carries every eval-derived repair-curriculum item
from that source summary.

Evidence-bundle validation reopens recorded `eval_summary` and
`serving_lifecycle` artifacts from the bundle file location before trusting
derived metrics, so cwd-relative lookalikes, missing files, and malformed source
artifacts cannot preserve stale bundle decisions.
Action-ledger validation reopens referenced evidence bundles from the ledger file
location and verifies every source `decision.next_actions` occurrence is
represented, including eval-summary bundle actions such as
`resolve_eval_summary_blockers`; cwd-relative lookalikes and malformed bundles
cannot validate the ledger.
Action-ledger gate validation reopens the referenced `action_ledger.json`,
recomputes source metrics and check actuals, and verifies persisted policy
requirements still have corresponding gate checks from the gate file location.
Decision-gate validation reopens the referenced source artifact from the gate
file's directory, so cwd-relative lookalikes, deleted sources, stale hashes, and
rewritten source decisions cannot validate as promotion-ready evidence. Gate
creation and validation also require a registered decision-bearing source type
that satisfies its bundled JSON Schema; arbitrary or wrong-type JSON cannot
authorize promotion.
Promotion-ledger validation applies the same file-relative rule to recorded
decision gates and revalidates each gate's current source contract before
trusting ledger metrics or longitudinal gate decisions.
Promotion-decision validation additionally replays the referenced evidence
bundle, eval summary, and external result set before trusting persisted lineage
checks or alias authorization.
Trainer-preflight validation applies the same file-relative rule to recorded
gates, validation summaries, schema contracts, and trainer artifacts before
trainer launch or archive handoff evidence can pass.
