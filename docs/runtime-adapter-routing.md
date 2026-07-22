# Governed Runtime Adapter Routing

Flight Recorder uses a hybrid adapter architecture: one generalist LoRA, up to
three initial specialists (`browser`, `database`, and `code_terminal`), and a
deterministic router outside the model. The router consumes trusted task
contracts; model text never chooses its own tools, adapter, or write authority.

This is preferable to both extremes:

- one monolithic adapter is simple but allows unrelated task families to
  interfere and gives weak fault isolation;
- one adapter per tool creates excessive routing, evaluation, and memory churn;
- a generalist plus a few broad specialists preserves cross-domain behavior
  while isolating the domains where specialized trajectories help.

Only one atomic adapter is active for a model call. A merged or stacked adapter
is a new candidate with its own immutable hash, evaluation, safety result,
promotion decision, and rollback target. Flight Recorder never treats several
independently promoted adapters as permission to compose them at runtime.

## Governed decisions

`hfr.tool_capability_selection.v1` records every evaluated tool, the exact
schema/version/hash, eligibility, rejection reasons, selected tool-set hash,
task contract, policy, and runtime environment.

`hfr.adapter_route_decision.v1` binds that selection to the router version,
task contract, exact base/tokenizer/template identities, one adapter hash,
independent training and evaluation references, promotion evidence, confidence,
fallback reason, and policy decision.

The runtime order is:

1. A trusted caller creates a task contract with required capabilities and
   domains.
2. Tool selection evaluates the whole catalog and fails closed on unknown,
   denied, malformed, or unavailable capabilities.
3. A single-domain task selects one uniquely compatible promoted specialist,
   then an independently promoted atomic composition when no specialist fits.
4. A cross-domain task selects one uniquely compatible promoted atomic
   composition before the promoted generalist. Multiple compatible candidates
   at either tier block as ambiguous. A missing or stale specialist may use the
   generalist only when policy explicitly permits it.
5. The serving layer activates exactly the selected atomic adapter.
6. Every tool call is checked again before dispatch. A write-capable call needs
   an external, expiring, content-bound, single-use approval. Prompt text cannot
   supply that approval.

## CLI

```bash
flightrecorder runtime-router tool-capabilities \
  --task-contract task_contract.json \
  --tool-catalog tool_catalog.json \
  --policy tool_policy.json \
  --environment runtime_environment.json \
  --out tool_capability_selection.json

flightrecorder runtime-router adapter \
  --task-contract task_contract.json \
  --capability-selection tool_capability_selection.json \
  --candidate-catalog candidate_catalog.json \
  --routing-policy routing_policy.json \
  --runtime-environment runtime_environment.json \
  --out adapter_route_decision.json

flightrecorder validate \
  --tool-capability-selection tool_capability_selection.json \
  --adapter-route-decision adapter_route_decision.json \
  --strict
```

Output creation is atomic and refuses to overwrite an existing artifact.
Semantic validation resolves only safe relative regular-file references and
rechecks file size, SHA-256, promotion contents, candidate identity, and the
independent training/evaluation bindings.

Run the offline end-to-end proof with:

```bash
python3.11 scripts/validate_runtime_adapter_router.py
```

It demonstrates a specialist route, a recorded stale-specialist rejection and
generalist fallback, an authorized read, a denied write whose handler is not
called, and an approved write dispatched exactly once.

## Training and promotion data

The case study contains 360 synthetic native action-SFT trajectories: 288
train, 36 development, and 36 sealed final. It includes exact tool schemas,
multi-call order and arguments, tool results, ambiguity, refusals, failure
recovery, external write approvals/denials, and cross-domain tasks. Candidate
trainer views contain train rows only; sealed-final tasks are excluded from
recipe choice and adapter training.

The initial local Qwen3-0.6B integration run trained the four intended LoRAs,
but all failed the strict sealed promotion gate. The generalist improved exact
tool-call order/arguments from 8/36 on the base to 12/36; specialist scoped
rates were 12/16 browser, 12/20 database, and 12/16 code/terminal. None passed
the exact safety and final-answer requirements, so the actual candidate catalog
must remain unroutable. That is the intended failure mode: training success is
not promotion.

The safe synthetic report is checked into
`examples/case_studies/runtime_adapter_router/evaluation/actual_local_evaluation.json`
under the registered `hfr.runtime_adapter_candidate_evaluation.v1` contract.
It remains `passed: false` with an empty promotion list; raw observations and
adapter weights remain in ignored local run storage.

The short 60-step generalist and 40-step specialist runs remain historical
integration proofs. The governed campaign runner below now demonstrates the
longer path: choose recipes on development evidence, rerun a promising scoped
candidate, and touch sealed final only once at promotion without lowering the
safety or promotion gates.

## Governed autoresearch campaign runner

`scripts/run_runtime_adapter_autoresearch.py` is the offline controller for the
longer recipe search. It keeps the creative search loop inside Flight Recorder
evidence boundaries:

- search inputs are a development suite manifest plus development JSONL only;
  paths that look like held-out, frozen, final, or sealed data are refused;
- every proposal gets a unique `attempts/trial-*` directory;
- a proposal/launch record is written before any subprocess is invoked;
- the trainer command requires registered model and dataset manifests,
  `--local-training`, `--local-model-path`, `--execute-local-training`,
  `--disable-trackio`, and a fixed per-trial wall-clock budget;
- `--push-to-hub` and remote tracking are never added by the controller;
- candidate JSON is content-addressed from the recipe, adapter fingerprint, and
  training result;
- model, revision, tokenizer revision, and chat-template hash are explicit
  campaign inputs and must exactly match the registered model manifest;
- tool order, names, URLs, identifiers, commands, and non-null values remain
  strict. A documented functional comparator accepts only a trailing
  `headline` refinement on `browser.search` and unused null bind keys on
  `database.query`; raw exactness remains a separate diagnostic;
- deterministic safety and recovery checks judge observable behavior: no tool
  call plus an approval denial, clarification, or sensitive-data refusal, and
  ordered recovery calls followed by a cache/mirror-grounded answer. Exact
  wording remains reported as a diagnostic but is not mislabeled as unsafe;
- development scoring records both functional and exact tool-call rates, and
  combines overall, functional tool-call, exact final-answer, safety,
  write-denial, and failure-recovery metrics into a bounded positive quality
  score, then subtracts one full point per actual unsafe call;
- the search does not access sealed final rows and does not lower promotion
  thresholds.

Before optimizer step one, the trainer inspects the tokenized labels after the
configured `max_length` boundary. A recipe is refused if any row would reach
the model with zero visible assistant labels. The training result records row
counts for complete, partially truncated, and zero-supervision examples. This
closes a subtle failure mode in which preprocessing sees a valid assistant mask
but the later data collator truncates every supervised token from a long row.

When development evidence shows correct tool selection but brittle literal
arguments, `--action-turn-repeats N` can add bounded weight to the first native
tool-call turn. The trainer derives those prefixes in memory from the same
governed, reviewed action-SFT rows; it keeps every full trajectory, does not
duplicate no-tool safety rows, records the source and effective counts in the
training plan, and rejects values outside `0..32`. The value is part of the
content-addressed autoresearch recipe, so it can be compared, resumed, and
audited like rank, dropout, or step count rather than becoming an unrecorded
data-loader tweak.

By default the runner does not update weights. It records blocked/not-run trial
evidence instead:

```bash
python3.11 scripts/run_runtime_adapter_autoresearch.py \
  --campaign-dir runs/runtime_adapter_autoresearch/dev-search \
  --development-suite path/to/development_suite_manifest.json \
  --development-jsonl examples/case_studies/runtime_adapter_router/data/development_action_sft.jsonl
```

Actual local training requires an explicit flag and local registered inputs:

```bash
python3.11 scripts/run_runtime_adapter_autoresearch.py \
  --campaign-dir runs/runtime_adapter_autoresearch/dev-search \
  --development-suite path/to/development_suite.json \
  --development-jsonl examples/case_studies/runtime_adapter_router/data/development_action_sft.jsonl \
  --model-manifest examples/case_studies/runtime_adapter_router/registry/model_candidate.json \
  --dataset-manifest examples/case_studies/runtime_adapter_router/registry/dataset_version.json \
  --local-model-path /path/to/local/qwen3-0.6b \
  --execute-local-training \
  --max-trials 6 \
  --campaign-max-duration-seconds 43200 \
  --trial-training-seconds 5400
```

Specialist campaigns keep both sides of the specialization explicit. A
repeatable `--task-family` accepts registered training families ending only in
`_train`; `--evaluation-scope` binds the candidate to one or more router task
scopes. The trainer filter, candidate identity, campaign record, development
report, and eventual sealed report all retain that boundary. For example:

```bash
python3.11 scripts/run_runtime_adapter_autoresearch.py \
  --campaign-dir runs/runtime_adapter_autoresearch/browser-dev-search \
  --development-suite path/to/development_suite.json \
  --development-jsonl examples/case_studies/runtime_adapter_router/data/development_action_sft.jsonl \
  --task-family runtime_adapter_router_browser_train \
  --evaluation-scope browser \
  --model-manifest examples/case_studies/runtime_adapter_router/registry/model_candidate.json \
  --dataset-manifest examples/case_studies/runtime_adapter_router/registry/browser_dataset_version.json \
  --local-model-path /path/to/local/model \
  --execute-local-training
```

For a different local base, the identity is never inferred from a directory
name. Bind all four values explicitly and use a manifest carrying the same
values, for example the included 4B candidate:

```bash
python3.11 scripts/run_runtime_adapter_autoresearch.py \
  --campaign-dir runs/runtime_adapter_autoresearch/dev-search-4b \
  --development-suite path/to/development_suite.json \
  --development-jsonl examples/case_studies/runtime_adapter_router/data/development_action_sft.jsonl \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --model-revision cdbee75f17c01a7cc42f958dc650907174af0554 \
  --tokenizer-revision cdbee75f17c01a7cc42f958dc650907174af0554 \
  --chat-template-sha256 64f85b198065d0fba2a81f37e10ed68161ce2c19a754c7100e67e0ca2ee9c326 \
  --model-manifest examples/case_studies/runtime_adapter_router/registry/model_candidate_qwen3_4b_instruct_2507.json \
  --dataset-manifest examples/case_studies/runtime_adapter_router/registry/generalist_dataset_version.json \
  --local-model-path /path/to/local/qwen3-4b-instruct-2507 \
  --execute-local-training \
  --device mps \
  --max-length 1024 \
  --action-turn-repeats 4
```

Resume is requested with `--resume`. The controller feature-detects the
installed `run_search` API and forwards `resume=True` only when that API exists;
older library versions still run deterministically from the immutable plan.

Sealed final is a separate, one-shot action. It selects only the champion
candidate and refuses to run unless that candidate already passes every
development promotion threshold with zero critical unsafe calls. It also
recomputes the persisted candidate and recipe content hashes, revalidates the
current adapter and training-result fingerprints, and refuses to run if
`sealed_final_receipt.json` already exists. All of those checks happen before
the receipt that authorizes the single sealed access is created:

```bash
python3.11 scripts/run_runtime_adapter_autoresearch.py \
  --campaign-dir runs/runtime_adapter_autoresearch/dev-search \
  --sealed-jsonl examples/case_studies/runtime_adapter_router/data/sealed_final_action_sft.jsonl \
  --finalize-sealed
```

Validate the campaign outcome with:

```bash
python3.11 scripts/validate_runtime_adapter_autoresearch.py \
  --campaign-dir runs/runtime_adapter_autoresearch/dev-search \
  --out runs/runtime_adapter_autoresearch/dev-search/validation.json
```

The validator replays the recipe-search result, checks that the sealed report
evaluated the exact champion adapter hash, requires one promotion-eligible
sealed candidate, requires zero critical safety failures, and can optionally
consume a separate release-check receipt. A passing validation result means the
campaign is promotable evidence; it does not itself mutate a registry alias or
deploy an adapter.

## Governed browser-specialist result

The completed `governed-browser-specialist-v9` campaign trained a local
Qwen3-4B-Instruct-2507 LoRA on 71 registered browser training rows. Four
bounded action-turn repeats produced 355 effective supervised rows; all 355
retained full visible supervision at `max_length=1024`. The 160-step MPS run
took 1,783.6 seconds and ended with training loss `0.0459749`.

The immutable development report passed all four browser tasks with ordered
functional tool calls, exact final answers, and zero critical failures. The
single sealed-final access then passed all nine browser-scoped tasks with the
same results. Raw byte-exact tool-call arguments remained 0/4 on development
and 0/9 sealed because every search query added only the documented trailing
`headline` refinement; URL, identifier, recency, extraction mode, call order,
and final answers matched the governed tasks.

The content identities are:

- adapter directory SHA-256:
  `1db6865cd4013baeb5370264bcfa8406a0f8456eb5eba93e32469fac56ea87de`;
- candidate identity SHA-256:
  `a1495c41471e0b09230fcf1d9ef2556e88ca856dcab489aa110b302b669d3dc7`;
- sealed report SHA-256:
  `6e1c9a9d889059f840bc8a6ab0d32e8482613c0aa26b77e8ef4a9609d435216c`.

The raw observations, model weights, and campaign directory remain ignored
local artifacts because they are potentially sensitive and large. This result
qualifies one read-only browser specialist; it does not establish database,
code/terminal, cross-domain, or write-capable quality, and it does not deploy
or promote a runtime registry alias automatically.
