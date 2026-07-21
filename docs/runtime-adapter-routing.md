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

The short 60-step generalist and 40-step specialist runs are integration
proofs, not a 12-hour autoresearch campaign. A longer campaign should search
recipes on development evidence, rerun promising candidates, and touch sealed
final only once at promotion. It must not lower the safety or promotion gates
to obtain a passing result.
