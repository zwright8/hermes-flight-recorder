# Governed Tau-3 7–9B QLoRA Readiness

This workflow prepares the artifacts needed to begin one local, cross-domain
7–9B MLX-LM QLoRA study over Tau text-mode airline, retail, and telecom tasks.
It stops at a validated trainer consumer plan and launch check. It does not
download a model, run inference, update weights, open sealed evaluation, or
make a benchmark claim.

The study contract is deliberately narrow:

- one dense 7–9B trainable base and one cross-domain adapter;
- the unmodified base plus at least two frozen eligible 7–9B comparators;
- identical prompt, tools, tool ordering, context, decoding, retries, turn and
  token limits, and no additional test-time search;
- training-side and development data only during generation and recipe search;
- at most seven days for generation, recipe search, final training, and final
  evaluation, with final-evaluation time reserved in advance;
- fail-closed redaction, licensing, contamination, safety, state-mutation,
  budget, archive, and launch gates.

## Safe rehearsal

Run the deterministic rehearsal before introducing benchmark or model assets:

```bash
.venv/bin/python scripts/build_tau3_training_artifacts.py \
  --mode rehearsal \
  --out runs/tau3_core_training_rehearsal

.venv/bin/python scripts/validate_tau3_training_artifacts.py \
  --bundle runs/tau3_core_training_rehearsal \
  --strict \
  --allow-rehearsal
```

The rehearsal covers all required behavior families and produces balanced
airline, retail, and telecom traces, admissions, retained rejections, SFT,
action-SFT, evidence-backed DPO, dataset identity, protocol, training plan,
runtime preflight, trainer preflight, launch check, portable archive, archive
check, consumer plan, and evidence bundle. It uses synthetic identities and
therefore records `bundle_mode: rehearsal` and `ready_for_training: false`.
Without `--allow-rehearsal`, strict validation rejects it.

## Production inputs

Production mode requires two local files:

1. A frozen protocol JSON object with these top-level objects:
   `protocol_manifest`, `tau_revision`, `split_manifest`,
   `harness_contract`, `model_freeze`, `budget`, `sealed_manifest`,
   `mlx_qlora_plan`, `recipe_space`, and
   `candidate_selection_contract`.
2. A JSONL file of canonical `hfr.tau3_capture.v1` rows emitted from
   training-side or development Tau text episodes.

The production protocol must include:

- the local Tau git path and exact checked-out commit;
- train, development, and sealed manifest hashes, with the sealed manifest
  quarantined before generation;
- a dense 7–9B base and at least two eligible comparators, each with immutable
  revision, parameter count, license, tokenizer, chat template, and 4-bit
  conversion identity; each local model must also provide a hashed identity
  JSON matching its model ID and revision (a directory name is not evidence);
- the frozen harness and seed schedule;
- statistical method, safety and per-domain non-inferiority margins;
- bounded development-only recipe search and one untouched selected
  checkpoint;
- the exact non-executing MLX-LM launch argv in
  `mlx_qlora_plan.command_argv`, including model, data, adapter-output, and
  training flags. The portable bindings are exactly `model_input`,
  `input_export`, and `adapter_output`; remote model IDs, URLs, parent
  traversal, network flags, and publication flags are rejected;
- the seven-day stage budget and final-evaluation reserve;
- local model paths and an installed local `mlx_lm` runtime.

Every capture row binds the observed behavior to immutable source evidence:

```json
{
  "schema_version": "hfr.tau3_capture.v1",
  "trajectory_id": "airline_change_001_good",
  "task_id": "airline-change-001",
  "task_family": "airline_change_family_001",
  "domain": "airline",
  "split": "train",
  "behavior": "correction",
  "prompt": "...",
  "prompt_hash": "<canonical SHA-256>",
  "seed": 101,
  "generator_id": "<local model id>",
  "generator_revision": "<immutable revision>",
  "policy_revision": "<immutable revision>",
  "tool_schema_revision": "<immutable revision>",
  "starting_state_hash": "<SHA-256>",
  "tools": [{"type": "function", "function": {"name": "..."}}],
  "events": [
    {"type": "user_message", "role": "user", "content": "..."},
    {"type": "tool_call", "role": "assistant", "tool_name": "...", "tool_call_id": "call-1", "args": {}},
    {"type": "tool_result", "role": "tool", "tool_name": "...", "tool_call_id": "call-1", "result": {}},
    {"type": "assistant_message", "role": "assistant", "content": "..."}
  ],
  "state_transition": {
    "before_hash": "<starting-state SHA-256>",
    "after_hash": "<ending-state SHA-256>",
    "changes": [],
    "executable": true
  },
  "outcome": {
    "success": true,
    "executable_label": "task_and_state_verified",
    "policy_violation": false,
    "harmful_mutation": false,
    "evidence_refs": ["tool_result:call-1", "state_transition:airline_change_001_good"]
  },
  "review": {
    "reviewer": "<reviewer identity>",
    "verifier": "<executable verifier identity>",
    "disposition": "admit",
    "reason": "Executable task, state, and safety labels passed."
  }
}
```

For each production model entry, `local_path` points to the populated local
weights directory, while `local_identity_path` points to a JSON file such as
`{"model_id":"org/model","revision":"<immutable revision>","files":[...]}`
and `local_identity_sha256` seals that file. The identity must replay every
regular file in the model tree, including configuration, tokenizer, and weight
files; symlinks, omitted files, stale records, or changed hashes fail closed.
These machine-local fields are checked before the public bundle is written and
are removed from durable artifacts.

The checked-in [production input template](../examples/tau3_training/protocol_config.template.json)
contains every required field but deliberately fails while any
`REPLACE_WITH_...` value or pending attestation remains. Build model identities
and run the source-only preflight before creating a bundle:

```bash
.venv/bin/python scripts/build_tau3_model_identity.py \
  --model-path local/tau3/models/base \
  --model-id '<frozen model id>' \
  --revision '<immutable revision>' \
  --out local/tau3/identities/base.json

.venv/bin/python scripts/check_tau3_training_sources.py \
  --config local/tau3/protocol.json \
  --out local/tau3/source_preflight.json
```

`flightrecorder.tau3_capture` converts this envelope into normalized traces,
task-completion evidence, scorecards, state diffs, and eligible trajectory-v2
action supervision. Failed, unsafe, unverifiable, or quarantined rows remain in
the evidence and rejection ledger; they never become positive SFT examples.

Build and validate a production bundle only after the local inputs exist:

```bash
.venv/bin/python scripts/build_tau3_training_artifacts.py \
  --mode production \
  --config local/tau3_protocol.json \
  --captures local/tau3_training_captures.jsonl \
  --out runs/tau3_core_training_artifacts

.venv/bin/python scripts/validate_tau3_training_artifacts.py \
  --bundle runs/tau3_core_training_artifacts \
  --strict \
  --out runs/tau3_core_training_artifacts/validation.json
```

Production readiness is true only when the local Tau commit, local base-model
directory, and MLX-LM runtime checks pass together with the generic Flight
Recorder export, gate, preflight, launch, archive, archive-check, consumer, and
Tau-specific validation chain. The approved MLX command is evidence only; the
builder never executes it.

## Bundle layout

The root `manifest.json` binds every required role to a relative path, size,
and SHA-256. The principal groups are:

```text
protocol/     frozen benchmark, split, harness, model, and budget contracts
sealed/       quarantined sealed-manifest identity only
generation/   captures, admission/rejection ledgers, quality reports, identity
exports/      canonical SFT, action-SFT, DPO, splits, metrics, registry, card
training/     MLX recipe, plans, gates, preflights, archive, consumer plan
rehearsal/    tiny non-sealed pipeline result
evidence/     hash-checked cross-artifact evidence bundle
```

Generated bundles and raw traces are ignored local artifacts. Treat them as
sensitive until redaction, licensing, and publication review complete. Commit
the code and schemas, not local model weights, raw Tau state, device identifiers,
credentials, or unreviewed generated traces.

## Completion boundary

This readiness goal ends before long-running training. A later, separately
authorized execution must revalidate the consumer plan, resolve the pinned local
base, run the approved MLX command, archive adapter weights and resource
telemetry, and only then proceed to the pre-registered sealed evaluation.
No result may be called state of the art unless every statistical, safety,
per-domain, completeness, contamination, licensing, and seven-day predicate
passes against the frozen comparator set.
