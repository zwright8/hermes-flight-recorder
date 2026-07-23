# Governed Tau-3 7–9B QLoRA Readiness

This workflow prepares the artifacts needed to begin one local, cross-domain
7–9B MLX-LM QLoRA study over Tau text-mode airline, retail, and telecom tasks.
It stops at a validated trainer consumer plan and launch check. Acquisition is
an explicit, separately authorized setup step; the preparation, capture,
freeze, build, and validation commands themselves do not contact model
providers, update weights, open sealed evaluation, or make a benchmark claim.

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

## Frozen study inputs

The first study lineage uses these immutable public inputs:

| Role | Local MLX snapshot | Revision | Upstream identity |
| --- | --- | --- | --- |
| Trainable base | [mlx-community/Qwen3.5-9B-4bit](https://huggingface.co/mlx-community/Qwen3.5-9B-4bit) | `8b2b98c00a6b4d291155e4890773ca8f769aee53` | `Qwen/Qwen3.5-9B@c202236235762e1c871ad0ccb60c8ee5ba337b9a` |
| Comparator 1 | [mlx-community/Qwen3-8B-4bit](https://huggingface.co/mlx-community/Qwen3-8B-4bit) | `545dc4251c05440727734bcd94334791f6ab0192` | `Qwen/Qwen3-8B@b968826d9c46dd6066d109eabc6255188de91218` |
| Comparator 2 | [mlx-community/granite-3.3-8b-instruct-4bit](https://huggingface.co/mlx-community/granite-3.3-8b-instruct-4bit) | `0751a30ba2420ecd5c2f142707dd4fcfacc4486e` | `ibm-granite/granite-3.3-8b-instruct@51dd4bc2ade4059a6bd87649d68aa11e4fb2529b` |

All three are dense 7–9B Apache-2.0 models. The benchmark source is the
official [sierra-research/tau2-bench](https://github.com/sierra-research/tau2-bench)
repository at `1d244f5dca42944b67a379b44bfeb9f5748f189d`. The local runtime freeze is
`mlx-lm==0.31.3` with `mlx==0.32.0`.

Network access is needed only for this explicit acquisition phase:

```bash
git clone https://github.com/sierra-research/tau2-bench.git local/tau3/repository
git -C local/tau3/repository checkout 1d244f5dca42944b67a379b44bfeb9f5748f189d

uv venv --python 3.12 local/tau3/venv
uv pip install --python local/tau3/venv/bin/python -e local/tau3/repository
.venv/bin/python -m pip install 'mlx-lm==0.31.3'

.venv/bin/hf download mlx-community/Qwen3.5-9B-4bit \
  --revision 8b2b98c00a6b4d291155e4890773ca8f769aee53 \
  --local-dir local/tau3/models/base
.venv/bin/hf download mlx-community/Qwen3-8B-4bit \
  --revision 545dc4251c05440727734bcd94334791f6ab0192 \
  --local-dir local/tau3/models/comparator-1
.venv/bin/hf download mlx-community/granite-3.3-8b-instruct-4bit \
  --revision 0751a30ba2420ecd5c2f142707dd4fcfacc4486e \
  --local-dir local/tau3/models/comparator-2
```

Run `hf cache verify` for each exact repository and revision. The local-dir
downloader creates `.cache` metadata that is not model content; move that
directory to ignored `local/tau3/download-metadata/<role>` and then require
both no missing and no extra files:

```bash
.venv/bin/hf cache verify mlx-community/Qwen3.5-9B-4bit \
  --revision 8b2b98c00a6b4d291155e4890773ca8f769aee53 \
  --local-dir local/tau3/models/base \
  --fail-on-missing-files --fail-on-extra-files
```

Repeat that verification for both comparator repositories. Build one complete
tree identity per model after the cache metadata is outside the model tree:

```bash
mkdir -p local/tau3/identities
.venv/bin/python scripts/build_tau3_model_identity.py \
  --model-path local/tau3/models/base \
  --model-id mlx-community/Qwen3.5-9B-4bit \
  --revision 8b2b98c00a6b4d291155e4890773ca8f769aee53 \
  --out local/tau3/identities/base.json
.venv/bin/python scripts/build_tau3_model_identity.py \
  --model-path local/tau3/models/comparator-1 \
  --model-id mlx-community/Qwen3-8B-4bit \
  --revision 545dc4251c05440727734bcd94334791f6ab0192 \
  --out local/tau3/identities/comparator-1.json
.venv/bin/python scripts/build_tau3_model_identity.py \
  --model-path local/tau3/models/comparator-2 \
  --model-id mlx-community/granite-3.3-8b-instruct-4bit \
  --revision 0751a30ba2420ecd5c2f142707dd4fcfacc4486e \
  --out local/tau3/identities/comparator-2.json
```

## Prepare governed Tau data

The partitioner verifies a clean exact checkout, verifies official split
invariants, assigns task families before the development split, and writes raw
payloads only for training and development. `sealed.json` contains hashes only;
the workflow never materializes official test payloads into a training file.

```bash
.venv/bin/python scripts/prepare_tau3_training_sources.py \
  --tau-repo local/tau3/repository \
  --expected-revision 1d244f5dca42944b67a379b44bfeb9f5748f189d \
  --development-fraction 0.2 \
  --salt hfr-tau3-core-v1 \
  --out local/tau3/source-v1
```

The capture generator replays official training-side reference actions in the
pinned Tau runtime and produces eight governed behavior variants. Replay errors
are retained as hashed source rejections, and the next deterministic candidate
is tried; quotas still fail closed if they cannot be filled. Default quotas
compensate for telecom's longer policy text, and generation fails if any domain
exceeds 45% of corpus tokens.

```bash
.venv/bin/python scripts/generate_tau3_training_captures.py \
  --tau-repo local/tau3/repository \
  --expected-revision 1d244f5dca42944b67a379b44bfeb9f5748f189d \
  --train-tasks local/tau3/source-v1/training_source/train_tasks.jsonl \
  --development-tasks local/tau3/source-v1/training_source/development_tasks.jsonl \
  --tau-python local/tau3/venv/bin/python \
  --generator-id hfr-tau3-reference-action-replay \
  --generator-revision 1d244f5dca42944b67a379b44bfeb9f5748f189d \
  --seed 8675309 \
  --train-domain-quotas airline=8,retail=8,telecom=3 \
  --development-domain-quotas airline=2,retail=2,telecom=1 \
  --sample-salt hfr-tau3-core-agent-study-v1 \
  --out local/tau3/captures-v1

install -m 600 local/tau3/captures-v1/captures.jsonl \
  local/tau3/training_captures.jsonl
```

This lineage produces 192 captures: 24 examples for each required behavior,
74 admitted trajectories, and 118 retained negatives. Its measured domain
token shares are airline `0.338407`, retail `0.310488`, and telecom `0.351104`.
The canonical capture file is sealed as
`eedac897ce432837506d33883d9e2444bc54b3d2a78636a54cf014cf3e1d3c1f`;
the freezer embeds this content identity and its 192-row count in the split
contract, so a different JSONL cannot pass production preflight.
These are corpus-construction statistics, not model-quality results.

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
- a common 16,384-token harness context and a 12,288-token default training
  sequence budget, both identical across the study rather than tuned per model;
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
remains useful for contract review, but deliberately fails while any
`REPLACE_WITH_...` value or pending attestation remains. The production path
should generate the config from evidence instead of editing the template:

```bash
.venv/bin/python scripts/freeze_tau3_training_protocol.py \
  --tau-repo local/tau3/repository \
  --tau-revision 1d244f5dca42944b67a379b44bfeb9f5748f189d \
  --source-manifest local/tau3/source-v1/manifest.json \
  --train-split local/tau3/source-v1/train.json \
  --development-split local/tau3/source-v1/development.json \
  --sealed-split local/tau3/source-v1/sealed.json \
  --train-tasks local/tau3/source-v1/training_source/train_tasks.jsonl \
  --development-tasks local/tau3/source-v1/training_source/development_tasks.jsonl \
  --base-identity local/tau3/identities/base.json \
  --base-model-path local/tau3/models/base \
  --comparator1-identity local/tau3/identities/comparator-1.json \
  --comparator1-model-path local/tau3/models/comparator-1 \
  --comparator2-identity local/tau3/identities/comparator-2.json \
  --comparator2-model-path local/tau3/models/comparator-2 \
  --captures local/tau3/training_captures.jsonl \
  --out local/tau3/protocol.json

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
  --config local/tau3/protocol.json \
  --captures local/tau3/training_captures.jsonl \
  --out runs/tau3_core_training_artifacts

.venv/bin/python scripts/validate_tau3_training_artifacts.py \
  --bundle runs/tau3_core_training_artifacts \
  --strict \
  --out runs/tau3_core_training_artifacts/validation.json
```

Production readiness is true only when the local Tau commit, local base-model
directory, and MLX-LM runtime checks pass together with the generic Flight
Recorder export, gate, preflight, launch, archive, archive-check, consumer, and
Tau-specific validation chain. The current frozen protocol SHA-256 is
`cf4737f9590344fb6bfbe90fafc45eae130215a307841e487b944576c451d1ba`;
its source preflight passes 31/31 checks. The approved MLX command is evidence
only; the builder never executes it.

The trainer-facing archive contains `training/input_export/train.jsonl` and
`training/input_export/valid.jsonl` in MLX chat format, derived from reviewed
action-SFT rows with a full canonical-capture fallback when generic SFT would
discard a recovery/tool sequence. It has 58 train and 16 development rows: 36
action-SFT-backed and 38 canonical fallbacks. Sixty-eight rows contain tool
trajectories. The pinned Qwen tokenizer renders all 74 rows successfully; the
observed range is 677–11,970 tokens, with zero rows over the 12,288-token
training limit or 16,384-token harness window. It never writes `test.jsonl`,
and sealed rows are forbidden from both views. The launch check passes while
recording `executed=false` and `training_started=false`; it exposes the exact
local `python -m mlx_lm lora --train ...` command but does not invoke it.

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
