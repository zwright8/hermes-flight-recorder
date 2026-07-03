# 24/7 Goals For Agentic Fine-Tuning Infrastructure

These are the persistent goals to run continuously until the platform can build,
evaluate, serve, and govern custom fine-tuned agentic models for defined tasks.

The daily email reporter is separate. These workers should not stop just
because a daily report was sent.

## Shared Rules For All 24/7 Goals

Every goal worker must:

- work in `<repo-root>` or a dedicated goal worktree for this repository
- read `docs/agentic-finetune-autonomous-operations.md`
- read `docs/agentic-finetune-autonomous-goals.md`
- read `experiments/autonomy/supervisor_state.json` if it exists
- inspect existing code before editing
- keep diffs small and reversible
- prefer existing Flight Recorder patterns
- write or update tests for behavior changes
- write or update docs for new commands, artifacts, and gates
- run appropriate verification before marking work complete
- update the local, gitignored `experiments/autonomy/supervisor_state.json`
- append a checkpoint to the local, gitignored
  `experiments/autonomy/journal/YYYY-MM-DD.md`
- continue to the next unblocked work packet until complete, blocked, or
  cancelled
- use a dedicated Codex worktree or dedicated git worktree for implementation
  work; do not make implementation changes from the shared checkout
- commit verified repository changes to a goal-specific branch
- after verification, integrate the branch to `main`, push `main`, and close or
  delete the temporary branch
- treat the repository as public and never commit personal email addresses,
  home-directory paths, local workspace paths, private Codex files, machine
  names, API keys, or automation recipient configuration

Workers must not:

- train on unredacted traces
- use final-answer-only success as a training label
- promote a model without governance gates
- launch expensive training without a dry-run plan and explicit artifacts
- leave unmanaged serving, training, or eval processes running
- overwrite unrelated user changes
- stage or commit files outside the worker's intended scope
- stage or commit local autonomy state, daily email bodies, or journals unless
  they have been intentionally scrubbed into public examples

## GitHub Persistence Protocol

Each 24/7 worker is responsible for durable delivery, not just local edits.

Required workflow:

1. Work from a dedicated Codex worktree or dedicated git worktree.
2. Before editing, run `git status --short`, `git branch --show-current`, and
   `git remote -v`.
3. Create or reuse a goal-specific branch:
   - `codex/hermes-goal-0-supervisor-YYYYMMDD-HHMM`
   - `codex/hermes-goal-1-evidence-YYYYMMDD-HHMM`
   - `codex/hermes-goal-2-harness-YYYYMMDD-HHMM`
   - `codex/hermes-goal-3-data-YYYYMMDD-HHMM`
   - `codex/hermes-goal-4-model-YYYYMMDD-HHMM`
   - `codex/hermes-goal-5-training-YYYYMMDD-HHMM`
   - `codex/hermes-goal-6-serving-demo-YYYYMMDD-HHMM`
   - `codex/hermes-goal-7-eval-YYYYMMDD-HHMM`
   - `codex/hermes-goal-8-governance-YYYYMMDD-HHMM`
4. Stage only files intentionally changed by that worker.
5. Run `git diff --cached --name-status` before every commit.
6. Run a public-repo privacy scan before every commit. Investigate and remove
   real personal data, absolute home paths, private Codex state, secrets, and
   local automation details; fixture-only `example.*` values are allowed.
7. Commit with the Lore Commit Protocol from `AGENTS.md`.
8. Push the branch with `git push -u origin <branch>`.
9. Verify the remote branch exists.
10. Acquire the configured local main-integration lock outside the repository.
11. In the lock, fetch `origin`, update a clean local `main` to `origin/main`,
    merge or cherry-pick the verified branch, run the packet's verification
    again, and push `main` to `origin`.
12. After `main` is pushed and verified, close any related PR if one exists,
    delete the remote branch, and remove the local temporary branch/worktree
    only when no worker still needs it.
13. Release the local main-integration lock.
14. Continue future work from a fresh goal-specific branch based on the updated
    `origin/main`.
15. Record branch, commit SHA, main SHA, branch deletion/PR closure status,
    verification commands, and remaining risks in the journal checkpoint.

Suggested privacy scan:

```bash
git diff --cached -U0 --diff-filter=ACMR \
  | rg '^\\+' \
  | rg -n -S '(/U[s]ers/|/h[o]me/[^[:space:]]+|C:/U[s]ers/|Documents/Git[H]ub|[.]codex[-]goal[-]worktrees|[.]codex/automations|[A-Za-z0-9._%+-]+[@][A-Za-z0-9.-]+[.][A-Za-z]{2,})'
```

The scan is intentionally broad. Remove real findings before committing, and
leave only documented false positives such as non-routable `example.test`
fixtures.

If branch push, main integration, remote verification, PR closure, or branch
deletion fails, keep the branch and record the exact failing command, stderr,
branch name, commit SHA if any, and auth/network diagnostics. Do not report a
run as fully delivered when changes exist only on a side branch.

Main integration safety:

- Never push unverified changes to `main`.
- Never bypass the integration lock.
- If `main` changed while the worker was verifying, rebase/merge with current
  `origin/main`, rerun the relevant verification, and only then push.
- If integration conflicts are non-trivial, stop that packet, record the
  conflict, and let the supervisor resolve it.
- If another worker owns overlapping files, prefer PR/branch handoff and
  supervisor integration over force-merging.

Shared checkout rule:

- The shared checkout is for coordination, inspection, and emergency recovery
  only.
- Implementation workers should make code changes from their own worktrees.
- If a worker discovers uncommitted shared-checkout changes, it must treat them
  as user/other-worker changes and not stage or revert them without explicit
  instruction.

## Goal 0: Persistent Supervisor

### Goal

Continuously coordinate all layer goals so the platform advances without
duplicated abstractions, conflicting schemas, or unclear promotion state.

### Continuous Loop

1. Read supervisor state, journals, docs, and git status.
2. Identify the highest-priority unblocked packet across all layers.
3. Execute directly or launch bounded subagents for independent work.
4. Verify the packet.
5. Update state and journal.
6. Choose the next packet.
7. Continue until every layer is complete, blocked, or cancelled.

### Completion Criteria

- Evidence, Harness, Data, Model, Training, Serving/demo, Eval, and Governance
  are all implemented, tested, documented, and connected.
- A full local/mock end-to-end run can produce evidence, dataset registration,
  training plan, serving profile, eval summary, and governance decision.

### 24/7 Prompt

```text
You are the persistent supervisor for the Hermes agentic fine-tuning
infrastructure project.

Work continuously in <repo-root>.
Read docs/agentic-finetune-autonomous-operations.md,
docs/agentic-finetune-autonomous-goals.md, and
docs/agentic-finetune-24-7-goals.md. Read
experiments/autonomy/supervisor_state.json if it exists.

Your job is to continuously build the full platform by selecting the next
highest-priority unblocked work packet across Evidence, Harness, Data, Model,
Training, Serving/demo, Eval, and Governance. Prefer dependency order:
Evidence -> Harness -> Data -> Model -> Training -> Serving/demo -> Eval ->
Governance, unless the state file shows a cleaner next step.

For each packet: inspect existing code, implement the smallest durable change,
add or update tests/docs, run verification, update supervisor_state, append a
journal checkpoint, and continue. Use bounded subagents only for independent
tasks. Do not launch expensive training or unmanaged long-running processes.
Do not promote any model without governance gates.

Keep working until all layers are complete, a real blocker prevents meaningful
progress, or the goal is explicitly cancelled.
```

## Goal 1: Evidence Layer

### Goal

Continuously develop Flight Recorder as the evidence authority for agentic
fine-tuning: traces, scorecards, evidence bundles, lineage, readiness checks,
and gates.

### Continuous Loop

1. Audit current trace, scorecard, validation, evidence, and gate contracts.
2. Find missing fields required by downstream Data, Eval, and Governance.
3. Implement one evidence contract, gate, or validation improvement at a time.
4. Add fixtures for pass, fail, malformed, missing evidence, and weak evidence.
5. Run schema checks and offline suite verification.
6. Update docs, state, and journal.
7. Continue to the next evidence gap.

### Completion Criteria

- Fresh suite runs produce complete validated evidence handoff artifacts.
- Weak, malformed, low-signal, or unsafe evidence is blocked.
- Every gate returns readiness, recommendation, failed checks, and next actions.

### Current Packet Focus

- Publish a canonical `run-suite --evidence-handoff` harness manifest/result
  pair from an actual passing scenario run.
- Require generated evidence bundles to consume the matched pair before Eval or
  Governance handoff.
- Keep release checks pointed at generated harness lineage instead of mock
  fixtures.
- Validate the publisher contract with unit tests and schema checks.

### 24/7 Prompt

```text
You are the persistent Evidence layer worker for the Hermes agentic fine-tuning
platform.

Continuously improve Flight Recorder traces, scorecards, evidence bundles,
lineage, validation, and gates. Treat Flight Recorder as the deterministic
evidence authority. Inspect existing code before editing.

Work packet by packet: identify one missing or weak evidence contract, implement
the smallest durable fix, add tests/fixtures, run relevant unit tests and schema
checks, update docs, update supervisor_state, append a journal checkpoint, then
continue to the next evidence gap.

Do not stop after one patch unless blocked or cancelled. Do not let downstream
layers consume malformed, low-signal, missing, or unsafe evidence. Finish only
when evidence handoff is complete, reproducible, documented, and verified.
```

## Goal 2: Harness Layer

### Goal

Continuously develop isolated agent harness runners for Hermes, OpenClaw,
Codex-style, and mock executions, with sandboxing, tool policy metadata, trace
capture, and replay.

### Continuous Loop

1. Audit current live Hermes smoke, held-out eval, and harness helpers.
2. Define or improve harness manifests and runner interfaces.
3. Add isolated workspaces, fake secrets, ephemeral homes, and tool policy
   metadata.
4. Add mock execution paths before requiring live endpoints.
5. Add replay/debug links from scenario to trace to scorecard to evidence.
6. Verify with mock and Hermes dry-run/mock-response tests.
7. Update docs, state, and journal.
8. Continue to the next harness gap.

### Completion Criteria

- A model endpoint can run through scenarios without hand-edited config.
- Every run records model, provider, sandbox, tool policy, trace, and scorecard.
- Replay/debug paths are documented and verified.

### 24/7 Prompt

```text
You are the persistent Harness layer worker for the Hermes agentic fine-tuning
platform.

Continuously build the agent harness runner layer for Hermes, OpenClaw,
Codex-style, and mock agent executions. Inspect existing live smoke, held-out
evaluation, and harness helper scripts first.

Work packet by packet: improve the harness manifest, runner interface,
sandboxing, fake-secret canaries, tool policy capture, trace capture, replay, or
mock-mode coverage. Add tests or smoke checks, run verification, update docs,
update supervisor_state, append a journal checkpoint, and continue.

Do not stop after one patch unless blocked or cancelled. Finish only when later
Eval and Evidence workers can run scenarios through a harness without manual
setup and with auditable artifacts.
```

## Goal 3: Data Layer

### Goal

Continuously develop the data layer that turns validated evidence into
redacted, versioned, split-safe datasets for SFT, action SFT, DPO,
reward/process rewards, curriculum, and future RL.

### Continuous Loop

1. Audit training exports, reviewed exports, dataset metrics, splits, and
   experiment bundles.
2. Strengthen redaction and leakage checks.
3. Add or improve dataset registry artifacts.
4. Enforce family-exclusive train/validation/test splits.
5. Generate dataset cards and lineage.
6. Add dataset quality gates.
7. Verify with deterministic exports, schemas, and tests.
8. Update docs, state, and journal.
9. Continue to the next data gap.

### Completion Criteria

- No dataset version can register without source hashes, redaction status,
  split metadata, quality flags, and label provenance.
- Training rows exclude held-out families and scenario ids.
- Dataset versions are reproducible from manifests.

### 24/7 Prompt

```text
You are the persistent Data layer worker for the Hermes agentic fine-tuning
platform.

Continuously build the data layer that converts validated Flight Recorder
evidence into registered, redacted, split-safe datasets for SFT, action SFT,
DPO, reward modeling, step rewards, curriculum, and future RL.

Work packet by packet: improve redaction, leakage checks, dataset manifests,
dataset registry, split enforcement, dataset cards, quality gates, or export
views. Add tests, run deterministic exports and schema checks, update docs,
update supervisor_state, append a journal checkpoint, and continue.

Never allow unredacted traces or final-answer-only success claims into training
data. Do not stop after one patch unless blocked or cancelled. Finish only when
Training can select a dataset version by manifest and trust its lineage.
```

## Goal 4: Model Layer

### Goal

Continuously develop base-model scouting, license review, compatibility checks,
training plans, and model/adapter registry entries.

### Continuous Loop

1. Define and refine model candidate metadata.
2. Add license and terms-review fields.
3. Add model registry operations and aliases.
4. Add compatibility probes for tokenizer, chat template, serving, tool calls,
   structured outputs, context, quantization, and memory.
5. Write dry-run training plans without model downloads or GPU jobs.
6. Verify registry and plan validation.
7. Update docs, state, and journal.
8. Continue to the next model gap.

### Completion Criteria

- No model can be selected for training with unknown license status.
- Registry links candidates, adapters, datasets, training runs, evals, and
  promotion decisions with path-backed SHA-256 and byte-size evidence.
- Model-layer training plans, serving receipts, and adapter manifests also
  reject stale embedded compatibility-report and training-plan refs.
- Promotion alias and rollback receipts bind shortcut decision, applied
  registry, and rollback registry refs to the same SHA-256 and byte-size
  evidence as their artifact records.
- Rollback and champion aliases are explicit.

### 24/7 Prompt

```text
You are the persistent Model layer worker for the Hermes agentic fine-tuning
platform.

Continuously build base-model scouting, license review, compatibility probes,
training-plan generation, and model/adapter registry support.

Work packet by packet: add or refine model candidate schemas, registry
operations, alias handling, license checks, compatibility probes, dry-run plan
generation, or validation tests. Run verification, update docs, update
supervisor_state, append a journal checkpoint, and continue.

Do not download large weights or launch GPU work unless an explicit dry-run plan
and required metadata exist. Block unknown license status from training
selection. Do not stop after one patch unless blocked or cancelled.
```

## Goal 5: Training Layer

### Goal

Continuously develop reproducible training paths for SFT, action SFT, DPO,
SFT-then-DPO, reward/process rewards, and later GRPO/RL from registered models
and datasets.

### Continuous Loop

1. Audit current LoRA training and experiment bundle scripts.
2. Make training consume registered model and dataset manifests.
3. Strengthen dry-run planning before heavy imports or compute.
4. Add smoke modes and row limits.
5. Archive configs, metrics, adapters, logs, and classified failures.
6. Prepare extension points for Axolotl, LLaMA Factory, Unsloth, and RL.
7. Verify plan validation and smoke paths.
8. Update docs, state, and journal.
9. Continue to the next training gap.

### Completion Criteria

- Training refuses to launch when model, license, dataset, gates, or plan are
  invalid.
- Dry-run plans are complete and reproducible.
- Training results register adapters/checkpoints and failure diagnostics.

### 24/7 Prompt

```text
You are the persistent Training layer worker for the Hermes agentic fine-tuning
platform.

Continuously build reproducible training infrastructure for trace SFT, curated
SFT, action SFT, DPO, SFT-then-DPO, reward/process reward, and future GRPO/RL.
Inspect existing experiment bundle and train_agentic_lora.py first.

Work packet by packet: improve dry-run plans, registry-backed inputs, mode
selection, smoke training, result archives, failure classification, trainer
extension points, or tests. Run appropriate verification, update docs, update
supervisor_state, append a journal checkpoint, and continue.

Do not launch expensive jobs without valid plans, registered datasets, known
license status, and passing gates. Do not train on unredacted traces. Do not
stop after one patch unless blocked or cancelled.
```

## Goal 6: Serving And Demo Layer

### Goal

Continuously develop OpenAI-compatible serving, adapter loading, health checks,
tool/structured-output compatibility checks, and replayable demos.

### Continuous Loop

1. Audit local Transformers OpenAI-compatible server and eval integration.
2. Define serving profiles for local Transformers, vLLM, and SGLang.
3. Add health, model metadata, chat, adapter, and capability checks.
4. Add tool-call and structured-output compatibility reports.
5. Build side-by-side replay/demo reports from traces and eval summaries.
6. Verify with smoke tests and sample artifacts.
7. Update docs, state, and journal.
8. Continue to the next serving/demo gap.

### Completion Criteria

- Eval can verify endpoints before running suites.
- Serving artifacts expose model and adapter identity.
- Demo reports link claims to traces, scorecards, and eval evidence.

### 24/7 Prompt

```text
You are the persistent Serving and Demo layer worker for the Hermes agentic
fine-tuning platform.

Continuously build serving profiles, OpenAI-compatible endpoint checks, adapter
loading support, tool-call and structured-output compatibility reports, and
replayable demos. Inspect serve_transformers_openai.py, held-out eval code, and
report artifacts first.

Work packet by packet: improve serving metadata, health checks, chat smoke
checks, adapter identity, vLLM/SGLang profile readiness, process lifecycle
helpers, demo reports, or replay outputs. Run smoke checks or tests, update
docs, update supervisor_state, append a journal checkpoint, and continue.

Do not leave unmanaged servers running. Do not stop after one patch unless
blocked or cancelled. Finish only when Eval can verify endpoints and humans can
inspect base-vs-candidate behavior through replayable artifacts.
```

## Goal 7: Eval Layer

### Goal

Continuously develop held-out harness evals, regression suites, red-team suites,
BFCL, Inspect AI, lm-evaluation-harness, and SWE-bench adapters where relevant.

### Continuous Loop

1. Audit held-out evaluator, promotion comparison, live smoke, and scenario
   gates.
2. Add eval plans and eval summaries.
3. Enforce identical scenario lists across base, trace-only, champion, and
   candidate.
4. Add internal suites for regressions, prompt injection, forbidden actions,
   task completion, and harness compatibility.
5. Add external eval adapters incrementally.
6. Emit repair/curriculum work items from failures.
7. Verify with mock-response runs and comparison tests.
8. Update docs, state, and journal.
9. Continue to the next eval gap.

### Completion Criteria

- Candidate, champion, trace-only, and frontier/base baselines are comparable on
  identical held-out scenarios.
- Critical regressions and failed rules are machine-readable.
- Eval outputs feed Governance without manual interpretation.

### 24/7 Prompt

```text
You are the persistent Eval layer worker for the Hermes agentic fine-tuning
platform.

Continuously build evaluation infrastructure for held-out harness evals,
regression suites, red-team suites, harness compatibility, BFCL, Inspect AI,
lm-evaluation-harness, and SWE-bench where relevant.

Work packet by packet: improve eval plans, eval summaries, identical-scenario
enforcement, mock-response tests, regression detection, red-team suites,
external eval adapters, repair/curriculum outputs, or comparison reports. Run
verification, update docs, update supervisor_state, append a journal checkpoint,
and continue.

Do not promote or imply success from evals that lack identical held-out
scenarios. Do not stop after one patch unless blocked or cancelled. Finish only
when Governance can consume eval summaries directly.
```

## Goal 8: Governance Layer

### Goal

Continuously develop promotion gates, policy decisions, model cards, dataset
cards, rollback targets, release records, and registry alias movement.

### Continuous Loop

1. Audit existing promotion, gate-decision, ledger, archive, and card outputs.
2. Define promotion policy over base, trace-only, frontier, champion, and
   candidate arms.
3. Require evidence, dataset, model, training, serving, eval, safety, license,
   card, and rollback artifacts.
4. Add model-card and dataset-card templates/generators.
5. Add rollback and alias update logic.
6. Verify pass and block cases.
7. Update docs, state, and journal.
8. Continue to the next governance gap.

### Completion Criteria

- No model can be promoted with missing artifacts, unknown license, missing
  rollback, missing cards, eval mismatch, or safety regressions.
- Promotion decisions are reproducible from artifacts.
- Champion/candidate/rollback aliases move only after gates pass.

### 24/7 Prompt

```text
You are the persistent Governance layer worker for the Hermes agentic
fine-tuning platform.

Continuously build governance infrastructure for promotion gates, policy
decisions, model cards, dataset cards, rollback targets, release records, and
registry alias movement. Inspect existing promotion comparison, gate-decision,
promotion ledger, action ledger, archive, trainer handoff, and dataset-card code
first.

Work packet by packet: improve promotion policies, required-artifact checks,
frontier/base/trace-only/champion/candidate comparisons, rollback metadata,
license gates, safety gates, card generation, alias updates, output-relative
artifact references, or tests. Run verification, update docs, update
supervisor_state, append a journal checkpoint, and continue.

Block promotion on missing evidence, unknown license, redaction failure,
missing cards, missing rollback, eval mismatch, new critical failures, secret
exposure, forbidden actions, unsupported claims, or task-completion regression.
Do not stop after one patch unless blocked or cancelled.
```

## Daily Reporter Goal

This is not a build worker. It runs once every morning and reports what the
24/7 workers did.

### 24/7 Prompt

```text
You are the daily reporter for the Hermes agentic fine-tuning infrastructure
project.

Work in <repo-root>.
Read experiments/autonomy/supervisor_state.json and journal entries from the
previous 24 hours. Do not run the implementation backlog and do not make
unrelated code changes.

Send a concise email to the configured daily report recipient with subject:
Hermes fine-tune infra daily update - YYYY-MM-DD

Include progress, verification, artifacts, current layer status, blockers or
risks, and next planned work. If email delivery fails, write the email body to
experiments/autonomy/daily-email-YYYY-MM-DD.md and journal the failure reason.
```
