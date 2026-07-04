# Autonomous Goals For Agentic Fine-Tuning Infrastructure

Use these goals as persistent autonomous work briefs. Each one is scoped to a
layer from the component map and can be run by Codex goal workers, local agent
loops, or task runners until the layer is complete, blocked, or cancelled.

The daily email is only a reporting mechanism. It summarizes what the agents
completed in the previous 24 hours; it is not the unit of work and should not
cause workers to stop.

For the recommended daily supervisor and email reporting pattern, see
`docs/agentic-finetune-autonomous-operations.md`.
For copy-pasteable 24/7 worker prompts, see
`docs/agentic-finetune-24-7-goals.md`.

The goals assume `hermes-flight-recorder` is the evidence nucleus. Each layer
should produce durable artifacts, tests, and verification evidence rather than
only a prose plan.

## Global Instructions For Every Goal

Every autonomous run should:

- inspect the existing repository before changing code
- prefer existing Flight Recorder patterns and schemas
- keep diffs small, reviewable, and reversible
- write or update docs for new commands and artifacts
- add regression tests before changing protected behavior
- avoid new dependencies unless the layer genuinely needs them
- write dry-run plans before expensive training, serving, or eval work
- never train on unredacted traces or final-answer-only success claims
- verify with unit tests, CLI smoke tests, schema checks, and artifact checks
- finish with changed files, commands run, verification evidence, and remaining
  risks
- perform implementation work in a dedicated Codex worktree or dedicated git
  worktree
- commit verified repository changes to a goal-specific branch, push that
  branch, integrate it to `main` under the configured local main-push lock,
  push `main`, and close or delete the temporary branch
- never push unverified changes to `main`
- treat the repository as public: never commit personal email addresses,
  home-directory paths, machine-specific workspace paths, private Codex state,
  API keys, or local automation configuration
- keep autonomy state, journals, and daily report recipient configuration in
  ignored local files unless a sanitized example is intentionally added

Suggested execution order:

1. Evidence layer
2. Harness layer
3. Data layer
4. Model layer
5. Training layer
6. Serving/demo layer
7. Eval layer
8. Governance layer

The order can overlap, but do not promote models until Evidence, Data, Eval,
Serving, and Governance have working gates.

## Goal 1: Evidence Layer

### Autonomous Objective

Create the evidence foundation for agentic fine-tuning: traces, normalized
traces, scorecards, evidence bundles, gates, lineage, and deterministic
readiness checks that decide whether a run can become training or promotion
evidence.

### Build Scope

- Audit current Flight Recorder trace, scorecard, evidence-bundle, and gate
  commands.
- Identify missing schemas or fields needed by later layers.
- Strengthen evidence coverage, trace observability, scenario quality, and
  gate outputs where needed.
- Validate harness manifest, result, replay, and suite handoffs before public
  use so preserved scenario, sandbox, trace, scorecard, report, lineage, or
  run-artifact paths warn under strict validation.
- Validate run lineage before public handoff so preserved output paths, replay
  args, commands, and input fingerprint paths warn under strict validation.
- Validate replay bundles before public handoff so preserved source lineage and
  copied-input source display paths warn under strict validation.
- Validate trainer preflights before public use so preserved trainer-command
  raw or argv tokens warn under strict validation.
- Validate trainer launch checks before public use so preserved approved-command
  raw, argv, or shell tokens are rejected when they carry local absolute paths.
- Validate trainer archives before public use so preserved archive source paths,
  approved-command raw, argv, or shell tokens warn under strict validation.
- Validate trainer archive checks before public use so preserved archive roots,
  external code roots, or resolved paths warn under strict validation, while
  portable command tokens are rejected when they carry local absolute paths.
- Validate trainer consumer plans before public use so preserved archive roots,
  external code roots, or argv tokens are rejected when they carry local
  absolute paths.
- Validate trainer wrapper dry-run receipts before public use so preserved
  would-run roots or argv tokens are rejected when they carry local absolute
  paths.
- Validate agentic training plans before public use so preserved external
  runner command tokens are rejected when they carry local absolute paths.
- Validate agentic training flow handoffs before public use so preserved
  delegated command cwd, archive roots, external code roots, or argv tokens
  are rejected when they contain absolute local paths.
- Validate cloud-training launch plans before public use so preserved dry-run
  command tokens are rejected when they contain absolute local paths.
- Validate scenario-check receipts before public handoff so preserved scenario,
  trace, or state source paths warn under strict validation.
- Validate review-calibration receipts before public handoff so preserved
  reviewed-export or reviewed-label source paths warn under strict validation.
- Validate review-export queues before public handoff so preserved per-item run,
  report, trace, scorecard, lineage, or regression refs warn under strict
  validation.
- Validate reviewed-label exports before public handoff so preserved per-row
  label files or inherited source artifact refs warn under strict validation.
- Add a single evidence-readiness command or documented command sequence that
  produces a complete handoff bundle from runs.
- Ensure failed evidence produces repair/curriculum work items instead of
  silently passing downstream.

### Required Deliverables

- Stable artifact contract for:
  - `normalized_trace`
  - `scorecard`
  - `run_digest`
  - `evidence_bundle`
  - `training_export`
  - `gate_result`
- CLI path or script for evidence handoff.
- Tests for pass, fail, missing evidence, weak evidence, and malformed artifact
  cases.
- Documentation showing the evidence chain from raw run to training handoff.

### Acceptance Criteria

- A fresh suite run can produce validated evidence artifacts in one documented
  sequence.
- Evidence gates block low-signal traces, missing required evidence, critical
  failures, and malformed bundles.
- Every gate result includes machine-readable recommendation, readiness,
  failed checks, and next actions.
- Later layers can consume artifact paths, hashes, and byte sizes without
  guessing.
- Repair queues keep replayed source fingerprints relative to the queue
  location and warn before public handoffs include absolute source artifact
  display paths or replay command tokens.
- Live-smoke summaries warn before runtime output paths or environment roots
  are published as public evidence.
- Scenario-quality artifacts warn before scenario, trace, or state source paths
  are published as public handoff evidence.
- Evidence bundles and improvement plans reject existing file artifacts that
  omit SHA-256 or byte-size evidence at the schema boundary, and validation
  rejects symlinked source paths before trusting those fingerprints.
- Evidence-bundle producers reject symlinked input artifacts, export manifests,
  and run-digest sources before reading JSON, hashing evidence, or summarizing
  handoff metrics.
- Agentic training loop plans skip symlinked source artifact payloads before
  deriving receipt state, lineage, or source-validation snapshots.
- Agentic loop ledgers reject symlinked source loop plan paths before trusting
  plan size, hash, artifact lineage, or receipt-state snapshots.
- Improvement ledgers bind each source improvement plan to SHA-256 plus byte
  size evidence before downstream gates trust recurring-work metrics, and
  reject symlinked source plan parents before reading or validating those refs.
- Action, improvement, and promotion ledger gates reject symlinked source
  ledger paths before replaying metrics, checks, or decisions.
- Decision gates reject symlinked source artifacts before trusting
  `source_artifact` fingerprints or replayed source decisions.
- Promotion ledgers reject symlinked recorded decision-gate paths before
  trusting record fingerprints or replayed gate contents.
- Promotion cards reject symlinked required input artifacts before reading or
  fingerprinting card source evidence.
- Promotion decisions and release records reject symlinked promotion-policy
  paths before reading or fingerprinting policy artifacts.
- Promotion decisions reject required source artifacts and card files that
  traverse symlinked parents before reading, hashing, or binding those refs.
- Promotion release records reject required source artifacts and release notes
  that traverse symlinked parents before reading or binding those refs.
- Promotion alias-apply and rollback receipts reject symlinked registry or
  promotion-decision inputs before hashing, replaying, or mutating aliases.
- Next-iteration schedules reject symlinked source ledger paths before trusting
  source ledger size, hash, metrics, or decision snapshots.
- Governance receipts reject symlinked source loop ledger paths before trusting
  replayed readiness digests, execution boundaries, or decisions.

### Verification

- Run unit tests covering changed evidence code.
- Run a deterministic offline suite.
- Run schema validation on generated artifacts.
- Run gate commands and confirm at least one passing and one blocking case.

### Ready-To-Run Prompt

```text
Build the Evidence layer for the agentic fine-tuning platform.

Work inside hermes-flight-recorder. Treat Flight Recorder as the deterministic
artifact authority. Inspect existing trace normalization, scorecard, evidence
bundle, validation, and gate code first. Then implement the smallest set of
changes needed so a fresh run suite can produce a complete, validated evidence
handoff for training and promotion loops.

Deliver stable artifacts for normalized traces, scorecards, run digests,
evidence bundles, training exports, and gate results. Gates must block
low-signal traces, missing evidence, malformed artifacts, and critical failures.
Every gate result must include readiness, recommendation, failed checks, and
next actions.

Add or update tests for pass, fail, missing evidence, weak evidence, and schema
failure cases. Update docs with the exact command sequence. Run the offline
suite, schema checks, and relevant tests. Finish only when the evidence handoff
is reproducible and verified.
```

## Goal 2: Harness Layer

### Autonomous Objective

Create a harness runner layer that can run Hermes, OpenClaw, Codex-style, or
mock agent tasks in isolated sessions, capture traces, apply tool policies, and
replay results for scoring and debugging.

### Build Scope

- Audit existing Hermes harness and live smoke scripts.
- Define a common harness-run manifest for model, provider, base URL, tools,
  scenario, workspace, policy, and output paths.
- Implement or document a runner interface for:
  - `run_scenario`
  - `run_suite`
  - `probe_model`
  - `replay_trace`
- Add sandbox defaults: ephemeral home, fake secrets, isolated workspace,
  network/tool policy metadata, and deterministic mock tools where available.
- Preserve compatibility with current live Hermes smoke/eval scripts.

### Required Deliverables

- Harness run manifest schema or documented JSON shape.
- CLI or script that can run at least mock and Hermes-backed scenarios.
- Tool policy configuration captured in artifacts.
- Replay/debug path that links traces back to scenario, scorecard, and evidence
  refs.
- Tests or smoke tests for mock execution, blocked action capture, and trace
  artifact generation.

### Acceptance Criteria

- A model endpoint can be evaluated through the harness without hand-edited
  config files.
- Each run uses isolated filesystem state and fake-secret canaries.
- Tool policy and environment metadata are recorded in the output artifacts.
- Replay can reproduce or inspect the scored evidence path.

### Verification

- Run mock harness smoke tests.
- Run Hermes held-out evaluator in dry-run or mock-response mode.
- Confirm generated artifacts contain scenario id, model id, provider, tool
  policy, sandbox paths, trace path, and scorecard path.

### Ready-To-Run Prompt

```text
Build the Harness layer for the agentic fine-tuning platform.

Work inside hermes-flight-recorder and inspect existing live Hermes smoke,
held-out evaluation, and harness helper scripts first. Create a common harness
run manifest and the minimal runner interface needed for run_scenario,
run_suite, probe_model, and replay_trace.

The runner must support isolated workspaces, ephemeral homes, fake-secret
canaries, model/provider/base-url configuration, tool policy metadata, and trace
capture. Preserve the current Hermes eval path, and add mock-mode support where
needed so the layer can be tested without external model calls.

Deliver docs, tests or smoke tests, and generated sample artifacts. Verify with
mock execution and a Hermes evaluator dry-run or mock-response run. Finish only
when a later eval loop can invoke the harness without manual setup.
```

## Goal 3: Data Layer

### Autonomous Objective

Create the data layer that turns validated traces into versioned, redacted,
split-safe datasets for SFT, action SFT, DPO, reward modeling, process rewards,
and future RL.

### Build Scope

- Audit existing training export, reviewed export, dataset metrics, dataset
  split, and experiment-bundle scripts.
- Strengthen redaction and contamination checks.
- Define a local dataset registry layout or SQLite schema.
- Ensure family-exclusive train/validation/test split metadata is enforced.
- Generate dataset cards and lineage records for every dataset version.
- Add quality gates for dataset size, balance, leakage, trace signal, and label
  provenance.

### Required Deliverables

- Dataset registry artifact with versions, hashes, source runs, split metadata,
  and labels.
- Redaction check or gate before dataset registration.
- Dataset-card generator or documented output.
- Exports for SFT, action SFT, DPO, reward model, step reward, and curriculum.
- Tests for redaction, split leakage, malformed rows, and registry updates.

### Acceptance Criteria

- No dataset can be registered without source artifact hashes and redaction
  status.
- Dataset registry schemas distinguish training exports from reviewed exports
  and reject missing variant-specific source evidence.
- Held-out task families and scenario ids are excluded from training rows.
- Dataset quality flags are visible to training and promotion loops.
- Dataset versions are reproducible from manifests.

### Verification

- Run training export on deterministic runs.
- Validate generated schemas and dataset metrics.
- Run tests covering split leakage and redaction blocking.
- Build an experiment bundle from the registered dataset.

### Ready-To-Run Prompt

```text
Build the Data layer for the agentic fine-tuning platform.

Work inside hermes-flight-recorder. Inspect existing training export,
reviewed export, dataset metrics, dataset split, redaction, and experiment
bundle code first. Implement the smallest durable dataset registry that records
dataset versions, source artifact hashes, split metadata, label provenance,
redaction status, quality flags, and dataset cards.

The layer must export SFT, action SFT, DPO, reward-model, step-reward, and
curriculum views only from validated evidence. Enforce family-exclusive heldout
splits and block train/eval leakage. No unredacted trace or final-answer-only
success claim may enter registered training data.

Add tests for redaction, split leakage, malformed rows, and registry updates.
Run deterministic training export, schema validation, and experiment bundle
generation. Finish only when a training loop can select a dataset version by
manifest and trust its lineage.
```

## Goal 4: Model Layer

### Autonomous Objective

Create the model layer that scouts base models, records license and runtime
compatibility, writes training plans, and maintains adapter/model registry
entries for candidates, champions, and rollback targets.

### Build Scope

- Define `model_candidate`, `model_registry_entry`, and `training_plan`
  schemas or documented JSON shapes.
- Implement a local registry under `experiments/registry/` or SQLite.
- Add a model-scout manifest with initial candidates and required metadata.
- Add compatibility probes for tokenizer/chat template, serving engine,
  structured output, tool-call behavior, context length, and memory notes.
- Add license-review fields and block unknown license status from training.

### Required Deliverables

- Model candidate manifest.
- Model registry with aliases such as `candidate`, `champion`, and `rollback`.
- Training-plan artifact format that references model, dataset, trainer,
  hyperparameters, output paths, compute assumptions, and dry-run status.
- CLI/script for registering, listing, and validating model candidates.
- Tests for registry insert/update, alias movement, missing license, and invalid
  training plans.

### Acceptance Criteria

- A model cannot be selected for training without license status, source, model
  id, and compatibility metadata.
- A training plan can be written without downloading weights or launching a GPU
  job.
- Registry entries link model candidates to datasets, training runs, evals, and
  promotion decisions with path-backed SHA-256 and byte-size evidence.
- Training plans, serving receipts, and adapter manifests bind their embedded
  compatibility-report and training-plan refs with matching byte-size evidence.
- Model-layer path-backed links and embedded refs reject symlinked parent
  components before trusting SHA-256 or byte-size evidence.
- Rollback target is always explicit for promoted models.

### Verification

- Validate sample model-candidate manifests.
- Run registry tests.
- Generate a dry-run training plan for at least one local/small candidate.
- Confirm missing license status blocks registration or training selection.

### Ready-To-Run Prompt

```text
Build the Model layer for the agentic fine-tuning platform.

Work inside hermes-flight-recorder. Create a local model registry and model
candidate format that records model id, source, license status, accepted terms,
context length, tokenizer/chat template notes, serving compatibility,
tool-calling/structured-output support, quantization options, memory notes, and
review status.

Implement a model-scout manifest and registry commands or scripts for
registering, listing, validating, and aliasing models. Add a training-plan
artifact that can be produced in dry-run mode before any heavy ML import,
download, or GPU job. Block candidates with unknown license status from
training selection.

Add tests for registry updates, alias movement, invalid candidates, missing
license status, and dry-run training-plan validation. Finish only when the
Training layer can choose a model candidate from the registry and produce a
verified plan without manual metadata lookup.
```

## Goal 5: Training Layer

### Autonomous Objective

Create the training layer that can launch reproducible SFT, action SFT, DPO,
reward/process reward, and later GRPO/RL jobs from registered models and
datasets, starting with dry-run and smoke training before expensive jobs.

### Build Scope

- Audit current `train_agentic_lora.py` and experiment-bundle scripts.
- Make training consume registry-backed model candidates and dataset versions.
- Keep TRL/PEFT local training path for MVP.
- Add dry-run validation for all modes before importing heavy ML dependencies.
- Add smoke mode with tiny row limits and small local model support if feasible.
- Add archive artifacts for configs, metrics, adapters, logs, and failures.
- Design extension points for Axolotl/LLaMA Factory/Unsloth recipes and later
  GRPO/RL.

### Required Deliverables

- Training plan schema or JSON shape.
- Training result schema or JSON shape.
- CLI/script path for:
  - dry-run plan
  - smoke training
  - full SFT
  - action SFT
  - DPO
  - SFT-then-DPO
- Registry update after training completion or classified failure.
- Tests for plan validation, missing data, mode selection, and result archive.

### Acceptance Criteria

- Training fails before launch if dataset gates, model license, or plan
  validation fail.
- Dry-run produces the exact model, dataset, mode, hyperparameters, output
  paths, tracking config, and compute assumptions.
- Smoke training can be run with a row limit without changing the full plan.
- Training results register adapters/checkpoints and failure diagnostics.
- Completed result registry links distinguish the training-run receipt from
  adapter artifact links; adapter links carry SHA-256 and byte-size evidence at
  the schema boundary.

### Verification

- Run dry-run plans for trace SFT and curated SFT-DPO.
- Run unit tests for training-plan and result validation.
- If local hardware permits, run a tiny smoke training job.
- Validate result archive and registry link.

### Ready-To-Run Prompt

```text
Build the Training layer for the agentic fine-tuning platform.

Work inside hermes-flight-recorder. Inspect the existing experiment bundle and
train_agentic_lora.py first. Refactor or extend the training path so it consumes
registered model candidates and registered dataset versions, writes a complete
training plan in dry-run mode, and archives a training result after success or
classified failure.

Support trace SFT, curated SFT, action SFT, DPO, and SFT-then-DPO for MVP.
Design the result format so reward-model, process-reward, and GRPO/RL jobs can
be added later without rewriting the registry. Do not launch expensive jobs
unless the plan, dataset gates, and license checks pass. Add smoke mode with
row limits.

Add tests for plan validation, missing data, invalid modes, failed gates, and
result archive. Run dry-run plans for trace SFT and curated SFT-DPO. Run tiny
smoke training only if local hardware and dependencies make it practical.
Finish only when training outputs are reproducible and linked back to model and
dataset registry entries.
```

## Goal 6: Eval Layer

### Autonomous Objective

Create the eval layer that runs candidates through held-out harness evals,
regression suites, red-team suites, and external eval adapters, then writes
comparison-ready summaries for promotion gates.

### Build Scope

- Audit held-out evaluator and promotion comparison scripts.
- Make eval consume registered model candidates, serving endpoints, scenario
  sets, and dataset split metadata.
- Add eval-plan and eval-result artifacts.
- Integrate internal suites first:
  - held-out scenarios
  - prompt-injection and forbidden-action suites
  - task-completion evidence suites
  - regression scenarios
  - harness compatibility smoke tests
- Add adapter stubs or first implementations for BFCL, Inspect AI,
  lm-evaluation-harness, and SWE-bench where relevant.

### Required Deliverables

- Eval plan artifact.
- Eval summary artifact with pass rate, average score, critical failures,
  failed rules, task-completion metrics, cost, latency, and scenario ids.
- CLI/script for baseline, trace-only, candidate, and champion comparisons.
- Red-team eval suite or command group.
- Tests for identical-scenario enforcement and regression detection.

### Acceptance Criteria

- Candidate evals are run on identical held-out scenarios as baseline and
  trace-only arms.
- Eval summaries are comparable without manual path edits.
- External eval plans bind scenario-manifest references to plan-relative
  SHA-256 and byte-size evidence.
- Any new critical failure is visible to governance gates.
- Failed evals generate repair/curriculum work items.

### Verification

- Run evaluator in mock-response mode.
- Compare mock baseline, trace-only, and candidate summaries.
- Run tests for scenario-list mismatch and critical-regression detection.
- Validate eval summary artifacts.

### Ready-To-Run Prompt

```text
Build the Eval layer for the agentic fine-tuning platform.

Work inside hermes-flight-recorder. Inspect the existing held-out evaluator,
promotion comparison, live smoke, and scenario/gate code first. Create eval
plan and eval summary artifacts that can compare baseline, trace-only,
champion, and candidate models on identical held-out scenarios.

Implement or wire internal eval suites for held-out scenarios, prompt injection,
forbidden actions, task-completion evidence, regressions, and harness
compatibility. Add adapter stubs or first-pass integrations for BFCL, Inspect
AI, lm-evaluation-harness, and SWE-bench only where the repo can support them
cleanly.

Eval summaries must include scenario ids, pass rate, average score, critical
failures, failed rules, task-completion metrics, cost, latency, model metadata,
and artifact hashes. Failed evals should emit repair/curriculum work items.

Verify with mock-response evaluator runs, comparison tests, scenario mismatch
tests, and schema/artifact validation. Finish only when Governance can consume
eval summaries without manual interpretation.
```

## Goal 7: Serving And Demo Layer

### Autonomous Objective

Create the serving and demo layer that exposes base models and fine-tuned
candidates through OpenAI-compatible endpoints, validates tool/structured output
compatibility, and provides replayable demos comparing baseline and candidate
behavior.

### Build Scope

- Audit the local Transformers OpenAI-compatible server.
- Add serving profiles for local Transformers and future vLLM/SGLang.
- Define `serving_profile` and `demo_run` artifact shapes.
- Support adapter loading, model metadata, health checks, streaming status,
  structured-output capability notes, and tool-call compatibility checks.
- Add side-by-side replay/demo output from existing traces and eval summaries.
- Add process lifecycle helpers for eval loops where practical.

### Required Deliverables

- Serving profile artifact.
- Server smoke checks for health, models, chat completion, and model metadata.
- Tool-call and structured-output compatibility report.
- Demo/replay report comparing base, trace-only, champion, and candidate.
- Docs for serving base model and adapter candidates.

### Acceptance Criteria

- Eval loops can start or reference a serving endpoint with known model id and
  profile metadata.
- Health and chat smoke checks pass before eval launch.
- Adapter and base model identity are visible in generated artifacts.
- Demo output links each behavior claim to trace, scorecard, and eval evidence.

### Verification

- Run local server help/dry-run where available.
- Run mock or lightweight server smoke tests.
- Validate serving profile artifacts.
- Generate a demo/replay report from existing sample runs.

### Ready-To-Run Prompt

```text
Build the Serving and Demo layer for the agentic fine-tuning platform.

Work inside hermes-flight-recorder. Inspect the existing
serve_transformers_openai.py, held-out evaluator, and report/replay artifacts
first. Define serving profile and demo run artifacts. Extend the local serving
path so eval loops can reliably identify base models and adapters, check
health, test chat completion, record metadata, and report tool-call or
structured-output compatibility.

Use the local Transformers server for MVP and design the profile so vLLM or
SGLang can be added later. Add demo/replay outputs that compare base,
trace-only, champion, and candidate behavior using traces, scorecards, and eval
summaries.

Add smoke tests or script checks for health, models, chat completion, metadata,
and artifact generation. Finish only when the Eval layer can verify an endpoint
before running a suite and humans can inspect a replayable candidate demo.
```

## Goal 8: Governance Layer

### Autonomous Objective

Create the governance layer that turns evidence, data, model, training, eval,
and serving artifacts into deterministic promotion, rejection, rollback, model
card, and dataset card decisions.

### Build Scope

- Audit promotion comparison, gate-decision, promotion ledger, action ledger,
  trainer handoff, and dataset-card outputs.
- Define a top-level promotion policy that compares:
  - base model
  - trace-only fine-tune
  - current champion
  - candidate
- Add model-card and dataset-card requirements.
- Add rollback target requirement.
- Add license, redaction, eval, safety, latency, and cost gates.
- Add promotion ledger entries and release notes.

### Required Deliverables

- Promotion policy artifact.
- Promotion decision artifact.
- Documented `promotion-decision` command and validation sequence.
- Documented `promotion-cards` command and validation sequence.
- Documented registry alias receipt command and validation sequence.
- Documented release-record command and validation sequence.
- Model card template/output.
- Dataset card template/output.
- Rollback metadata and champion/candidate alias update path.
- Tests for promotion pass, blocked critical regression, missing card, missing
  rollback, license failure, eval mismatch, card generation, and alias movement
  receipts.
- Tests for release records binding decisions, cards, alias receipts, rollback,
  evals, and release notes.

### Acceptance Criteria

- No model can be promoted without evidence bundle, dataset version, model
  registry entry, training result, eval summaries, serving profile, model card,
  dataset card, and rollback target.
- Candidate must beat or tie configured baselines according to policy.
- New secret-exposure, forbidden-action, unsupported-claim, or critical
  task-completion regressions block promotion.
- Promotion decisions are reproducible from artifacts.
- Evidence-bundle validation reopens recorded eval-summary and serving-lifecycle
  artifacts from the bundle file location before trusting derived metrics, and
  rejects existing artifact paths that are symlinks or traverse symlinked parent
  directories before trusting file or directory evidence.
- Action-ledger validation reopens recorded evidence bundles from the ledger
  file location, requires SHA-256 plus byte-size evidence, and rejects missing,
  moved, stale, malformed, symlinked, or cwd-substituted bundle records.
- Scenario-authored trace and state snapshot paths resolve from the scenario
  file location; only explicit CLI overrides resolve from the process working
  directory.
- State-snapshot schemas reject existing file records and directory-listing
  file entries that omit SHA-256 or byte-size evidence while preserving
  missing-file and directory diagnostics; validation rejects stale captured
  file sizes when paths resolve.
- Lineage replay fingerprints include input sizes; replay and replay-bundle
  validation reject size drift or symlinked copied inputs between recorded
  inputs and replay metadata.
- Run and runs-directory validation reject symlinked roots before trusting
  generated trace, scorecard, report, or lineage artifacts.
- Suite-summary validation rejects run artifact refs that traverse symlinked
  parents before trusting recorded report, scorecard, digest, or lineage
  fingerprints.
- Harness replay receipts bind lineage and scorecard paths to SHA-256 plus
  byte-size evidence and require non-symlink replay output directories;
  validation rejects stale replay references before trusting pass/fail summaries.
- Harness run results from both the common publishers and installed
  `hermes-harness` entry point bind required artifact paths to SHA-256 plus
  byte-size evidence; validation rejects stale trace, scorecard, digest, report,
  or lineage artifact references before trusting the run receipt.
- Suite summaries bind each run's report, scorecard, run digest, and lineage
  refs to SHA-256 plus byte-size evidence; validation rejects stale resolvable
  run artifact refs before downstream exports or bundles trust the suite.
- Training-export and suite-comparison source fingerprints include input sizes;
  verified source-fingerprint coverage requires scenario and trace hashes plus
  sizes.
- Agentic training-result lineage refs include input sizes; validation rejects
  regular plan or runtime-preflight lineage refs without matching SHA-256 and
  byte-size evidence resolved from the receipt location, rejects refs that
  traverse symlinked parent directories before replaying flow or plan-derived
  manifest checks, and rejects model or dataset manifest summaries that no
  longer match the verified training plan.
- Result-receipt archiving rejects symlinked plan, runtime-preflight, or
  delegated-flow source inputs before reading them or emitting lineage hashes.
- Agentic training-result registry proposals include adapter/checkpoint byte
  sizes; validation rejects hashed registry links that are not backed by a
  supplied artifact ref with matching size evidence and expected registry
  target, and detects missing duplicate output-artifact links.
- Agentic training plans and runtime preflights also bind registered manifests
  and selected trainer views to SHA-256 plus byte-size evidence.
- Runtime-preflight generation rejects symlinked plan inputs and blocks selected
  trainer views that resolve through symlinks without hashing those views.
- Delegated-flow generation rejects symlinked plan, runtime-preflight, or
  trainer-consumer-plan source inputs before emitting source-artifact hashes.
- Trainer-consumer-plan generation rejects symlinked archive-check source
  inputs before reading them or emitting source-archive-check hashes.
- Trainer archive checks reject passed external-code file checks that omit
  SHA-256 or byte-size evidence while preserving missing-code diagnostics.
- Trainer archive checks reject passed trainer-input checks that omit SHA-256,
  byte-size, or directory file-count evidence while preserving failed-input
  diagnostics.
- Trainer preflight artifact records reject regular file or directory artifacts
  that omit SHA-256 and byte-size evidence while preserving non-regular artifact
  diagnostics.
- Trainer preflight schema-contract records reject regular files that omit
  SHA-256 or byte-size evidence while preserving non-regular contract
  diagnostics.
- Trainer preflight validation-summary records reject regular files that omit
  SHA-256 or byte-size evidence while preserving non-regular summary
  diagnostics.
- Trainer preflight gate records reject existing gate files that omit SHA-256
  or byte-size evidence while preserving missing-gate diagnostics.
- Trainer consumer plans reject passed trainer-input and external-code records
  that omit SHA-256 or byte-size evidence, or carry malformed expected
  SHA-256 values, while preserving blank failed-input diagnostics.
- Trainer wrapper dry-run receipts reject passed trainer-input and external-code
  records that omit SHA-256, byte-size, or directory file-count evidence, or
  carry malformed expected SHA-256 values, while preserving blank failed-input
  diagnostics.
- Harness run manifests reject fake-secret canary declarations whose stored
  SHA-256 fingerprints are missing or malformed.
- Harness suite receipts bind each run's manifest and result references to
  SHA-256 and byte-size evidence, and validation rejects stale or symlinked
  suite-level references before evidence handoff.
- Harness run results reject fake-secret canary checked-artifact records whose
  existing files omit, forge, go missing, or stale out SHA-256 and byte-size
  evidence, reject symlinked run/replay/canary artifact refs, and reject canary
  summaries whose counts, leak records, or pass flags disagree with the checked
  artifacts; refresh older harness-result fixtures before validating them.
- Model registry entry validation rejects path-backed link records whose
  SHA-256 fingerprints are not lowercase hex digests.
- External-eval plans reject scenario-manifest references whose verified file
  is not a held-out scenario manifest.
- External-eval plans reject scenario-manifest readiness and scenario-count
  metadata that no longer matches the verified manifest file.
- Evidence bundles reject stale existing file-artifact fingerprints and reject
  missing or non-regular existing file and directory artifact paths when those
  paths are expected to exist during validation; decision next-actions must
  also reference declared bundle artifacts, metrics, or the bundle itself
  before handoff instructions are trusted.
- Evidence-bundle decisions must mirror the failed bundle checks and gate
  metrics they summarize, so blocking-check and blocking-gate rows cannot be
  stale, renamed, or substituted while keeping only aggregate blocker counts
  correct.
- Evidence-bundle decision key metrics are recomputed from bundle metrics during
  validation, blocking stale or hand-edited executive summaries that disagree
  with the underlying evidence sections.
- Evidence-bundle decision summary text is recomputed from bundle readiness and
  failed checks during validation, blocking hand-edited prose that claims a
  safer or clearer handoff state than the evidence supports.
- Evidence-bundle next actions are recomputed from blocking checks, failed
  gates, and metrics during validation, blocking forged remediation plans even
  when their fingerprints and routing keys have been refreshed.
- Evidence-bundle public notes are validated against the producer-defined text,
  blocking hand-edited caveats that overstate scoring, mutation, sandboxing, or
  training guarantees.
- Evidence-bundle strict validation now warns on absolute top-level bundle paths,
  catching preserved or forged local paths before public evidence packages are
  accepted.
- Action-ledger strict validation now warns on absolute ledger, bundle, metric,
  and occurrence paths, catching preserved local evidence coordinates before
  public action-review packets are accepted.
- Action-ledger gate strict validation now warns on absolute source ledger and
  policy paths, catching preserved local gate coordinates before public
  action-review decisions are accepted.
- Improvement-ledger strict validation now warns on absolute ledger, plan,
  metric, and occurrence paths, catching preserved local improvement-plan
  coordinates before public iteration-review packets are accepted.
- Improvement-ledger gate validation reopens and replays the referenced source
  ledger, rejecting stale or forged metrics, checks, decisions, missing source
  ledgers, and absolute public source/policy paths.
- Eval summaries bind suite, compare-manifest, compare-gate,
  external-adapter, and serving-preflight source refs to SHA-256 and byte-size
  evidence, and validation rejects stale source artifacts before Governance
  consumes summarized claims.
- Eval-summary strict validation now warns on absolute suite, compare-manifest,
  compare-gate, external-adapter, and serving-preflight refs before public
  governance summaries are accepted.
- Suite-summary strict validation now warns on absolute scenario, trace, run,
  and run-artifact paths before public eval summaries are accepted.
- Evidence-coverage and trace-observability strict validation now warn on
  absolute per-run directories before public diagnostics are accepted.
- External-eval strict validation now warns on absolute heldout-manifest and
  source-plan refs before adapter plans or dry-run receipts become public
  governance evidence.
- Heldout-manifest validation replays source suite summaries from the manifest
  location, rejecting stale scenario sets or fingerprints while strict
  validation warns on absolute source refs.
- Rollout-receipt strict validation now warns on absolute source-plan refs
  before mock rollout evidence is admitted to rejection-sampling gates.
- Rollout-plan strict validation now warns on absolute plan, scenario, and
  verifier-config refs before mock rollout batches are admitted to receipts.
- Cloud-training artifact manifests, preflights, launch plans, launch receipts,
  and status receipts reopen path-backed upload/source refs from their own file
  location and reject symlinked, stale SHA-256, or byte-size evidence before
  handoff; derived source-readiness and provider-chain state also skip refs
  that traverse symlinked components. Launch-plan validation also rejects
  absolute dry-run command tokens before cloud-training handoffs become public
  evidence.
- Agentic training loop plans reopen existing file source refs from the plan
  location and reject missing, moved, stale, symlinked, or cwd-substituted loop
  inputs before orchestration trusts phase readiness.
- Repair queue items include source artifact fingerprints for normalized
  traces, scorecards, and reports; validation reopens those refs from the queue
  location and rejects stale or moved repair evidence before work dispatch.
- Model-grader rubric, dry-run, override, and gate artifacts emit
  output-relative source refs with SHA-256 plus byte-size evidence; validation
  reopens those refs from the artifact location while training, review,
  reviewed, rubric, and calibration entrypoints reject symlinked roots or
  top-level files, and producer commands reject symlinked output destinations,
  so optional refs, top-level receipts, missing fingerprint blocks, anchor
  manifests or registries, stale source artifacts, or redirected generated
  evidence are blocked before labels can be trusted.
- Model-grader strict validation now warns on absolute source-file refs across
  dry-run, gate, and override artifacts before public grader labels are
  admitted to downstream curation.
- Model-grader rubric, dry-run, and override receipt validation recompute
  review-item fingerprints, label hashes, and override row hashes from source
  row contents, so stale or tampered grader evidence is blocked even when the
  hash still has a SHA-256 shape.
- Model-grader dry-run validation also binds disagreement-queue rows to the
  labels that require human review, so queued override work cannot drift from
  the label identities that triggered review.
- Model-grader dry-run validation binds label rows back to the current
  review-export items and requires its review export to match the referenced
  rubric's review export, blocking self-consistent but provenance-mismatched
  grader receipts.
- Training and reviewed dataset selection keys hash pathless fingerprint
  identity while manifests keep display paths as provenance, so registry keys
  remain stable across cwd and path-redaction presentation changes.
- Review item label hashes ignore source-artifact display paths while still
  binding artifact-role presence and review content, so benign path-redaction
  changes do not invalidate completed labels.
- Agentic training flow receipts reopen plan, runtime-preflight, and trainer
  consumer-plan refs from the flow artifact location and reject missing,
  non-regular, symlinked, stale SHA-256, or stale byte-size source evidence
  before delegated trainer execution can be trusted.
- Trainer consumer plans reopen visible trainer archive-check source refs and
  reject missing, non-regular, symlinked, stale SHA-256, or stale byte-size
  handoff evidence before an external trainer wrapper can trust command inputs.
- Promotion archives and trainer archives reject artifact paths that are
  symlinks or resolve through symlinked parent components before trusting
  recorded file hashes, tree hashes, or self-contained handoff claims.
- Trainer archive checks, consumer-plan execution refs, and wrapper dry-run
  refs reject missing visible resolved paths before accepting external trainer
  code or trainer inputs as ready for wrapper handoff.
- Improvement plans reject missing, non-regular, stale SHA-256, or stale
  byte-size file source-artifact records during validation.
- Action-ledger gate validation reopens the referenced action ledger from the
  gate file location before trusting gate metrics, checks, or decisions.
- Decision-gate validation reopens the referenced source artifact from the gate
  file location and rejects missing, moved, stale, or cwd-substituted sources.
- Decision-gate strict validation now warns on absolute artifact and
  source-artifact paths, catching preserved local gate coordinates before
  public promotion decisions are accepted.
- Rejection-sampling gates now write their own gate path relative to the gate
  output and strict validation warns on absolute gate or input-artifact refs
  before public dataset-curation admission is accepted.
- Dataset-curation receipts reopen rejection-sampling gate and training-export
  manifest refs from the receipt location, rejecting stale byte-size or SHA-256
  evidence while strict validation warns on absolute receipt or input-artifact
  refs.
- Promotion-ledger validation reopens recorded decision gates from the ledger
  file location and rejects missing, moved, stale, cwd-substituted, or symlinked
  records before hashing or replaying them.
- Promotion-ledger gate validation reopens the referenced promotion ledger,
  replays gate evaluation, and rejects stale metrics, checks, decisions,
  symlinked source ledger refs, or omitted policy checks.
- Promotion-ledger gate strict validation now warns on absolute source ledger
  and policy paths, catching preserved local promotion coordinates before
  public release decisions are accepted.
- Promotion cards, promotion decisions, alias receipts, rollback receipts, and
  release records write referenced artifacts relative to their own output files;
  validation reopens those references from the source file location and rejects
  symlinked or cwd-substituted governance artifacts.
- Promotion decisions and release records now reject symlinked promotion-policy
  inputs during generation; release records also refuse symlinked-parent source
  artifacts and release notes before they are fingerprinted or bound.
- Promotion-decision generation refuses symlinked-parent required artifacts and
  model/dataset card files before hashing them or reading JSON/card claims.
- Promotion-card generation refuses symlinked-parent evidence, training export,
  compare, redaction, and safety inputs before hashing them or reading JSON
  readiness signals.
- Promotion alias receipts and rollback receipts bind human-facing
  promotion-decision, applied-registry, and rollback-registry refs to the same
  SHA-256 and byte-size evidence carried by their fingerprinted artifact
  records.
- Promotion alias-apply and rollback receipt generation now refuse symlinked
  registry or promotion-decision inputs before hashing, replaying, or writing
  registry alias movement.
- Promotion archives reject symlinked source inputs and recorded source refs
  before reading, hashing, or copying promotion evidence into portable bundles.
- Promotion archives reopen ledger-recorded decision gates from the ledger file
  location and decision source artifacts from the decision file location; archive
  builds reject cwd-substituted members instead of copying lookalike evidence.
- Promotion archives compare ledger-recorded decision gate hashes and
  decision-recorded source artifact hashes before copying, blocking evidence that
  changed after approval.
- Promotion-archive strict validation now warns on preserved archive and
  original artifact paths before public promotion handoffs are accepted.
- Trainer-preflight validation reopens gates, validation summaries, schema
  contracts, and trainer artifacts from the preflight file location and rejects
  missing, moved, stale, or cwd-substituted records.
- Trainer archives compare preflight-recorded file hashes, sizes, and directory
  fingerprints before copying handoff inputs, blocking source evidence that
  changed after preflight approval.

### Verification

- Run promotion comparison using sample or mock eval summaries.
- Run governance tests for pass and blocking cases.
- Validate generated model and dataset cards.
- Validate release records and release notes.
- Confirm aliases are updated only after promotion passes.

### Ready-To-Run Prompt

```text
Build the Governance layer for the agentic fine-tuning platform.

Work inside hermes-flight-recorder. Inspect existing promotion comparison,
gate-decision, promotion ledger, action ledger, trainer handoff, and dataset
card code first. Create a top-level promotion policy and promotion decision
artifact that consumes evidence bundles, dataset registry entries, model
registry entries, training results, eval summaries, serving profiles, model
cards, dataset cards, and rollback metadata.

Promotion must compare candidate against base model, trace-only fine-tune, and
current champion on identical held-out scenarios. Block promotion on missing
artifacts, unknown license, missing rollback, missing cards, redaction failure,
eval mismatch, new critical failures, secret exposure, forbidden actions,
unsupported claims, or task-completion regression.

Add model-card and dataset-card templates or generators. Add tests for
promotion pass, critical regression, missing card, missing rollback, license
failure, and scenario mismatch. Finish only when promotion and rollback
decisions are reproducible from artifacts and aliases move only after passing
gates.
```

## Optional Meta Goal: End-To-End Orchestrator

Run this only after several layer goals have working artifacts.

```text
Build the first end-to-end autonomous orchestrator for the agentic fine-tuning
platform.

Use the existing layer artifacts instead of inventing a large workflow engine.
Coordinate the flow: evidence handoff, dataset registration, model candidate
selection, dry-run training plan, optional smoke training, serving profile,
held-out eval, promotion comparison, governance decision, and registry update.

The orchestrator must be resumable from artifacts, dry-run by default, and
strictly gated. It should not launch paid or heavy compute unless the required
inputs, licenses, dataset gates, and explicit runtime configuration are present.

Verify with a fully local/mock pipeline first. Finish only when a single command
can produce an auditable end-to-end report from existing sample runs.
```
