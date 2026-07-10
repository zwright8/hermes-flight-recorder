# Agentic Fine-Tuning Infrastructure Blueprint

This document defines the platform components and autonomous loops needed to
build fine-tuned agentic models from open or open-weight base models and use
them inside agent harnesses such as Hermes, OpenClaw, Codex-like runners, or
other tool-using runtimes.

For copy-pasteable autonomous work briefs, see
`docs/agentic-finetune-autonomous-goals.md`.
For the recommended daily supervisor and email reporting pattern, see
`docs/agentic-finetune-autonomous-operations.md`.

The core thesis: do not optimize for chat quality alone. Optimize for
trajectory-level agent behavior: task completion, correct tool use, evidence,
safety, cost, latency, and generalization across held-out tasks.

## Existing Nucleus

`hermes-flight-recorder` already provides the evidence/data-contract layer:

- trace normalization and scenario scoring
- scenario quality, evidence coverage, and trace observability gates
- RL-ready exports for episodes, rewards, step rewards, preferences, SFT, DPO,
  reward-model rows, dataset metrics, and family-exclusive splits
- experiment bundle generation for trace-only versus Flight-Recorder-gated
  fine-tuning arms
- delegated training plans and fail-closed handoff receipts covering trace
  SFT, curated SFT, DPO, and SFT-then-DPO; Flight Recorder does not execute
  model-weight training
- held-out evaluation planning, local deterministic replay, and promotion
  comparison contracts
- OpenAI-compatible serving lifecycle and capability-check contracts; serving
  engines remain external to Flight Recorder

Treat Flight Recorder as the deterministic evidence layer, not as the entire
trainer, registry, scheduler, or serving platform.

## North-Star System

The system should continuously produce candidates that can be dropped into an
agent harness and proven better than:

1. the base model,
2. a naive trace-only fine-tune,
3. the current production/promoted model,
4. any prior candidate on security and regression suites.

Promotion requires identical held-out scenarios, no new critical regressions,
and measurable improvement on task-completion evidence, tool-call behavior,
and cost/latency.

## Canonical Artifacts

Every loop should read and write stable artifacts. The minimum set:

- `trace`: raw runtime events from an agent harness
- `normalized_trace`: canonical event stream
- `scenario`: contract describing allowed actions, required evidence, scoring,
  and fixture/replay inputs
- `scorecard`: deterministic pass/fail and rule-level results
- `run_digest`: compact explanation of a run and its failure modes
- `evidence_bundle`: hashes, gates, readiness, and next actions
- `dataset_export`: SFT, DPO, reward-model, step-reward, curriculum, split, and
  dataset-card outputs
- `experiment_bundle`: model id, arms, datasets, held-out split, gates, and
  next commands
- `training_plan`: exact model, adapter, hyperparameters, data paths, tracking,
  and compute assumptions
- `training_result`: metrics, checkpoint ids, adapter locations, logs, and
  failure diagnostics
- `model_candidate`: base model, adapter/full model reference, license,
  quantization, context length, serving profile, and compatibility tags
- `eval_summary`: per-suite pass rate, score, critical failures, cost, latency,
  and task-completion proof metrics
- `promotion_decision`: deterministic gate result and rollback target
- `model_card` and `dataset_card`: rights, limitations, intended use, evals,
  safety notes, and lineage

## Platform Components

### 1. Harness Runner

Runs tasks through one or more agent harnesses.

Responsibilities:

- launch isolated workspaces and per-run homes
- configure provider/model/base URL consistently
- expose runtime tools, skills, memory, cron, subagents, or browser adapters
- capture raw observer events
- preserve prompt, tool config, environment metadata, and model settings

Minimum interfaces:

- `run_scenario(model_candidate, scenario) -> trace`
- `run_suite(model_candidate, scenario_set) -> suite_summary`
- `probe_model(model_candidate) -> compatibility_report`

### 2. Sandboxing And Tool Policy

Prevents the model from turning training and eval runs into ambient side
effects.

Responsibilities:

- ephemeral filesystem roots
- network allow/deny policies
- fake secrets and canary tokens
- command and URL policy enforcement
- deterministic mock tools for high-signal evals
- optional live-tool profile for integration tests

This is separate from model training. A good model can still be unsafe in a bad
sandbox.

### 3. Trace Normalization And Redaction

Turns runtime-specific logs into stable training and eval evidence.

Responsibilities:

- normalize messages, tool calls, tool results, file diffs, state snapshots,
  subagent activity, approvals, cost, and token usage
- redact secrets, PII, credentials, and customer data
- compute content hashes and lineage
- reject low-signal traces before they become labels

### 4. Scenario And Curriculum Factory

Creates and evolves tasks that teach and test agent behavior.

Responsibilities:

- convert failures into regression scenarios
- draft new scenarios for undercovered task families
- maintain train/validation/test family-exclusive splits
- classify scenarios by tool family, risk class, context length, difficulty,
  and expected evidence
- prevent train/eval leakage

### 5. Scoring, Reward, And Preference Engine

Converts traces into labels without trusting final-answer claims.

Responsibilities:

- final outcome rewards
- rule-level failure modes
- step-level reward attribution where evidence supports it
- chosen/rejected preference pairs
- task-completion and evidence-coverage scoring
- reward-hacking checks

Start with deterministic verifier rewards. Add learned reward models only after
the deterministic labels are stable.

### 6. Review And Labeling Queue

Human or model-assisted review for ambiguous data.

Responsibilities:

- label accepted SFT examples
- approve/reject DPO pairs
- calibrate evaluator outputs
- flag weak scenarios and ambiguous evidence
- store reviewer identity, rationale, and confidence

The queue should bias toward reviewing high-impact failures, not randomly
reviewing easy passes.

### 7. Dataset Lake And Registry

Stores all training data with lineage.

Responsibilities:

- object storage for raw and normalized traces
- dataset manifests with hashes
- versioned train/validation/test splits
- contamination checks against held-out suites
- dataset cards
- retention and deletion policies

Good default: local filesystem plus manifest JSONL for MVP; object storage plus
Postgres/SQLite metadata when runs scale. DVC or lakeFS can provide data
versioning if the team wants Git-like dataset provenance.

### 8. Model Scout And Base-Model Registry

Continuously evaluates candidate base models before expensive fine-tuning.

Responsibilities:

- discover model candidates
- check license and commercial rights
- check context length, tokenizer, chat template, tool-call support, structured
  output support, quantization formats, and serving engines
- baseline each candidate on a small agentic suite
- select top candidates for fine-tuning

Seed families to evaluate as of 2026-07-02:

- Qwen open-weight models, especially strong Apache-licensed Qwen3-family
  variants when the license is verified
- Mistral/Magistral/Devstral open models when Apache licensing and tool-use
  behavior fit the target harness
- DeepSeek reasoning or distill models where MIT licensing and runtime support
  fit
- OpenAI `gpt-oss` open-weight models where Apache licensing and hardware
  requirements fit
- Llama and Gemma as open-weight candidates only after accepting their specific
  license terms; do not call them strict open-source without a license review

The scout loop should refresh this list. Do not bake permanent model rankings
into the code.

### 9. Training Orchestrator

Launches reproducible post-training jobs.

Training modes:

- SFT on reviewed successful trajectories
- action-level SFT on high-quality intermediate tool decisions
- DPO/IPO/ORPO/KTO on chosen/rejected pairs
- reward-model or process-reward-model training from labeled outcomes
- GRPO/RL only for tasks with verifiable rewards and strong anti-gaming checks
- distillation from stronger teacher traces into smaller deployable models

Implementation options:

- TRL plus PEFT for direct Python control
- Axolotl or LLaMA Factory for YAML-driven multi-model training recipes
- Unsloth for fast single-node LoRA/QLoRA experiments
- Ray, SkyPilot, Slurm, or Kubernetes for distributed compute scheduling

Every job writes a training plan before importing heavy ML dependencies or
renting GPUs.

### 10. Experiment Tracker And Model Registry

Tracks lineage from base model and data to adapter and eval result.

Responsibilities:

- run metrics, logs, and artifacts
- checkpoint and adapter references
- model aliases such as `candidate`, `champion`, `rollback`
- approval status and promotion gates
- cost and hardware metadata

Good defaults: Trackio or W&B for experiment tracking, MLflow for registry
semantics, and a simple local JSON/SQLite registry for MVP reproducibility.

### 11. Serving And Compatibility Layer

Makes candidates usable by agent harnesses.

Responsibilities:

- OpenAI-compatible `/v1/chat/completions`
- tool/function calling
- structured outputs
- streaming
- quantized deployment profiles
- adapter merge/load strategy
- health checks and model metadata
- latency, throughput, memory, and context-length benchmarking

Use the existing local Transformers shim for smoke tests. Use vLLM or SGLang
for serious throughput and production-like serving.

### 12. Evaluation Farm

Runs candidate models through internal and external evals.

Internal suites:

- Flight Recorder held-out scenarios
- failure regressions
- prompt-injection and forbidden-action suites
- task-completion evidence suites
- harness-compatibility suites
- long-context and multi-turn tool-use suites

External suites to integrate:

- lm-evaluation-harness for general language benchmarks
- Inspect AI for agentic, tool-use, safety, and custom evals
- BFCL for function/tool calling
- SWE-bench or mini-SWE-agent style coding-agent evals when coding is a target
- domain-specific evals for any specialized agent workflows

### 13. Promotion And Release Gate

Decides whether a candidate can become usable.

Promotion checks:

- same held-out scenarios as baseline and challenger arms
- higher pass rate and average score than base and trace-only arms
- fewer or no more critical failures
- no new forbidden-action, secret-exposure, or unsupported-claim regressions
- task-completion evidence improves
- latency and cost remain within budget
- license and dataset rights pass
- model card and dataset card exist
- rollback model is defined

The gate should block by default when evidence is missing.

### 14. Demo And Replay Workbench

Shows what the model can do without hiding failure details.

Responsibilities:

- compare base, trace-only, and curated fine-tuned candidates
- replay traces side by side
- inspect tool calls, state diffs, scorecards, and evidence refs
- run interactive demo tasks against a served candidate
- generate a shareable report for each promoted model

### 15. Security, Legal, And Safety Layer

Runs continuously across data, training, serving, and eval.

Responsibilities:

- license scanning for base models and datasets
- PII and secret detection
- prompt-injection evals
- tool-abuse evals
- malicious fine-tune checks
- model card limitations and misuse warnings
- canary testing to ensure secrets never enter training data

## Reference Stack To Track

These are not permanent dependencies. They are the first tools to evaluate when
building each loop:

- Hugging Face TRL for SFT, DPO, GRPO, reward modeling, and other post-training
  methods: <https://huggingface.co/docs/trl/en/index>
- Hugging Face PEFT for LoRA/QLoRA adapter training:
  <https://huggingface.co/docs/peft/package_reference/lora>
- Axolotl for YAML-driven full fine-tuning, LoRA/QLoRA, preference tuning, and
  RL recipes: <https://docs.axolotl.ai/>
- Unsloth for fast local LoRA/QLoRA experiments:
  <https://unsloth.ai/docs/get-started/fine-tuning-llms-guide>
- vLLM for high-throughput OpenAI-compatible serving, structured outputs, and
  tool calling: <https://docs.vllm.ai/>
- SkyPilot or Ray for GPU job scheduling once local/manual launches become the
  bottleneck: <https://docs.skypilot.co/> and
  <https://docs.ray.io/en/latest/train/train.html>
- MLflow, Trackio, or W&B for experiment tracking and registry metadata:
  <https://mlflow.org/docs/latest/ml/model-registry/>
- lm-evaluation-harness for general benchmark sanity checks:
  <https://github.com/EleutherAI/lm-evaluation-harness>
- Inspect AI for agentic, tool-use, safety, and custom evals:
  <https://inspect.aisi.org.uk/>
- BFCL for function/tool-calling evaluation:
  <https://gorilla.cs.berkeley.edu/leaderboard.html>
- SWE-bench or mini-SWE-agent style evals if coding agents are a target:
  <https://www.swebench.com/>

## Autonomous Loops

### Loop A: Data Flywheel

Purpose: turn agent runs into validated training and regression artifacts.

Trigger: cron, new harness runs, failed evals, or manually selected tasks.

Steps:

1. run scenarios or live tasks in isolated harness sessions
2. capture traces and state snapshots
3. normalize, redact, and hash artifacts
4. score with scenario contracts
5. generate run digests, evidence bundles, repair queues, and training exports
6. gate trace quality, evidence coverage, scenario quality, and dataset quality

Output: validated dataset export plus repair/curriculum work items.

Stop condition: export passes gates or blocked work items are written.

### Loop B: Scenario And Curriculum Expansion

Purpose: keep the task distribution ahead of the model.

Trigger: recurring failures, low coverage, new tool families, or model
overfitting signs.

Steps:

1. mine failure modes and unsupported claims
2. draft new scenario contracts
3. create positive and negative fixtures where possible
4. run scenario quality and observability gates
5. assign train/validation/test split by task family
6. update curriculum priorities

Output: new scenarios, held-out split updates, and curriculum metadata.

Stop condition: new scenarios pass quality gates and do not leak into training.

### Loop C: Model Scout

Purpose: choose base models worth fine-tuning.

Trigger: weekly, new model releases, or failed training result.

Steps:

1. discover candidate models
2. verify license, weights availability, tokenizer/chat template, and serving
   support
3. run a smoke harness compatibility probe
4. baseline on a small internal eval suite
5. benchmark serving memory, latency, and context behavior
6. rank candidates for full training

Output: model-candidate manifest and shortlist.

Stop condition: top candidates are registered or rejected with reasons.

### Loop D: Training

Purpose: produce model candidates from validated data.

Trigger: dataset gate passes and model candidate is selected.

Steps:

1. build experiment bundle
2. write dry-run training plan
3. run tiny data smoke training
4. run selected full SFT, DPO, SFT-DPO, or RL job
5. archive adapter/checkpoints/logs/configs
6. register candidate with lineage

Output: trained adapter or merged model plus training result.

Stop condition: candidate is registered or training failure is classified.

### Loop E: Evaluation And Promotion

Purpose: prove that a candidate is better in the harness, not just lower loss.

Trigger: new candidate registered.

Steps:

1. serve base model, trace-only fine-tune, current champion, and candidate
2. run identical held-out scenarios
3. run security and regression suites
4. run external evals selected for the target domain
5. compare pass rate, score, failures, task completion, cost, and latency
6. promote, reject, or send failures to curriculum loop

Output: promotion decision, report, and rollback target.

Stop condition: candidate is promoted or rejected with evidence.

### Loop F: Harness Compatibility

Purpose: ensure models work with real agent runtimes.

Trigger: new candidate, new serving engine, or harness change.

Steps:

1. test chat template and system-prompt behavior
2. test tool/function-call schema compliance
3. test structured output parsing
4. test multi-turn memory and context pressure
5. test subagent/delegation behavior where relevant
6. test error recovery from tool failures

Output: compatibility report and serving profile.

Stop condition: candidate is compatible or rejected for the target harness.

### Loop G: Red-Team And Reward-Hacking

Purpose: prevent the data flywheel from training the model to game the gates.

Trigger: before promotion, after scenario changes, and on high-performing
candidates.

Steps:

1. run prompt-injection, secret-exposure, and forbidden-action scenarios
2. test fake evidence, unsupported claims, and premature completion
3. mutate scenarios to detect brittle reward exploitation
4. evaluate with canary secrets and controlled malicious tool outputs
5. add any failures to regression and curriculum queues

Output: red-team report and blocked/passed gate.

Stop condition: no blocking safety regressions remain.

### Loop H: Demo And Documentation

Purpose: produce evidence-backed demos for humans.

Trigger: candidate promotion or release candidate.

Steps:

1. select representative tasks
2. run demo traces through the promoted model
3. generate side-by-side baseline/champion/candidate reports
4. write model card, dataset card, and limitations
5. publish local demo package or hosted endpoint

Output: demo report, replay artifacts, and release notes.

Stop condition: demo artifacts match the promoted model and eval evidence.

### Loop I: Cost And Capacity

Purpose: keep training and serving economically sane.

Trigger: each training batch, candidate promotion, or infra change.

Steps:

1. benchmark throughput, latency, VRAM, and max context
2. test quantization profiles
3. estimate training and eval cost per candidate
4. choose serving engine and replica count
5. update budget gates

Output: cost profile and deployment recommendation.

Stop condition: model meets budget or is rejected for cost.

### Loop J: Registry And Governance

Purpose: keep all decisions reversible and auditable.

Trigger: any new dataset, model, eval, or promotion decision.

Steps:

1. record artifact hashes and lineage
2. update model and dataset registries
3. attach eval summaries and promotion decisions
4. mark aliases such as `candidate`, `champion`, and `rollback`
5. verify cards and license records

Output: registry update and audit trail.

Stop condition: every promoted artifact has lineage, eval evidence, and a
rollback target.

## Suggested MVP Build Order

### Phase 0: Freeze The Contract

- keep Flight Recorder as the artifact authority
- define `model_candidate`, `training_plan`, `training_result`, and
  `promotion_decision` schemas
- create a local registry directory or SQLite database
- make all loops write plans before doing expensive work

### Phase 1: Close The Existing Experiment Loop

- generate a Flight Recorder training export
- build the existing Qwen experiment bundle
- run dry-run training plans for trace SFT and curated SFT-DPO
- serve the base model and adapters through an OpenAI-compatible endpoint
- run held-out evaluations for base, trace-only, and curated arms
- run the existing promotion comparison gate

### Phase 2: Add Model Scout And Registry

- add a declarative model-candidate manifest
- baseline multiple base models on the same small harness suite
- record license, serving, memory, latency, and compatibility metadata
- pick one small model for fast iteration and one frontier open-weight model
  for serious experiments

### Phase 3: Add External Evals

- wire lm-evaluation-harness for general sanity checks
- wire BFCL for tool calling
- wire Inspect AI for custom agentic/security tasks
- wire SWE-bench style evals only if coding agents are a target use case

### Phase 4: Add Production Training Orchestration

- keep the current TRL/PEFT script for local experiments
- add Axolotl or LLaMA Factory recipes for repeatable larger jobs
- add SkyPilot/Ray/Slurm/Kubernetes job launchers when GPU scheduling becomes
  the bottleneck
- archive every training run into the registry before evaluation

### Phase 5: Add RL And Process Rewards

- use GRPO/RL only after deterministic rewards are stable
- start with verifiable tasks where reward hacking is easy to detect
- keep SFT/DPO baselines for every RL candidate
- require red-team mutation checks before promotion

## Minimum Viable Loop Runner

A single orchestrator can start as a thin command runner over existing scripts:

```yaml
pipeline_id: agentic-qwen-smoke
base_model: Qwen/Qwen3-4B-Instruct-2507
arms:
  - id: baseline
    kind: base
  - id: trace_only
    kind: lora
    train_mode: trace_sft
  - id: flightrecorder
    kind: lora
    train_mode: fr_sft_dpo
datasets:
  runs_dir: runs
  experiment_dir: experiments/qwen3_4b_flightrecorder
evals:
  heldout_split: heldout
  require_same_scenarios: true
gates:
  require_higher_pass_rate_than: [baseline, trace_only]
  require_higher_score_than: [baseline, trace_only]
  forbid_new_critical_failures: true
  forbid_new_secret_exposure: true
  require_task_completion_improvement: true
```

The first implementation should only coordinate:

- data export and gates
- dry-run training plans
- optional training launch
- serving process lifecycle
- held-out evals
- promotion comparison
- artifact registration

Do not start with a complex workflow engine. Start with explicit artifacts and
strict gates.

## Non-Negotiable Gates

- no raw trace enters training until redaction passes
- no final-answer-only success labels
- no train/eval leakage by task family
- no promotion without base and trace-only comparisons
- no model promoted without a rollback target
- no model downloaded or trained without license metadata
- no live-tool eval without sandbox and fake-secret canaries
- no RL loop until deterministic SFT/DPO baselines exist

## Immediate Next Work Items

1. Add schemas for `model_candidate`, `training_plan`, `training_result`, and
   `promotion_decision`.
2. Add a local registry under `experiments/registry/` or SQLite.
3. Create a loop runner that executes the existing Flight Recorder experiment
   chain end to end.
4. Add a model-scout manifest with 3 to 5 candidate base models and a license
   review field.
5. Add external eval adapters for BFCL, lm-evaluation-harness, and Inspect AI.
6. Promote only after base, trace-only, curated, and current-champion arms are
   compared on identical held-out scenarios.
