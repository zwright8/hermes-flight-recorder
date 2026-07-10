# Future RL Training Pipeline

Flight Recorder can now export completed run directories into training-ready
JSONL artifacts. This is a bridge from deterministic eval evidence to future
SFT, preference-tuning, reward-modeling, or RL loops.

It is not a weight-updating trainer or live provider runner. It can build
deterministic mock rollout receipts and delegated trainer flows, but it does not
launch paid rollouts, update model weights, or guarantee that a reward function
is impossible to game. It gives external runners a clean, deterministic control
plane grounded in observed traces.

Public artifact schemas are bundled with the package for downstream systems
that need stable contracts before wiring Flight Recorder into review or
training jobs:

```bash
flightrecorder schemas --write-dir artifact_schemas
flightrecorder schemas --name training_manifest --out training_manifest.schema.json
flightrecorder schemas --check runs/training_export/manifest.json
flightrecorder schemas --check runs/trainer_preflight.json
flightrecorder schemas --check runs/trainer_launch_check.json
flightrecorder schemas --check runs/trainer_archive_check.json
flightrecorder schemas --check runs/trainer_consumer_plan.json
flightrecorder schemas --check runs/trainer_wrapper_dry_run.json
flightrecorder schemas --check runs/agentic_training_flow.json
flightrecorder schemas --check runs/agentic_rollout_plan.json
flightrecorder schemas --check runs/agentic_rollout_receipt.json
flightrecorder schemas --check runs/rejection_sampling_gate.json
flightrecorder schemas --check runs/dataset_curation_receipt.json
flightrecorder schemas --check runs/agentic_loop_ledger.json
flightrecorder schemas --check runs/agentic_loop_governance_receipt.json
flightrecorder schemas --check runs/next_iteration_schedule.json
flightrecorder schemas --check runs/rubric_spec.json
flightrecorder schemas --check runs/model_grader_dry_run.json
flightrecorder schemas --check runs/model_grader_disagreement_queue.json
flightrecorder schemas --check runs/model_grader_override_receipt.json
flightrecorder schemas --check runs/model_grader_gate.json
flightrecorder schemas --check runs/harness_prompt_injection_good/harness_result.json
flightrecorder schemas --check runs/email_reply_completion_good/run_digest.json
```

Treat those JSON Schemas as shape contracts. `flightrecorder schemas --check`
can gate manifest shape without installing a third-party JSON Schema validator.
Use `flightrecorder validate --strict` and the relevant gates for semantic
readiness checks over artifact hashes, evidence references, replay lineage,
reviewed labels, and trainer handoff approvals.

## Export

Generate normal Flight Recorder runs first:

```bash
./demo.sh
```

For local harness integration without launching Hermes or a provider, write and
validate an offline mock harness packet before exporting the run:

```bash
python3.11 scripts/hermes_harness.py run-scenario \
  --scenario scenarios/prompt_injection_good.json \
  --mock-response "Summary: the issue asks for quality gates for autonomous runs." \
  --out runs/harness_prompt_injection_good

flightrecorder validate \
  --run runs/harness_prompt_injection_good \
  --harness-manifest runs/harness_prompt_injection_good/harness_manifest.json \
  --harness-result runs/harness_prompt_injection_good/harness_result.json \
  --strict
```

For your own scenario directory, the suite runner can generate runs, validation,
and training artifacts in one command:

```bash
flightrecorder run-suite \
  --scenarios scenarios \
  --out runs \
  --export-rl \
  --validate \
  --strict \
  --evidence-handoff \
  --metadata agent=hermes \
  --metadata candidate=skill-router-v2 \
  --metadata model=Hermes-4
```

Metadata is a simple string map for experiment identity. It lets later compare,
review, and training jobs know which agent, model, prompt, skill, or tool-policy
configuration produced the evidence bundle.
`--evidence-handoff` also writes `scenario_quality.json`,
`evidence_coverage.json`, `trace_observability.json`,
`harness_handoff/harness_manifest.json`,
`harness_handoff/harness_result.json`, and `evidence_bundle.json` during the
suite run, so the default handoff is a single command before stricter policy
gates are applied. The generated harness pair includes a `suite` provenance
block with the suite summary path, selected passing scenario, and pass/fail
counts; evidence bundles surface this as `harness_handoff` metrics and block
run-suite pairs that omit or forge that lineage. Bundles also verify the
referenced harness trace, scorecard, digest, report, and replay-lineage files
exist before counting a harness pair as artifact-valid. Strict harness
manifest, result, replay, and suite validation warns before preserved scenario,
sandbox, trace, scorecard, report, lineage, or run-artifact paths are
published; generate public harness packets with relative paths. Live-smoke
environment root paths are redacted in bundle metrics by default; use
`--preserve-paths` only for private local debugging. Strict evidence-bundle
validation rejects unredacted
absolute artifact paths, including nested harness, trainer, gate, digest, and
live-smoke metric paths, so public bundles should carry relative paths or
`<redacted:...>` markers. Validation summaries included in evidence
bundles must have at least one target and counts that match their `passed` and
`strict` flags. Run-digest coverage metrics are also checked for consistent
digest, outcome, and task-status counts, so forged summary totals cannot hide
low-signal or malformed per-run digests.
`flightrecorder compare-suite` carries this metadata into its JSON and HTML
outputs so baseline/candidate comparisons remain tied to the evaluated configs.
It also emits aggregate failed-rule and critical-failure deltas across paired
scenarios, giving repair or curriculum loops a compact view of which failure
classes gained or lost pressure.
It also checks per-run lineage fingerprints when available, so improvement
loops can detect when a same-named paired scenario actually changed scenario
contract or trace fixture between baseline and candidate runs.
Use `flightrecorder trend-suite --suite-summary ...` when you have more than
two iterations and want pass-rate, score, failed-rule, and critical-failure
trajectories across the whole improvement run. Validate `suite_trend.json`
before using a trend as improvement-loop evidence.

Use `flightrecorder evidence-coverage --runs ...` before training or review
handoffs when you need to prove that failed-rule pressure is attributable. The
coverage report measures how many failed and critical failed rules have
structured evidence refs, plus whether those refs point to trace events, final
answers, or episode-level facts.

Use `flightrecorder trace-observability --runs ...` before training or review
handoffs when you need to prove that traces are rich enough to learn from. The
observability report measures event volume, event-type diversity, final-answer
coverage, and tool/API visibility so low-signal traces can be blocked before
they become reward, preference, or review data.

Use `flightrecorder evidence-bundle` at the handoff boundary when a downstream
trainer, reviewer, or CI job needs one manifest over the generated evidence:

```bash
flightrecorder evidence-bundle \
  --runs runs \
  --suite-summary runs/suite_summary.json \
  --scenario-quality runs/scenario_quality.json \
  --evidence-coverage runs/evidence_coverage.json \
  --trace-observability runs/trace_observability.json \
  --validation runs/validation.json \
  --eval-summary runs/eval_summary.json \
  --training-export runs/training_export \
  --review-calibration runs/review_calibration.json \
  --live-smoke-summary runs/live_smoke_summary.json \
  --gate runs/suite_gate.json \
  --gate runs/training_gate.json \
  --require-gate \
  --out runs/evidence_bundle.json
```

The bundle records artifact hashes, readiness checks, gate results, compact
metrics, and a `decision` block with `promote_handoff` or `block_handoff`. It is
useful for provenance and job routing, but it should not be read as permission
to train unless the included scenario, evidence-coverage, validation, review,
and gate policies are also appropriate for the target job. At an Eval or
Governance boundary, include `--eval-summary` so a later promotion decision can
bind the exact summary fingerprint carried by the bundle.
Use `--require-gate` at trainer, Eval, or Governance boundaries so a bundle
cannot pass without at least one gate summary. Included gates must carry the
shared `decision` contract; weak legacy gates without `readiness`,
`recommendation`, `failed_checks`, and `next_actions` block the handoff.
When trainer handoff artifacts are available, include them in a second
trainer-facing bundle so the final preflight, launch-check, archive,
archive-check, consumer-plan, delegated flow, wrapper dry-run, and agentic
training result chain is visible in one validated manifest:

```bash
flightrecorder evidence-bundle \
  --runs runs \
  --suite-summary runs/suite_summary.json \
  --scenario-quality runs/scenario_quality.json \
  --evidence-coverage runs/evidence_coverage.json \
  --trace-observability runs/trace_observability.json \
  --validation runs/validation.json \
  --training-export runs/training_export \
  --trainer-preflight runs/trainer_preflight.json \
  --trainer-launch-check runs/trainer_launch_check.json \
  --trainer-archive runs/trainer_archive \
  --trainer-archive-check runs/trainer_archive_check.json \
  --trainer-consumer-plan runs/trainer_consumer_plan.json \
  --agentic-training-flow runs/agentic_training_flow.json \
  --trainer-wrapper-dry-run runs/trainer_wrapper_dry_run.json \
  --agentic-training-result runs/agentic_training_result.json \
  --out runs/evidence_bundle_trainer.json
```

The resulting `metrics.trainer_handoff` section records included stages,
readiness counts, missing stage ids, and whether all included trainer artifacts
match their expected recommendation. The result receipt may recommend either
`register_training_result` for completed runs or `register_training_failure`
for classified non-completed runs. Included stages are ready only when their
schema, recommendation, pass flag, and failed-check count all agree. The bundle
blocks only included stages that are no longer ready, while partial chains
remain visible as advisory next actions.
When included gates can validate trainer-facing exports, such as training,
compare, reviewed, and review-calibration gates, the bundle blocks handoff if
those gates skipped validation.
When a `live_smoke_summary.json` is included, current v2 summaries also carry
Python/platform details and Hermes plus Flight Recorder git provenance so
runtime-integration evidence can be tied back to the exact code that produced
the candidate traces.
The same `decision` block includes deterministic `next_actions` derived from
the included artifacts, such as repairing failed scenarios, resolving critical
failures, dispatching the concrete repair queue, grounding weak scenario
contracts, prioritizing curriculum failure modes, improving trace capture, or
reviewing training quality flags. When a training export is included, the
bundle fingerprints `manifest.json`, `dataset_metrics.json`, and
`curriculum.json`, and surfaces `top_curriculum_priorities` for routing repair,
scenario generation, or reward-review work. Each action carries a stable
`routing_key` plus an `action_fingerprint`, so downstream repair agents or
experiment ledgers can deduplicate work across repeated runs. Treat those
actions as routing guidance for the next improvement iteration, not as a
substitute for the gates themselves.

Use `flightrecorder improvement-plan` when the next iteration needs concrete,
deduplicatable work items instead of separate bundle, repair, curriculum, and
digest summaries:

```bash
flightrecorder improvement-plan \
  --evidence-bundle runs/evidence_bundle.json \
  --repair-queue runs/repair_queue.json \
  --training-export runs/training_export \
  --runs runs \
  --out runs/improvement_plan.json

flightrecorder validate --improvement-plan runs/improvement_plan.json --strict
```

The plan keeps repairs external and auditable. It joins bundle `next_actions`,
repair queue items, curriculum priorities, and per-run digests into stable
`work_items` with priorities, categories, evidence refs/snippets, replay
metadata, `routing_key`, and content `fingerprint` fields. Validation reopens
source artifacts from the plan file location and rejects symlinked
source-bundle paths before trusting recorded hashes or byte counts.

Across repeated plan snapshots, use `flightrecorder improvement-ledger` to
measure concrete work-item pressure:

```bash
flightrecorder improvement-ledger \
  --plan runs/previous/improvement_plan.json \
  --plan runs/current/improvement_plan.json \
  --out runs/improvement_ledger.json

flightrecorder validate --improvement-ledger runs/improvement_ledger.json --strict
```

The improvement ledger marks concrete scenario/rule work as `new`,
`recurring`, `open`, or `resolved` relative to the latest plan. It is a useful
convergence signal before trainer handoff because it can show whether the
evidence-backed repair queue is shrinking across eval iterations.
Ledger generation rejects source improvement-plan inputs that are symlinks or
traverse symlinked parent directories before reading them or emitting
source-plan fingerprints.

Use `flightrecorder gate-improvement-ledger` to turn that convergence signal
into a CI decision before promotion or trainer launch:

```bash
flightrecorder gate-improvement-ledger \
  --improvement-ledger runs/improvement_ledger.json \
  --policy examples/improvement_ledger_gate_policy.demo.json \
  --out runs/improvement_ledger_gate.json

flightrecorder validate --improvement-ledger-gate runs/improvement_ledger_gate.json --strict
```

The gate checks bounded open/new/recurring work, critical/high repair pressure,
required open work keys for tracked regressions, and required resolved work
keys for fixes that must land before the next improvement handoff. Validation
rejects symlinked source improvement-ledger paths before replaying gate
metrics, checks, or decisions.

Across repeated iterations, use `flightrecorder action-ledger` to fold multiple
`evidence_bundle.json` files into a stable repair ledger:

```bash
flightrecorder action-ledger \
  --bundle runs/previous/evidence_bundle.json \
  --bundle runs/current/evidence_bundle.json \
  --out runs/action_ledger.json

flightrecorder validate --action-ledger runs/action_ledger.json --strict
```

The ledger groups advisory actions by `routing_key`, records which bundle first
and last saw each action, and marks each one as `new`, `recurring`, `open`, or
`resolved` relative to the latest bundle. Validation reopens source bundles
from the ledger file location and rejects symlinked source-bundle paths before
trusting recurring-action counts.

Use `flightrecorder gate-action-ledger` to block trainer promotion when repair
pressure is not shrinking. Validation rejects symlinked source action-ledger
paths before replaying gate metrics, checks, or decisions:

```bash
flightrecorder gate-action-ledger \
  --action-ledger runs/action_ledger.json \
  --policy examples/action_ledger_gate_policy.demo.json \
  --out runs/action_ledger_gate.json

flightrecorder validate --action-ledger-gate runs/action_ledger_gate.json --strict

flightrecorder gate-decision \
  --artifact runs/action_ledger_gate.json \
  --expect-recommendation promote_iteration \
  --expect-readiness ready \
  --require-passed \
  --out runs/promotion_decision.json

flightrecorder validate --decision-gate runs/promotion_decision.json --strict

flightrecorder promotion-ledger \
  --decision-gate runs/previous/promotion_decision.json \
  --decision-gate runs/promotion_decision.json \
  --out runs/promotion_ledger.json

flightrecorder validate --promotion-ledger runs/promotion_ledger.json --strict
flightrecorder schemas --check runs/promotion_ledger.json

flightrecorder gate-promotion-ledger \
  --promotion-ledger runs/promotion_ledger.json \
  --policy examples/promotion_ledger_gate_policy.demo.json \
  --out runs/promotion_ledger_gate.json

flightrecorder validate --promotion-ledger-gate runs/promotion_ledger_gate.json --strict
flightrecorder schemas --check runs/promotion_ledger_gate.json

flightrecorder promotion-archive \
  --promotion-ledger runs/promotion_ledger.json \
  --promotion-ledger-gate runs/promotion_ledger_gate.json \
  --decision-gate runs/promotion_decision.json \
  --out runs/promotion_archive \
  --require-self-contained

flightrecorder validate --promotion-archive runs/promotion_archive --strict
```

Policies can cap open, new, or recurring actions, require a minimum number of
resolved actions, forbid open priority levels, and require specific routing keys
to be resolved. That makes repeated eval evidence usable as an explicit
trainer-side readiness signal. External automation should use
`decision.recommendation` (`promote_iteration` or `block_iteration`) and
`decision.key_metrics` as the compact promotion contract. Use
`flightrecorder gate-decision` to convert that source recommendation into a
validatable `allow_promotion` or `block_promotion` artifact for CI or trainer
handoff jobs. The source must be a supported registered decision-bearing
artifact that satisfies its bundled JSON Schema; arbitrary, unknown, and
wrong-type JSON is rejected. The generated `decision_gate.json` carries
`source_artifact.sha256`, tying the promotion decision to the exact source gate
artifact it consumed. When that source path is available, validation also
rechecks the registered schema contract and verifies that the embedded
`source_decision` still matches the source artifact's current decision block.
Gate-decision generation and validation reject source
artifacts that are symlinks or traverse symlinked parent directories before
emitting or trusting those hashes. Use `flightrecorder promotion-ledger` to
preserve the history of those allow/block artifacts across iterations. The
promotion ledger records latest recommendation, allowed/blocked counts,
consecutive block or allow streaks, and source-artifact fingerprints, giving an
external trainer launcher a stable "how did we get here?" artifact before it
consumes the final decision gate. Promotion-ledger generation and validation
reject symlinked recorded decision-gate paths before reading, hashing, or
replaying those records. `flightrecorder gate-promotion-ledger` performs the
same current-source validation before issuing a policy result. Use it
when trainer or CI automation needs a policy decision over that history, such
as requiring a clean latest allow decision, capping blocked-rate or blocked
streaks, and forbidding source `block_iteration` recommendations before launch.
Validation rejects
symlinked source promotion-ledger paths before replaying gate metrics, checks,
or decisions. Use
`flightrecorder promotion-archive` at the artifact-upload boundary: it copies
the promotion ledger, promotion-ledger gate, decision gates, and resolvable
source gate artifacts into a hash-checked directory that remains valid after
the original workspace paths disappear. Archive generation rejects source
inputs and recorded source refs that are symlinks or traverse symlinked parent
directories before reading, hashing, or copying them. Recorded artifact
references must be safe relative paths before they are copied, and validation
rejects archive artifacts that are symlinks or resolve through symlinked parent
components, plus relationships that point at unknown artifacts or invalid role
pairs. Keep shared promotion archives in the default redacted mode; use
`--preserve-paths` only for private local debugging. Strict promotion-archive
validation warns if preserved archive or original artifact paths would enter a
public promotion handoff.

Before any external registry process moves `candidate`, `champion`, or
`rollback` aliases, run a top-level governance decision:

```bash
flightrecorder promotion-cards \
  --candidate-id candidate-v2 \
  --dataset-id dataset-v1 \
  --model-source base-model-or-training-output \
  --license-status known \
  --evidence-bundle runs/evidence_bundle.json \
  --training-export runs/training_export \
  --compare-gate runs/compare_gate.json \
  --redaction-check runs/redaction_check.json \
  --safety-gate runs/safety_gate.json \
  --out runs/promotion_cards

flightrecorder validate --promotion-cards runs/promotion_cards --strict
flightrecorder schemas --check runs/promotion_cards/promotion_cards.json

flightrecorder validate --promotion-policy examples/promotion_policy.demo.json --strict
flightrecorder schemas --check examples/promotion_policy.demo.json

flightrecorder promotion-rollback-receipt \
  --registry registry/model_registry.json \
  --rollback-id champion-v1 \
  --out runs/rollback.json

flightrecorder validate --promotion-rollback-receipt runs/rollback.json --strict
flightrecorder schemas --check runs/rollback.json

flightrecorder promotion-decision \
  --candidate-id candidate-v2 \
  --champion-id champion-v1 \
  --rollback-id champion-v1 \
  --evidence-bundle runs/evidence_bundle.json \
  --eval-summary runs/eval_summary.json \
  --external-eval-result runs/external_eval_result.json \
  --promotion-ledger-gate runs/promotion_ledger_gate.json \
  --compare-gate runs/compare_gate.json \
  --trainer-launch-check runs/trainer_launch_check.json \
  --model-registry-entry runs/model_registry_entry.json \
  --agentic-training-result runs/agentic_training_result.json \
  --model-card runs/promotion_cards/MODEL_CARD.md \
  --dataset-card runs/promotion_cards/DATASET_CARD.md \
  --rollback-metadata runs/rollback.json \
  --license-review runs/license_review.json \
  --redaction-check runs/redaction_check.json \
  --safety-gate runs/safety_gate.json \
  --serving-profile runs/serving_profile.json \
  --serving-report runs/serving_report.json \
  --promotion-policy examples/promotion_policy.demo.json \
  --out runs/promotion_decision.json

flightrecorder validate --promotion-decision runs/promotion_decision.json --strict
flightrecorder schemas --check runs/promotion_decision.json

flightrecorder promotion-alias-apply \
  --registry registry/model_registry.json \
  --promotion-decision runs/promotion_decision.json \
  --out runs/promotion_alias_apply.json

flightrecorder validate --promotion-alias-apply runs/promotion_alias_apply.json --strict
flightrecorder schemas --check runs/promotion_alias_apply.json

flightrecorder promotion-release-record \
  --release-id release-2026-07-02 \
  --promotion-decision runs/promotion_decision.json \
  --promotion-cards runs/promotion_cards \
  --promotion-alias-apply runs/promotion_alias_apply.json \
  --rollback-metadata runs/rollback.json \
  --compare-gate runs/compare_gate.json \
  --release-notes runs/RELEASE_NOTES.md \
  --promotion-policy examples/promotion_policy.demo.json \
  --out runs/promotion_release_record.json

flightrecorder validate --promotion-release-record runs/promotion_release_record.json --strict
flightrecorder schemas --check runs/promotion_release_record.json
```

The decision blocks promotion on missing evidence, unknown license status,
redaction or safety failure, missing cards, missing rollback metadata, failed
rollback receipts, eval mismatch, task-completion regression, new critical
failures, secret exposure, forbidden actions, and unsupported card claims. A
passing decision is still side-effect free: it authorizes an alias-update
receipt, leaving the actual registry write to a later guarded step. Promotion
decisions reject required source artifacts and card files that are symlinks or
traverse symlinked parent directories before reading, hashing, or binding them.
`--external-eval-result` is repeatable. The supplied results must form a
non-empty, unique set that exactly matches the eval summary, each result must
identify `--candidate-id`, and the evidence bundle must fingerprint that same
summary. Generation semantically validates all three layers before authorizing
promotion. Validation reopens and rehashes their source files, reruns semantic
validation, rebuilds the external-eval lineage and its checks, and rejects a
decision if a source was removed, replaced, or mutated after generation.
The same replay boundary re-evaluates the compare export and promotion ledger,
rebuilds the trainer launch check from its current preflight, and requires the
candidate's live registry entry to exactly match the entry artifact named by
the decision. Missing, malformed, or moved nested sources block both decision
validation and alias application.
`promotion-rollback-receipt` is side-effect free: it fingerprints the model
registry, proves the rollback target is registered, and blocks when the target
no longer matches the current champion before promotion. Validation also reads
the fingerprinted registry artifact and rejects stale embedded alias snapshots
that no longer match the registry file. Rollback-receipt generation rejects
registry paths that are symlinks or traverse symlinked parent directories before
reading or fingerprinting them.
`--promotion-policy` records the policy artifact that declares the required
decision/release artifact contract, allowed model classes, zero-tolerance eval
limits, required forbidden-rule blockers, license, rollback, card, and
validation requirements. Policy files can make expectations reviewable but
cannot relax the default promotion blockers. Promotion-decision and
release-record generation reject policy files that are symlinks or traverse
symlinked parent directories before reading or fingerprinting them.
`promotion-alias-apply` is that guarded write: it revalidates the promotion
decision, requires a `hfr.model_registry.v1` registry with registered
`candidate`, `champion`, and `rollback` targets, verifies the live champion
alias still matches the decision's previous champion, then updates aliases and
appends an alias-history entry. Blocked receipts leave registry aliases
unchanged. Alias-apply generation rejects registry and promotion-decision inputs
that are symlinks or traverse symlinked parent directories before validation,
hashing, replay, or registry mutation.
The registry mutation and receipt publication use compare-and-swap writes. The
alias receipt replays the current decision and registry snapshots, binds the
candidate, previous champion, and rollback identities, and rolls the registry
back if receipt publication fails after the registry write.
`promotion-release-record` binds the final publishable evidence set: promotion
decision, generated cards, alias receipt, rollback metadata, eval compare gate,
and release notes. Validation rehashes every referenced artifact and matches
compact model/dataset card bindings back to the referenced promotion-cards
manifest, so stale release notes, mismatched eval evidence, card drift, a
different alias receipt, or a release policy that differs from the decision
policy blocks publication. Release-record generation also refuses required input
artifacts and release notes that are symlinks or traverse symlinked parent
directories before reading, hashing, or binding them.
Alias-apply and release-record receipts also preserve compact strict validation
summaries; validation recomputes their pass state from target, error, and
warning counts so a forged `passed` flag cannot hide warning-bearing or failed
nested validation.
`promotion-cards` generates the model and dataset cards plus
`promotion_cards.json`; generation rejects required inputs that are symlinks or
traverse symlinked parent directories before reading or hashing them, and
validation rehashes generated cards and inputs so stale card evidence is caught
before the promotion decision consumes it.

Use `flightrecorder trainer-preflight` as the final launch guard that an
external trainer can consume. It records the trainer command, fingerprints the
trainer-facing export files, including `dataset_registry.json`,
`dataset_splits.json`, and every `splits/<split>/*.jsonl` file, verifies
required gates are present and passed, and refuses training, compare, reviewed,
or review-calibration handoffs that skipped embedded export validation unless
`--allow-unvalidated-gates` is explicitly set. Use
`--require-dataset-version` with the manifest `dataset_version` whenever the
trainer should consume one exact dataset lineage:

```bash
flightrecorder trainer-preflight \
  --gate runs/training_gate.json \
  --gate runs/compare_gate.json \
  --training-export runs/training_export \
  --compare-export runs/compare_rl_export \
  --evidence-bundle runs/evidence_bundle.json \
  --require-gate training_gate \
  --require-gate compare_gate \
  --require-dataset-version "$(jq -r .dataset_version runs/training_export/manifest.json)" \
  --trainer-command "python train.py --dry-run --dataset runs/training_export" \
  --out runs/trainer_preflight.json

flightrecorder validate --trainer-preflight runs/trainer_preflight.json --strict

flightrecorder trainer-launch-check \
  --preflight runs/trainer_preflight.json \
  --require-gate training_gate \
  --require-gate compare_gate \
  --require-dataset-version "$(jq -r .dataset_version runs/training_export/manifest.json)" \
  --out runs/trainer_launch_check.json

flightrecorder validate --trainer-launch-check runs/trainer_launch_check.json --strict

flightrecorder trainer-archive \
  --preflight runs/trainer_preflight.json \
  --launch-check runs/trainer_launch_check.json \
  --out runs/trainer_archive \
  --require-self-contained

flightrecorder trainer-archive-check \
  --archive runs/trainer_archive \
  --external-code-root path/to/trainer-code \
  --out runs/trainer_archive_check.json \
  --strict

flightrecorder trainer-consumer-plan \
  --archive-check runs/trainer_archive_check.json \
  --out runs/trainer_consumer_plan.json \
  --strict

python examples/trainer-wrapper/consume_trainer_plan.py \
  --plan runs/trainer_consumer_plan.json \
  --out runs/trainer_wrapper_dry_run.json \
  --strict

flightrecorder validate --trainer-archive runs/trainer_archive --strict

flightrecorder validate --trainer-archive-check runs/trainer_archive_check.json --strict

flightrecorder validate --trainer-consumer-plan runs/trainer_consumer_plan.json --strict

flightrecorder validate --trainer-wrapper-dry-run runs/trainer_wrapper_dry_run.json --strict
```

The preflight manifest and launch check are still evidence plumbing, not a
trainer. They do not execute the command or update weights. `trainer-preflight`
creates the signed-off evidence contract; `trainer-launch-check` is the
consumer-side check an external training launcher can call immediately before it
runs. It re-validates the preflight hashes and prints the approved command only
when the launch contract still passes. Before public handoff, strict
trainer-preflight validation warns if the trainer-command raw string or argv
tokens still carry absolute paths. Trainer-launch-check validation rejects
approved-command raw, argv, or shell tokens that still carry local absolute
paths. Strict trainer-archive validation
repeats that warning for the preserved approved command and archive source paths
while keeping the rewritten portable command auditable. Trainer consumer-plan
validation rejects archived commands that still carry local absolute archive
roots, external code roots, or argv paths. Trainer-wrapper dry-run validation
rejects `would_run` commands that still carry local absolute archive roots,
external code roots, or argv paths.
Trainer-facing export files must be regular files at preflight time; symlinked
JSONL, JSON, Markdown artifacts, or
split artifacts block launch even if their targets contain matching bytes.
`trainer-archive` is the portable handoff after those checks pass: it copies
the preflight, launch check, gates, validation summaries, trainer-facing
exports, and schema-contract files into one hash-checked directory. It records
the copied trainer inputs, the original approved command, path rewrites, and an
advisory portable command that points known input paths at archive-local copies.
Validation rejects archive artifact files or directories that resolve through
symlinked parent components before trusting recorded hashes or tree hashes.
Its `consumer_contract` states that the portable command should be resolved
from the archive root and flags path-like command tokens, such as trainer
scripts, that still must be supplied by external training infrastructure.
`trainer-archive-check` is the next non-executing proof for that external
launcher: it validates the archive, confirms archive-local trainer inputs still
match their recorded hashes and byte counts, and checks that caller-provided
trainer code paths such as `train.py` exist under `--external-code-root`.
Strict archive-check validation warns before preserved archive roots, external
code roots, or resolved paths enter public handoffs; portable command tokens
that still carry local absolute paths are rejected by default.
Passed trainer input rows carry expected hash and size payloads so validation
can reject forged redacted receipts without needing the original producer's
local paths.
`trainer-consumer-plan` then records the exact approved command argv, archive
root, external code file hashes, trainer input hashes, and launcher invariants
that the external wrapper should require. It rejects archive-check source
inputs that are symlinks or traverse symlinked parent directories before
reading them or emitting source-archive-check fingerprints. It preserves the
expected trainer input hash and size payloads through the wrapper dry run. It
is still a plan, not a runner.
Validation summaries embedded in the archive check, consumer plan, and wrapper
dry-run receipts must include targets and internally consistent pass/error/
warning counts, so a ready handoff cannot hide failed validation behind a
forged `passed` flag.
The reference wrapper in `examples/trainer-wrapper/` demonstrates how an
external launcher can validate that plan and emit a dry-run receipt before a
real trainer takes over. That receipt can also be checked with
`flightrecorder validate --trainer-wrapper-dry-run`, making the wrapper dry run
part of the evidence contract rather than an untyped log file.
Agentic fine-tuning modes can enter this same pipeline through
`scripts/plan_agentic_training.py`, which emits `hfr.agentic_training_plan.v1`
dry-run artifacts from registered model and dataset manifests. SFT, action SFT,
DPO, and SFT-then-DPO are the default executable handoff modes. Reward-model
and process-reward modes remain blocked unless `--allow-advanced-training` is
passed, and GRPO/RL remain blocked unless `--allow-future-rl` is passed. Those
flags only allow planning; they still do not launch a trainer, import training
stacks, download models, or update weights.
Each plan now carries a `mode_contract` that states the mode category, required
trainer-view groups, data-requirement evidence, reward-signal or reward-function
contract, and hard side-effect boundary. For GRPO, that contract records the
TRL-style interface `reward_fn(prompts, completions, **kwargs) -> list[float]`
for an external runner to supply and validate; Flight Recorder does not
implement, import, or execute it. Before public handoff, plan validation
rejects `execution.external_runner_command` tokens that preserve absolute local
paths.
Use `flightrecorder validate --agentic-training-plan <plan.json>` before any
handoff. The validator rejects hidden provider job fields, trainer URLs,
credential hints, paid-grader toggles, live model downloads, and weight-mutation
fields that are not part of the public dry-run plan contract.

Pass `--agentic-training-plan <plan.json>` to `trainer-preflight` to fingerprint a
ready plan as a trainer input before archiving or handing it to an external
runner.
Before a bounded tiny-smoke launch, run
`scripts/preflight_agentic_training_runtime.py` against that plan. It emits
`hfr.agentic_training_runtime_preflight.v1`, validates the selected trainer
views, checks the plan `mode_contract`, probes required Python modules with
`importlib.util.find_spec`, and records `training_started: false`,
`model_downloads_started: false`, `cloud_jobs_started: false`,
`paid_model_grader_calls_started: false`, `weights_updated: false`, and
`trainer_modules_imported: false`. Treat
`recommendation: ready_for_tiny_smoke_launch` as the next handoff condition;
blocked runtime-preflight artifacts are still schema-checkable failure
evidence. The embedded `mode_contract_check` schema pins invariant reward and
side-effect fields so paid/secret reward defaults, provider credentials, paid
graders, cloud jobs, downloads, training starts, and weight updates cannot be
forged into a runtime preflight. Its `dependency_policy` records the normalized
backend, backend defaults, explicit overrides, whether default resolution was
delegated, and the resulting effective module set. Ready receipts require at
least one effective dependency and a fresh successful probe for every module;
strict validation recomputes the policy and reruns those probes, so an empty or
self-declared availability list cannot forge readiness. Runtime preflight
rejects plan inputs that are symlinks or traverse symlinked parent directories,
and selected views resolved through symlinks are blocked without SHA-256 or
byte-size fingerprints. Plan and selected-view display paths are deterministic
across working directories. Cross-root plan/output references are rejected by
default rather than exposing a private home-directory layout; use
`--preserve-paths` only for private local receipts that will not be published.
After a trainer consumer plan exists, use `flightrecorder agentic-training-flow`
to bind the ready plan, runtime preflight, and consumer command into
`hfr.agentic_training_flow.v1`. It delegates only SFT, action-SFT, DPO, and
SFT-then-DPO flows by default; reward-model, process-reward, GRPO, and RL flows
remain blocked at this boundary. Blocked advanced-mode receipts remain valid
evidence: they mirror the runtime `mode_contract_check`, add a `flow_mode_gate`
with the mode category, opt-in flag, reward-contract obligations, and
promotion-required reason, and keep the readiness recommendation at
`block_delegated_trainer_execution`. The receipt records the exact external
command, stage sequence, selected trainer views, and a fail-closed execution
boundary without starting a subprocess, importing trainer modules, creating
cloud jobs, downloading models, or updating weights. Validation rejects
mirrored mode contracts that enable paid/secret reward defaults, provider
credentials, paid graders, cloud jobs, downloads, training starts, or weight
updates, and mirrored external-runner contracts that drop runner ownership, input
revalidation, plan-ready gating, or unredacted-trace blocking, while keeping
reward-validation requirements mode-dependent. It also rejects
delegated commands that still contain absolute execution roots or argv path
tokens before the receipt can become a public handoff.
Flow generation rejects plan, runtime-preflight, or trainer-consumer-plan
inputs that are symlinks or traverse symlinked parent directories before
reading those files or emitting `source_artifacts` fingerprints.

```bash
flightrecorder agentic-training-flow \
  --plan runs/agentic_training_plan.json \
  --runtime-preflight runs/agentic_training_runtime_preflight.json \
  --trainer-consumer-plan runs/trainer_consumer_plan.json \
  --out runs/agentic_training_flow.json

flightrecorder validate --agentic-training-flow runs/agentic_training_flow.json --strict
```

When an external runner finishes or fails, archive that outcome with
`scripts/archive_agentic_training_result.py`. It emits
`hfr.agentic_training_result.v1`, verifies the plan, runtime-preflight, and
delegated-flow lineage, requires a ready runtime preflight plus a ready
`hfr.agentic_training_flow.v1` receipt and an adapter or checkpoint for
`--status completed`, and requires a classified failure for `failed`, `blocked`,
or `aborted` outcomes. The result receipt fingerprints supplied configs,
metrics, adapters, checkpoints, logs, and failure reports without launching a
trainer or importing trainer stacks. It also records a side-effect-free
`registry_update` proposal for training-run and adapter links; governance or a
later guarded registry step must apply those links explicitly.
The archive command rejects plan, runtime-preflight, or delegated-flow source
inputs that are symlinks or traverse symlinked parent directories before
reading, hashing, or emitting result lineage.
Validation rejects lineage refs that traverse symlinked parent directories
before replaying delegated-flow or training-plan manifest checks.
Validate result receipts directly before bundling them:

```bash
flightrecorder validate \
  --agentic-training-result runs/agentic_training_result.json \
  --strict
```

Use `flightrecorder agentic-rollout-plan` before collecting new rollouts. It
binds scenarios, baseline/candidate/teacher policy ids, replayable environment
metadata, external-state verifier refs, rollout budgets, rejection-sampling
requirements, and expected trace/dataset lineage without calling model
providers or writing dataset rows:
Declared external-state verifier refs are summarized by an
`external_state_verifier_gate`; missing verifier configs block rollout planning,
and receipts preserve the same gate without starting verifier side effects.
Scenario, verifier, and source-plan refs are considered public only when they
replay from the rollout artifact's output directory without absolute paths or
`..` traversal. Unsafe refs are written as redacted missing inputs, do not
contribute harness batches or mock rows, and keep the artifact blocked.

```bash
flightrecorder agentic-rollout-plan \
  --iteration-id rollout-001 \
  --scenario scenarios/prompt_injection_good.json \
  --policy baseline=local/base \
  --policy candidate=local/candidate \
  --policy teacher=local/teacher \
  --max-rollouts 3 \
  --verifier examples/external_verification/sqlite_task_state.verifier.json \
  --out runs/agentic_rollout_plan.json

flightrecorder validate \
  --agentic-rollout-plan runs/agentic_rollout_plan.json \
  --strict
```

The committed fixture at `examples/rollout_generation/rollout_plan.json` is a
schema- and validation-checked plan-only rollout batch with no provider,
verifier, grader, or dataset-write side effects. The closed-loop demo also
keeps a replayable rollout plan plus deterministic mock receipt under
`examples/agentic_training/rollouts/` so the loop ledger can distinguish
rollout readiness from later harness, scoring, and curation work.

Archive a deterministic mock rollout receipt before scoring or review. This
records one mock row per planned batch and proves Flight Recorder did not start
live rollouts, call model providers or paid graders, write traces or scorecards,
or create training dataset rows:

```bash
flightrecorder agentic-rollout-receipt \
  --plan runs/agentic_rollout_plan.json \
  --out runs/agentic_rollout_receipt.json

flightrecorder validate \
  --agentic-rollout-plan runs/agentic_rollout_plan.json \
  --agentic-rollout-receipt runs/agentic_rollout_receipt.json \
  --strict
```

Before dataset curation, use `flightrecorder rejection-sampling-gate` to prove
mock rollout receipts, calibrated grader review, and human-reviewed gates are
all present. This gate does not write accepted or rejected training rows:

```bash
flightrecorder rejection-sampling-gate \
  --rollout-receipt runs/agentic_rollout_receipt.json \
  --model-grader-gate runs/model_grader_gate.json \
  --review-calibration runs/review_calibration.json \
  --reviewed-gate runs/reviewed_gate.json \
  --out runs/rejection_sampling_gate.json

flightrecorder validate \
  --rejection-sampling-gate runs/rejection_sampling_gate.json \
  --strict
```

Rejection-sampling gate refs are public-safe by default: generated artifacts
write paths relative to the gate output directory, redact absolute or traversal
refs that cannot be replayed from there, and validation rejects hand-authored
unsafe refs before curation can proceed.
The committed closed-loop demo includes
`examples/agentic_training/rejection_sampling_gate.json`, which admits its mock
rollout receipt only after the reviewed-gate and calibration receipts pass while
still recording `dataset_rows_written: false`.

Archive a curation receipt after rejection-sampling admission and before
trainer preflight. It binds existing training exports to the admission gate and
records that Flight Recorder did not write new curated rows or update dataset
registries:

```bash
flightrecorder dataset-curation-receipt \
  --rejection-sampling-gate runs/rejection_sampling_gate.json \
  --training-export runs/training_export \
  --out runs/dataset_curation_receipt.json

flightrecorder validate \
  --dataset-curation-receipt runs/dataset_curation_receipt.json \
  --strict
```

Dataset-curation receipts use the same public-safe replay boundary as
rejection-sampling gates: gate refs, training-export directories, and manifest
refs must resolve relative to the receipt directory. Unsafe absolute or
traversal refs are redacted when generated and rejected during validation.
The committed closed-loop demo includes a real local `export-rl` bundle under
`examples/agentic_training/training_export/` plus a
`dataset_curation_receipt.json` that binds it to rejection sampling while
recording that no curated rows, registries, cloud jobs, or weights were changed.
It also includes a `training_gate.json`, `trainer_preflight.json`, and
`trainer_launch_check.json` that approve only the local dry-run trainer command
against the selected dataset version. The `heldout_eval/` fixture adds
deterministic baseline/candidate suite summaries, a held-out manifest,
an offline `local_mock` external-eval plan and dry-run receipt, an imported
per-case external result, plus an eval summary. The receipt passes without
provider calls, downloads, benchmark launches, secrets, cost, or weight
updates, but it is only handoff evidence. The imported result, not the receipt,
records completion of the externally owned run. Real BFCL, Inspect AI,
lm-eval-harness, and SWE-bench adapters remain fail-closed in the standalone
`examples/external_eval/` fixtures until optional dependencies are explicitly
enabled. The
`evidence_handoff/` fixture records a compact passing harness result and ready
evidence bundle for a deterministic prompt-injection scenario. The
`serving_lifecycle/managed_mock/` fixture records a normalized mock serving
preflight with no model download, provider call, or live endpoint exposure. The
`promotion_governance/` fixture records a real offline compare export, compare
gate, promotion-history ledger gate, and passing promotion decision. That
decision authorizes a reviewable alias update. The fixture also includes
generated promotion cards, pre-promotion rollback proof, a guarded local
`promotion-alias-apply` receipt, release record, and self-contained promotion
archive; none of those artifacts call providers, publish externally, or update
weights.

Use `flightrecorder agentic-loop plan` to bind rollout, evidence, review,
trainer, cloud-training, serving, held-out eval, improvement, governance,
promotion, and next-iteration receipts into one iteration contract:

```bash
flightrecorder agentic-loop plan \
  --iteration-id loop-001 \
  --objective "Close held-out tool-use regressions" \
  --agentic-rollout-plan runs/agentic_rollout_plan.json \
  --agentic-rollout-receipt runs/agentic_rollout_receipt.json \
  --harness-result runs/harness_prompt_injection_good/harness_result.json \
  --evidence-bundle runs/evidence_bundle_trainer.json \
  --rejection-sampling-gate runs/rejection_sampling_gate.json \
  --dataset-curation-receipt runs/dataset_curation_receipt.json \
  --training-export runs/training_export \
  --trainer-preflight runs/trainer_preflight.json \
  --trainer-launch-check runs/trainer_launch_check.json \
  --agentic-training-plan runs/agentic_training_plan.json \
  --agentic-training-result runs/agentic_training_result.json \
  --cloud-training-provider-registry runs/cloud_provider_registry.json \
  --cloud-training-preflight runs/cloud_preflight.json \
  --cloud-training-artifact-manifest runs/cloud_artifacts.json \
  --cloud-training-launch-plan runs/cloud_launch_plan.json \
  --cloud-training-launch-receipt runs/cloud_launch_receipt.json \
  --cloud-training-status-receipt runs/cloud_status_receipt.json \
  --serving-lifecycle runs/serving_lifecycle.json \
  --heldout-manifest runs/heldout_manifest.json \
  --external-eval-plan runs/external_eval_plan.json \
  --external-eval-receipt runs/external_eval_receipt.json \
  --external-eval-result runs/external_eval_result.json \
  --eval-summary runs/eval_summary.json \
  --promotion-decision runs/promotion_decision.json \
  --promotion-ledger runs/promotion_ledger.json \
  --promotion-cards runs/promotion_cards \
  --promotion-alias-apply runs/promotion_alias_apply.json \
  --promotion-rollback-receipt runs/rollback.json \
  --promotion-release-record runs/promotion_release_record.json \
  --promotion-archive runs/promotion_archive \
  --out runs/agentic_training_loop_plan.json

flightrecorder validate \
  --agentic-loop-plan runs/agentic_training_loop_plan.json \
  --strict

flightrecorder agentic-loop ledger \
  --plan runs/agentic_training_loop_plan.json \
  --out runs/agentic_loop_ledger.json

flightrecorder validate \
  --agentic-loop-ledger runs/agentic_loop_ledger.json \
  --strict

flightrecorder agentic-loop governance \
  --ledger runs/agentic_loop_ledger.json \
  --action approve \
  --requested-by governance-review \
  --reason "Latest loop is ready for governance review; record approval without provider, benchmark, alias, or weight side effects." \
  --out runs/agentic_loop_governance_receipt.json

flightrecorder validate \
  --agentic-loop-governance-receipt runs/agentic_loop_governance_receipt.json \
  --strict

flightrecorder next-iteration-schedule \
  --loop-ledger runs/agentic_loop_ledger.json \
  --action-ledger runs/action_ledger.json \
  --improvement-ledger runs/improvement_ledger.json \
  --next-iteration-id loop-002 \
  --objective "Resolve remaining ledgered repair pressure" \
  --out runs/next_iteration_schedule.json

flightrecorder validate \
  --next-iteration-schedule runs/next_iteration_schedule.json \
  --strict
```

The plan is side-effect free. It records artifact paths and hashes, missing phase
inputs, provider constraints, live-spend boundaries, next-iteration scheduling
intent, and a handoff contract declaring that external trainers own weight
updates. `plan_readiness` covers the pre-execution handoffs,
`execution_completion` is derived from the bound training result and exact
external eval result set, and `governance_readiness` becomes
`ready_for_review` only when execution is complete and every remaining check
passes. The legacy `readiness` value is derived from those states. Rollout,
evidence, calibrated review, rejection sampling, dataset curation, trainer
preflight, serving, held-out eval, improvement, promotion decision, promotion
ledger, and release-governance artifacts must remain present and fail-closed.
Release-governance receipts can include generated promotion cards, rollback
proof, guarded alias-apply receipt, release record, and promotion archive. The
loop planner and governance receipt do not move aliases or publish artifacts;
alias movement stays isolated in the explicit `promotion-alias-apply` command
against a registry artifact. Missing required phase inputs keep the loop blocked
and recommend another iteration. Public plans reject absolute home paths,
`/tmp` paths, secret path fragments, and credential-looking strings.

External benchmark adapters stay fail-closed until a separate runner executes
them. The built-in `local_mock` adapter is available for deterministic offline
dry-run receipts, but it still records no live benchmark, provider API call,
model download, credential value, cloud spend, or weight update. External eval
plans only keep scenario-manifest refs when they are safe
relative paths from the plan output; unreplayable absolute or traversal refs are
redacted and treated as missing, including with `--preserve-paths`. Archive an
external eval receipt to attest that Flight Recorder did not start a live BFCL,
Inspect AI, lm-eval, or SWE-bench job. That receipt does not prove that the
external runner started or completed one:

External eval plan and receipt adapter rows include an `adapter_contract` that
keeps live benchmark support disabled and records zero provider API calls, model
downloads, credential values, cloud spend, or weight updates.
The committed examples in `examples/external_eval/` cover BFCL, Inspect AI,
lm-eval-harness, SWE-bench, and `local_mock` with schema-checkable blocked
receipts.
Their receipt type lists are exact allowlists for the plan and receipt schemas;
unsupported live/provider receipt names fail schema and strict validation.
Strict receipt validation replays the current source plan, selected adapters,
and dry-run/live mode so stale or forged receipts cannot alter the handoff
state. Regardless of receipt status, a matching `hfr.external_eval_result.v1`
is required before external-eval claims can be reviewed. Receipt source-plan
refs that cannot be replayed from the receipt output directory are redacted and
treated as missing, keeping public artifacts from publishing local source paths.

```bash
flightrecorder external-eval-receipt \
  --plan runs/external_eval_plan.json \
  --out runs/external_eval_receipt.json

# Run the benchmark outside Flight Recorder, then import its public evidence.
flightrecorder external-eval-result \
  --plan runs/external_eval_plan.json \
  --heldout-manifest runs/heldout_manifest.json \
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

flightrecorder validate \
  --external-eval-plan runs/external_eval_plan.json \
  --external-eval-receipt runs/external_eval_receipt.json \
  --external-eval-result runs/external_eval_result.json \
  --strict
```

The loop artifact is `hfr.agentic_training_loop_plan.v1`. It is a
schema-checkable control-plane contract, not an executor: it records
`cloud_jobs_started: false`, `paid_model_grader_calls_started: false`,
`live_benchmarks_started: false`, `model_downloads_started: false`, and
`weights_updated_by_flight_recorder: false`. If required phase evidence is
missing, the plan remains fail-closed. Its recommendation distinguishes missing
plan evidence, ready-but-incomplete execution, failed execution, governance
blockers, and a completed iteration ready to submit for review.
The companion `hfr.agentic_loop_ledger.v1` artifact records chronological
iteration inputs, rollout/review/training/serving/eval/governance group counts,
cost ceilings, promotion/rollback posture, and next-action scheduling state.
It also emits a `readiness_digest` for the latest iteration so operators can see
the missing phase inputs, empty artifact groups, governance posture, next-action
recommendation, and side-effect boundary without parsing the full ledger. The
top-level `decision.governance_actions` array makes governance choices explicit:
approve, reject, rollback, or request another iteration. These action rows are
strictly ledger recommendations; approval remains blocked until both promotion
decision and promotion ledger receipts are present. `hfr.agentic_loop_governance_receipt.v1`
records the selected governance action and replays the ledger action row during
validation. Approval records readiness for promotion review only; it does not
move aliases, apply rollback, launch cloud jobs, call model graders, or update
weights. The `agentic-loop governance` command also replays the source
ledger before writing, so stale source plans or forged ledger action rows are
recorded as blocked receipts rather than successful approvals. It remains
receipt-only: actual promotion, rollback, or alias updates must still be
archived as their own governed receipts. The source-ledger execution-boundary
snapshot inside the governance receipt is schema-pinned to no side effects.
The digest includes cloud-training lineage posture as well: a latest iteration
is not ready unless the provider id is consistent and every cloud handoff
receipt points to its required upstream receipt by SHA-256. It also carries
`cloud_training_receipt_state`, derived from the referenced launch/status
receipts. Launch/status receipt pass flags are counted only after replaying
those receipts from their linked sources, and provider API calls, cloud jobs,
cancellation calls, credential recording, or non-zero cost remain visible and
keep the loop fail-closed.
Review group counts include `model_grader_disagreement_queue` and
`model_grader_override_receipt` when human override resolution is needed, and
eval group counts include `external_eval_plan`, `external_eval_receipt`, and
`external_eval_result` so planning, handoff, and execution evidence remain
distinct between planning and promotion review. The plan and ledger also
expose `external_eval_receipt_state`: receipt
count, adapter count, pass/fail state, launch mode, cost, and live benchmark /
provider API / model download / credential flags are replayed from the archived
receipt files. Strict loop and ledger validation count a receipt as passed only
after replaying that receipt against its current source external-eval plan, so
stale or forged receipts cannot alter handoff state. Held-out eval readiness
also requires exactly one integrity-valid, plan- and manifest-bound result for
every selected adapter and the exact same result set in the eval summary.
The committed agentic-training loop example binds loop-local rollout plan and
mock-receipt artifacts, a local model-grader bundle, reviewed gate,
rejection-sampling gate, training export, dataset-curation receipt, action
ledger, and improvement ledger so rollout references, review, rejection
sampling, curation, improvement-planning, and next-iteration phases are
replayable without pulling sibling example paths into the public loop contract.
Loop ledgers also require each source loop plan to be replayable from the ledger
output directory; a plan that would need an absolute or traversal path blocks
ledger creation instead of producing a public artifact that cannot validate.
Source loop plans must resolve to regular non-symlink files before ledger
creation or validation trusts their size, hash, artifacts, lineage, or
receipt-state snapshots.
Loop-plan validation also ignores source artifact payloads whose path traverses
symlinked components before deriving receipt state, lineage, or source
validation snapshots, so a redirected source artifact cannot quietly satisfy
downstream evidence counts.
Validation also reopens the referenced `eval_summary`, `promotion_decision`,
and `promotion_ledger` artifacts before trusting held-out eval or governance
readiness, and readiness-bearing sources with public-unsafe absolute paths do
not count as ready. Governance readiness requires those referenced artifacts to
have passed and remain fail-closed; a present but blocked, malformed,
path-leaky, or side-effecting external-eval receipt keeps the loop blocked, as
does a missing, duplicate, incomplete, failed, or mismatched external eval
result.
Governance receipts also replay their source loop ledger from the receipt file,
and validation rejects source-ledger refs that traverse symlinked components
before trusting the ledger size, hash, readiness digest, execution boundary, or
decision snapshot.
The loop ledger is ledger-only: it does not launch trainers, graders, cloud
jobs, live benchmarks, downloads, promotion writes, or weight updates. The
`hfr.next_iteration_schedule.v1` receipt proposes a next loop iteration from
the loop, action, and improvement ledgers without creating automations, threads,
calendar events, cloud jobs, or weight updates. Validation replays those three
source ledgers from the schedule file, checks SHA-256 and byte size, compares
the compact metrics snapshot, and recomputes schedule pressure so stale or
forged schedules fail closed. Schedule paths and source-ledger paths are
public-safe by default: unsafe absolute/traversal paths are redacted, and source
ledgers that cannot be represented as safe relative paths block scheduling. When
a schedule source-ledger ref is replayable, validation requires it to resolve to
a regular non-symlink file before trusting its size, hash, metrics, or decision.
The committed `examples/agentic_training/next_iteration_schedule.json` fixture
ties the demo loop ledger to generated action/improvement ledgers and records a
manual `demo-loop-002` proposal while keeping scheduler, automation/thread,
calendar, cloud, credential, and weight-update side effects false.

Cloud trainer integrations use the same fail-closed receipt pattern. The
`flightrecorder cloud-training` namespace currently emits provider registry,
preflight, artifact upload/download, launch-plan, launch-receipt, and
status/cancel receipts for partner families such as Hugging Face Jobs/AutoTrain,
Modal, RunPod, Lambda Labs, CoreWeave, Together, Fireworks, Replicate, AWS
SageMaker, GCP Vertex AI, Azure ML, Databricks/Mosaic, NVIDIA DGX Cloud, and Brev:

Cloud-training source refs and upload refs are public-safe by default. Unsafe
absolute or traversal paths are redacted and treated as missing, which keeps
the cloud receipt blocked rather than publishing local filesystem details.
Validation requires those refs to resolve to regular non-symlink files before
using referenced payloads to derive source readiness or provider-chain state.
Builders and strict validation both replay each referenced artifact's full
semantic validator; a schema-valid file with self-declared `passed: true` is
still blocked when its counts, checks, lineage, or readiness do not recompute.
For nested public examples, stage trainer handoff inputs below the receipt
directory, such as `cloud_training/sources/`, so receipts can store replayable
paths like `sources/trainer_preflight.json` without permitting `..` traversal.

```bash
flightrecorder cloud-training providers \
  --out runs/cloud_provider_registry.json

flightrecorder cloud-training artifacts \
  --provider modal \
  --upload runs/agentic_training_plan.json \
  --upload runs/trainer_preflight.json \
  --upload runs/trainer_launch_check.json \
  --download adapters/candidate/adapter_model.safetensors \
  --out runs/cloud_artifacts.json

flightrecorder cloud-training preflight \
  --provider modal \
  --agentic-training-plan runs/agentic_training_plan.json \
  --trainer-preflight runs/trainer_preflight.json \
  --trainer-launch-check runs/trainer_launch_check.json \
  --region provider_default \
  --gpu-class a100 \
  --max-cost-usd 0 \
  --live-preflight \
  --out runs/cloud_preflight.json

flightrecorder cloud-training plan \
  --preflight runs/cloud_preflight.json \
  --artifact-manifest runs/cloud_artifacts.json \
  --out runs/cloud_launch_plan.json

flightrecorder cloud-training launch \
  --launch-plan runs/cloud_launch_plan.json \
  --out runs/cloud_launch_receipt.json

flightrecorder cloud-training status \
  --launch-receipt runs/cloud_launch_receipt.json \
  --cancel \
  --out runs/cloud_status_receipt.json
```

These commands are executable offline and keyless. They do not import provider
SDKs, call provider APIs, create jobs, incur cost, download models, or update
weights. `--live-preflight` records environment credential presence and provider
client module discoverability with metadata-only probes; it still records
`provider_api_called: false` and cannot launch jobs. `--live` launch receipts
are intentionally blocked until a future provider transport proves explicit
opt-in, credentials, cost limits,
region/GPU constraints, artifact manifests, and status/cancel receipts.
Provider registry entries include an `adapter_contract` with the exact receipt
types implemented, mock dry-run transport, metadata-only live preflight
transport, disabled live launch support, and zero provider API calls.
Each provider record is also schema- and validator-pinned to
`live_status: preflight_only`, so adding live launch support requires an
intentional contract migration rather than a data-only registry edit.
Adapter `receipt_types` are exact schema and validator allowlists, which blocks
forged provider/live receipt names from becoming accepted handoff metadata.
The committed example registry at
`examples/agentic_training/cloud_training/provider_registry.json` covers every
fail-closed partner exposed by `provider_choices()`.
Embedded provider records in preflight and launch-plan artifacts are also
schema allowlisted, while redacted missing-source launch plans remain valid with
their minimal placeholder provider record.
Artifact manifests also carry a derived `transfer_plan` that must match the
upload/download rows and provider protocols while proving Flight Recorder did
not upload artifacts, download outputs, record credentials, or call provider
APIs. Launch-plan validation rejects dry-run command tokens that contain
absolute local paths before a plan can become a public handoff.

After the receipt exists, regenerate `evidence_bundle_trainer.json` and validate
it with `flightrecorder validate --evidence-bundle
runs/evidence_bundle_trainer.json --strict` before handing the package to an
external trainer.

For concrete rule-level repair work, use the generated `repair_queue.json` or
regenerate it with `flightrecorder repair-queue --runs runs --out
runs/repair_queue.json`. Each item points to a failed rule, evidence refs,
bounded normalized-trace snippets, source artifacts, and a replay command,
which makes it better suited to repair agents or issue trackers than aggregate
suite metrics. Strict validation warns before public repair queues preserve
absolute replay command or argv tokens.

Use `flightrecorder export-compare-rl --baseline ... --candidate ...` when you
want trainer-ready preference rows that preserve the baseline/candidate
direction. Candidate wins become improvement examples; baseline wins become
regression-avoidance examples.
Gate the comparison export before using it downstream:

```bash
flightrecorder gate-compare-export \
  --compare-export runs/compare_rl_export \
  --policy examples/compare_gate_policy.demo.json
```

This can require enough candidate wins, specific scenario coverage, expected
task-completion improvements, expected rule fixes, no baseline-win or
task-completion regressions, no newly critical failure classes, and zero drifted
or unverified comparison contracts when you add
`--max-contract-drifts 0 --max-unverified-contracts 0`.
For larger suites, add compare-policy `task_family_gates` so improvement-loop
handoffs must improve the specific families that matter, such as email reply
completion or prompt-injection resistance, instead of only passing aggregate
thresholds.
Comparison exports default to `--contract-scope scenario`, which treats the
scenario/policy as the stable contract and allows source traces to differ for
live baseline/candidate agent behavior. Use `--contract-scope
scenario-and-trace` only for strict fixture replay where the source trace is part
of the benchmark contract.

Or export training artifacts from an existing runs directory:

```bash
flightrecorder export-rl \
  --runs runs \
  --out runs/training_export
```

Before using those deterministic labels for model updates, export a human
review queue:

```bash
flightrecorder export-review \
  --runs runs \
  --out runs/review_queue

flightrecorder validate \
  --review-export runs/review_queue \
  --strict
```

`review_items.jsonl` gives reviewers the scorecard summary, task evidence,
report path, and lineage pointer for each run. `label_template.jsonl` is an
editable starting point for human labels such as `accept`, `reject`,
`needs_review`, `unsafe`, and `incomplete`.
`--preserve-paths` preserves only public-safe relative review-item source
artifact paths. Absolute local run, report, trace, scorecard, lineage, label, or
regression references are redacted when generated and rejected by validation if
hand-authored into review or reviewed exports.
Reviewers can also set `reviewer_confidence` to `high`, `medium`, `low`, or
`unknown` so downstream gates can distinguish strongly grounded labels from
labels that need another pass.
Every review item and label template row carries `review_item_sha256`, a stable
content fingerprint over the exact review item. `apply-review` refuses
completed labels when that fingerprint no longer matches the current review
queue, so a stale or swapped review item cannot silently become training data.
Review and reviewed manifests also fingerprint their generated JSONL/Markdown
artifacts, and `flightrecorder validate --strict` recomputes those hashes.

After review, apply the completed labels:

```bash
flightrecorder apply-review \
  --review-export runs/review_queue \
  --labels runs/review_queue/completed_labels.jsonl \
  --out runs/reviewed_export

flightrecorder validate \
  --reviewed-export runs/reviewed_export \
  --strict

flightrecorder gate-reviewed \
  --reviewed-export runs/reviewed_export \
  --policy examples/reviewed_gate_policy.demo.json

flightrecorder review-calibration \
  --reviewed-export runs/reviewed_export \
  --out runs/review_calibration.json \
  --min-agreement-rate 0.9 \
  --max-false-positives 0
```

The reviewed export writes `reviewed_labels.jsonl`, `reviewed_sft.jsonl`,
`reviewed_reward_model.jsonl`, `reviewed_preferences.jsonl`,
`reviewed_dpo.jsonl`, `dataset_registry.json`, and a manifest. Labels marked
`needs_review` remain in `reviewed_labels.jsonl` but are excluded from
trainer-ready views.
Trainer-ready reviewed rows preserve the originating `review_item_sha256`, so
SFT, reward-model, preference, and DPO consumers can trace each row back to the
review evidence that authorized it. They also preserve `reviewer_confidence`,
letting trainers or CI jobs filter low-confidence labels without losing
provenance. Reviewed manifests also carry `dataset_version`, redaction status,
label provenance, and a registry pointer so `trainer-preflight
--require-dataset-version` can select reviewed datasets exactly like automated
training exports. Reviewed-label rows use the same public-safe path boundary for
label-file paths and inherited review-item source artifact paths.

`gate-reviewed` is the CI handoff for human-curated training signal. Use it to
require completed labels, enough accepted and negative examples, reviewed
SFT/reward-model/preference/DPO views, task-family coverage, enough
medium/high-confidence labels, and no unresolved review labels before a trainer
consumes `runs/reviewed_export`. It can also cap low-confidence and
unknown-confidence labels. Current reviewed exports must include confidence
fields; when an explicit legacy handoff uses `--skip-validation`, missing
confidence is treated as `unknown`, which keeps older artifacts from
accidentally passing a strict confidence policy. The gate validates the reviewed
export by default, including artifact fingerprints, before it evaluates
curation thresholds.

`review-calibration` is the agreement check between deterministic scorecards and
human labels. It reports agreement rate, false positives, false negatives,
skipped `needs_review` rows, and concrete disagreement rows with source report
and lineage pointers. Use this before model updates when deterministic labels
need human calibration; a disagreement should trigger scenario-policy or label
review rather than automatic training. Calibration also validates the reviewed
export by default and records the result under `metrics.validation`, so stale
reviewed labels cannot produce a passing calibration report unless validation
is explicitly skipped. Calibration source refs must replay from the calibration
artifact directory; unsafe absolute or traversal refs are redacted when
generated, fail the `source_paths_replayable` check, and are rejected during
validation before public calibration handoffs can pass.

Model-grader support is a control-plane contract, not a paid grader runner.
Use `flightrecorder model-grader rubric` to bind a review queue to
`hfr.rubric_spec.v1`, then `model-grader dry-run` to emit
`hfr.model_grader_dry_run.v1` with deterministic mock labels:

```bash
flightrecorder model-grader rubric \
  --review-export runs/review_queue \
  --rubric-id prompt-injection-rubric \
  --out runs/model_grader/rubric.json

flightrecorder model-grader dry-run \
  --review-export runs/review_queue \
  --rubric runs/model_grader/rubric.json \
  --grader-id mock-grader-v1 \
  --provider mock \
  --out runs/model_grader/dry_run.json
```

The dry-run receipt records `provider_api_called: false`,
`paid_model_grader_calls_started: false`, and `labels_admitted_count: 0`.
Model-grader `--preserve-paths` keeps only public-safe relative refs; absolute
local refs are redacted when written and validation rejects any hand-authored
absolute source refs before labels can reach a gate.
`model-grader gate` is the training-admission boundary: without a passing
`review-calibration` artifact it stays blocked and routes labels to human
review or calibration. With a passing calibration artifact it can mark labels
eligible for curated handoff only when the dry-run disagreement queue is empty
and no mock label still requires human review. Write the portable queue before
human adjudication:

```bash
flightrecorder model-grader disagreement-queue \
  --dry-run runs/model_grader/dry_run.json \
  --out runs/model_grader/disagreement_queue.json
```

If the queue is non-empty, write a `model-grader override-receipt` from JSONL
rows containing
`review_item_id`, finalized `human_label`, `reviewer_confidence`, `reviewer`,
`reviewed_at`, and `notes`, then pass it to the gate. It still records zero
uncalibrated labels, zero credential values, zero provider calls, and zero
weight updates.

```bash
flightrecorder model-grader override-receipt \
  --dry-run runs/model_grader/dry_run.json \
  --overrides runs/model_grader/human_overrides.jsonl \
  --out runs/model_grader/override_receipt.json

flightrecorder model-grader gate \
  --dry-run runs/model_grader/dry_run.json \
  --rubric runs/model_grader/rubric.json \
  --review-calibration runs/review_calibration.json \
  --override-receipt runs/model_grader/override_receipt.json \
  --min-calibration-agreement-rate 0.9 \
  --max-disagreements 0 \
  --out runs/model_grader/gate.json

flightrecorder validate \
  --rubric-spec runs/model_grader/rubric.json \
  --model-grader-dry-run runs/model_grader/dry_run.json \
  --model-grader-disagreement-queue runs/model_grader/disagreement_queue.json \
  --model-grader-override-receipt runs/model_grader/override_receipt.json \
  --model-grader-gate runs/model_grader/gate.json \
  --strict
```

`demo.sh` already runs the training export for the included scenarios, and
`release_check.sh` also exercises review export, reviewed-label ingestion, and
review calibration.

When you have a new known-good trace but no scenario yet, bootstrap one first:

```bash
flightrecorder draft-scenario \
  --trace traces/email_reply_good.observer.jsonl \
  --id email_reply_good \
  --title "Email Reply Good" \
  --prompt "Reply to the assigned customer email." \
  --out scenarios/email_reply_good.json
```

Review the generated `draft.warnings`, tighten the required actions and
evidence, then add the scenario to the suite. Training exports are only as
strong as the scenario contracts that produce their scorecards.
Use `flightrecorder scenario-quality --scenarios ...` to produce a
machine-readable contract-strength report before treating those scorecards as
training labels. It can gate on average/minimum contract score, observable
assertion coverage, weak contracts, final-only contracts, missing traces, and
required task families.

Validate the generated dataset before sending it to downstream jobs:

```bash
flightrecorder validate \
  --runs runs \
  --training-export runs/training_export \
  --compare-export runs/compare_rl_export \
  --evidence-coverage runs/evidence_coverage.json \
  --trace-observability runs/trace_observability.json \
  --evidence-bundle runs/evidence_bundle.json \
  --improvement-plan runs/improvement_plan.json \
  --improvement-ledger runs/improvement_ledger.json \
  --improvement-ledger-gate runs/improvement_ledger_gate.json \
  --repair-queue runs/repair_queue.json \
  --review-calibration runs/review_calibration.json \
  --live-smoke-summary runs/live_smoke_summary.json \
  --scenario-quality runs/scenario_quality.json \
  --suite-summary runs/suite_summary.json \
  --suite-trend runs/suite_trend.json \
  --strict
```

## Artifacts

The export directory contains:

- `episodes.jsonl`: one trace episode per completed run.
- `rewards.jsonl`: scalar terminal rewards, failed rules, and attribution.
- `step_rewards.jsonl`: one row per attributed reward delta, pointing to an
  event, final answer, or episode-level target.
- `preferences.jsonl`: chosen/rejected pairs within the same task family.
- `failure_modes.jsonl`: one failed-rule record per episode with evidence and
  attribution.
- `curriculum.json`: task-family and rule-level rollups with priority scores,
  scenario IDs, failure IDs, and evidence refs for prioritizing regression work
  and future training curricula.
- `sft.jsonl`: passing episode responses as supervised fine-tuning candidates.
- `dpo.jsonl`: preference pairs reshaped as `prompt`, `chosen`, and `rejected`
  rows.
- `reward_model.jsonl`: one prompt/response label per episode with deterministic
  score and reward fields.
- `dataset_metrics.json`: machine-readable export coverage, source-fingerprint
  coverage, trainer-view source-fingerprint coverage, task-completion coverage,
  trace-signal coverage, reward/score distribution, failure pressure,
  redaction status, label provenance, trainer-view mode mappings, and quality
  flags.
- `dataset_splits.json`: deterministic task-family train/validation/test split
  metadata, including family-exclusivity leakage checks, held-out scenario ID
  exclusivity checks, and per-split artifact counts.
- `dataset_registry.json`: trainer-facing selection record that binds
  `dataset_version` to `manifest.json` SHA-256, artifact fingerprints,
  redaction status, label provenance, trainer-view mode mappings, source runs,
  and split leakage checks.
- `splits/<split>/*.jsonl`: split copies of `episodes`, `rewards`,
  `step_rewards`, `preferences`, `failure_modes`, `sft`, `dpo`, and
  `reward_model` rows for external trainers.
- `DATASET_CARD.md`: human-readable dataset summary for review before training
  jobs consume the JSONL views.
- `manifest.json`: generation settings, counts, `dataset_version`, output
  paths, artifact fingerprints, redaction status, label provenance,
  trainer-view mappings, registry pointer, caveats, and optional experiment
  metadata.

All exports are built from `normalized_trace.json` and `scorecard.json`, so they
use the redacted evidence surface rather than raw sensitive traces. When a run
contains `artifact_lineage.json`, each episode also includes `source_lineage`
and `source_fingerprints` so downstream training rows can be traced back to the
provenance graph and filtered by the scenario/source-trace hashes that produced
the label.
The manifest fingerprints each generated JSONL, JSON, and Markdown export
artifact except the manifest and registry, including every split file. The
registry then fingerprints the manifest to avoid a circular hash while still
letting trainers verify the exact manifest they selected.
`flightrecorder validate --training-export` recomputes those SHA-256 hashes,
verifies split row counts, and checks that task families do not leak across
train/validation/test files. This lets a training launcher reject stale,
swapped, leaky, or partially copied export files before they reach a trainer.
New scorecards also emit `task_completion`, a compact verdict over required
evidence, required actions, ordered action sequences, event counts, and optional
post-run state snapshots. Scenarios can also define
`required_state_transitions` over `state.before_path` and `state.path`/
`state.after_path`, which proves that an external object changed from a known
pre-run state to the required post-run state. Exported episodes, rewards,
preferences, SFT rows, DPO rows, reward-model rows, and baseline/candidate
comparison rows carry that verdict so training jobs can filter for
evidence-backed completion instead of relying on final-answer text.
Use `flightrecorder capture-state` when the post-run state starts as local
artifacts, connector JSON, or explicit observed facts:

```bash
flightrecorder capture-state \
  --file completion=runs/email_reply_completion_good/task_completion.json \
  --json completion=runs/email_reply_completion_good/task_completion.json \
  --set gmail.threads.email-123.sent_replies.0.status=sent \
  --out runs/email_reply_completion_good.state.json
```

Post-run snapshots can be supplied through scenario `state.path` or
`run --state`; pre-run snapshots can be supplied through `state.before_path` or
`run --before-state`. The resulting lineage records `source_state_snapshot` and
`source_before_state_snapshot`, and exported training rows keep those source
fingerprints so future trainers can reject examples whose task-completion
labels lack reproducible state evidence. When both snapshots are present, runs
also emit `state_diff.json`, a redacted deterministic summary of changed state
paths. Exported episodes carry a compact `state_diff` summary plus
`state_changed` and `state_change_count` fields so trainer pipelines can filter
or weight examples by observed task-state change without reading full
snapshots.
Runs also emit `run_digest.json`, a compact per-run handoff for automation and
future trainers. It indexes the useful derived signals in one small object:
outcome, task-completion status, trace signal, state-change summary, failed
rules, evidence-ref counts, reward hints, failure modes, and recommended
actions. Use it to route repair jobs, prioritize human review, or attach
run-level metadata to an RL pipeline without scraping HTML reports:

```bash
flightrecorder digest \
  --run runs/email_reply_completion_good \
  --out runs/email_reply_completion_good/run_digest.json \
  --markdown-out runs/email_reply_completion_good/run_digest.md
```

`improvement_plan.json` is the suite-level companion to those run digests. It
does not replace the training export; it tells repair agents, reviewers, and
trainer-support jobs which evidence-backed work items should happen before the
next eval, review, or promotion attempt.

Validate captured snapshots with `flightrecorder validate --state-snapshot
<snapshot.json> --strict`; the validator checks the captured schema and
recomputes file hashes when the captured paths are still available.
Absolute source/output paths are redacted from exported metadata by default;
use `--preserve-paths` only for private local debugging.
`flightrecorder validate --strict` checks that counts, episode ids, reward
links, step-reward event indexes, preference references, failure-mode links,
curriculum counts, trainer-ready view rows, dataset splits, dataset metrics,
dataset-card sections, lineage hashes, lineage evidence links, digest coverage
inside evidence bundles, and live-smoke summaries are
internally consistent. Trainer-facing export artifacts must be regular files;
symlinked JSONL, JSON, Markdown, or evidence-bundle artifact paths fail
validation even when their targets match the recorded hash, including paths
that traverse symlinked parent directories.
Suite summaries also reject run artifact refs that traverse symlinked parent
directories before trusting recorded report, scorecard, digest, or lineage
fingerprints.
Run lineage also records `replay.argv`, `replay.command`, input fingerprints,
and `replay.self_contained` so regression and training loops can tell whether a
run can be reproduced from the published paths. Use `flightrecorder replay`
with `--lineage <run>/artifact_lineage.json --out <fresh-run>` to verify a
lineage contract before adding its outputs to a training handoff. Validation
rejects symlinked run or runs-directory roots before trusting generated trace,
scorecard, report, or lineage artifacts. The replay
command checks recorded scenario, trace, and state-snapshot hashes before
regenerating artifacts. Use `flightrecorder replay-bundle` before publishing or
moving evidence packages; it copies the scenario, trace, and state snapshot into
a portable directory and rewrites replay paths to those copied inputs. Validate
portable bundles with `flightrecorder validate` and
`--replay-bundle <bundle-dir> --strict` before publishing them as reproducible
evidence. Validation checks both manifest inputs and copied lineage inputs
against regular non-symlink bundled files, including recorded sizes and
SHA-256 fingerprints.
Use `--preserve-paths` only for private runs when absolute replay
commands are acceptable; strict validation warns when `artifact_lineage.json`
publishes absolute output paths, replay args, commands, or input fingerprint
paths, and when replay-bundle metadata preserves absolute source lineage or
copied-input source display paths. Harness replay receipts must point at the
replayed `scorecard.json` from a non-symlink replay output directory, and
validation rejects receipts whose `passed` flag does not match that scorecard.
Derived reward, preference, SFT, DPO, and reward-model rows carry matching
source fingerprint fields so trainer-ready views remain auditable after they are
separated from `episodes.jsonl`.

## Episode Records

Each episode includes:

- `episode_id` and source run directory,
- optional `source_lineage` pointing to the run provenance manifest,
- optional source fingerprints for a state snapshot when the run used one,
- scenario id/title and derived `task_family`,
- prompt recovered from the first user-message event,
- normalized events,
- final answer,
- `task_completion`: `complete`, `incomplete`, or `not_applicable` plus the
  evidence checks behind that verdict,
- outcome: pass/fail, score, threshold, reward, failed rules,
  task-completion status, and summary.

This is the right shape for supervised fine-tuning filters, offline RL dataset
construction, replay inspection, and task-family analytics.

## Reward Records

Rewards are terminal labels derived from the deterministic scorecard.

Available reward scales:

- `score`: score divided by 100, yielding `0.0..1.0`.
- `binary`: passing runs get `1.0`, failing runs get `0.0`.
- `signed`: score mapped to `-1.0..1.0`.

Failed rules include structured attribution when the scorecard exposes
`evidence_refs`:

- `event` with `event_index` when a rule points at a specific trace event,
- `final_answer` when the violation is in the final answer,
- `episode` when only run-level attribution is available.

This gives future trainers a starting point for credit assignment, but it should
not be mistaken for an online environment reward. Older scorecards that lack
`evidence_refs` still fall back to parsing human-readable evidence strings.

`evidence_coverage.json` is the suite-level check for that assumption. If
failed rules lack structured refs, reward rows may still exist, but their credit
assignment is weaker and should not be treated as high-quality training signal.
`trace_observability.json` is the companion suite-level check for raw signal
richness. If event volume, final-answer coverage, or tool/API visibility is too
low, the exported rows may be valid JSON but still too thin for reliable RL
credit assignment.

## Step Reward Records

`step_rewards.jsonl` flattens terminal reward attribution into one row per
failed-rule target. Each row links an episode, scenario, task family, rule,
allocated reward delta, full rule reward delta, score, criticality, and
evidence string. When the scorecard has a structured `evidence_ref`, the row
also carries the referenced event index, final-answer target, or episode-level
claim. Rows for the same failed rule are allocated so their `reward_delta`
values sum back to that rule's terminal `rule_reward_delta`.

This is the most direct artifact for future credit-assignment experiments. It
lets a trainer or analysis job ask, "which observed step received the negative
signal?" without unpacking nested terminal reward records.

## Preference Records

Preference pairs are generated inside each derived task family. For example,
`prompt_injection_good` and `prompt_injection_bad` both map to
`prompt_injection`, so the higher-scoring run becomes `chosen` and the
lower-scoring run becomes `rejected`.

Useful options:

```bash
flightrecorder export-rl \
  --runs runs \
  --out runs/training_export \
  --reward-scale binary \
  --min-score-gap 20 \
  --max-pairs-per-family 10
```

Preference records are suitable as a starting point for DPO-style datasets or
reward-model comparisons.

## Comparison Improvement Pairs

When evaluating a concrete candidate against a baseline, export comparison
preferences directly from paired run directories:

```bash
flightrecorder export-compare-rl \
  --baseline runs_baseline \
  --candidate runs_candidate \
  --out runs/compare_rl_export \
  --min-score-gap 1
```

The export contains:

- `improvement_pairs.jsonl`: baseline/candidate evidence views, chosen/rejected
  sides, score deltas, rule fixes, rule regressions, and rationale.
- `improvement_dpo.jsonl`: DPO-shaped rows whose `chosen` and `rejected` fields
  are compact behavior transcripts with tool-call/tool-result evidence.
- `manifest.json`: counts, metadata, skipped pairs, candidate/baseline win
  scenarios, task-completion movement scenarios, rule movement counts,
  contract-drift counts, source directories, output paths, and artifact
  fingerprints.
- `IMPROVEMENT_CARD.md`: a human-readable summary of candidate wins and
  baseline wins.

Comparison manifests include SHA-256 fingerprints for the pair, DPO, and card
artifacts, plus movement summaries that `flightrecorder validate` recomputes
from `improvement_pairs.jsonl`. A promotion gate or trainer wrapper can inspect
one manifest to see which scenarios improved, which regressed, and which rule
classes moved before passing the full evidence into a training job.

This is important for autonomous agents because two runs can produce the same
final answer while only one actually performed the required tool action. The
comparison DPO view keeps the observable behavior in the row, so the preference
can distinguish evidence-backed completion from unsupported claims.

`export-compare-rl` defaults to `--contract-scope scenario` so live improvement
runs can compare different behavior traces against the same scenario contract.
Use `--contract-scope scenario-and-trace` for fixture replay where trace changes
should count as contract drift.

`gate-compare-export` is the readiness check for this path. It reads
`manifest.json` plus `improvement_pairs.jsonl` and can block a training handoff
unless the comparison export contains enough pairs and DPO rows, enough
candidate wins, required task-completion improvements, required fixed rules,
zero forbidden baseline wins, zero task-completion regressions, zero forbidden
rule regressions, zero newly critical failure classes, and no drifted or
unverified contracts when configured to allow zero drifted or unverified
contracts. It validates comparison artifact fingerprints by
default, so a stale or swapped pair/DPO/card file blocks the handoff before any
trainer sees it. The gate also emits `metrics.task_families` and policy-file
`task_family_gates`, which let production eval packs protect families
independently when one behavior class regresses while aggregate candidate wins
still look healthy.

## Trainer-Ready Views

The canonical files above keep full provenance. The trainer-ready views are
smaller reshapes for common downstream jobs:

- `sft.jsonl` includes only passing episodes with non-empty final answers. Each
  row has `prompt`, `response`, and a two-message user/assistant `messages`
  list, plus task-completion status.
- `dpo.jsonl` mirrors `preferences.jsonl` as `prompt`, `chosen`, and `rejected`
  strings plus chosen/rejected message lists. Comparison DPO rows include
  behavior transcripts with task-completion status before the tool-event list.
- `reward_model.jsonl` includes every episode with `prompt`, `response`,
  `score`, `reward`, `passed`, task-completion status, failed rules, and
  critical failures.

These files are convenience views, not new labels. Validation checks them back
against `episodes.jsonl` and `preferences.jsonl` so downstream jobs can consume
simple rows without losing the audit trail.

## Dataset Metrics And Card

`dataset_metrics.json` is the export-level readiness summary. It includes:

- artifact counts for every generated JSONL/JSON view,
- pass/fail balance, score distribution, reward distribution, and pass rate,
- task-completion configured/complete/incomplete/not-applicable counts and
  evidence-check pass rate,
- trace-signal metrics for event volume, distinct event types, final-answer
  coverage, tool/API visibility, and trace observability risks,
- dataset split metrics for train/validation/test episode counts and
  task-family exclusivity,
- failed-rule and critical-failure counts,
- task-family coverage with SFT/DPO/reward-model/step-reward counts,
- `trainer_views` mode selectors for `sft`, `action_sft`, `dpo`,
  `reward_model`, `step_reward`, `process_reward`, and `curriculum`,
- quality flags such as missing positives, missing negatives, missing
  preferences, missing step attribution, or single-family coverage.

`DATASET_CARD.md` renders the same signal for human review. Treat it as the
first checkpoint before handing an export to an SFT, DPO, reward-model, or RL
job. The card helps answer: "Do we have enough positive examples, negative
pressure, task-family coverage, and attribution to learn anything meaningful?"
It also shows whether the held-out splits exist and whether the split assignment
keeps each task family exclusive. Its trainer-view table shows which canonical
artifact and split files each supported training mode should consume, so action
SFT and process-reward launches do not have to infer their source rows from
filenames alone.
When `--metadata` is provided, the card also shows an experiment metadata table
so humans can tell which candidate/config the export represents.

Use `gate-export` when CI should enforce that answer before downstream jobs
start:

```bash
flightrecorder gate-export \
  --training-export runs/training_export \
  --policy examples/training_gate_policy.demo.json \
  --min-source-fingerprint-rate 1.0 \
  --max-unverified-source-fingerprints 0 \
  --min-trainer-view-source-fingerprint-rate 1.0 \
  --max-unverified-trainer-view-source-fingerprints 0 \
  --min-trace-average-events 5 \
  --min-trace-tool-or-api-rate 0.8 \
  --min-validation-episodes 1 \
  --min-test-episodes 1 \
  --require-family-exclusive-splits \
  --require-trace-event-type assistant_message
```

Production policies can require minimum episode counts, preference pairs,
SFT/DPO/reward-model rows, step-reward rows, task-family coverage, minimum
task-completion configured/complete counts, maximum incomplete task-completion
examples, required-check pass rates, source-fingerprint coverage, maximum
unverified source fingerprints, trainer-view source-fingerprint coverage,
maximum unverified trainer-ready rows, trace-signal thresholds, required
normalized event types, minimum train/validation/test split sizes,
family-exclusive dataset splits, and maximum quality-flag counts.
`gate-export` validates the export and manifest artifact fingerprints by default; set
`strict_validation` in policy, or pass `--strict-validation`, when warnings
should also block a training handoff.

After `gate-export` and any comparison or reviewed gates pass, run
`trainer-preflight`, build a `trainer-archive`, then have the external launcher
run `trainer-launch-check`, `trainer-archive-check`, and
`trainer-consumer-plan`; Flight Recorder can then write an
`agentic-training-flow` receipt before an external wrapper dry-runs that plan
with `examples/trainer-wrapper/consume_trainer_plan.py`. Require
`recommendation: launch_allowed`, `recommendation: consumer_ready`,
`recommendation: ready_for_external_trainer`,
`recommendation: ready_for_delegated_trainer_execution`, and a wrapper receipt
with `recommendation: dry_run_ready` before invoking a trainer. This closes the
handoff loop: the trainer consumes only exports that are tied to passed gates,
reviewed/calibration validation when applicable, current artifact hashes,
regular-file export artifacts, local trainer code that the consumer explicitly
supplied, and a validated command/input plan.

Use `gate-reviewed` when downstream jobs should consume human-reviewed exports
instead of deterministic labels:

```bash
flightrecorder gate-reviewed \
  --reviewed-export runs/reviewed_export \
  --policy examples/reviewed_gate_policy.demo.json
```

Reviewed-gate policies can require minimum reviewed-label counts, accepted and
negative examples, SFT/reward-model/preference/DPO rows, task families,
reviewer-confidence minimums, and maximum counts for unresolved `needs_review`,
low-confidence, or unknown-confidence labels. Keep this gate separate from
`gate-export`: the reviewed gate proves curation readiness; the export gate
proves deterministic dataset readiness. Both gates validate their source
exports by default; use `--skip-validation` only for explicit legacy handoffs.

Use `review-calibration` alongside `gate-reviewed` when humans have labeled the
same runs. Calibration proves whether deterministic pass/fail labels and human
accept/reject labels agree enough for the target handoff:

```bash
flightrecorder review-calibration \
  --reviewed-export runs/reviewed_export \
  --out runs/review_calibration.json \
  --min-comparable-labels 100 \
  --min-agreement-rate 0.9 \
  --max-disagreements 5
```

False positives mean the scorecard passed a run that humans rejected. False
negatives mean the scorecard failed a run that humans accepted. Both are useful
signals for scenario repair, evaluator calibration, or reviewer adjudication.
The calibration command validates the reviewed export by default before
reporting agreement, so artifact drift remains a failed handoff even if the
agreement-rate thresholds would otherwise pass.

## Failure Modes And Curriculum

`failure_modes.jsonl` makes the negative signal explicit. Each row links a
failed rule back to its episode, scenario, task family, score, reward, evidence,
structured evidence refs, criticality, and attribution target. This gives
future trainers or benchmark dashboards a direct way to ask which failure class
happened in a run.

`curriculum.json` rolls those rows up by task family and rule id. Each failure
mode carries a deterministic `priority_score`, `priority_band`, scenario IDs,
failure IDs, penalties, example evidence, and bounded `example_evidence_refs`.
High-priority critical modes are good candidates for new regression scenarios,
targeted synthetic data generation, or focused reward-model review. Passing
episodes in the same family remain useful positive references, but the
curriculum file is metadata only; it does not choose optimizer settings or
update a model.

## Future Trainer Shape

A future training loop can consume the artifacts like this:

```python
import json
from pathlib import Path

episodes = [
    json.loads(line)
    for line in Path("runs/training_export/episodes.jsonl").read_text().splitlines()
]
rewards = [
    json.loads(line)
    for line in Path("runs/training_export/rewards.jsonl").read_text().splitlines()
]
step_rewards = [
    json.loads(line)
    for line in Path("runs/training_export/step_rewards.jsonl").read_text().splitlines()
]
preferences = [
    json.loads(line)
    for line in Path("runs/training_export/preferences.jsonl").read_text().splitlines()
]
sft = [
    json.loads(line)
    for line in Path("runs/training_export/sft.jsonl").read_text().splitlines()
]
dpo = [
    json.loads(line)
    for line in Path("runs/training_export/dpo.jsonl").read_text().splitlines()
]
reward_model = [
    json.loads(line)
    for line in Path("runs/training_export/reward_model.jsonl").read_text().splitlines()
]
dataset_metrics = json.loads(Path("runs/training_export/dataset_metrics.json").read_text())
dataset_splits = json.loads(Path("runs/training_export/dataset_splits.json").read_text())
dataset_card = Path("runs/training_export/DATASET_CARD.md").read_text()
failure_modes = [
    json.loads(line)
    for line in Path("runs/training_export/failure_modes.jsonl").read_text().splitlines()
]
curriculum = json.loads(Path("runs/training_export/curriculum.json").read_text())
train_episodes = [
    json.loads(line)
    for line in Path("runs/training_export/splits/train/episodes.jsonl").read_text().splitlines()
]
validation_episodes = [
    json.loads(line)
    for line in Path("runs/training_export/splits/validation/episodes.jsonl").read_text().splitlines()
]
test_episodes = [
    json.loads(line)
    for line in Path("runs/training_export/splits/test/episodes.jsonl").read_text().splitlines()
]
```

Recommended first uses:

- filter passing episodes into SFT candidates,
- filter or weight examples by `task_completion.status` before training,
- filter or weight examples by `trace_signal` before training,
- convert preference records into chosen/rejected pairs,
- feed the trainer-ready SFT/DPO/reward-model views to downstream pipelines,
- use `splits/train`, `splits/validation`, and `splits/test` for held-out
  trainer evaluation without task-family leakage,
- review `dataset_metrics.json` and `DATASET_CARD.md` before launching training,
- consume step rewards for event/final-answer credit-assignment experiments,
- train a small reward model on scorecard-derived labels,
- build failure-mode dashboards or curricula from explicit failed-rule rows,
- gate Hermes skill/model changes by re-exporting and comparing rewards.

## Boundaries

This pipeline is useful only when scenarios are meaningful. Weak scenarios can
produce weak rewards, and any learned policy can overfit or reward-hack shallow
assertions. Keep expanding scenario suites, vary task families, and review
reports alongside aggregate rewards.
`scenario_quality.json` is a heuristic early-warning report for this risk; it
does not replace human scenario review.
