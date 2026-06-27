# Future RL Training Pipeline

Flight Recorder can now export completed run directories into training-ready
JSONL artifacts. This is a bridge from deterministic eval evidence to future
SFT, preference-tuning, reward-modeling, or RL loops.

It is not a trainer. It does not generate rollouts, update model weights, or
guarantee that the reward function is impossible to game. It gives a future
trainer a clean, deterministic data contract grounded in observed traces.

Public artifact schemas are bundled with the package for downstream systems
that need stable contracts before wiring Flight Recorder into review or
training jobs:

```bash
flightrecorder schemas --write-dir artifact_schemas
flightrecorder schemas --name training_manifest --out training_manifest.schema.json
flightrecorder schemas --check runs/training_export/manifest.json
```

Treat those JSON Schemas as shape contracts. `flightrecorder schemas --check`
can gate manifest shape without installing a third-party JSON Schema validator.
Use `flightrecorder validate --strict` and the relevant gates for stronger
readiness checks over artifact hashes, evidence references, replay lineage,
reviewed labels, and trainer preflight handoffs.

## Export

Generate normal Flight Recorder runs first:

```bash
./demo.sh
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
`evidence_coverage.json`, `trace_observability.json`, and
`evidence_bundle.json` during the suite run, so the default handoff is a single
command before stricter policy gates are applied.
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
  --training-export runs/training_export \
  --review-calibration runs/review_calibration.json \
  --live-smoke-summary runs/live_smoke_summary.json \
  --gate runs/suite_gate.json \
  --gate runs/training_gate.json \
  --out runs/evidence_bundle.json
```

The bundle records artifact hashes, readiness checks, gate results, compact
metrics, and a `decision` block with `promote_handoff` or `block_handoff`. It is
useful for provenance and job routing, but it should not be read as permission
to train unless the included scenario, evidence-coverage, validation, review,
and gate policies are also appropriate for the target job.
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
`resolved` relative to the latest bundle.

Use `flightrecorder gate-action-ledger` to block trainer promotion when repair
pressure is not shrinking:

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

flightrecorder gate-promotion-ledger \
  --promotion-ledger runs/promotion_ledger.json \
  --policy examples/promotion_ledger_gate_policy.demo.json \
  --out runs/promotion_ledger_gate.json

flightrecorder validate --promotion-ledger-gate runs/promotion_ledger_gate.json --strict

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
handoff jobs. The generated `decision_gate.json` carries
`source_artifact.sha256`, tying the promotion decision to the exact source gate
artifact it consumed. When that source path is available, validation also
checks that the embedded `source_decision` still matches the source artifact's
current decision block. Use `flightrecorder promotion-ledger` to preserve the
history of those allow/block artifacts across iterations. The promotion ledger
records latest recommendation, allowed/blocked counts, consecutive block or
allow streaks, and source-artifact fingerprints, giving an external trainer
launcher a stable "how did we get here?" artifact before it consumes the final
decision gate. Use `flightrecorder gate-promotion-ledger` when trainer or CI
automation needs a policy decision over that history, such as requiring a clean
latest allow decision, capping blocked-rate or blocked streaks, and forbidding
source `block_iteration` recommendations before launch. Use
`flightrecorder promotion-archive` at the artifact-upload boundary: it copies
the promotion ledger, promotion-ledger gate, decision gates, and resolvable
source gate artifacts into a hash-checked directory that remains valid after
the original workspace paths disappear. Recorded artifact references must be
safe relative paths before they are copied, and validation rejects symlinked
archive artifacts. Keep shared promotion archives in the default redacted mode;
use `--preserve-paths` only for private local debugging.

Use `flightrecorder trainer-preflight` as the final launch guard that an
external trainer can consume. It records the trainer command, fingerprints the
trainer-facing export files, verifies required gates are present and passed, and
refuses training, compare, reviewed, or review-calibration handoffs that skipped
embedded export validation unless `--allow-unvalidated-gates` is explicitly set:

```bash
flightrecorder trainer-preflight \
  --gate runs/training_gate.json \
  --gate runs/compare_gate.json \
  --training-export runs/training_export \
  --compare-export runs/compare_rl_export \
  --evidence-bundle runs/evidence_bundle.json \
  --require-gate training_gate \
  --require-gate compare_gate \
  --trainer-command "python train.py --dry-run --dataset runs/training_export" \
  --out runs/trainer_preflight.json

flightrecorder validate --trainer-preflight runs/trainer_preflight.json --strict

flightrecorder trainer-launch-check \
  --preflight runs/trainer_preflight.json \
  --require-gate training_gate \
  --require-gate compare_gate \
  --out runs/trainer_launch_check.json

flightrecorder validate --trainer-launch-check runs/trainer_launch_check.json --strict
```

The preflight manifest and launch check are still evidence plumbing, not a
trainer. They do not execute the command or update weights. `trainer-preflight`
creates the signed-off evidence contract; `trainer-launch-check` is the
consumer-side check an external training launcher can call immediately before it
runs. It re-validates the preflight hashes and prints the approved command only
when the launch contract still passes. Trainer-facing export files must be
regular files at preflight time; symlinked JSONL, JSON, or Markdown artifacts
block launch even if their targets contain matching bytes.

For concrete rule-level repair work, use the generated `repair_queue.json` or
regenerate it with `flightrecorder repair-queue --runs runs --out
runs/repair_queue.json`. Each item points to a failed rule, evidence refs,
bounded normalized-trace snippets, source artifacts, and a replay command,
which makes it better suited to repair agents or issue trackers than aggregate
suite metrics.

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
`reviewed_dpo.jsonl`, and a manifest. Labels marked `needs_review` remain in
`reviewed_labels.jsonl` but are excluded from trainer-ready views.
Trainer-ready reviewed rows preserve the originating `review_item_sha256`, so
SFT, reward-model, preference, and DPO consumers can trace each row back to the
review evidence that authorized it.

`gate-reviewed` is the CI handoff for human-curated training signal. Use it to
require completed labels, enough accepted and negative examples, reviewed
SFT/reward-model/preference/DPO views, task-family coverage, and no unresolved
review labels before a trainer consumes `runs/reviewed_export`. It validates
the reviewed export by default, including artifact fingerprints, before it
evaluates curation thresholds.

`review-calibration` is the agreement check between deterministic scorecards and
human labels. It reports agreement rate, false positives, false negatives,
skipped `needs_review` rows, and concrete disagreement rows with source report
and lineage pointers. Use this before model updates when deterministic labels
need human calibration; a disagreement should trigger scenario-policy or label
review rather than automatic training. Calibration also validates the reviewed
export by default and records the result under `metrics.validation`, so stale
reviewed labels cannot produce a passing calibration report unless validation
is explicitly skipped.

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
  trace-signal coverage, reward/score distribution, failure pressure, and
  quality flags.
- `DATASET_CARD.md`: human-readable dataset summary for review before training
  jobs consume the JSONL views.
- `manifest.json`: generation settings, counts, output paths, artifact
  fingerprints, caveats, and optional experiment metadata.

All exports are built from `normalized_trace.json` and `scorecard.json`, so they
use the redacted evidence surface rather than raw sensitive traces. When a run
contains `artifact_lineage.json`, each episode also includes `source_lineage`
and `source_fingerprints` so downstream training rows can be traced back to the
provenance graph and filtered by the scenario/source-trace hashes that produced
the label.
The manifest also fingerprints each generated JSONL, JSON, and Markdown export
artifact except the manifest itself. Validation recomputes those SHA-256 hashes,
which lets a training launcher reject stale, swapped, or partially copied export
files before they reach a trainer.
New scorecards also emit `task_completion`, a compact verdict over required
evidence, required actions, ordered action sequences, event counts, and optional
post-run state snapshots. Exported episodes, rewards, preferences, SFT rows, DPO
rows, reward-model rows, and baseline/candidate comparison rows carry that
verdict so training jobs can filter for evidence-backed completion instead of
relying on final-answer text.
Use `flightrecorder capture-state` when the post-run state starts as local
artifacts, connector JSON, or explicit observed facts:

```bash
flightrecorder capture-state \
  --file completion=runs/email_reply_completion_good/task_completion.json \
  --json completion=runs/email_reply_completion_good/task_completion.json \
  --set gmail.threads.email-123.sent_replies.0.status=sent \
  --out runs/email_reply_completion_good.state.json
```

Those snapshots can be supplied through scenario `state.path` or `run --state`.
The resulting lineage records `source_state_snapshot`, and exported training
rows keep the source fingerprint so future trainers can reject examples whose
task-completion labels lack reproducible post-run state evidence.
Validate captured snapshots with `flightrecorder validate --state-snapshot
<snapshot.json> --strict`; the validator checks the captured schema and
recomputes file hashes when the captured paths are still available.
Absolute source/output paths are redacted from exported metadata by default;
use `--preserve-paths` only for private local debugging.
`flightrecorder validate --strict` checks that counts, episode ids, reward
links, step-reward event indexes, preference references, failure-mode links,
curriculum counts, trainer-ready view rows, dataset metrics, dataset-card
sections, lineage hashes, lineage evidence links, and live-smoke summaries are
internally consistent. Trainer-facing export artifacts must be regular files;
symlinked JSONL, JSON, or Markdown artifacts fail validation even when their
targets match the recorded hash.
Run lineage also records `replay.argv`, `replay.command`, input fingerprints,
and `replay.self_contained` so regression and training loops can tell whether a
run can be reproduced from the published paths. Use `flightrecorder replay`
with `--lineage <run>/artifact_lineage.json --out <fresh-run>` to verify a
lineage contract before adding its outputs to a training handoff. The replay
command checks recorded scenario, trace, and state-snapshot hashes before
regenerating artifacts. Use `flightrecorder replay-bundle` before publishing or
moving evidence packages; it copies the scenario, trace, and state snapshot into
a portable directory and rewrites replay paths to those copied inputs. Validate
portable bundles with `flightrecorder validate` and
`--replay-bundle <bundle-dir> --strict` before publishing them as reproducible
evidence. Use `--preserve-paths` only for private runs when absolute replay
commands are acceptable.
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
- `manifest.json`: counts, metadata, skipped pairs, contract-drift counts,
  source directories, output paths, and artifact fingerprints.
- `IMPROVEMENT_CARD.md`: a human-readable summary of candidate wins and
  baseline wins.

Comparison manifests include SHA-256 fingerprints for the pair, DPO, and card
artifacts, so a promotion gate can verify the exact improvement evidence before
passing it into a training job.

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
- failed-rule and critical-failure counts,
- task-family coverage with SFT/DPO/reward-model/step-reward counts,
- quality flags such as missing positives, missing negatives, missing
  preferences, missing step attribution, or single-family coverage.

`DATASET_CARD.md` renders the same signal for human review. Treat it as the
first checkpoint before handing an export to an SFT, DPO, reward-model, or RL
job. The card helps answer: "Do we have enough positive examples, negative
pressure, task-family coverage, and attribution to learn anything meaningful?"
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
  --require-trace-event-type assistant_message
```

Production policies can require minimum episode counts, preference pairs,
SFT/DPO/reward-model rows, step-reward rows, task-family coverage, minimum
task-completion configured/complete counts, maximum incomplete task-completion
examples, required-check pass rates, source-fingerprint coverage, maximum
unverified source fingerprints, trainer-view source-fingerprint coverage,
maximum unverified trainer-ready rows, trace-signal thresholds, required
normalized event types, and maximum quality-flag counts. `gate-export` validates
the export and manifest artifact fingerprints by default; set
`strict_validation` in policy, or pass `--strict-validation`, when warnings
should also block a training handoff.

After `gate-export` and any comparison or reviewed gates pass, run
`trainer-preflight`, then have the external launcher run `trainer-launch-check`
and require `recommendation: launch_allowed` before invoking a trainer. This
closes the handoff loop: the trainer consumes only exports that are tied to
passed gates, reviewed/calibration validation when applicable, current artifact
hashes, and regular-file export artifacts.

Use `gate-reviewed` when downstream jobs should consume human-reviewed exports
instead of deterministic labels:

```bash
flightrecorder gate-reviewed \
  --reviewed-export runs/reviewed_export \
  --policy examples/reviewed_gate_policy.demo.json
```

Reviewed-gate policies can require minimum reviewed-label counts, accepted and
negative examples, SFT/reward-model/preference/DPO rows, task families, and a
maximum number of unresolved `needs_review` labels. Keep this gate separate from
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
dataset_card = Path("runs/training_export/DATASET_CARD.md").read_text()
failure_modes = [
    json.loads(line)
    for line in Path("runs/training_export/failure_modes.jsonl").read_text().splitlines()
]
curriculum = json.loads(Path("runs/training_export/curriculum.json").read_text())
```

Recommended first uses:

- filter passing episodes into SFT candidates,
- filter or weight examples by `task_completion.status` before training,
- filter or weight examples by `trace_signal` before training,
- convert preference records into chosen/rejected pairs,
- feed the trainer-ready SFT/DPO/reward-model views to downstream pipelines,
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
