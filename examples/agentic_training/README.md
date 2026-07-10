# Agentic Training Plan Example

This directory contains a tiny registered-input fixture for the Goal 5 Training
Layer. It is safe to commit because it uses synthetic, redacted rows and a local
test model manifest.

Regenerate the sample plan with:

```bash
python3 scripts/plan_agentic_training.py \
  --mode sft_then_dpo \
  --model-manifest examples/agentic_training/model_manifest.json \
  --dataset-manifest examples/agentic_training/dataset_manifest.json \
  --trainer-backend axolotl \
  --output-dir runs/agentic_training/adapters \
  --limit 2 \
  --created-at 2026-07-02T00:00:00+00:00 \
  --out examples/agentic_training/plans/sft_then_dpo_plan.json
```

The plan is a handoff contract only. It does not import trainer packages,
download model weights, mutate aliases, or launch training.

Check the local tiny-smoke runtime boundary without importing trainer stacks:

```bash
python3 scripts/preflight_agentic_training_runtime.py \
  --plan examples/agentic_training/plans/sft_then_dpo_plan.json \
  --skip-default-modules \
  --require-module json \
  --created-at 2026-07-02T00:00:00+00:00 \
  --out examples/agentic_training/runtime_preflight/ready.json
```

The runtime preflight remains side-effect-free. It validates the plan and
selected trainer-view JSONL files, checks module discoverability with
`importlib.util.find_spec`, and records that Flight Recorder did not start
training or model downloads.

Record a synthetic external trainer result receipt with the committed
trainer-output fixture:

```bash
python3 scripts/archive_agentic_training_result.py \
  --plan examples/agentic_training/plans/sft_then_dpo_plan.json \
  --runtime-preflight examples/agentic_training/runtime_preflight/ready.json \
  --agentic-training-flow examples/agentic_training/agentic_training_flow.json \
  --status completed \
  --runner-id synthetic-example-runner \
  --run-id synthetic-completed-001 \
  --output-dir examples/agentic_training/trainer_outputs/adapter \
  --config examples/agentic_training/trainer_outputs/adapter/adapter_config.json \
  --adapter examples/agentic_training/trainer_outputs/adapter/adapter_model.safetensors \
  --metrics examples/agentic_training/trainer_outputs/metrics.json \
  --log examples/agentic_training/trainer_outputs/trainer.log \
  --created-at 2026-07-02T00:00:00+00:00 \
  --out examples/agentic_training/completed_result.json
```

The result receipt proposes size-bound model-registry links but does not mutate
registry entries, move aliases, download models, or train weights.

Generate the loop-local rollout bundle before binding the loop contract:

```bash
flightrecorder agentic-rollout-plan \
  --iteration-id demo-loop-001 \
  --scenario examples/agentic_training/rollouts/scenarios/prompt_injection_good.json \
  --scenario examples/agentic_training/rollouts/scenarios/email_reply_completion_good.json \
  --policy baseline=local/mock-baseline \
  --policy candidate=local/mock-candidate \
  --policy teacher=local/mock-teacher \
  --max-rollouts 6 \
  --verifier examples/agentic_training/rollouts/verifiers/sqlite_task_state.verifier.json \
  --created-at 2026-07-03T00:00:00+00:00 \
  --out examples/agentic_training/rollouts/rollout_plan.json

flightrecorder agentic-rollout-receipt \
  --plan examples/agentic_training/rollouts/rollout_plan.json \
  --created-at 2026-07-03T00:00:00+00:00 \
  --out examples/agentic_training/rollouts/rollout_receipt.json
```

The rollout receipt records deterministic mock rows only. It does not call model
providers, start live rollouts, write traces or scorecards, invoke paid graders,
or create training rows.

Gate the tiny reviewed export and admit the mock rollouts to curation without
writing datasets:

```bash
flightrecorder gate-reviewed \
  --reviewed-export examples/agentic_training/model_grader/reviewed \
  --min-reviewed-labels 2 \
  --min-accepted 1 \
  --min-rejected 1 \
  --min-sft 1 \
  --min-reward-model 2 \
  --min-preferences 1 \
  --min-dpo 1 \
  --min-medium-or-high-confidence-labels 2 \
  --max-needs-review 0 \
  --max-low-confidence-labels 0 \
  --max-unknown-confidence-labels 0 \
  --out examples/agentic_training/model_grader/reviewed_gate.json

flightrecorder rejection-sampling-gate \
  --rollout-receipt examples/agentic_training/rollouts/rollout_receipt.json \
  --model-grader-gate examples/agentic_training/model_grader/passing_gate.json \
  --review-calibration examples/agentic_training/model_grader/review_calibration.json \
  --reviewed-gate examples/agentic_training/model_grader/reviewed_gate.json \
  --created-at 2026-07-03T00:00:00+00:00 \
  --out examples/agentic_training/rejection_sampling_gate.json
```

The rejection-sampling gate is ready for dataset curation, but it remains
gate-only: it does not write accepted/rejected rows or update weights.

Build the real trainer-view export and archive the side-effect-free curation
receipt:

```bash
flightrecorder run-suite \
  --scenarios scenarios \
  --out examples/agentic_training/training_runs \
  --export-rl \
  --training-export-out examples/agentic_training/training_export \
  --metadata agent=hermes-demo \
  --metadata candidate=mock-candidate \
  --metadata model=local-mock

rm -rf examples/agentic_training/training_runs

flightrecorder dataset-curation-receipt \
  --rejection-sampling-gate examples/agentic_training/rejection_sampling_gate.json \
  --training-export examples/agentic_training/training_export \
  --created-at 2026-07-03T00:00:00+00:00 \
  --out examples/agentic_training/dataset_curation_receipt.json
```

The committed export is a deterministic local `export-rl` bundle. The curation
receipt binds it to rejection sampling, but records that Flight Recorder did not
write curated rows, update registries, start cloud jobs, or update weights.

Gate the curated export and record the local dry-run trainer handoff:

```bash
dataset_version="$(python3.11 -c 'import json, pathlib; print(json.loads(pathlib.Path("examples/agentic_training/training_export/manifest.json").read_text())["dataset_version"])')"

flightrecorder gate-export \
  --training-export examples/agentic_training/training_export \
  --policy examples/training_gate_policy.demo.json \
  --out examples/agentic_training/training_gate.json

flightrecorder trainer-preflight \
  --gate examples/agentic_training/training_gate.json \
  --training-export examples/agentic_training/training_export \
  --agentic-training-plan examples/agentic_training/plans/sft_then_dpo_plan.json \
  --require-gate training_gate \
  --require-dataset-version "$dataset_version" \
  --trainer-command "python train.py --dataset training_export --dry-run" \
  --metadata launcher=dry-run \
  --out examples/agentic_training/trainer_preflight.json

flightrecorder trainer-launch-check \
  --preflight examples/agentic_training/trainer_preflight.json \
  --require-gate training_gate \
  --require-dataset-version "$dataset_version" \
  --require-metadata launcher=dry-run \
  --out examples/agentic_training/trainer_launch_check.json \
  --strict
```

Refresh the dry-run cloud-training handoff from replayable source snapshots:

```bash
mkdir -p examples/agentic_training/cloud_training/sources/plans
cp examples/agentic_training/plans/sft_then_dpo_plan.json \
  examples/agentic_training/cloud_training/sources/plans/sft_then_dpo_plan.json

# Build a cloud-local, self-contained trainer guard. Copying the repository-root
# preflight would strand its relative training-export references after nesting.
flightrecorder trainer-preflight \
  --gate examples/agentic_training/cloud_training/sources/plans/sft_then_dpo_plan.json \
  --agentic-training-plan examples/agentic_training/cloud_training/sources/plans/sft_then_dpo_plan.json \
  --require-gate agentic_training_plan \
  --trainer-command "python train.py --agentic-plan plans/sft_then_dpo_plan.json --dry-run" \
  --metadata launcher=cloud-dry-run \
  --out examples/agentic_training/cloud_training/sources/trainer_preflight.json

flightrecorder trainer-launch-check \
  --preflight examples/agentic_training/cloud_training/sources/trainer_preflight.json \
  --require-gate agentic_training_plan \
  --require-metadata launcher=cloud-dry-run \
  --out examples/agentic_training/cloud_training/sources/trainer_launch_check.json \
  --strict

flightrecorder cloud-training preflight \
  --provider modal \
  --agentic-training-plan examples/agentic_training/cloud_training/sources/plans/sft_then_dpo_plan.json \
  --trainer-preflight examples/agentic_training/cloud_training/sources/trainer_preflight.json \
  --trainer-launch-check examples/agentic_training/cloud_training/sources/trainer_launch_check.json \
  --region provider_default \
  --gpu-class a100 \
  --max-cost-usd 0 \
  --created-at 2026-07-03T00:00:00+00:00 \
  --out examples/agentic_training/cloud_training/preflight.json

flightrecorder cloud-training artifacts \
  --provider modal \
  --upload examples/agentic_training/cloud_training/sources/plans/sft_then_dpo_plan.json \
  --upload examples/agentic_training/cloud_training/sources/trainer_preflight.json \
  --upload examples/agentic_training/cloud_training/sources/trainer_launch_check.json \
  --created-at 2026-07-03T00:00:00+00:00 \
  --out examples/agentic_training/cloud_training/artifact_manifest.json

flightrecorder cloud-training plan \
  --preflight examples/agentic_training/cloud_training/preflight.json \
  --artifact-manifest examples/agentic_training/cloud_training/artifact_manifest.json \
  --created-at 2026-07-03T00:00:00+00:00 \
  --out examples/agentic_training/cloud_training/launch_plan.json

flightrecorder cloud-training launch \
  --launch-plan examples/agentic_training/cloud_training/launch_plan.json \
  --created-at 2026-07-03T00:00:00+00:00 \
  --out examples/agentic_training/cloud_training/launch_receipt.json

flightrecorder cloud-training status \
  --launch-receipt examples/agentic_training/cloud_training/launch_receipt.json \
  --created-at 2026-07-03T00:00:00+00:00 \
  --out examples/agentic_training/cloud_training/status_receipt.json
```

Seed the held-out eval lane with a fail-closed external adapter handoff and an
import-only result:
the committed `baseline_suite_summary.json` and `candidate_suite_summary.json`
cover the held-out scenario IDs excluded from the training export.

```bash
(cd examples/agentic_training/heldout_eval && \
  flightrecorder run-suite \
    --scenarios scenarios \
    --suite-manifest heldout_suite_manifest.json \
    --out heldout_runs \
    --summary-out baseline_suite_summary.json \
    --no-index \
    --junit \
    --markdown \
    --validate \
    --strict && \
  cp baseline_suite_summary.json candidate_suite_summary.json)

flightrecorder heldout-manifest \
  --suite-summary baseline=examples/agentic_training/heldout_eval/baseline_suite_summary.json \
  --suite-summary candidate=examples/agentic_training/heldout_eval/candidate_suite_summary.json \
  --out examples/agentic_training/heldout_eval/heldout_manifest.json

flightrecorder external-eval-plan \
  --adapter local_mock \
  --scenario-manifest examples/agentic_training/heldout_eval/heldout_manifest.json \
  --model-endpoint local/mock-candidate \
  --model local/mock-candidate \
  --allow-installed \
  --out examples/agentic_training/heldout_eval/external_eval_plan.json

flightrecorder external-eval-receipt \
  --plan examples/agentic_training/heldout_eval/external_eval_plan.json \
  --created-at 2026-07-03T00:00:00+00:00 \
  --out examples/agentic_training/heldout_eval/external_eval_receipt.json

flightrecorder external-eval-result \
  --plan examples/agentic_training/heldout_eval/external_eval_plan.json \
  --heldout-manifest examples/agentic_training/heldout_eval/heldout_manifest.json \
  --raw-result examples/agentic_training/heldout_eval/external_eval_raw_result.json \
  --runner-metadata examples/agentic_training/heldout_eval/external_eval_runner.json \
  --adapter local_mock \
  --execution-id local-mock-eval-001 \
  --model-id local/mock-candidate \
  --normalizer-id hfr.local_mock.per_case_json \
  --normalizer-version 1 \
  --raw-format json \
  --status completed \
  --created-at 2026-07-03T00:00:00+00:00 \
  --out examples/agentic_training/heldout_eval/external_eval_result.json

flightrecorder eval-summary \
  --suite-summary baseline=examples/agentic_training/heldout_eval/baseline_suite_summary.json \
  --suite-summary candidate=examples/agentic_training/heldout_eval/candidate_suite_summary.json \
  --external-adapter-plan local_mock=examples/agentic_training/heldout_eval/external_eval_plan.json \
  --external-adapter-result local_mock=examples/agentic_training/heldout_eval/external_eval_result.json \
  --out examples/agentic_training/heldout_eval/eval_summary.json \
  --markdown-out examples/agentic_training/heldout_eval/eval_summary.md
```

The dry-run receipt proves only that Flight Recorder prepared a side-effect-free
handoff; it is not benchmark-completion evidence. The committed runner metadata
and per-case raw result represent synthetic output owned by the external local
runner.
`external-eval-result` imports and fingerprints those files without loading an
adapter package or launching a benchmark. A completed import can truthfully
carry either a passed or failed benchmark outcome, but a failed outcome blocks
claims and governance readiness while remaining separate from receipt
integrity.

The agentic-training fixture uses the built-in `local_mock` adapter so its plan
and per-case result can be reproduced offline while Flight Recorder itself
makes zero provider API calls, model downloads, benchmark launches, credential
recordings, cloud spend, or weight updates.

Real BFCL, Inspect AI, lm-eval-harness, and SWE-bench adapters remain
fail-closed in the standalone `examples/external_eval/` fixtures until their
dependencies, inputs, and explicit opt-in flags are present; each selected
adapter must then supply exactly one plan-bound result.

Generate the evidence handoff harness result and bundle:

```bash
(cd examples/agentic_training && \
  flightrecorder run-suite \
    --scenarios evidence_handoff/scenarios \
    --pattern prompt_injection_good.json \
    --out evidence_handoff \
    --no-index \
    --evidence-handoff \
    --validate \
    --strict)
```

The committed evidence handoff uses one deterministic passing scenario to keep
the public fixture compact. It emits `harness_handoff/harness_result.json` and
`evidence_bundle.json` without raw sensitive traces or live external calls.

Refresh the offline promotion-governance gates:

```bash
flightrecorder run \
  --scenario examples/agentic_training/promotion_governance/compare_scenarios/email_reply_completion.json \
  --trace fixtures/email_reply_completion_bad.observer.jsonl \
  --before-state fixtures/email_reply_completion_before.state.json \
  --state fixtures/email_reply_completion_bad.state.json \
  --out examples/agentic_training/promotion_governance/compare_runs/baseline/email_reply_completion

flightrecorder run \
  --scenario examples/agentic_training/promotion_governance/compare_scenarios/email_reply_completion.json \
  --trace fixtures/email_reply_completion_good.observer.jsonl \
  --before-state fixtures/email_reply_completion_before.state.json \
  --state fixtures/email_reply_completion_good.state.json \
  --out examples/agentic_training/promotion_governance/compare_runs/candidate/email_reply_completion

flightrecorder export-compare-rl \
  --baseline examples/agentic_training/promotion_governance/compare_runs/baseline \
  --candidate examples/agentic_training/promotion_governance/compare_runs/candidate \
  --out examples/agentic_training/promotion_governance/compare_export \
  --metadata baseline=local/mock-baseline \
  --metadata candidate=local/mock-candidate \
  --metadata contract=shared-email-reply-scenario

flightrecorder gate-compare-export \
  --compare-export examples/agentic_training/promotion_governance/compare_export \
  --policy examples/compare_gate_policy.demo.json \
  --out examples/agentic_training/promotion_governance/compare_gate.json

flightrecorder gate-decision \
  --artifact examples/agentic_training/promotion_governance/compare_gate.json \
  --expect-recommendation promote_iteration \
  --expect-readiness ready \
  --require-passed \
  --out examples/agentic_training/promotion_governance/promotion_history_decision_gate.json

flightrecorder promotion-ledger \
  --decision-gate examples/agentic_training/promotion_governance/promotion_history_decision_gate.json \
  --out examples/agentic_training/promotion_governance/promotion_ledger.json

flightrecorder gate-promotion-ledger \
  --promotion-ledger examples/agentic_training/promotion_governance/promotion_ledger.json \
  --policy examples/promotion_ledger_gate_policy.demo.json \
  --out examples/agentic_training/promotion_governance/promotion_ledger_gate.json

flightrecorder promotion-cards \
  --candidate-id local/mock-candidate \
  --dataset-id agentic-training-export \
  --model-source examples/agentic_training/completed_result.json \
  --license-status known \
  --evidence-bundle examples/agentic_training/evidence_handoff/evidence_bundle.json \
  --training-export examples/agentic_training/training_export \
  --compare-gate examples/agentic_training/promotion_governance/compare_gate.json \
  --redaction-check examples/agentic_training/promotion_governance/redaction_check.json \
  --safety-gate examples/agentic_training/promotion_governance/safety_gate.json \
  --out examples/agentic_training/promotion_governance/promotion_cards

flightrecorder promotion-decision \
  --candidate-id local/mock-candidate \
  --champion-id local/mock-baseline \
  --rollback-id local/mock-baseline \
  --candidate-class trace-only \
  --champion-class trace-only \
  --evidence-bundle examples/agentic_training/evidence_handoff/evidence_bundle.json \
  --promotion-ledger-gate examples/agentic_training/promotion_governance/promotion_ledger_gate.json \
  --compare-gate examples/agentic_training/promotion_governance/compare_gate.json \
  --trainer-launch-check examples/agentic_training/trainer_launch_check.json \
  --model-registry-entry examples/agentic_training/promotion_governance/model_registry_entry.json \
  --agentic-training-result examples/agentic_training/completed_result.json \
  --model-card examples/agentic_training/promotion_governance/promotion_cards/MODEL_CARD.md \
  --dataset-card examples/agentic_training/promotion_governance/promotion_cards/DATASET_CARD.md \
  --rollback-metadata examples/agentic_training/promotion_governance/rollback_metadata.json \
  --license-review examples/agentic_training/promotion_governance/license_review.json \
  --redaction-check examples/agentic_training/promotion_governance/redaction_check.json \
  --safety-gate examples/agentic_training/promotion_governance/safety_gate.json \
  --serving-profile examples/agentic_training/serving_lifecycle/managed_mock/preflight/serving_profile.json \
  --serving-report examples/agentic_training/serving_lifecycle/managed_mock/preflight/serving_check.json \
  --promotion-policy examples/promotion_policy.demo.json \
  --out examples/agentic_training/promotion_governance/promotion_decision.json

flightrecorder gate-decision \
  --artifact examples/agentic_training/promotion_governance/promotion_decision.json \
  --expect-recommendation apply_alias_update \
  --expect-readiness ready \
  --require-passed \
  --out examples/agentic_training/promotion_governance/promotion_decision_gate.json

flightrecorder promotion-rollback-receipt \
  --registry examples/agentic_training/promotion_governance/model_registry_before_alias_apply.json \
  --rollback-id local/mock-baseline \
  --champion-id local/mock-baseline \
  --out examples/agentic_training/promotion_governance/promotion_rollback_receipt.json

cp examples/agentic_training/promotion_governance/model_registry_before_alias_apply.json \
  examples/agentic_training/promotion_governance/model_registry.json

flightrecorder promotion-alias-apply \
  --registry examples/agentic_training/promotion_governance/model_registry.json \
  --promotion-decision examples/agentic_training/promotion_governance/promotion_decision.json \
  --out examples/agentic_training/promotion_governance/promotion_alias_apply.json

flightrecorder promotion-release-record \
  --release-id demo-loop-001-release \
  --promotion-decision examples/agentic_training/promotion_governance/promotion_decision.json \
  --promotion-cards examples/agentic_training/promotion_governance/promotion_cards \
  --promotion-alias-apply examples/agentic_training/promotion_governance/promotion_alias_apply.json \
  --rollback-metadata examples/agentic_training/promotion_governance/promotion_rollback_receipt.json \
  --compare-gate examples/agentic_training/promotion_governance/compare_gate.json \
  --release-notes examples/agentic_training/promotion_governance/RELEASE_NOTES.md \
  --promotion-policy examples/promotion_policy.demo.json \
  --out examples/agentic_training/promotion_governance/promotion_release_record.json

flightrecorder promotion-archive \
  --promotion-ledger examples/agentic_training/promotion_governance/promotion_ledger.json \
  --promotion-ledger-gate examples/agentic_training/promotion_governance/promotion_ledger_gate.json \
  --decision-gate examples/agentic_training/promotion_governance/promotion_history_decision_gate.json \
  --decision-gate examples/agentic_training/promotion_governance/promotion_decision_gate.json \
  --promotion-release-record examples/agentic_training/promotion_governance/promotion_release_record.json \
  --out examples/agentic_training/promotion_governance/promotion_archive \
  --require-self-contained \
  --force
```

The committed promotion decision consumes that compare gate and promotion
history gate. It authorizes a reviewable alias update, but it does not update
weights or call a provider. Alias movement is represented only by the separate
guarded `promotion-alias-apply` command against a local mock
`model_registry.json`; the release record and archive bind those receipts
without publishing external artifacts.

Bind the example receipts into a fail-closed loop contract:

```bash
flightrecorder agentic-loop plan \
  --iteration-id demo-loop-001 \
  --objective "Close held-out tool-use regressions with fail-closed external trainers." \
  --baseline local/mock-baseline \
  --candidate local/mock-candidate \
  --teacher local/mock-teacher \
  --provider mock \
  --region local \
  --gpu-class none \
  --budget max_cloud_cost_usd=0 \
  --budget max_gpu_hours=0 \
  --agentic-rollout-plan examples/agentic_training/rollouts/rollout_plan.json \
  --agentic-rollout-receipt examples/agentic_training/rollouts/rollout_receipt.json \
  --harness-result examples/agentic_training/evidence_handoff/harness_handoff/harness_result.json \
  --evidence-bundle examples/agentic_training/evidence_handoff/evidence_bundle.json \
  --reviewed-gate examples/agentic_training/model_grader/reviewed_gate.json \
  --rejection-sampling-gate examples/agentic_training/rejection_sampling_gate.json \
  --dataset-curation-receipt examples/agentic_training/dataset_curation_receipt.json \
  --training-export examples/agentic_training/training_export \
  --trainer-preflight examples/agentic_training/cloud_training/sources/trainer_preflight.json \
  --trainer-launch-check examples/agentic_training/cloud_training/sources/trainer_launch_check.json \
  --agentic-training-plan examples/agentic_training/plans/sft_then_dpo_plan.json \
  --agentic-training-runtime-preflight examples/agentic_training/runtime_preflight/ready.json \
  --agentic-training-flow examples/agentic_training/agentic_training_flow.json \
  --agentic-training-result examples/agentic_training/completed_result.json \
  --cloud-training-provider-registry examples/agentic_training/cloud_training/provider_registry.json \
  --cloud-training-preflight examples/agentic_training/cloud_training/preflight.json \
  --cloud-training-artifact-manifest examples/agentic_training/cloud_training/artifact_manifest.json \
  --cloud-training-launch-plan examples/agentic_training/cloud_training/launch_plan.json \
  --cloud-training-launch-receipt examples/agentic_training/cloud_training/launch_receipt.json \
  --cloud-training-status-receipt examples/agentic_training/cloud_training/status_receipt.json \
  --serving-lifecycle examples/agentic_training/serving_lifecycle/managed_mock/serving_lifecycle.json \
  --heldout-manifest examples/agentic_training/heldout_eval/heldout_manifest.json \
  --external-eval-plan examples/agentic_training/heldout_eval/external_eval_plan.json \
  --external-eval-receipt examples/agentic_training/heldout_eval/external_eval_receipt.json \
  --external-eval-result examples/agentic_training/heldout_eval/external_eval_result.json \
  --eval-summary examples/agentic_training/heldout_eval/eval_summary.json \
  --rubric-spec examples/agentic_training/model_grader/rubric.json \
  --model-grader-dry-run examples/agentic_training/model_grader/dry_run.json \
  --model-grader-disagreement-queue examples/agentic_training/model_grader/disagreement_queue.json \
  --model-grader-gate examples/agentic_training/model_grader/passing_gate.json \
  --review-calibration examples/agentic_training/model_grader/review_calibration.json \
  --improvement-plan examples/agentic_training/iteration_ledgers/improvement_plan.json \
  --improvement-ledger examples/agentic_training/iteration_ledgers/improvement_ledger.json \
  --action-ledger examples/agentic_training/iteration_ledgers/action_ledger.json \
  --promotion-decision examples/agentic_training/promotion_governance/promotion_decision.json \
  --promotion-ledger examples/agentic_training/promotion_governance/promotion_ledger.json \
  --promotion-cards examples/agentic_training/promotion_governance/promotion_cards \
  --promotion-alias-apply examples/agentic_training/promotion_governance/promotion_alias_apply.json \
  --promotion-rollback-receipt examples/agentic_training/promotion_governance/promotion_rollback_receipt.json \
  --promotion-release-record examples/agentic_training/promotion_governance/promotion_release_record.json \
  --promotion-archive examples/agentic_training/promotion_governance/promotion_archive \
  --created-at 2026-07-03T00:00:00+00:00 \
  --out examples/agentic_training/loop_plan.json
```

The committed plan separates three states: `plan_readiness` is
`ready_to_execute`, `execution_completion` is `completed`, and
`governance_readiness` is `ready_for_review`. Its derived legacy `readiness` is
therefore `ready_for_governance_review`. This example binds a completed,
integrity-valid external result for every selected adapter and the exact same
result set in the eval summary; it does not infer execution from the dry-run
receipt. It also binds compare, promotion-history, promotion-decision,
promotion-ledger, rollback, alias-apply, release-record, and promotion-archive
receipts, plus a loop-local rollout plan
and mock receipt, harness/evidence handoff artifacts, a managed mock serving
lifecycle, nested model-grader review, rejection-sampling, dataset-curation, training-export,
trainer-preflight, trainer-launch-check, cloud-training, held-out eval,
action-ledger, improvement-ledger, and promotion-governance receipts without
Flight Recorder starting a provider call, dataset write, benchmark, scheduler,
or weight update.
The loop plan and governance receipt do not move aliases; the explicit
alias-apply receipt is a local JSON-registry fixture. The
`cloud_training_receipt_state` block is derived from the referenced launch and
status receipts, so forged loop summaries cannot hide provider API calls, cloud
jobs, cancellation calls, or incurred cost. The `cloud_training_lineage` block
records the SHA-256 chain from preflight through launch plan, launch receipt,
and status receipt. The example ledger's governance decision recommends
`approve`, while still listing `approve`, `reject`, `rollback`, and
`request_another_iteration` as schema-checkable action rows. Because the
rollback receipt is present, rollback is also available as a governance action.
Approval is a receipt-only governance choice; it does not move aliases or
update weights.

Validate the committed receipt before including it in a trainer-facing evidence
bundle:

```bash
flightrecorder validate \
  --agentic-training-result examples/agentic_training/completed_result.json \
  --strict

flightrecorder schemas --check examples/agentic_training/loop_plan.json
flightrecorder schemas --check examples/agentic_training/heldout_eval/external_eval_result.json
flightrecorder schemas --check examples/agentic_training/model_grader/reviewed_gate.json
flightrecorder schemas --check examples/agentic_training/training_gate.json
flightrecorder schemas --check examples/agentic_training/promotion_governance/compare_gate.json
flightrecorder validate \
  --external-eval-result examples/agentic_training/heldout_eval/external_eval_result.json \
  --strict

flightrecorder validate \
  --agentic-rollout-plan examples/agentic_training/rollouts/rollout_plan.json \
  --agentic-rollout-receipt examples/agentic_training/rollouts/rollout_receipt.json \
  --harness-result examples/agentic_training/evidence_handoff/harness_handoff/harness_result.json \
  --evidence-bundle examples/agentic_training/evidence_handoff/evidence_bundle.json \
  --rejection-sampling-gate examples/agentic_training/rejection_sampling_gate.json \
  --training-export examples/agentic_training/training_export \
  --dataset-curation-receipt examples/agentic_training/dataset_curation_receipt.json \
  --trainer-preflight examples/agentic_training/trainer_preflight.json \
  --trainer-launch-check examples/agentic_training/trainer_launch_check.json \
  --cloud-training-provider-registry examples/agentic_training/cloud_training/provider_registry.json \
  --cloud-training-preflight examples/agentic_training/cloud_training/preflight.json \
  --cloud-training-artifact-manifest examples/agentic_training/cloud_training/artifact_manifest.json \
  --cloud-training-launch-plan examples/agentic_training/cloud_training/launch_plan.json \
  --cloud-training-launch-receipt examples/agentic_training/cloud_training/launch_receipt.json \
  --cloud-training-status-receipt examples/agentic_training/cloud_training/status_receipt.json \
  --serving-lifecycle examples/agentic_training/serving_lifecycle/managed_mock/serving_lifecycle.json \
  --heldout-manifest examples/agentic_training/heldout_eval/heldout_manifest.json \
  --external-eval-plan examples/agentic_training/heldout_eval/external_eval_plan.json \
  --external-eval-receipt examples/agentic_training/heldout_eval/external_eval_receipt.json \
  --external-eval-result examples/agentic_training/heldout_eval/external_eval_result.json \
  --eval-summary examples/agentic_training/heldout_eval/eval_summary.json \
  --promotion-cards examples/agentic_training/promotion_governance/promotion_cards \
  --promotion-decision examples/agentic_training/promotion_governance/promotion_decision.json \
  --promotion-alias-apply examples/agentic_training/promotion_governance/promotion_alias_apply.json \
  --promotion-rollback-receipt examples/agentic_training/promotion_governance/promotion_rollback_receipt.json \
  --promotion-release-record examples/agentic_training/promotion_governance/promotion_release_record.json \
  --decision-gate examples/agentic_training/promotion_governance/promotion_history_decision_gate.json \
  --decision-gate examples/agentic_training/promotion_governance/promotion_decision_gate.json \
  --promotion-ledger examples/agentic_training/promotion_governance/promotion_ledger.json \
  --promotion-ledger-gate examples/agentic_training/promotion_governance/promotion_ledger_gate.json \
  --promotion-archive examples/agentic_training/promotion_governance/promotion_archive \
  --agentic-loop-plan examples/agentic_training/loop_plan.json \
  --strict

flightrecorder agentic-loop ledger \
  --plan examples/agentic_training/loop_plan.json \
  --out examples/agentic_training/loop_ledger.json

flightrecorder validate \
  --agentic-loop-ledger examples/agentic_training/loop_ledger.json \
  --strict

flightrecorder agentic-loop governance \
  --ledger examples/agentic_training/loop_ledger.json \
  --action approve \
  --requested-by example-governance \
  --reason "Demo ledger is ready for promotion review; this approval receipt records review readiness without provider, benchmark, alias, or weight side effects." \
  --created-at 2026-07-03T00:00:00+00:00 \
  --out examples/agentic_training/loop_governance_receipt.json

flightrecorder validate \
  --agentic-loop-governance-receipt examples/agentic_training/loop_governance_receipt.json \
  --strict

(cd examples/agentic_training/iteration_ledgers && \
  flightrecorder evidence-bundle \
    --runs runs \
    --gate failed_gate.json \
    --out evidence_bundle.json || test $? -eq 1)

(cd examples/agentic_training/iteration_ledgers && \
  flightrecorder action-ledger \
    --bundle evidence_bundle.json \
    --out action_ledger.json)

(cd examples/agentic_training/iteration_ledgers && \
  flightrecorder improvement-plan \
    --evidence-bundle evidence_bundle.json \
    --out improvement_plan.json)

(cd examples/agentic_training/iteration_ledgers && \
  flightrecorder improvement-ledger \
    --plan improvement_plan.json \
    --out improvement_ledger.json)

flightrecorder next-iteration-schedule \
  --loop-ledger examples/agentic_training/loop_ledger.json \
  --action-ledger examples/agentic_training/iteration_ledgers/action_ledger.json \
  --improvement-ledger examples/agentic_training/iteration_ledgers/improvement_ledger.json \
  --next-iteration-id demo-loop-002 \
  --objective "Resolve remaining ledgered repair pressure after governance review." \
  --schedule cadence=\"manual\" \
  --created-at 2026-07-03T00:00:00+00:00 \
  --out examples/agentic_training/next_iteration_schedule.json

flightrecorder validate \
  --next-iteration-schedule examples/agentic_training/next_iteration_schedule.json \
  --strict
```
