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
cp examples/agentic_training/trainer_preflight.json \
  examples/agentic_training/cloud_training/sources/trainer_preflight.json
cp examples/agentic_training/trainer_launch_check.json \
  examples/agentic_training/cloud_training/sources/trainer_launch_check.json

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

Seed the held-out eval lane with fail-closed external adapter receipts:
the committed `baseline_suite_summary.json` and `candidate_suite_summary.json`
cover the held-out scenario IDs excluded from the training export.

```bash
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

flightrecorder eval-summary \
  --suite-summary baseline=examples/agentic_training/heldout_eval/baseline_suite_summary.json \
  --suite-summary candidate=examples/agentic_training/heldout_eval/candidate_suite_summary.json \
  --external-adapter-plan local_mock=examples/agentic_training/heldout_eval/external_eval_plan.json \
  --out examples/agentic_training/heldout_eval/eval_summary.json \
  --markdown-out examples/agentic_training/heldout_eval/eval_summary.md
```

The agentic-training fixture uses the built-in `local_mock` adapter so held-out
eval can pass offline while still proving zero provider API calls, model
downloads, benchmark launches, credential recording, cost, or weight updates.
Real BFCL, Inspect AI, lm-eval-harness, and SWE-bench adapters remain
fail-closed in the standalone `examples/external_eval/` fixtures until their
dependencies, inputs, and explicit opt-in flags are present.

Generate the evidence handoff harness result and bundle:

```bash
flightrecorder run-suite \
  --scenarios scenarios \
  --pattern prompt_injection_good.json \
  --out examples/agentic_training/evidence_handoff \
  --no-index \
  --evidence-handoff \
  --validate \
  --strict
```

The committed evidence handoff uses one deterministic passing scenario to keep
the public fixture compact. It emits `harness_handoff/harness_result.json` and
`evidence_bundle.json` without raw sensitive traces or live external calls.

Bind the example receipts into a fail-closed loop contract:

```bash
flightrecorder agentic-loop plan \
  --iteration-id demo-loop-001 \
  --objective "Demonstrate a fail-closed closed-loop agentic training iteration contract." \
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
  --trainer-preflight examples/agentic_training/trainer_preflight.json \
  --trainer-launch-check examples/agentic_training/trainer_launch_check.json \
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
  --created-at 2026-07-03T00:00:00+00:00 \
  --out examples/agentic_training/loop_plan.json
```

The committed plan is intentionally `planned_fail_closed` because this example
binds a blocked promotion decision and promotion ledger. It does bind a passing
offline held-out eval receipt through `local_mock`, loop-local rollout plan
and mock receipt, harness/evidence handoff artifacts, a managed mock serving
lifecycle, nested model-grader review, rejection-sampling, dataset-curation, training-export,
trainer-preflight, trainer-launch-check, cloud-training, held-out eval,
action-ledger, improvement-ledger, and promotion-governance receipts without provider, dataset-write,
benchmark-launch, or scheduler side effects. The
`cloud_training_receipt_state` block is derived from the referenced launch and
status receipts, so forged loop summaries cannot hide provider API calls, cloud
jobs, cancellation calls, or incurred cost. The `cloud_training_lineage` block
records the SHA-256 chain from preflight through launch plan, launch receipt,
and status receipt. Because this demo is intentionally incomplete, the example
ledger's governance decision recommends `request_another_iteration` while still
listing `approve`, `reject`, `rollback`, and `request_another_iteration` as
schema-checkable action rows.

Validate the committed receipt before including it in a trainer-facing evidence
bundle:

```bash
flightrecorder validate \
  --agentic-training-result examples/agentic_training/completed_result.json \
  --strict

flightrecorder schemas --check examples/agentic_training/loop_plan.json
flightrecorder schemas --check examples/agentic_training/model_grader/reviewed_gate.json
flightrecorder schemas --check examples/agentic_training/training_gate.json
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
  --eval-summary examples/agentic_training/heldout_eval/eval_summary.json \
  --promotion-decision examples/agentic_training/promotion_governance/promotion_decision.json \
  --decision-gate examples/agentic_training/promotion_governance/promotion_decision_gate.json \
  --promotion-ledger examples/agentic_training/promotion_governance/promotion_ledger.json \
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
  --action request_another_iteration \
  --requested-by example-governance \
  --reason "Demo ledger is fail-closed, so governance requests another iteration." \
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
  --objective "Collect missing closed-loop receipts and resolve ledgered repair pressure." \
  --schedule cadence=\"manual\" \
  --created-at 2026-07-03T00:00:00+00:00 \
  --out examples/agentic_training/next_iteration_schedule.json

flightrecorder validate \
  --next-iteration-schedule examples/agentic_training/next_iteration_schedule.json \
  --strict
```
