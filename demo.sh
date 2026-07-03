#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if [[ -z "${PYTHON:-}" ]]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON=python
  elif command -v python3.11 >/dev/null 2>&1; then
    PYTHON=python3.11
  else
    PYTHON=python3
  fi
fi
export PYTHON

rm -rf runs
mkdir -p runs

"$PYTHON" -m flightrecorder run-suite \
  --scenarios scenarios \
  --out runs \
  --junit \
  --markdown \
  --export-rl \
  --validate \
  --strict \
  --evidence-handoff \
  --metadata agent=hermes-fixture \
  --metadata candidate=offline-demo \
  --metadata eval_pack=core
"$PYTHON" -m flightrecorder improvement-plan \
  --evidence-bundle runs/evidence_bundle.json \
  --repair-queue runs/repair_queue.json \
  --training-export runs/training_export \
  --runs runs \
  --out runs/improvement_plan.json
"$PYTHON" -m flightrecorder improvement-ledger \
  --plan runs/improvement_plan.json \
  --plan runs/improvement_plan.json \
  --out runs/improvement_ledger.json
"$PYTHON" -m flightrecorder gate-improvement-ledger \
  --improvement-ledger runs/improvement_ledger.json \
  --policy examples/improvement_ledger_gate_policy.demo.json \
  --out runs/improvement_ledger_gate.json
"$PYTHON" -m flightrecorder compare \
  --baseline runs/prompt_injection_good \
  --candidate runs/prompt_injection_bad \
  --out runs/prompt_injection_compare.json \
  --html-out runs/prompt_injection_compare.html
"$PYTHON" -m flightrecorder compare-suite \
  --baseline runs \
  --candidate runs \
  --out runs/suite_compare.json \
  --html-out runs/suite_compare.html
"$PYTHON" -m flightrecorder trend-suite \
  --suite-summary runs/suite_summary.json \
  --suite-summary runs/suite_summary.json \
  --out runs/suite_trend.json \
  --html-out runs/suite_trend.html
"$PYTHON" -m flightrecorder action-ledger \
  --bundle runs/evidence_bundle.json \
  --bundle runs/evidence_bundle.json \
  --out runs/action_ledger.json
"$PYTHON" -m flightrecorder gate-action-ledger \
  --action-ledger runs/action_ledger.json \
  --policy examples/action_ledger_gate_policy.demo.json \
  --out runs/action_ledger_gate.json
"$PYTHON" -m flightrecorder gate-decision \
  --artifact runs/action_ledger_gate.json \
  --expect-recommendation promote_iteration \
  --expect-readiness ready \
  --require-passed \
  --out runs/promotion_decision.json
"$PYTHON" -m flightrecorder promotion-ledger \
  --decision-gate runs/promotion_decision.json \
  --out runs/promotion_ledger.json
"$PYTHON" -m flightrecorder gate-promotion-ledger \
  --promotion-ledger runs/promotion_ledger.json \
  --policy examples/promotion_ledger_gate_policy.demo.json \
  --out runs/promotion_ledger_gate.json
"$PYTHON" -m flightrecorder promotion-archive \
  --promotion-ledger runs/promotion_ledger.json \
  --promotion-ledger-gate runs/promotion_ledger_gate.json \
  --decision-gate runs/promotion_decision.json \
  --out runs/promotion_archive \
  --require-self-contained \
  --force
"$PYTHON" -m flightrecorder gate-export \
  --training-export runs/training_export \
  --policy examples/training_gate_policy.demo.json \
  --out runs/training_gate.json
"$PYTHON" -m flightrecorder trainer-preflight \
  --gate runs/training_gate.json \
  --training-export runs/training_export \
  --require-gate training_gate \
  --trainer-command "python train.py --dataset runs/training_export" \
  --out runs/trainer_preflight.json
"$PYTHON" -m flightrecorder trainer-launch-check \
  --preflight runs/trainer_preflight.json \
  --require-gate training_gate \
  --out runs/trainer_launch_check.json
"$PYTHON" -m flightrecorder trainer-archive \
  --preflight runs/trainer_preflight.json \
  --launch-check runs/trainer_launch_check.json \
  --out runs/trainer_archive \
  --require-self-contained \
  --force
mkdir -p runs/trainer_code
printf "print('trainer placeholder; Flight Recorder never executes this file')\n" > runs/trainer_code/train.py
"$PYTHON" -m flightrecorder trainer-archive-check \
  --archive runs/trainer_archive \
  --external-code-root runs/trainer_code \
  --out runs/trainer_archive_check.json \
  --strict
"$PYTHON" -m flightrecorder trainer-consumer-plan \
  --archive-check runs/trainer_archive_check.json \
  --out runs/trainer_consumer_plan.json \
  --strict
"$PYTHON" scripts/preflight_agentic_training_runtime.py \
  --plan examples/agentic_training/plans/sft_then_dpo_plan.json \
  --skip-default-modules \
  --require-module json \
  --out runs/agentic_training_runtime_preflight.json
"$PYTHON" -m flightrecorder agentic-training-flow \
  --plan examples/agentic_training/plans/sft_then_dpo_plan.json \
  --runtime-preflight runs/agentic_training_runtime_preflight.json \
  --trainer-consumer-plan runs/trainer_consumer_plan.json \
  --out runs/agentic_training_flow.json
"$PYTHON" examples/trainer-wrapper/consume_trainer_plan.py \
  --plan runs/trainer_consumer_plan.json \
  --out runs/trainer_wrapper_dry_run.json \
  --strict
"$PYTHON" -m flightrecorder evidence-bundle \
  --runs runs \
  --suite-summary runs/suite_summary.json \
  --scenario-quality runs/scenario_quality.json \
  --evidence-coverage runs/evidence_coverage.json \
  --trace-observability runs/trace_observability.json \
  --repair-queue runs/repair_queue.json \
  --validation runs/validation.json \
  --training-export runs/training_export \
  --trainer-preflight runs/trainer_preflight.json \
  --trainer-launch-check runs/trainer_launch_check.json \
  --trainer-archive runs/trainer_archive \
  --trainer-archive-check runs/trainer_archive_check.json \
  --trainer-consumer-plan runs/trainer_consumer_plan.json \
  --agentic-training-flow runs/agentic_training_flow.json \
  --trainer-wrapper-dry-run runs/trainer_wrapper_dry_run.json \
  --out runs/evidence_bundle_trainer.json
"$PYTHON" -m flightrecorder index --runs runs --out runs/index.html
echo "Demo reports written to $ROOT/runs/index.html"
