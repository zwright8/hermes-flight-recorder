#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

rm -rf runs
mkdir -p runs

python -m flightrecorder run-suite \
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
python -m flightrecorder improvement-plan \
  --evidence-bundle runs/evidence_bundle.json \
  --repair-queue runs/repair_queue.json \
  --training-export runs/training_export \
  --runs runs \
  --out runs/improvement_plan.json
python -m flightrecorder improvement-ledger \
  --plan runs/improvement_plan.json \
  --plan runs/improvement_plan.json \
  --out runs/improvement_ledger.json
python -m flightrecorder compare \
  --baseline runs/prompt_injection_good \
  --candidate runs/prompt_injection_bad \
  --out runs/prompt_injection_compare.json \
  --html-out runs/prompt_injection_compare.html
python -m flightrecorder compare-suite \
  --baseline runs \
  --candidate runs \
  --out runs/suite_compare.json \
  --html-out runs/suite_compare.html
python -m flightrecorder trend-suite \
  --suite-summary runs/suite_summary.json \
  --suite-summary runs/suite_summary.json \
  --out runs/suite_trend.json \
  --html-out runs/suite_trend.html
python -m flightrecorder action-ledger \
  --bundle runs/evidence_bundle.json \
  --bundle runs/evidence_bundle.json \
  --out runs/action_ledger.json
python -m flightrecorder gate-action-ledger \
  --action-ledger runs/action_ledger.json \
  --policy examples/action_ledger_gate_policy.demo.json \
  --out runs/action_ledger_gate.json
python -m flightrecorder gate-decision \
  --artifact runs/action_ledger_gate.json \
  --expect-recommendation promote_iteration \
  --expect-readiness ready \
  --require-passed \
  --out runs/promotion_decision.json
python -m flightrecorder promotion-ledger \
  --decision-gate runs/promotion_decision.json \
  --out runs/promotion_ledger.json
python -m flightrecorder gate-promotion-ledger \
  --promotion-ledger runs/promotion_ledger.json \
  --policy examples/promotion_ledger_gate_policy.demo.json \
  --out runs/promotion_ledger_gate.json
python -m flightrecorder promotion-archive \
  --promotion-ledger runs/promotion_ledger.json \
  --promotion-ledger-gate runs/promotion_ledger_gate.json \
  --decision-gate runs/promotion_decision.json \
  --out runs/promotion_archive \
  --require-self-contained \
  --force
echo "Demo reports written to $ROOT/runs/index.html"
