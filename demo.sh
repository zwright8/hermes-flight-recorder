#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

rm -rf runs
mkdir -p runs

python -m flightrecorder run --scenario scenarios/prompt_injection_good.json --out runs/prompt_injection_good
python -m flightrecorder run \
  --scenario scenarios/email_reply_completion_good.json \
  --out runs/email_reply_completion_good \
  --junit-out runs/email_reply_completion_good/scorecard.junit.xml \
  --markdown-out runs/email_reply_completion_good/scorecard.md
python -m flightrecorder run --scenario scenarios/prompt_injection_bad.json --out runs/prompt_injection_bad
python -m flightrecorder run --scenario scenarios/subagent_claim_bad.json --out runs/subagent_claim_bad
python -m flightrecorder run --scenario scenarios/budget_runaway_bad.json --out runs/budget_runaway_bad
python -m flightrecorder compare \
  --baseline runs/prompt_injection_good \
  --candidate runs/prompt_injection_bad \
  --out runs/prompt_injection_compare.json \
  --html-out runs/prompt_injection_compare.html
python -m flightrecorder export-rl --runs runs --out runs/training_export
python -m flightrecorder index --runs runs --out runs/index.html

echo "Demo reports written to $ROOT/runs/index.html"
