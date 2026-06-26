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
echo "Demo reports written to $ROOT/runs/index.html"
