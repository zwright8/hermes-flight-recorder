#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

rm -rf runs
mkdir -p runs

python -m flightrecorder run --scenario scenarios/prompt_injection_good.json --out runs/prompt_injection_good
python -m flightrecorder run --scenario scenarios/prompt_injection_bad.json --out runs/prompt_injection_bad
python -m flightrecorder run --scenario scenarios/subagent_claim_bad.json --out runs/subagent_claim_bad
python -m flightrecorder run --scenario scenarios/budget_runaway_bad.json --out runs/budget_runaway_bad
python -m flightrecorder index --runs runs --out runs/index.html

echo "Demo reports written to $ROOT/runs/index.html"
