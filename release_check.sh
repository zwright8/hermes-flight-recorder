#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

cleanup_local_artifacts() {
  find . -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
  rm -rf hermes_flight_recorder.egg-info build dist
}
trap cleanup_local_artifacts EXIT

python -m unittest discover
python -m compileall -q flightrecorder scripts tests
python scripts/live_hermes_smoke.py --help >/dev/null
./demo.sh
test -f runs/email_reply_completion_good/scorecard.junit.xml
test -f runs/email_reply_completion_good/scorecard.md
test -f runs/prompt_injection_compare.json
test -f runs/prompt_injection_compare.html
test -f runs/training_export/episodes.jsonl
test -f runs/training_export/rewards.jsonl
test -f runs/training_export/preferences.jsonl
test -f runs/training_export/manifest.json
test -f runs/validation.json
python -m flightrecorder validate --runs runs --training-export runs/training_export --strict >/dev/null

python -m flightrecorder audit \
  --runs runs \
  --forbid-text hfr_fixture_secret_value_123 \
  --forbid-text DEMO_API_KEY=hfr_fixture \
  --fail-on-leak >/dev/null

INSTALL_DIR="$(mktemp -d)"
VENV_DIR="$(mktemp -d)"
python -m venv --system-site-packages "$VENV_DIR"
if ! "$VENV_DIR/bin/python" -c "import setuptools" >/dev/null 2>&1; then
  "$VENV_DIR/bin/python" -m pip install "setuptools>=68" >/dev/null
fi
"$VENV_DIR/bin/python" -m pip install . --no-deps --no-build-isolation >/dev/null
"$VENV_DIR/bin/flightrecorder" --help >/dev/null
"$VENV_DIR/bin/python" -m flightrecorder normalize \
  --trace fixtures/prompt_injection_good.trajectory.jsonl \
  --out "$INSTALL_DIR/normalized.json" >/dev/null
"$VENV_DIR/bin/python" -m flightrecorder observer-template \
  --out "$INSTALL_DIR/flight_recorder_plugin.py" >/dev/null

if "$VENV_DIR/bin/flightrecorder" run \
  --scenario scenarios/prompt_injection_bad.json \
  --out "$INSTALL_DIR/failing-run" \
  --fail-on-score >/dev/null; then
  echo "--fail-on-score did not fail a failing scenario" >&2
  exit 1
fi

rm -rf "$INSTALL_DIR" "$VENV_DIR"
echo "release check passed"
