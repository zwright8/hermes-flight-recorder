# Harness Examples

This directory contains scrubbed public examples for the common harness layer.
They are safe to run from a fresh checkout without live provider credentials.

Run every checked-in scenario through the mock harness:

```bash
python3.11 scripts/hermes_harness.py run-suite \
  --scenarios harness/scenarios \
  --out runs/harness_examples \
  --relative-paths \
  --force
```

The suite is expected to include one passing run and one failing policy-canary
run. The failing run is intentional: it keeps blocked terminal, command, URL,
and replay behavior visible in generated artifacts.

Run one manifest directly:

```bash
python3.11 scripts/hermes_harness.py run-scenario \
  --manifest harness/mock_manifest.json \
  --relative-paths \
  --force
```

Replay a generated example:

```bash
python3.11 scripts/hermes_harness.py replay-trace \
  --lineage runs/harness_examples/harness_mock_success/artifact_lineage.json \
  --out runs/harness_examples/harness_mock_success_replay \
  --relative-paths
```
