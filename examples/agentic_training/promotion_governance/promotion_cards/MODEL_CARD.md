# Model Card

- Candidate model: `local/mock-candidate`
- Source: `examples/agentic_training/completed_result.json`
- License status: `known`
- Promotion-card readiness: `ready`

## Required Evidence

- evidence_bundle: present; path `../../evidence_handoff/evidence_bundle.json`
- training_export: present; path `../../training_export`
- compare_gate: present; path `../compare_gate.json`
- redaction_check: present; path `../redaction_check.json`
- safety_gate: present; path `../safety_gate.json`

## Evaluation Movement

- Task-completion regressions: `0`
- New critical failures: `0`

## Limitations

- Promotion requires a separate validated promotion-decision artifact before aliases move.
- This card summarizes local governance evidence and should be regenerated when any referenced artifact changes.
