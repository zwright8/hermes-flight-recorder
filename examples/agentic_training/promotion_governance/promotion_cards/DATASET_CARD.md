# Dataset Card

- Dataset: `agentic-training-export`
- Candidate model: `local/mock-candidate`
- Training export: `../../training_export`
- Promotion-card readiness: `ready`

## Governance Inputs

- evidence_bundle: present; path `../../evidence_handoff/evidence_bundle.json`
- redaction_check: present; path `../redaction_check.json`
- safety_gate: present; path `../safety_gate.json`
- compare_gate: present; path `../compare_gate.json`

## Quality Signals

- Task-completion regressions: `0`
- New critical failures: `0`

## Use

- Use this dataset card only with the matching promotion_cards.json manifest.
- Regenerate the card if redaction, safety, evidence, training, or eval artifacts change.
