# Agentic Fine-Tune Governance

The governance layer converts cross-layer proof into a deterministic promotion
decision. The decision artifact is intentionally side-effect free: it can allow
promotion, but it does not train, serve, or move registry aliases.

## Promotion Decision

Use `promotion-decision` after evidence, data, model, training, serving, eval,
and safety gates have produced artifacts.

```bash
flightrecorder promotion-decision \
  --policy examples/promotion_policy.demo.json \
  --artifact evidence_bundle=path/to/evidence_bundle.json \
  --artifact dataset_manifest=path/to/manifest.json \
  --artifact dataset_card=path/to/DATASET_CARD.md \
  --artifact model_registry_entry=path/to/model_registry_entry.json \
  --artifact training_result=path/to/training_result.json \
  --artifact serving_profile=path/to/serving_profile.json \
  --artifact model_card=path/to/MODEL_CARD.md \
  --artifact rollback=path/to/rollback.json \
  --eval base=path/to/base/evaluation_summary.json \
  --eval trace_only=path/to/trace_only/evaluation_summary.json \
  --eval frontier=path/to/frontier/evaluation_summary.json \
  --eval champion=path/to/champion/evaluation_summary.json \
  --eval candidate=path/to/candidate/evaluation_summary.json \
  --gate training_gate=path/to/training_gate.json \
  --gate compare_gate=path/to/compare_gate.json \
  --gate safety_gate=path/to/safety_gate.json \
  --out path/to/promotion_decision.json
```

Then validate the saved decision:

```bash
flightrecorder validate --promotion-decision path/to/promotion_decision.json --strict
```

## Default Policy

`examples/promotion_policy.demo.json` requires:

- artifacts: evidence bundle, dataset manifest, dataset card, model registry
  entry, training result, serving profile, model card, rollback metadata
- eval arms: `base`, `trace_only`, `frontier`, `champion`, `candidate`
- gates: `training_gate`, `compare_gate`, `safety_gate`
- identical held-out scenario ids across every eval arm
- candidate pass rate and average score to beat or tie each baseline arm
- passed gates, passed evidence bundle, approved model license, declared passed
  redaction status, required card sections, and rollback target

Promotion is blocked for missing evidence, unknown license, redaction failure,
missing cards, missing rollback, eval mismatch, new critical failures, secret
exposure, forbidden actions, unsupported claims, task-completion regression, or
failed gates.

## Promotion Cards

Generate the required model and dataset cards before building the final
promotion decision:

```bash
flightrecorder promotion-cards \
  --policy examples/promotion_policy.demo.json \
  --out-dir path/to/governance/cards \
  --artifact dataset_manifest=path/to/manifest.json \
  --artifact model_registry_entry=path/to/model_registry_entry.json \
  --artifact training_result=path/to/training_result.json \
  --artifact serving_profile=path/to/serving_profile.json \
  --artifact rollback=path/to/rollback.json \
  --eval base=path/to/base/evaluation_summary.json \
  --eval trace_only=path/to/trace_only/evaluation_summary.json \
  --eval frontier=path/to/frontier/evaluation_summary.json \
  --eval champion=path/to/champion/evaluation_summary.json \
  --eval candidate=path/to/candidate/evaluation_summary.json \
  --gate training_gate=path/to/training_gate.json \
  --gate compare_gate=path/to/compare_gate.json \
  --gate safety_gate=path/to/safety_gate.json
```

The command writes `MODEL_CARD.md`, `DATASET_CARD.md`, and
`promotion_cards.json`. Validate the manifest before using the cards:

```bash
flightrecorder validate --promotion-cards path/to/governance/cards/promotion_cards.json --strict
```

Then pass the generated cards into `promotion-decision` as `model_card` and
`dataset_card` artifacts.

## Alias Movement

Registry alias movement should consume `promotion_decision.json` and only move
`candidate`, `champion`, or `rollback` aliases when:

- `passed` is `true`
- `decision.recommendation` is `promote_candidate`
- validation passes with `flightrecorder validate --promotion-decision --strict`
- the rollback artifact names a non-empty target

Before any side-effectful registry write, generate a dry-run alias receipt:

```bash
flightrecorder model-registry alias-receipt \
  --registry experiments/registry/model_registry.json \
  --promotion-decision path/to/promotion_decision.json \
  --alias champion \
  --target candidate-v2 \
  --rollback-target champion-v1 \
  --reason "promote candidate-v2 after governance approval" \
  --out path/to/registry_alias_receipt.json
```

Then validate the receipt:

```bash
flightrecorder validate --registry-alias-receipt path/to/registry_alias_receipt.json --strict
```

The receipt records the current registry fingerprint, promotion-decision
fingerprint, previous alias target, proposed target, rollback target, planned
alias-history rows, and a checks list. It never mutates the registry; a later
side-effectful alias writer must revalidate the receipt immediately before
applying the move.

## Release Records

Before final promotion, bind the passed decision, generated cards, alias
receipt, rollback target, eval summaries, and release notes into one release
record:

```bash
flightrecorder promotion-release-record \
  --release-id release-001 \
  --policy examples/promotion_policy.demo.json \
  --promotion-decision path/to/promotion_decision.json \
  --promotion-cards path/to/governance/cards/promotion_cards.json \
  --registry-alias-receipt path/to/registry_alias_receipt.json \
  --rollback path/to/rollback.json \
  --eval base=path/to/base/evaluation_summary.json \
  --eval trace_only=path/to/trace_only/evaluation_summary.json \
  --eval frontier=path/to/frontier/evaluation_summary.json \
  --eval champion=path/to/champion/evaluation_summary.json \
  --eval candidate=path/to/candidate/evaluation_summary.json \
  --out path/to/promotion_release_record.json \
  --notes-out path/to/RELEASE_NOTES.md
```

Validate it before any side-effectful promotion step:

```bash
flightrecorder validate --promotion-release-record path/to/promotion_release_record.json --strict
```

The release record is still side-effect free. It records component fingerprints
and generated release notes so a later promotion applier can revalidate exactly
what was approved.
