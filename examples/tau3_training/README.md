# Tau-3 production inputs

`protocol_config.template.json` documents the contract and is deliberately not
runnable as production. The supported production flow generates that contract
from a pinned Tau checkout, partition manifests, reviewed capture JSONL, and
complete local model-tree identities; it does not fill the template by hand.
Every local path should remain under ignored `local/`.

After the model directories exist and their exact Hub revisions pass
`hf cache verify`, create one content-addressed identity per base, comparator,
and optional teacher:

```bash
.venv/bin/python scripts/build_tau3_model_identity.py \
  --model-path local/tau3/models/base \
  --model-id '<frozen model id>' \
  --revision '<immutable revision>' \
  --out local/tau3/identities/base.json
```

The identity covers every regular file in the model tree and rejects missing or
changed weights, tokenizer files, unrecorded additions, and symlinks. Generate
the source and capture evidence, then let the freezer bind the reported
identity hashes into `local/tau3/protocol.json`:

```bash
.venv/bin/python scripts/prepare_tau3_training_sources.py \
  --tau-repo local/tau3/repository \
  --expected-revision 1d244f5dca42944b67a379b44bfeb9f5748f189d \
  --out local/tau3/source-v1

.venv/bin/python scripts/generate_tau3_training_captures.py \
  --tau-repo local/tau3/repository \
  --expected-revision 1d244f5dca42944b67a379b44bfeb9f5748f189d \
  --train-tasks local/tau3/source-v1/training_source/train_tasks.jsonl \
  --development-tasks local/tau3/source-v1/training_source/development_tasks.jsonl \
  --tau-python local/tau3/venv/bin/python \
  --out local/tau3/captures-v1 \
  --seed 8675309 \
  --sample-salt hfr-tau3-core-agent-study-v1
```

Use the exact protocol-freeze command in
[`docs/tau3-training-readiness.md`](../../docs/tau3-training-readiness.md); it
binds all source paths, the hashes-only sealed manifest, three identities, the
fixed harness, budget, recipe bounds, and non-executing MLX command.

Check all sources before generating any bundle:

```bash
.venv/bin/python scripts/check_tau3_training_sources.py \
  --config local/tau3/protocol.json \
  --out local/tau3/source_preflight.json
```

This preflight is read-only except for the optional new receipt. It does not
clone Tau, download models, contact a provider, generate examples, or train.
It fails until the Tau commit, split hashes, complete model-tree identities,
MLX-LM runtime, contamination/redaction attestations, licenses, and frozen
local launch command all pass.

Only then provide the reviewed canonical capture JSONL to the production
builder described in the guide. The resulting MLX input has only `train.jsonl`
and `valid.jsonl`; there is no test or sealed trainer view. The production
builder also renders every row with the pinned local tokenizer and blocks the
launch handoff if any row exceeds the frozen sequence/context budget. Never
commit the populated local protocol, identity paths, raw captures, weights, or source
preflight; they are local evidence and may be sensitive.
