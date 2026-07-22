# Tau-3 production inputs

`protocol_config.template.json` is deliberately not runnable as production.
Every `REPLACE_WITH_...` value must be resolved, both approval fields must be
set only after their evidence passes, and all local paths should remain under
an ignored `local/` directory.

After the model directories exist, create one content-addressed identity per
base, comparator, and optional teacher:

```bash
.venv/bin/python scripts/build_tau3_model_identity.py \
  --model-path local/tau3/models/base \
  --model-id '<frozen model id>' \
  --revision '<immutable revision>' \
  --out local/tau3/identities/base.json
```

Copy the reported identity-file SHA-256 into the matching model entry. The
identity covers every regular file in the model tree and rejects missing or
changed weights, tokenizer files, unrecorded additions, and symlinks.

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

Only then provide the separately reviewed canonical capture JSONL to the
production builder described in `docs/tau3-training-readiness.md`. Never commit
the populated local protocol, identity paths, raw captures, weights, or source
preflight; they are local evidence and may be sensitive.
